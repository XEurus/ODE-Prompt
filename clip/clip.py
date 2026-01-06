"""
CLIP (Contrastive Language-Image Pre-training) 模型加载和工具函数

本模块提供了 CLIP 模型的加载、预处理和标记化功能。
这是 ODE-Prompt 框架使用的核心视觉-语言预训练模型。

主要功能：
    - available_models(): 列出所有可用的 CLIP 模型
    - load(): 加载指定的 CLIP 模型和预处理器
    - tokenize(): 将文本转换为 CLIP 可处理的 token 序列

支持的模型架构：
    - ResNet 系列: RN50, RN101, RN50x4, RN50x16, RN50x64
    - Vision Transformer 系列: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px

CLIP 模型特点：
    - 在 4 亿图像-文本对上预训练
    - 强大的零样本迁移能力
    - 对齐的视觉和文本特征空间

使用示例：
    >>> import clip
    >>> model, preprocess = clip.load("ViT-B/16")
    >>> image = preprocess(PIL_image).unsqueeze(0)
    >>> text = clip.tokenize(["a dog", "a cat"])
    >>> image_features = model.encode_image(image)
    >>> text_features = model.encode_text(text)
"""

import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
import packaging.version

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

# 处理 torchvision 版本兼容性
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# PyTorch 版本检查
if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]

# 全局 tokenizer 实例
_tokenizer = _Tokenizer()

# 可用的 CLIP 模型及其下载 URL
# 格式: {模型名称: 模型文件 URL}
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    """
    下载模型文件到指定目录
    
    参数：
        url: 模型文件的 URL
        root: 下载目标目录
    
    返回：
        str: 下载文件的本地路径
    
    功能：
        - 自动创建目标目录
        - 使用 SHA256 校验文件完整性
        - 支持断点续传（如果文件存在但校验失败则重新下载）
        - 显示下载进度条
    """
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    # 从 URL 中提取预期的 SHA256 校验和
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    # 检查目标路径是否为有效文件
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    # 如果文件已存在，验证校验和
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    # 下载文件并显示进度
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    # 验证下载文件的校验和
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    """
    将图像转换为 RGB 格式
    
    处理可能的 RGBA、L（灰度）等格式
    """
    return image.convert("RGB")


def _transform(n_px):
    """
    构建 CLIP 标准图像预处理管道
    
    参数：
        n_px: 目标图像尺寸（像素）
    
    返回：
        Compose: torchvision 预处理管道
    
    预处理步骤：
        1. Resize: 将短边缩放到 n_px，保持宽高比
        2. CenterCrop: 中心裁剪到 n_px x n_px
        3. RGB 转换: 确保图像为 RGB 格式
        4. ToTensor: 转换为 PyTorch 张量
        5. Normalize: 使用 CLIP 的归一化参数
    
    归一化参数说明：
        - mean: [0.48145466, 0.4578275, 0.40821073] - CLIP 预训练数据的均值
        - std: [0.26862954, 0.26130258, 0.27577711] - CLIP 预训练数据的标准差
    """
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """
    返回所有可用的 CLIP 模型名称
    
    返回：
        List[str]: 模型名称列表
    
    示例：
        >>> clip.available_models()
        ['RN50', 'RN101', 'RN50x4', ..., 'ViT-L/14@336px']
    """
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """
    加载 CLIP 模型
    
    参数：
        name: 模型名称（如 'ViT-B/16'）或本地模型路径
        device: 运行设备 ('cuda' 或 'cpu')
        jit: 是否加载 JIT 优化版本
        download_root: 模型下载目录，默认为 ~/.cache/clip
    
    返回：
        model: CLIP 模型实例 (torch.nn.Module)
        preprocess: 图像预处理函数 (Callable[[PIL.Image], torch.Tensor])
    
    使用示例：
        >>> model, preprocess = clip.load("ViT-B/16", device="cuda")
        >>> image = preprocess(PIL.Image.open("image.jpg")).unsqueeze(0).to("cuda")
        >>> text = clip.tokenize(["a dog", "a cat"]).to("cuda")
        >>> 
        >>> with torch.no_grad():
        >>>     image_features = model.encode_image(image)
        >>>     text_features = model.encode_text(text)
        >>>     
        >>> # 计算相似度
        >>> logits_per_image = (image_features @ text_features.T).softmax(dim=-1)
    """
    # 确定模型路径
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # 加载模型文件
    with open(model_path, 'rb') as opened_file:
        try:
            # 尝试加载 JIT 归档
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # 加载普通状态字典
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    # 非 JIT 模式：构建模型并加载权重
    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # JIT 模式：修补设备名称
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """
        获取节点属性的多态函数
        
        来源: https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        """修补模块中的设备引用"""
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # CPU 模式：将 dtype 修补为 float32
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            """修补模块中的数据类型"""
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    将文本转换为 CLIP 可处理的 token 序列
    
    参数：
        texts: 单个字符串或字符串列表
        context_length: 上下文长度，CLIP 使用 77
        truncate: 是否截断过长的文本
    
    返回：
        tokens: 形状为 (n_texts, context_length) 的 token 张量
    
    Token 格式：
        [SOT, token1, token2, ..., EOT, 0, 0, ...]
        - SOT: Start of Text token
        - EOT: End of Text token
        - 0: Padding token
    
    示例：
        >>> tokens = clip.tokenize("a photo of a dog")
        >>> tokens.shape
        torch.Size([1, 77])
        >>> 
        >>> tokens = clip.tokenize(["a dog", "a cat"])
        >>> tokens.shape
        torch.Size([2, 77])
    """
    # 统一处理为列表
    if isinstance(texts, str):
        texts = [texts]

    # 获取特殊 token
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    
    # 编码所有文本
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    # 根据 PyTorch 版本选择数据类型
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    # 填充 token 序列
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                # 截断并保留 EOT
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
