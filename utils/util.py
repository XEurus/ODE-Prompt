"""
工具函数模块

本模块提供了模型加载、图像预处理和数学工具函数。
支持多种预训练模型和归一化方案。

主要功能：
    - 模型加载：支持 ResNet, ViT, CLIP, ALBEF, MAE 等
    - 图像归一化：CLIP, ImageNet, 通用归一化方案
    - 对抗样本工具：L2/Linf 范数约束

支持的模型：
    - 标准模型: ResNet18/50/101, ViT-B16, EfficientNet, MobileNet, DenseNet
    - CLIP 模型: RN50, ViT-B/16, ViT-B/32
    - 自监督模型: MAE, MoCo, SimCLR
    - 多模态模型: ALBEF, BEiT

归一化参数：
    - clip: 使用 CLIP 预训练时的归一化参数
    - imagenet: 使用 ImageNet 预训练时的归一化参数
    - general: 通用 [0.5, 0.5, 0.5] 归一化
"""

import torch
from .enumType import NormType
from torchvision import transforms
import timm
import ruamel.yaml as yaml
from pathlib import Path
from torchvision.models import ResNet101_Weights, ResNet50_Weights, ViT_B_16_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights, DenseNet121_Weights
from torchvision._internally_replaced_utils import load_state_dict_from_url


# 不同预训练模型的归一化参数
normalize_list = {
    'clip': transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],   # CLIP 训练数据均值
        [0.26862954, 0.26130258, 0.27577711]   # CLIP 训练数据标准差
    ),
    'general': transforms.Normalize(
        [0.5, 0.5, 0.5],                       # 通用归一化
        [0.5, 0.5, 0.5]
    ),
    'imagenet': transforms.Normalize(
        [0.485, 0.456, 0.406],                 # ImageNet 数据均值
        [0.229, 0.224, 0.225]                  # ImageNet 数据标准差
    )
}


def clamp_by_l2(x, max_norm):
    """
    将张量的 L2 范数限制在 max_norm 以内
    
    参数：
        x: 输入张量，形状 (batch, channels, height, width)
        max_norm: 最大 L2 范数
    
    返回：
        缩放后的张量，保证 ||x||_2 <= max_norm
    
    公式：
        x' = x * min(1, max_norm / ||x||_2)
    """
    norm = torch.norm(x, dim=(1,2,3), p=2, keepdim=True)
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    return x * factor


def random_init(x, norm_type, epsilon):
    """
    对抗扰动的随机初始化
    
    参数：
        x: 原始图像张量
        norm_type: 范数类型 (NormType.Linf 或 NormType.L2)
        epsilon: 扰动预算
    
    返回：
        delta: 初始扰动张量
    
    初始化策略：
        - Linf: 均匀分布 [0, epsilon]
        - L2: 均匀分布后约束到 L2 球内
    """
    delta = torch.zeros_like(x)
    if norm_type == NormType.Linf:
        delta.data.uniform_(-1.0, 1.0)
        delta.data = delta.data * epsilon
    elif norm_type == NormType.L2:
        delta.data.uniform_(-1.0, 1.0)
        delta.data = delta.data - x
        delta.data = clamp_by_l2(delta.data, epsilon)
    return delta


def is_image_file(filename):
    """
    检查文件是否为图像文件
    
    参数：
        filename: 文件名
    
    返回：
        bool: 是否为支持的图像格式
    """
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_model(name, num_classes=2, model_config='config/surrogate.yaml'):
    """
    获取预训练模型
    
    参数：
        name: 模型名称
        num_classes: 分类头输出类别数（用于微调）
        model_config: 模型配置文件路径
    
    返回：
        model: 加载了预训练权重的模型
    
    支持的模型：
        标准视觉模型：
            - resnet18, resnet50, resnet101
            - efficientnet_b0, mobilenet_v2, densenet121
            - ViT-B16
        
        CLIP 模型：
            - Clip-RN50, Clip-ViT-B16, Clip-ViT-B32
            - Clip-RN50-arcface (微调版)
        
        其他预训练模型：
            - BeiT_v2-B16, MAE-ViT-B16
            - ALBEF-ViT-B16
            - SimCLR-RN50, MoCo-ViT-B16
    
    使用示例：
        >>> model = get_model('resnet50', num_classes=1000)
        >>> model = get_model('Clip-ViT-B16', num_classes=100)
    """
    model_config = read_yaml(model_config)
    
    # 根据名称创建模型
    if name == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif name == 'resnet101':
        model = resnet101(num_classes=num_classes)
    elif name == 'resnet50':
        model = resnet50(num_classes=num_classes)
    elif name == 'efficientnet_b0':
        model = efficientnet_b0(num_classes=num_classes)
    elif name == 'mobilenet_v2':
        model = mobilenet_v2(num_classes=num_classes)
    elif name == 'densenet121':
        model = densenet121(num_classes=num_classes)
    elif name == 'regnet_x_1_6gf':
        model = regnet_x_1_6gf(num_classes=num_classes)
    elif name == 'ViT-B16':
        model = vit_b_16(num_classes=num_classes)
    elif name == 'Clip-RN50':
        model = ClipResnet(name='RN50', num_classes=num_classes)
    elif name == 'Clip-RN50-arcface':
        model = ClipModel(name='RN50', num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/clip/RN50.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'Clip-RN50x64-arcface':
        model = ClipModel(name='RN50x64', num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/clip/RN50x64.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'resnet50-arcface':
        model = resnet50(num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/imagenet/resnet50.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'resnet50-scratch':
        model = resnet50(num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/scratch/resnet50.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'Clip-ViT-B32':
        model = ClipViT(name='ViT-B/32', num_classes=num_classes)
    elif name == 'Clip-ViT-B16-arcface':
        model = ClipViT(name='ViT-B/16', num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/clip/ViT-B-16.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'OpenClip-ViT-H14-arcface':
        model = ClipOpen(name='ViT-H-14', num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/clip/ViT-H-14.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'convnext_xxlarge-arcface':
        model = ClipOpen(name='hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup', num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/open_clip/convnext_xxlarge.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'Clip-ViT-B16':
        model = ClipViT(name='ViT-B/16', num_classes=num_classes)
    elif name == 'BeiT_v2-B16':
        model = BeitViT(name='beitv2_base_patch16_224', num_classes=num_classes)
    elif name == 'ALBEF-ViT-B16':
        model = ALBEFViT(name='ViT-B/16', num_classes=num_classes)
    elif name == 'ALBEF-VE-ViT-B16':
        model = ALBEFViT(name='ViT-B/16', num_classes=1000)
    elif name == 'MAE-ViT-B16':
        model = MAEViT(name='ViT-B/16', num_classes=num_classes)
    elif name == 'SimCLR-RN50':
        model = resnet50(num_classes=num_classes)
    elif name == 'MoCo-ViT-B16':
        model = MoCoViT(name='ViT-B/16', num_classes=num_classes)
    else:
        raise (f'Model {name} Not Found')

    # 加载预训练权重
    if name == 'resnet101':
        state_dict = load_state_dict_from_url(ResNet101_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['fc.weight'], state_dict['fc.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'resnet50':
        state_dict = load_state_dict_from_url(ResNet50_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['fc.weight'], state_dict['fc.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'ViT-B16':
        state_dict = load_state_dict_from_url(ViT_B_16_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['heads.head.weight'], state_dict['heads.head.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'efficientnet_b0':
        state_dict = load_state_dict_from_url(EfficientNet_B0_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['classifier.weight'], state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'mobilenet_v2':
        state_dict = load_state_dict_from_url(MobileNet_V2_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['classifier.weight'], state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'densenet121':
        state_dict = load_state_dict_from_url(DenseNet121_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['classifier.weight'], state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'Clip-RN50':
        model.load_pretrain(tmp_dir='cache')
    elif name == 'Clip-ViT-B32':
        model.load_pretrain(tmp_dir='cache')
    elif name == 'Clip-ViT-B16':
        model.load_pretrain(tmp_dir='cache')
    elif name == 'BeiT_v2-B16':
        visual_encoder = timm.models.create_model('beitv2_base_patch16_224', pretrained=True, num_classes=0)
        model.visual_encoder = visual_encoder
    elif name == 'ALBEF-ViT-B16':
        state_dict = torch.load(model_config[name]['path'], map_location='cpu')['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        model.load_state_dict(state_dict, strict=False)
    elif name == 'ALBEF-VE-ViT-B16':
        state_dict = torch.load(model_config[name]['path'], map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'MAE-ViT-B16':
        state_dict = torch.load(model_config[name]['path'], map_location='cpu')['model']
        model.visual_encoder.load_state_dict(state_dict, strict=False)
    elif name == 'SimCLR-RN50':
        state_dict = torch.load(model_config[name]['path'], map_location='cpu')['state_dict']
        new_state_dict = model.state_dict()
        del new_state_dict['fc.weight'], new_state_dict['fc.bias']
        for k in new_state_dict.keys():
            new_state_dict[k] = state_dict['convnet.'+ k]
        model.load_state_dict(new_state_dict, strict=False)
    elif name == 'MoCo-ViT-B16':
        state_dict = torch.load(model_config[name]['path'], map_location='cpu')['state_dict']
        new_state_dict = model.visual_encoder.state_dict()
        del new_state_dict['head.weight'], new_state_dict['head.bias']
        for k in new_state_dict.keys():
            new_state_dict[k] = state_dict['module.base_encoder.'+ k]
        model.load_state_dict(new_state_dict, strict=False)

    return model


def read_yaml(path):
    """
    读取 YAML 配置文件
    
    参数：
        path: YAML 文件路径
    
    返回：
        解析后的配置字典
    """
    return yaml.load(open(path, 'r'), Loader=yaml.Loader)


def dir_check(path):
    """
    检查并创建目录
    
    如果目录不存在则创建，包括所有父目录。
    
    参数：
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def distance(A, B):
    """
    计算两组向量之间的欧几里得距离矩阵
    
    参数：
        A: 第一组向量，形状 (m, d)
        B: 第二组向量，形状 (n, d)
    
    返回：
        距离矩阵，形状 (m, n)
        dist[i,j] = ||A[i] - B[j]||^2
    
    公式：
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
    """
    prod = A @ B.T

    prod_A = A @ A.T
    norm_A = prod_A.diag().unsqueeze(1).expand_as(prod)

    prod_B = B @ B.T
    norm_B = prod_B.diag().unsqueeze(0).expand_as(prod)

    res = norm_A + norm_B - 2 * prod
    return res


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    """
    插值位置嵌入
    
    当预训练模型和目标模型的图像尺寸不同时，需要对位置嵌入进行插值。
    
    参数：
        pos_embed_checkpoint: 检查点中的位置嵌入
        visual_encoder: 目标视觉编码器
    
    返回：
        插值后的位置嵌入
    
    处理逻辑：
        1. 保留 class token 和 dist token 不变
        2. 对位置 token 进行双三次插值
        3. 拼接返回
    """
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    
    # 原始和目标尺寸
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # 分离 class token 和位置 token
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        
        # 重塑并插值
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        
        # 拼接
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d' % (orig_size ** 2, new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint
