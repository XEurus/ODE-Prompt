import os
import os.path as osp
import torchvision
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import copy
import time

from dass.engine import TRAINER_REGISTRY, TrainerX
from dass.metrics import compute_accuracy
from dass.utils import load_pretrained_weights, load_checkpoint
from dass.optim import build_optimizer, build_lr_scheduler
from utils.adv_utils import ImageNormalizer, get_model

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# ============================================================================
# ResNet10-Prompt: 基于 ResNet-10 的对抗提示学习
# 核心思想: p(T) = p(0) + ResNet10_θ([p(0); z_v])，其中 z_v 是对抗图像的视觉特征
# ============================================================================


# CUSTOM_TEMPLATES = {
#     "OxfordPets": "a photo of a {}, a type of pet.",
#     "OxfordFlowers": "a photo of a {}, a type of flower.",
#     "FGVCAircraft": "a photo of a {}, a type of aircraft.",
#     "DescribableTextures": "{} texture.",
#     "EuroSAT": "a centered satellite photo of {}.",
#     "StanfordCars": "a photo of a {}.",
#     "Food101": "a photo of {}, a type of food.",
#     "SUN397": "a photo of a {}.",
#     "Caltech101": "a photo of a {}.",
#     "UCF101": "a photo of a person doing {}.",
#     "ImageNet": "a photo of a {}.",
#     "ImageNetSketch": "a photo of a {}.",
#     "ImageNetV2": "a photo of a {}.",
#     "ImageNetA": "a photo of a {}.",
#     "ImageNetR": "a photo of a {}.",
# }

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, '/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/clip')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # 启用 gradient checkpointing 以节省显存
        if self.training:
            # 必须用 torch.utils.checkpoint.checkpoint 包装变压器
            # 为了使用 checkpointing，我们需要确保 x requires_grad
            if not x.requires_grad:
                x.requires_grad_(True)
            x = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
        else:
            x = self.transformer(x)
            
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class ResidualBlock(nn.Module):
    """ResNet 基本残差块 (1D 版本)"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class PromptResNet10(nn.Module):
    """
    ResNet-10 提示变换网络 f_θ
    
    公式:
        p(T) = p(0) + f_θ([p(0); z_v])
    
    其中:
        - p(0): 初始提示状态，形状 (n_ctx, dim)
        - z_v: 对抗图像的视觉特征，形状 (batch_size, dim)
    
    网络结构 (10 层):
        - 1 个输入投影层
        - 4 个残差块 (每块 2 层，共 8 层)
        - 1 个输出投影层
    """
    def __init__(self, prompt_dim, visual_dim):
        super(PromptResNet10, self).__init__()
        self.prompt_dim = prompt_dim
        self.visual_dim = visual_dim
        
        self.hidden_dim = prompt_dim * 2
        
        # 输入投影 (第 1 层)
        self.input_proj = nn.Linear(prompt_dim + visual_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.act = nn.GELU()
        
        # 4 个残差块 (第 2~9 层, 每块 2 层)
        self.res_blocks = nn.Sequential(
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim),
        )
        
        # 输出投影 (第 10 层)
        self.output_proj = nn.Linear(self.hidden_dim, prompt_dim)
        
        # 零初始化输出层，使初始时网络输出接近零（残差连接保持 p(0)）
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, p, z_v):
        """
        ResNet-10 前向传播: p(T) = p(0) + f_θ([p(0); z_v])
        
        参数:
            p: 初始提示状态，形状 (batch_size, n_ctx, prompt_dim)
            z_v: 视觉特征，形状 (batch_size, visual_dim)
        
        返回:
            p_out: 变换后的提示，形状 (batch_size, n_ctx, prompt_dim)
        """
        # 将 z_v 扩展到与 p 的 n_ctx 维度匹配
        # 扩展后形状: (batch_size, n_ctx, visual_dim)
        z_v_expanded = z_v.unsqueeze(1).expand(-1, p.shape[1], -1)
        
        # 拼接: [p(0); z_v]
        # 形状: (batch_size, n_ctx, prompt_dim + visual_dim)
        inp = torch.cat([p, z_v_expanded], dim=-1)
        
        # ResNet-10 前向传播
        x = self.input_proj(inp)
        x = self.input_norm(x)
        x = self.act(x)
        x = self.res_blocks(x)
        
        # 输出层 + 残差连接
        delta = self.output_proj(x)
        p_out = p + delta
        
        return p_out


class PromptLearner(nn.Module):
    """
    ResNet10-Prompt 的提示学习器
    
    核心改变:
        - 原始 AdvPT: ctx 是可学习参数，直接用于 prompt
        - ResNet10-Prompt: ctx 作为固定初始状态 p(0)，通过 ResNet-10 变换得到 p(T)
    
    流程:
        1. p(0) = "a photo of a" 的文本嵌入 (固定)
        2. 获取视觉特征 z_v = E_v(x_adv)
        3. ResNet-10 变换: p(T) = p(0) + ResNet10_θ([p(0); z_v])
        4. 拼接: [SOS, p(T), class_name, EOS]
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.ADV.N_CTX
        ctx_init = cfg.TRAINER.ADV.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # prompt 嵌入维度 (e.g., 512 for ViT-B/16)
        visual_dim = clip_model.visual.output_dim  # 视觉特征维度 (e.g., 512 for ViT-B/16)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # =========================================================
        # 初始化 p(0): 固定为 "a photo of a" 的文本嵌入
        # =========================================================
        if ctx_init:
            # 使用给定的词语初始化 (e.g., "a photo of a")
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            # 提取 token 嵌入 (跳过 SOS token)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]  # 形状: (n_ctx, ctx_dim)
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.ADV.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        # else:
        #     # 使用随机正态分布初始化 (Random Initialization)
        #     n_ctx = cfg.TRAINER.ADV.N_CTX
        #     ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        #     nn.init.normal_(ctx_vectors, std=0.02)
        #     prompt_prefix = "<random_init>"

        print(f'[ResNet10-Prompt] Initial prompt p(0): "{prompt_prefix}"')
        print(f"[ResNet10-Prompt] Number of context tokens: {n_ctx}")
        print(f"[ResNet10-Prompt] Prompt dimension: {ctx_dim}, Visual dimension: {visual_dim}")

        # =========================================================
        # p(0) 作为固定的 buffer，不参与梯度更新
        # 这是与原始 AdvPT 的关键区别！
        # =========================================================
        self.register_buffer("p0", ctx_vectors)  # 固定初始状态
        
        # =========================================================
        # ResNet-10 变换网络 f_θ (这是唯一的可学习部分!)
        # =========================================================
        self.resnet_func = PromptResNet10(ctx_dim, visual_dim).float()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.ADV.CLASS_TOKEN_POSITION


    def forward(self, z_v):
        """
        ResNet10-Prompt 的前向传播
        
        参数:
            z_v: 对抗图像的视觉特征，形状 (batch_size, visual_dim)
                 来自 Adversarial Embedding Bank
        
        返回:
            prompts: 完整的提示嵌入，形状 (batch_size, n_cls, seq_len, dim)
        
        流程:
            1. 获取初始提示 p(0)
            2. ResNet-10 变换: p(T) = p(0) + ResNet10_θ([p(0); z_v])
            3. 拼接 prompt: [SOS, p(T), class_name, EOS]
        """
        # =========================================================
        # Step 1: 获取 batch size
        # =========================================================
        bs = z_v.shape[0]
        
        # =========================================================
        # Step 2: ResNet-10 变换 - 核心步骤!
        # =========================================================
        # 初始状态: p(0) = "a photo of a" 的嵌入
        # 形状: (n_ctx, prompt_dim) -> (batch_size, n_ctx, prompt_dim)
        p0 = self.p0.unsqueeze(0).expand(bs, -1, -1)
        
        # 将输入转换为 float32 以匹配 ResNet-10 网络
        p0_float = p0.float()
        z_v_float = z_v.float()
        
        # ResNet-10 变换: p(T) = p(0) + ResNet10_θ([p(0); z_v])
        ctx = self.resnet_func(p0_float, z_v_float)
        
        # 将输出转回原始 dtype (与 prefix/suffix 匹配)
        ctx = ctx.type(self.p0.dtype)
        
        # =========================================================
        # Step 3: 扩展到所有类别
        # =========================================================
        # ctx 形状: (batch_size, n_ctx, prompt_dim) -> (batch_size, n_cls, n_ctx, prompt_dim)
        ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        # =========================================================
        # Step 4: 拼接完整 prompt
        # =========================================================
        # prefix: SOS token，形状 (n_cls, 1, dim) -> (batch_size, n_cls, 1, dim)
        # suffix: [class_name, EOS]，形状 (n_cls, *, dim) -> (batch_size, n_cls, *, dim)
        prefix = self.token_prefix.unsqueeze(0).expand(bs, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(bs, -1, -1, -1)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (bs, n_cls, 1, dim)
                    ctx,     # (bs, n_cls, n_ctx, dim)
                    suffix,  # (bs, n_cls, *, dim)
                ],
                dim=2,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:, i : i + 1, :, :]
                class_i = suffix[:, i : i + 1, :name_len, :]
                suffix_i = suffix[:, i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[:, i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[:, i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (bs, 1, 1, dim)
                        ctx_i_half1,  # (bs, 1, n_ctx//2, dim)
                        class_i,      # (bs, 1, name_len, dim)
                        ctx_i_half2,  # (bs, 1, n_ctx//2, dim)
                        suffix_i,     # (bs, 1, *, dim)
                    ],
                    dim=2,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=1)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:, i : i + 1, :, :]
                class_i = suffix[:, i : i + 1, :name_len, :]
                suffix_i = suffix[:, i : i + 1, name_len:, :]
                ctx_i = ctx[:, i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (bs, 1, 1, dim)
                        class_i,   # (bs, 1, name_len, dim)
                        ctx_i,     # (bs, 1, n_ctx, dim)
                        suffix_i,  # (bs, 1, *, dim)
                    ],
                    dim=2,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=1)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    """
    ResNet10-Prompt 的自定义 CLIP 模型
    
    组件:
        - prompt_learner: ResNet10-Prompt 提示学习器
        - image_encoder: 冻结的 CLIP 图像编码器
        - text_encoder: 冻结的 CLIP 文本编码器
    
    可学习部分:
        - 仅 prompt_learner.resnet_func (即 ResNet-10 变换网络 f_θ)
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        """
        正常推理模式 (使用干净图像)
        
        流程:
            1. 编码图像 -> 视觉特征
            2. ResNet-10 变换提示 (z_v 作为条件)
            3. 编码提示 -> 文本特征
            4. 计算相似度
        """
        # Step 1: 编码图像
        image_features = self.image_encoder(image.type(self.dtype))
        
        # Step 2: ResNet-10 变换提示
        # 注意: 这里使用干净图像的特征作为 z_v
        # 在训练时，会使用 forward_embedding 传入对抗特征
        prompts = self.prompt_learner(image_features) # (bs, n_cls, seq_len, dim)
        
        # Step 3: 编码提示
        # 需要将 prompts 和 tokenized_prompts 展平以适配 TextEncoder
        bs, n_cls, n_ctx, dim = prompts.shape
        prompts_flat = prompts.reshape(bs * n_cls, n_ctx, dim)
        
        tokenized_prompts = self.tokenized_prompts # (n_cls, seq_len)
        tokenized_prompts_flat = tokenized_prompts.unsqueeze(0).expand(bs, -1, -1).reshape(bs * n_cls, -1)
        
        text_features = self.text_encoder(prompts_flat, tokenized_prompts_flat) # (bs*n_cls, dim)
        text_features = text_features.view(bs, n_cls, -1) # (bs, n_cls, dim)

        # Step 4: 归一化并计算相似度
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (bs, dim)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # (bs, n_cls, dim)

        logit_scale = self.logit_scale.exp()
        
        # 计算相似度: (bs, dim) 与 (bs, n_cls, dim) 的交互
        # logits[i, j] = image_features[i] @ text_features[i, j]
        logits = logit_scale * torch.einsum("bd,bnd->bn", image_features, text_features)

        return logits

    def forward_embedding(self, image_features):
        """
        使用预存的对抗嵌入进行训练 (来自 Embedding Bank)
        
        这是 ResNet10-Prompt 训练的核心方法!
        
        参数:
            image_features: 对抗图像嵌入，形状 (batch_size, visual_dim)
                           来自 Adversarial Embedding Bank
        
        返回:
            logits: 预测分数，形状 (batch_size, n_cls)
        
        流程:
            1. 将对抗嵌入 z_v 传给 ResNet-10 网络
            2. ResNet-10 变换: p(T) = p(0) + ResNet10_θ([p(0); z_v])
            3. 编码提示 -> 文本特征
            4. 计算对抗嵌入与文本特征的相似度
        """
        # 类型转换
        image_features = image_features.type(self.dtype)
        
        # =========================================================
        # 核心: 将对抗图像嵌入传给 PromptLearner (ResNet-10 变换)
        # 这里 z_v = image_features 来自 Embedding Bank
        # =========================================================
        prompts = self.prompt_learner(image_features)  # (bs, n_cls, seq_len, dim)
        
        # 编码提示
        bs, n_cls, n_ctx, dim = prompts.shape
        prompts_flat = prompts.reshape(bs * n_cls, n_ctx, dim)
        
        tokenized_prompts = self.tokenized_prompts
        tokenized_prompts_flat = tokenized_prompts.unsqueeze(0).expand(bs, -1, -1).reshape(bs * n_cls, -1)
        
        text_features = self.text_encoder(prompts_flat, tokenized_prompts_flat) # (bs*n_cls, dim)
        text_features = text_features.view(bs, n_cls, -1) # (bs, n_cls, dim)

        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算相似度分数
        logit_scale = self.logit_scale.exp()
        # logits[i, j] = image_features[i] @ text_features[i, j]
        logits = logit_scale * torch.einsum("bd,bnd->bn", image_features, text_features)

        return logits


@TRAINER_REGISTRY.register()
class resnet10(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADV.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        """
        构建 ResNet10-Prompt 模型
        
        核心组件:
            - CustomCLIP: 包含 PromptLearner (ResNet10-Prompt)
            - PromptLearner 中的 resnet_func 是唯一可学习的部分
        
        冻结部分:
            - CLIP 图像编码器
            - CLIP 文本编码器
            - p(0) 初始状态
        """
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.ADV.PREC == "fp32" or cfg.TRAINER.ADV.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building ResNet10-Prompt CustomCLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # =========================================================
        # 只允许 prompt_learner 中的 ResNet-10 网络进行梯度更新
        # =========================================================
        print("[ResNet10-Prompt] Turning off gradients in image/text encoders")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        
        # =========================================================
        # 关键修复：确保 image_encoder 和 text_encoder 始终为 eval 模式
        # 对于含有 BatchNorm/Dropout 的模型（如 ResNet），train/eval 模式行为不同
        # 由于这些编码器是冻结的，必须始终使用 eval 模式以保持一致性
        # =========================================================
        self.model.image_encoder.eval()
        self.model.text_encoder.eval()
        print("[ResNet10-Prompt] Set image_encoder and text_encoder to eval mode (frozen)")
        
        # 统计可学习参数
        n_params = sum(p.numel() for p in self.model.prompt_learner.parameters() if p.requires_grad)
        print(f"[ResNet10-Prompt] Trainable parameters: {n_params:,} (only ResNet-10 network f_\u03b8)")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.ADV.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def set_model_mode(self, mode="train", names=None):
        """
        重写 set_model_mode 以确保冻结的编码器始终处于 eval 模式，
        并在训练时将不使用的 image_encoder 卸载到 CPU 以节省显存。
        """
        # 调用父类方法设置 prompt_learner 的模式
        super().set_model_mode(mode, names)
        
        # 获取实际的模型（处理 DataParallel 包装的情况）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.image_encoder.eval()
        model.text_encoder.eval()

        if mode == "train":
            # 训练时我们只使用前向嵌入（forward_embedding），不需要 image_encoder
            model.image_encoder.to("cpu")
            torch.cuda.empty_cache()
        else:
            # 测试/验证时需要使用 image_encoder
            model.image_encoder.to(self.device)

    def forward_backward_adv(self, batch_dict):
        batch, embedding_adv = batch_dict['batch'], batch_dict['images_adv']
        label = batch["label"].to(self.device)
        use_loss_mix = 1

        # 计算对抗损失的嵌入
        output_adv = self.model.forward_embedding(embedding_adv)
        loss_adv = torch.nn.CrossEntropyLoss()(output_adv, label)

        if use_loss_mix:
            # 计算与干净图像的损失（如果clean_pkl可用）
            #if 'images_clean' in batch_dict and batch_dict['images_clean'] is not None:
            embedding_clean = batch_dict['images_clean'].to(self.device)
            output_clean = self.model.forward_embedding(embedding_clean)
            loss_clean = torch.nn.CrossEntropyLoss()(output_clean, label)
            # 1:1 比例混合
            loss = 0.4 * loss_adv + 0.6 * loss_clean
        else:
            loss = loss_adv

        # 非有限 loss 直接跳过更新，避免训练中断
        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss detected; skip update for this batch")
            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output_adv, label)[0].item(),
            }
            return loss_summary

        # 手动实现 backward + 梯度裁剪 + update，防止梯度爆炸
        self.model_zero_grad()
        loss.backward()

        model = self.model.module if hasattr(self.model, 'module') else self.model
        torch.nn.utils.clip_grad_norm_(
            model.prompt_learner.parameters(),
            max_norm=1.0
        )

        self.model_update()

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output_adv, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            if self.cfg.OPTIM.LR_SCHEDULER != "plateau":
                self.update_lr()

        return loss_summary

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.ADV.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            if self.cfg.OPTIM.LR_SCHEDULER != "plateau":
                self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label


    def after_epoch(self):
        """
        每个 epoch 结束后进行验证（仅使用对抗数据）
        
        显示：
        1. 训练集对抗准确率
        2. 验证集对抗准确率
        """
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        
        if not do_test:
            return
        
        print(f"\n{'='*60}")
        print(f"Epoch {self.epoch + 1}/{self.max_epoch} - Adversarial Validation")
        print(f"{'='*60}")
        
        # 获取每个 epoch 测试的 batch 数量
        max_batches = getattr(self.cfg.TEST, 'EPOCH_TEST_BATCHES', 2)
        partial_test_batches = getattr(self.cfg.TEST, 'PARTIAL_TEST_BATCHES', 10)
        
        # 1. 训练集对抗准确率 - 直接使用 run_epoch_adv 中统计的平均值（避免重新遍历）
        train_acc = getattr(self, '_epoch_train_acc', None)
        if train_acc is not None:
            print(f'\n[1/2] Train Adversarial Accuracy (from training):')
            print(f"      Train Adv Acc: {train_acc:.2f}%")
        else:
            print(f'\n[1/2] Train adversarial accuracy not available (no acc in loss_summary)')
        
        # 2. 验证集对抗准确率
        val_acc = None
        val_loss = None
        if hasattr(self, 'val_pkl') and self.val_pkl is not None:
            print(f'\n[2/2] Validation Adversarial Accuracy:')
            val_acc = self._eval_adv_embedding(
                self.val_pkl,
                self.val_loader,
                max_batches=max_batches
            )
            print(f"      Val Adv Acc: {val_acc:.2f}%")
            # if self.cfg.OPTIM.LR_SCHEDULER == "plateau":
            val_loss = self._eval_adv_embedding_loss(
                self.val_pkl,
                self.val_loader,
                max_batches=max_batches
            )
        elif hasattr(self, 'test_pkl') and self.test_pkl is not None:
            # 如果没有验证集对抗嵌入，使用测试集
            print(f'\n[2/2] Test Adversarial Accuracy (no val_pkl):')
            val_acc = self.test_adv_partial(split="test", max_batches=max_batches)
            print(f"      Test Adv Acc: {val_acc:.2f}%")
            #if self.cfg.OPTIM.LR_SCHEDULER == "plateau":
            val_loss = self._eval_adv_embedding_loss(
                self.test_pkl,
                self.test_loader,
                max_batches=max_batches
            )
        else:
            print(f'\n[2/2] Validation adversarial test skipped')

        # 3. 每个 epoch 额外评估部分 test（用于观察 val/test 偏差）
        test_partial_acc = None
        if hasattr(self, 'test_pkl') and self.test_pkl is not None:
            print(f'\n[Extra] Partial Test Adversarial Accuracy ({partial_test_batches} batches):')
            test_partial_acc = self.test_adv_partial(split="test", max_batches=partial_test_batches)
            print(f"      Test Partial Adv Acc: {test_partial_acc:.2f}%")
        else:
            print(f'\n[Extra] Partial test adversarial evaluation skipped (test_pkl not prepared)')
        
        # 打印摘要
        summary_parts = []
        if train_acc is not None:
            summary_parts.append(f"Train: {train_acc:.2f}%")
        if val_acc is not None:
            summary_parts.append(f"Val: {val_acc:.2f}%")
        if test_partial_acc is not None:
            summary_parts.append(f"Test@{partial_test_batches}b: {test_partial_acc:.2f}%")
        if summary_parts:
            print(f"\n[Summary] {' | '.join(summary_parts)}")
        
        # 记录到 TensorBoard
        if train_acc is not None:
            self.write_scalar("epoch/train_adv_acc", train_acc, self.epoch)
        if val_acc is not None:
            self.write_scalar("epoch/val_adv_acc", val_acc, self.epoch)
        if val_loss is not None:
            self.write_scalar("epoch/val_adv_loss", val_loss, self.epoch)
        if test_partial_acc is not None:
            self.write_scalar("epoch/test_partial_adv_acc", test_partial_acc, self.epoch)
        if val_acc is not None and test_partial_acc is not None:
            gap = val_acc - test_partial_acc
            #self.write_scalar("epoch/val_test_gap", gap, self.epoch)
            self.write_scalar("epoch/val_test_gap_abs", abs(gap), self.epoch)

        # 统一打印一行结构化指标，便于日志解析/画图
        print(
            f"[EpochMetrics] epoch={self.epoch + 1} "
            f"train_adv_acc={train_acc if train_acc is not None else 'NA'} "
            f"val_adv_acc={val_acc if val_acc is not None else 'NA'} "
            f"test_partial_adv_acc={test_partial_acc if test_partial_acc is not None else 'NA'}"
        )

        # 使用验证集损失驱动学习率调整（ReduceLROnPlateau）
        if self.cfg.OPTIM.LR_SCHEDULER == "plateau" and val_loss is not None:
            self.sched.step(val_loss)
        
        # 保存最佳模型（基于验证集对抗准确率）
        if val_acc is not None:
            is_best = val_acc > self.best_result
            if is_best:
                self.best_result = val_acc
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=val_acc,
                    is_best=True
                )
                print(f"\n{'*'*60}")
                print(f"*** NEW BEST MODEL SAVED ***")
                print(f"    Epoch: {self.epoch + 1}")
                print(f"    Val Adv Acc: {val_acc:.2f}%")
                print(f"{'*'*60}")
        
        # 定期保存检查点
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )
        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
            print(f"[Checkpoint Saved] Epoch {self.epoch + 1}")
        
        print(f"{'='*60}\n")

    @torch.no_grad()
    def _eval_adv_embedding(self, embedding_pkl, data_loader, max_batches=None):
        """
        使用预计算的对抗嵌入评估模型
        
        参数:
            embedding_pkl: 对抗嵌入张量
            data_loader: 对应的数据加载器（用于获取标签）
            max_batches: 最多评估的 batch 数量
        
        返回:
            准确率 (%)
        """
        self.set_model_mode("eval")
        self.evaluator.reset()
        
        # 获取实际模型（处理 DataParallel 包装）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        for batch_idx, batch in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
            
            label = batch["label"].to(self.device)
            
            # 获取对应的对抗嵌入
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + label.shape[0]
            embedding_adv = embedding_pkl[start_idx:end_idx]
            
            embedding_adv = embedding_adv.to(self.device)
            
            # 使用对抗嵌入进行推理
            output = model.forward_embedding(embedding_adv)
            self.evaluator.process(output, label)
        
        results = self.evaluator.evaluate()
        return list(results.values())[0]

    @torch.no_grad()
    def _eval_adv_embedding_loss(self, embedding_pkl, data_loader, max_batches=None):
        """
        使用预计算的对抗嵌入评估验证集损失

        返回:
            平均交叉熵损失
        """
        self.set_model_mode("eval")

        # 获取实际模型（处理 DataParallel 包装）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        total_loss = 0.0
        total_count = 0

        for batch_idx, batch in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break

            label = batch["label"].to(self.device)
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + label.shape[0]
            embedding_adv = embedding_pkl[start_idx:end_idx]
                
            embedding_adv = embedding_adv.to(self.device)

            output = model.forward_embedding(embedding_adv)
            loss = F.cross_entropy(output, label, reduction="sum")
            total_loss += loss.item()
            total_count += label.shape[0]

        if total_count == 0:
            return None

        return total_loss / total_count

    @torch.no_grad()
    def _test_impl(self, split=None, max_batches=None, use_adv=False):
        """
        统一的测试实现
        
        参数:
            split: 数据集划分 ('test' or 'val')
            max_batches: 最多测试的 batch 数量（None 表示测试全部）
            use_adv: 是否使用对抗样本
        
        返回:
            准确率 (%)
        """
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        # 初始化归一化器（仅对抗测试时需要）
        if use_adv:
            if not hasattr(self, 'normalizer'):
                self.normalizer = ImageNormalizer(device=self.device)
            else:
                self.normalizer.to(self.device)
            test_eps = self.cfg.DATASET.TEST_EPS / 255.0
            array_to_pkl = self.test_pkl

        # 打印测试信息
        batch_info = f"first {max_batches} batches" if max_batches else "all"
        adv_info = "adversarial" if use_adv else "clean"
        print(f"Evaluate {adv_info} on the *{split}* set ({batch_info})")

        for batch_idx, batch in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
            
            input, label = self.parse_batch_test(batch)
            
            if use_adv:
                # 获取对应的对抗样本
                start_idx = batch_idx * data_loader.batch_size
                end_idx = start_idx + input.shape[0]
                input_adv = array_to_pkl[start_idx:end_idx].to(input.device)
                
                # 使用 normalizer 限制扰动
                input_to_eval = self.normalizer.clamp_perturbation(input_adv, input, test_eps)
            else:
                input_to_eval = input
            
            output = self.model_inference(input_to_eval)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        return list(results.values())[0]

    @torch.no_grad()
    def test_partial(self, split=None, max_batches=20):
        """部分数据集测试（干净样本）"""
        return self._test_impl(split, max_batches=max_batches, use_adv=False)

    @torch.no_grad()
    def test_adv_partial(self, split=None, max_batches=20):
        """部分数据集对抗测试"""
        return self._test_impl(split, max_batches=max_batches, use_adv=True)

    @torch.no_grad()
    # def generate_test_embedding(self, path):
    #     """
    #     将测试集的对抗图像转换为嵌入向量
        
    #     这确保测试时使用与训练相同的代码路径 (forward_embedding)
    #     """
    #     pkl_path = '{}/{}_{}_{}_embedding.pkl'.format(
    #         path, self.cfg.DATASET.NAME,
    #         self.cfg.MODEL.BACKBONE.NAME.replace("/", "_"), 'PGD'
    #     )
        
    #     if os.path.isfile(pkl_path):
    #         self.test_embedding_pkl = torch.load(pkl_path, map_location='cpu')
    #         print(f'[generate_test_embedding] Loaded test_embedding_pkl from {pkl_path}')
    #         return
        
    #     print("[generate_test_embedding] Converting test adversarial images to embeddings...")
        
    #     # 确保 test_pkl 存在（对抗图像）
    #     if not hasattr(self, 'test_pkl') or self.test_pkl is None:
    #         raise RuntimeError("test_pkl not found. Call before_adv_test first.")
        
    #     model = self.model.module if hasattr(self.model, 'module') else self.model
    #     image_encoder = model.image_encoder
    #     dtype = model.dtype
        
    #     # test_pkl 已经是归一化后的图像
    #     num_samples = self.test_pkl.shape[0]
    #     embedding_dim = image_encoder.output_dim
    #     self.test_embedding_pkl = torch.empty(size=[num_samples, embedding_dim])
        
    #     batch_size = self.cfg.DATALOADER.TEST.BATCH_SIZE
    #     num_batches = (num_samples + batch_size - 1) // batch_size
        
    #     image_encoder.eval()
        
    #     for batch_idx in range(num_batches):
    #         start_idx = batch_idx * batch_size
    #         end_idx = min(start_idx + batch_size, num_samples)
            
    #         images = self.test_pkl[start_idx:end_idx].to(self.device)
            
    #         with torch.no_grad():
    #             embedding = image_encoder(images.type(dtype))
            
    #         self.test_embedding_pkl[start_idx:end_idx] = embedding.cpu().float()
            
    #         if (batch_idx + 1) % 20 == 0:
    #             print(f"  Processed {batch_idx + 1}/{num_batches} batches")
        
    #     torch.save(self.test_embedding_pkl, pkl_path)
    #     print(f'[generate_test_embedding] Saved test_embedding_pkl to {pkl_path}')

    @torch.no_grad()
    def test_adv_embedding(self, split="test", max_batches=None):
        """
        使用预计算嵌入测试（与训练使用相同的 forward_embedding 路径）
        
        这是统一的测试方法，确保训练和测试使用完全相同的代码路径
        """
        self.set_model_mode("eval")
        self.evaluator.reset()
        
        if split == "val" and self.val_loader is not None:
            embedding_pkl = self.val_pkl
            data_loader = self.val_loader
        else:
            split = "test"
            # if hasattr(self, 'test_embedding_pkl') and self.test_embedding_pkl is not None:
            #     embedding_pkl = self.test_embedding_pkl
            # else:
            #     raise RuntimeError("test_embedding_pkl not found. Call generate_test_embedding first.") 
            # 废弃的测试路径
            data_loader = self.test_loader
        
        batch_info = f"first {max_batches} batches" if max_batches else "all"
        print(f"Evaluate adversarial (embedding path) on the *{split}* set ({batch_info})")
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        for batch_idx, batch in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
            
            label = batch["label"].to(self.device)
            
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + label.shape[0]
            embedding_adv = embedding_pkl[start_idx:end_idx].to(self.device)
            
            output = model.forward_embedding(embedding_adv)
            self.evaluator.process(output, label)
        
        results = self.evaluator.evaluate()
        return list(results.values())[0]

    def load_model(self, directory, epoch=None, model_file=None):
        """
        加载模型权重
        
        ResNet10-Prompt 的可学习部分:
            - resnet_func: ResNet-10 变换网络 f_θ
        
        固定部分 (应该忽略):
            - p0: 固定初始状态
            - token_prefix: SOS token
            - token_suffix: [class_name, EOS]
        """
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        if model_file is None:
            raise ValueError("model_file is required")
            # model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = model_file + "-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # 忽略固定的 token 向量 (这些应该使用当前类别名称计算)
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            
            # 忽略固定的初始状态 p(0) (这些应该使用当前配置计算)
            if "p0" in state_dict:
                del state_dict["p0"]
            
            # 兼容旧版本: 如果有 ctx 参数，也忽略
            if "ctx" in state_dict:
                del state_dict["ctx"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            
            # Debug: 检查键名匹配情况
            model_keys = set(self._models[name].state_dict().keys())
            loaded_keys = set(state_dict.keys())
            missing_keys = model_keys - loaded_keys
            unexpected_keys = loaded_keys - model_keys
            
            if missing_keys:
                print(f"[Warning] Missing keys in state_dict: {list(missing_keys)[:5]} ... (Total: {len(missing_keys)})")
            if unexpected_keys:
                print(f"[Warning] Unexpected keys in state_dict: {list(unexpected_keys)[:5]} ... (Total: {len(unexpected_keys)})")
            
            # set strict=False 以允许缺失的键
            self._models[name].load_state_dict(state_dict, strict=False)
            
            # =========================================================
            # 验证 ResNet-10 网络权重是否正确加载
            # =========================================================
            resnet_func = self._models[name].resnet_func
            print(f"\n[Verify] ResNet-10 network loaded successfully")
            print(f"[Verify] input_proj weight sum: {resnet_func.input_proj.weight.sum().item():.6f}")
            print(f"[Verify] input_proj bias sum: {resnet_func.input_proj.bias.sum().item():.6f}")
            print(f"[Verify] output_proj weight sum: {resnet_func.output_proj.weight.sum().item():.6f}")
            print(f"[Verify] output_proj bias sum: {resnet_func.output_proj.bias.sum().item():.6f}")
            
            # 检查是否所有 ResNet-10 相关的键都被加载
            resnet_keys = [k for k in state_dict.keys() if 'resnet_func' in k]
            print(f"[Verify] Loaded {len(resnet_keys)} ResNet-10-related keys from checkpoint")
            
            # 详细列出 ResNet-10 网络的各层权重统计
            print("[Verify] ResNet-10 network layer statistics:")
            for layer_name, param in resnet_func.named_parameters():
                print(f"  - {layer_name}: shape={list(param.shape)}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")