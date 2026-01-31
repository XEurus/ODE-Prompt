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
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
from utils.adv_utils import ImageNormalizer, get_model

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# ============================================================================
# ODE-Prompt: 连续时间对抗提示学习
# 核心思想: dp(t)/dt = f_θ(p(t), z_v)，其中 z_v 是对抗图像的视觉特征
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
    model_path = clip._download(url, '/home/dji/Project/ODE-Prompt/ODE-Adversarial-Prompt-Tuning/clip')

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
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class ODEFunc(nn.Module):
    """
    ODE 动力学网络 f_θ
    
    根据论文公式 (3.2):
        dp(t)/dt = f_θ(p(t), z_v)
    
    其中:
        - p(t): 当前时刻的提示状态，形状 (n_ctx, dim)
        - z_v: 对抗图像的视觉特征，形状 (batch_size, dim)
        
    网络设计:
        f_θ(p, z_v) = MLP_θ([p; z_v])  # 论文公式
        
    为了处理 batch 维度的不匹配，我们采用以下策略:
        - 在训练时，z_v 的 batch 均值作为全局视觉条件
        - 这样 ODE 为所有类别生成统一的提示演化
    """
    def __init__(self, prompt_dim, visual_dim):
        super(ODEFunc, self).__init__()
        self.prompt_dim = prompt_dim
        self.visual_dim = visual_dim
        
        # 视觉特征会被存储在这里，供 forward 使用
        # 这是因为 odeint 只允许 forward(t, x) 签名
        self.z_v = None  
        
        # 主网络: 使用 Residual MLP 替代简单的 MLP
        # 增加网络容量，有助于学习更复杂的动力学
        self.hidden_dim = prompt_dim * 2
        
        self.input_proj = nn.Linear(prompt_dim + visual_dim, self.hidden_dim)
        # self.norm_in = nn.LayerNorm(self.hidden_dim)
        self.act = nn.GELU()

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),nn.GELU(),
        )
        
        self.res_blocks = nn.ModuleList()
        for _ in range(2):  # 2 个残差块
            block = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU()
            )
            # 关键修复：零初始化残差块的最后一层，确保初始时残差接近零
            #nn.init.zeros_(block[3].weight)
            #nn.init.zeros_(block[3].bias)
            self.res_blocks.append(block)
        
        # 输出投影
        self.output_proj = nn.Linear(self.hidden_dim, prompt_dim)
        
        # 关键修复：使用极小的高斯初始化而不是全零
        # 全零会导致 res_blocks 在初期梯度为0（梯度阻断）
        # 极小值 (1e-5) 既能保证 ODE 初始接近恒等，又能打通梯度
        nn.init.zeros_(self.output_proj.weight)
        # nn.init.normal_(self.output_proj.weight, std=1e-5)
        nn.init.zeros_(self.output_proj.bias)
    
    def set_visual_feature(self, z_v):
        """
        设置视觉特征条件
        
        参数:
            z_v: 对抗图像嵌入，形状 (batch_size, visual_dim)
        """
        # 移除 batch 均值，保留每个样本的独立特征
        self.z_v = z_v  # (batch_size, visual_dim)

    def forward(self, t, p):
        """
        计算 ODE 导数 dp/dt = f_θ(p, z_v)
        
        参数:
            t: 当前时间点 (标量，ODE 求解器需要，但我们的动力学是时间无关的)
            p: 当前提示状态，形状 (batch_size, n_ctx, prompt_dim)
        
        返回:
            dp/dt: 提示状态的变化率，形状 (batch_size, n_ctx, prompt_dim)
        """
        if self.z_v is None:
            raise RuntimeError("必须先调用 set_visual_feature() 设置视觉特征！")
        
        # p 的形状: (batch_size, n_ctx, prompt_dim)
        # z_v 的形状: (batch_size, visual_dim)
        
        # 将 z_v 扩展到与 p 的 n_ctx 维度匹配
        # 扩展后形状: (batch_size, n_ctx, visual_dim)
        z_v_expanded = self.z_v.unsqueeze(1).expand(-1, p.shape[1], -1)
        
        # 拼接: [p(t); z_v]
        # 形状: (batch_size, n_ctx, prompt_dim + visual_dim)
        inp = torch.cat([p, z_v_expanded], dim=-1)
        
        # Residual MLP 前向传播
        x = self.input_proj(inp)
        x = self.act(x) 
        #x = self.mlp(x)
        for block in self.res_blocks:
            x = x + block(x)
        
        # 输出层
        dp_dt = self.output_proj(x)
        
        return dp_dt


class PromptLearner(nn.Module):
    """
    ODE-Prompt 的提示学习器
    
    核心改变:
        - 原始 AdvPT: ctx 是可学习参数，直接用于 prompt
        - ODE-Prompt: ctx 作为固定初始状态 p(0)，通过 ODE 演化得到 p(T)
    
    流程:
        1. p(0) = "a photo of a" 的文本嵌入 (固定)
        2. 设置视觉特征 z_v = E_v(x_adv)
        3. ODE 求解: p(T) = ODEsolve(f_θ, p(0), [0, T])
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

        print(f'[ODE-Prompt] Initial prompt p(0): "{prompt_prefix}"')
        print(f"[ODE-Prompt] Number of context tokens: {n_ctx}")
        print(f"[ODE-Prompt] Prompt dimension: {ctx_dim}, Visual dimension: {visual_dim}")

        # =========================================================
        # p(0) 作为固定的 buffer，不参与梯度更新
        # 这是与原始 AdvPT 的关键区别！
        # =========================================================
        self.register_buffer("p0", ctx_vectors)  # 固定初始状态
        
        # =========================================================
        # ODE 动力学网络 f_θ (这是唯一的可学习部分!)
        # 关键修复：ODE 网络必须使用 float32 以保证数值稳定性
        # torchdiffeq 在 fp16 下会出现严重的数值问题
        # =========================================================
        self.ode_func = ODEFunc(ctx_dim, visual_dim).float()  # 强制 float32
        
        # ODE 求解器参数
        self.ode_t = torch.tensor([0.0, 1.0])  # 时间范围 [0, T]，T=1

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
        ODE-Prompt 的前向传播
        
        参数:
            z_v: 对抗图像的视觉特征，形状 (batch_size, visual_dim)
                 来自 Adversarial Embedding Bank
        
        返回:
            prompts: 完整的提示嵌入，形状 (batch_size, n_cls, seq_len, dim)
        
        流程:
            1. 设置视觉特征到 ODE 网络
            2. 求解 ODE: p(T) = ODEsolve(f_θ, p(0), [0,1])
            3. 拼接 prompt: [SOS, p(T), class_name, EOS]
        """
        # =========================================================
        # Step 1: 获取 batch size
        # =========================================================
        bs = z_v.shape[0]
        
        # =========================================================
        # Step 2: ODE 求解 - 核心步骤!
        # =========================================================
        # 初始状态: p(0) = "a photo of a" 的嵌入
        # 形状: (n_ctx, prompt_dim) -> (batch_size, n_ctx, prompt_dim)
        p0 = self.p0.unsqueeze(0).expand(bs, -1, -1)
        
        # 时间点: [0, 1]  -> 表示从 t=0 演化到 t=1
        t = self.ode_t.to(p0.device)
        
        # 求解 ODE:
        # p(T) = p(0) + ∫_0^T f_θ(p(t), z_v) dt
        # 
        # odeint 返回形状: (len(t), batch_size, n_ctx, prompt_dim)
        # 我们取 [1] 即 t=1 时刻的状态
        # 关键修复：将输入转换为 float32 以匹配 ODE 网络
        # ODE 求解过程在 float32 下进行，之后再转回原始 dtype
        p0_float = p0.float()
        z_v_float = z_v.float()
        self.ode_func.set_visual_feature(z_v_float)  # 重新设置为 float32 版本
        
        # 使用普通 odeint 而非 adjoint 版本
        # adjoint 在反向传播时需要重新求解 ODE，数值上更不稳定
        # 普通 odeint 直接通过计算图反向传播，更稳定
        p_trajectory = odeint_adjoint(
            self.ode_func,      # 导数函数 dp/dt = f_θ(p, z_v)
            p0_float,           # 初始状态 p(0)，float32
            t,                  # 时间点 [0, 1]
            method='dopri5',    # Dormand-Prince 5 (自适应步长)
            rtol=1e-3,          # 放宽容差以提高数值稳定性
            atol=1e-4,          
            options={'min_step': 1e-5}  # 适当的最小步长
        )
        
        # 取终端状态 p(T)
        # 形状: (batch_size, n_ctx, prompt_dim)
        ctx = p_trajectory[1]  # t=1 时刻的状态
        
        # 关键修复：将 ODE 输出转回原始 dtype (与 prefix/suffix 匹配)
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
    ODE-Prompt 的自定义 CLIP 模型
    
    组件:
        - prompt_learner: ODE-Prompt 提示学习器
        - image_encoder: 冻结的 CLIP 图像编码器
        - text_encoder: 冻结的 CLIP 文本编码器
    
    可学习部分:
        - 仅 prompt_learner.ode_func (即 ODE 动力学网络 f_θ)
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
            2. ODE 演化提示 (z_v 作为条件)
            3. 编码提示 -> 文本特征
            4. 计算相似度
        """
        # Step 1: 编码图像
        image_features = self.image_encoder(image.type(self.dtype))
        
        # Step 2: ODE 演化提示
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
        
        这是 ODE-Prompt 训练的核心方法!
        
        参数:
            image_features: 对抗图像嵌入，形状 (batch_size, visual_dim)
                           来自 Adversarial Embedding Bank
        
        返回:
            logits: 预测分数，形状 (batch_size, n_cls)
        
        流程:
            1. 将对抗嵌入 z_v 传给 ODE 网络
            2. ODE 演化: p(T) = ODEsolve(f_θ, p(0), [0,1])
            3. 编码提示 -> 文本特征
            4. 计算对抗嵌入与文本特征的相似度
        """
        # 类型转换
        image_features = image_features.type(self.dtype)
        
        # =========================================================
        # 核心: 将对抗图像嵌入传给 PromptLearner
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
class AdvPT(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADV.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        """
        构建 ODE-Prompt 模型
        
        核心组件:
            - CustomCLIP: 包含 PromptLearner (ODE-Prompt)
            - PromptLearner 中的 ode_func 是唯一可学习的部分
        
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

        print("Building ODE-Prompt CustomCLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # =========================================================
        # 只允许 prompt_learner 中的 ODE 网络进行梯度更新
        # =========================================================
        print("[ODE-Prompt] Turning off gradients in image/text encoders")
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
        print("[ODE-Prompt] Set image_encoder and text_encoder to eval mode (frozen)")
        
        # 统计可学习参数
        n_params = sum(p.numel() for p in self.model.prompt_learner.parameters() if p.requires_grad)
        print(f"[ODE-Prompt] Trainable parameters: {n_params:,} (only ODE network f_θ)")

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
        重写 set_model_mode 以确保冻结的编码器始终处于 eval 模式
        
        无论是训练还是测试模式，image_encoder 和 text_encoder 都必须保持 eval 模式
        这对于含有 BatchNorm 的模型（如 ResNet backbone）尤为重要
        """
        # 调用父类方法设置 prompt_learner 的模式
        super().set_model_mode(mode, names)
        
        # 关键修复：确保冻结的编码器始终为 eval 模式
        # 获取实际的模型（处理 DataParallel 包装的情况）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.image_encoder.eval()
        model.text_encoder.eval()

    def forward_backward_adv(self, batch_dict):
        batch, embedding_adv = batch_dict['batch'], batch_dict['images_adv']
        label = batch["label"].to(self.device)

        output = self.model.forward_embedding(embedding_adv)
        loss_adv = torch.nn.CrossEntropyLoss()(output, label)
        loss = loss_adv
        self.model_backward_and_update(loss)
                
        # # 手动实现 backward + 梯度裁剪 + update，防止梯度爆炸
        # self.model_zero_grad()
        # loss.backward()
        
        # # 梯度裁剪：防止 ODE 网络梯度爆炸导致 NaN
        # model = self.model.module if hasattr(self.model, 'module') else self.model
        # torch.nn.utils.clip_grad_norm_(
        #     model.prompt_learner.ode_func.parameters(), 
        #     max_norm=1.0
        # )
        
        # self.model_update()

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
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
        
        # 1. 训练集对抗准确率
        train_acc = None
        if hasattr(self, 'train_pkl') and self.train_pkl is not None:
            print(f'\n[1/2] Train Adversarial Accuracy:')
            train_acc = self._eval_adv_embedding(
                self.train_pkl,
                self.train_loader_x_noshuffle,
                max_batches=max_batches
            )
            print(f"      Train Adv Acc: {train_acc:.2f}%")
        else:
            print(f'\n[1/2] Train adversarial test skipped (train_pkl not prepared)')
        
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
            if self.cfg.OPTIM.LR_SCHEDULER == "plateau":
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
            if self.cfg.OPTIM.LR_SCHEDULER == "plateau":
                val_loss = self._eval_adv_embedding_loss(
                    self.test_pkl,
                    self.test_loader,
                    max_batches=max_batches
                )
        else:
            print(f'\n[2/2] Validation adversarial test skipped')
        
        # 打印摘要
        summary_parts = []
        if train_acc is not None:
            summary_parts.append(f"Train: {train_acc:.2f}%")
        if val_acc is not None:
            summary_parts.append(f"Val: {val_acc:.2f}%")
        if summary_parts:
            print(f"\n[Summary] {' | '.join(summary_parts)}")
        
        # 记录到 TensorBoard
        if train_acc is not None:
            self.write_scalar("epoch/train_adv_acc", train_acc, self.epoch)
        if val_acc is not None:
            self.write_scalar("epoch/val_adv_acc", val_acc, self.epoch)
        if val_loss is not None:
            self.write_scalar("epoch/val_adv_loss", val_loss, self.epoch)

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
            embedding_adv = embedding_pkl[start_idx:end_idx].to(self.device)
            
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
            embedding_adv = embedding_pkl[start_idx:end_idx].to(self.device)

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

    def load_model(self, directory, epoch=None, model_file=None):
        """
        加载模型权重
        
        ODE-Prompt 的可学习部分:
            - ode_func: ODE 动力学网络 f_θ
        
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
            # 验证 ODE 网络权重是否正确加载
            # =========================================================
            ode_func = self._models[name].ode_func
            print(f"\n[Verify] ODE network loaded successfully")
            print(f"[Verify] input_proj weight sum: {ode_func.input_proj.weight.sum().item():.6f}")
            print(f"[Verify] input_proj bias sum: {ode_func.input_proj.bias.sum().item():.6f}")
            print(f"[Verify] output_proj weight sum: {ode_func.output_proj.weight.sum().item():.6f}")
            print(f"[Verify] output_proj bias sum: {ode_func.output_proj.bias.sum().item():.6f}")
            
            # 检查是否所有 ODE 相关的键都被加载
            ode_keys = [k for k in state_dict.keys() if 'ode_func' in k]
            print(f"[Verify] Loaded {len(ode_keys)} ODE-related keys from checkpoint")
            
            # 详细列出 ODE 网络的各层权重统计
            print("[Verify] ODE network layer statistics:")
            for layer_name, param in ode_func.named_parameters():
                print(f"  - {layer_name}: shape={list(param.shape)}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")