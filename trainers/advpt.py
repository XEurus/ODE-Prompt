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
from torchdiffeq import odeint_adjoint as odeint

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
    model_path = clip._download(url, '/home/dji/Project/ODE-Prompt/Adversarial-Prompt-Tuning/clip')

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
        self.hidden_dim = prompt_dim * 8
        
        self.input_proj = nn.Linear(prompt_dim + visual_dim, self.hidden_dim)
        # self.norm_in = nn.LayerNorm(self.hidden_dim)
        self.act = nn.GELU()

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),nn.GELU(),
        )
        
        # self.res_blocks = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(self.hidden_dim, self.hidden_dim),
        #         nn.LayerNorm(self.hidden_dim),
        #         nn.GELU(),
        #         nn.Linear(self.hidden_dim, self.hidden_dim),
        #         nn.LayerNorm(self.hidden_dim)
        #     ) for _ in range(1)  # 1 个残差块
        # ])
        
        # 输出投影
        self.output_proj = nn.Linear(self.hidden_dim, prompt_dim)
        
        # 零初始化最后一层，确保 ODE 初始时接近恒等映射
        # 这是 Neural ODE 的常见技巧，有助于训练稳定性
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def set_visual_feature(self, z_v):
        """
        设置视觉特征条件
        
        参数:
            z_v: 对抗图像嵌入，形状 (batch_size, visual_dim)
                 我们取 batch 均值作为全局条件
        """
        # 取 batch 均值，得到形状 (visual_dim,)
        self.z_v = z_v.mean(dim=0)  

    def forward(self, t, p):
        """
        计算 ODE 导数 dp/dt = f_θ(p, z_v)
        
        参数:
            t: 当前时间点 (标量，ODE 求解器需要，但我们的动力学是时间无关的)
            p: 当前提示状态，形状 (n_ctx, prompt_dim)
        
        返回:
            dp/dt: 提示状态的变化率，形状 (n_ctx, prompt_dim)
        """
        if self.z_v is None:
            raise RuntimeError("必须先调用 set_visual_feature() 设置视觉特征！")
        
        # p 的形状: (n_ctx, prompt_dim)
        # z_v 的形状: (visual_dim,)
        
        # 将 z_v 扩展到与 p 的 n_ctx 维度匹配
        # 扩展后形状: (n_ctx, visual_dim)
        z_v_expanded = self.z_v.unsqueeze(0).expand(p.shape[0], -1)
        
        # 拼接: [p(t); z_v]
        # 形状: (n_ctx, prompt_dim + visual_dim)
        inp = torch.cat([p, z_v_expanded], dim=-1)
        
        # Residual MLP 前向传播
        x = self.input_proj(inp)
        x = self.act(x) 
        x = self.mlp(x)
        
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
        # =========================================================
        self.ode_func = ODEFunc(ctx_dim, visual_dim).type(dtype)
        
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
            prompts: 完整的提示嵌入，形状 (n_cls, seq_len, dim)
        
        流程:
            1. 设置视觉特征到 ODE 网络
            2. 求解 ODE: p(T) = ODEsolve(f_θ, p(0), [0,1])
            3. 拼接 prompt: [SOS, p(T), class_name, EOS]
        """
        # =========================================================
        # Step 1: 设置视觉特征条件
        # =========================================================
        # z_v 形状: (batch_size, visual_dim)
        # ODE 网络会取 batch 均值作为全局条件
        self.ode_func.set_visual_feature(z_v)
        
        # =========================================================
        # Step 2: ODE 求解 - 核心步骤!
        # =========================================================
        # 初始状态: p(0) = "a photo of a" 的嵌入
        # 形状: (n_ctx, prompt_dim)
        p0 = self.p0
        
        # 时间点: [0, 1]  -> 表示从 t=0 演化到 t=1
        t = self.ode_t.to(p0.device)
        
        # 求解 ODE:
        # p(T) = p(0) + ∫_0^T f_θ(p(t), z_v) dt
        # 
        # odeint 返回形状: (len(t), n_ctx, prompt_dim)
        # 我们取 [1] 即 t=1 时刻的状态
        p_trajectory = odeint(
            self.ode_func,      # 导数函数 dp/dt = f_θ(p, z_v)
            p0,                 # 初始状态 p(0)
            t,                  # 时间点 [0, 1]
            method='dopri5',    # Dormand-Prince 5 (自适应步长)
            rtol=1e-3,          # 放宽误差容限以避免 underflow
            atol=1e-3,          
            adjoint_method='dopri5',        
            adjoint_rtol=1e-3,
            adjoint_atol=1e-3,
            options={'safety': 0.1, 'min_step': 1e-3} # 强制设置最小步长，防止 underflow
        )
        
        # 取终端状态 p(T)
        # 形状: (n_ctx, prompt_dim)
        ctx = p_trajectory[1]  # t=1 时刻的状态
        
        # =========================================================
        # Step 3: 扩展到所有类别
        # =========================================================
        # ctx 形状: (n_ctx, prompt_dim) -> (n_cls, n_ctx, prompt_dim)
        # 所有类别共享相同的演化后提示
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # =========================================================
        # Step 4: 拼接完整 prompt
        # =========================================================
        # prefix: SOS token，形状 (n_cls, 1, dim)
        # suffix: [class_name, EOS]，形状 (n_cls, *, dim)
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

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
        prompts = self.prompt_learner(image_features)
        
        # Step 3: 编码提示
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # Step 4: 归一化并计算相似度
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

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
        prompts = self.prompt_learner(image_features)  # ODE 演化!
        
        # 编码提示
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算相似度分数
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

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

    def forward_backward_adv(self, batch_dict):
        batch, embedding_adv = batch_dict['batch'], batch_dict['images_adv']
        label = batch["label"].to(self.device)

        output = self.model.forward_embedding(embedding_adv)
        loss_adv = torch.nn.CrossEntropyLoss()(output, label)
        loss = loss_adv
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
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
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
