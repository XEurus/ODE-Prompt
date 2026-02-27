"""
对抗训练通用工具模块

提供统一的图像归一化、对抗攻击相关工具函数。
确保训练/测试/攻击流程中的归一化操作一致。
"""

import torch
import torch.nn as nn
from torchvision import transforms

# CLIP 默认归一化参数
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class ImageNormalizer:
    """
    统一的图像归一化器
    
    用于处理 CLIP 模型的图像归一化/反归一化操作。
    所有涉及 mean/std 的操作都应该通过此类完成，避免硬编码。
    
    使用示例:
        normalizer = ImageNormalizer(device='cuda')
        
        # 反归一化（用于攻击前）
        x_pixel = normalizer.denormalize(x_normalized)
        
        # 归一化（用于攻击后）
        x_normalized = normalizer.normalize(x_pixel)
        
        # 限制扰动幅度
        x_adv = normalizer.clamp_perturbation(x_adv, x_clean, eps=16/255)
    """
    
    def __init__(self, mean=None, std=None, device='cpu'):
        """
        初始化归一化器
        
        参数:
            mean: 均值列表，默认为 CLIP_MEAN
            std: 标准差列表，默认为 CLIP_STD
            device: 张量设备
        """
        if mean is None:
            mean = CLIP_MEAN
        if std is None:
            std = CLIP_STD
            
        self.mean_list = mean
        self.std_list = std
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)
        self.normalize_transform = transforms.Normalize(mean, std)
    
    def to(self, device):
        """移动到指定设备"""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self
    
    def normalize(self, x):
        """
        归一化: [0,1] 像素空间 -> 归一化空间
        
        参数:
            x: 像素值在 [0,1] 范围的图像张量
        
        返回:
            归一化后的图像张量
        """
        return (x - self.mean) / self.std
    
    def denormalize(self, x):
        """
        反归一化: 归一化空间 -> [0,1] 像素空间
        
        参数:
            x: 归一化后的图像张量
        
        返回:
            像素值在 [0,1] 范围的图像张量
        """
        return x * self.std + self.mean
    
    def get_transform(self):
        """获取 torchvision 变换（用于攻击器的 preprocess）"""
        return self.normalize_transform
    
    def clamp_perturbation(self, x_adv, x_clean, eps):
        """
        限制对抗扰动幅度（在像素空间操作）
        
        确保对抗样本与干净样本的差距在 L_inf 范围内。
        
        参数:
            x_adv: 对抗样本（归一化空间）
            x_clean: 干净样本（归一化空间）
            eps: 扰动幅度（像素空间，如 16/255）
        
        返回:
            限制后的对抗样本（归一化空间）
        """
        # 反归一化到像素空间
        x_adv_pixel = self.denormalize(x_adv)
        x_clean_pixel = self.denormalize(x_clean)
        
        # 限制扰动在 L_inf ball 内
        noise = x_adv_pixel - x_clean_pixel
        noise = torch.clamp(noise, -eps, eps)
        x_adv_pixel = x_clean_pixel + noise
        
        # 限制图像在 [0,1] 范围
        x_adv_pixel = torch.clamp(x_adv_pixel, 0, 1)
        
        # 重新归一化
        return self.normalize(x_adv_pixel)
    
    def verify_perturbation(self, x_adv, x_clean, eps, tol=1e-6):
        """
        验证对抗扰动是否满足约束
        
        参数:
            x_adv: 对抗样本（归一化空间）
            x_clean: 干净样本（归一化空间）
            eps: 扰动幅度
            tol: 容差
        
        返回:
            bool: 是否满足约束
        """
        x_adv_pixel = self.denormalize(x_adv)
        x_clean_pixel = self.denormalize(x_clean)
        diff = x_adv_pixel - x_clean_pixel
        
        max_diff = torch.max(diff).item()
        min_diff = torch.min(diff).item()
        
        return max_diff < (eps + tol) and min_diff > (-eps - tol)


class ClipModel(nn.Module):
    """
    CLIP 视觉编码器包装器
    
    用于对抗攻击的代理模型。将 CLIP 的视觉编码器包装成
    一个简单的分类器，便于使用标准的对抗攻击方法。
    
    结构:
        visual_encoder (CLIP.visual) -> fc (Linear)

    根据 Johnson-Lindenstrauss 引理，高维空间中的点映射到低维空间（这里是随机投影），
    其相对距离在一定程度上是被保持的。

    如果攻击算法使得图像在经过随机 fc 层后的输出分布（2维）与原始图像的输出分布产生巨大的 KL 散度差异，
    那么反向传播回去，必然要求原始的图像特征（Embedding）发生巨大的变化。
    """
    
    def __init__(self, model, num_classes=2):
        """
        初始化 ClipModel
        
        参数:
            model: CLIP 的视觉编码器 (clip_model.visual)
            num_classes: 输出类别数（默认为 2，用于对抗攻击）
        """
        super(ClipModel, self).__init__()
        self.visual_encoder = model
        output_dim = self.visual_encoder.output_dim
        self.fc = nn.Linear(output_dim, num_classes)
    
    def forward(self, image):
        """前向传播"""
        x = self.visual_encoder(image)
        x = self.fc(x)
        return x


def get_model(model):
    """
    获取实际的模型对象
    
    如果是 DataParallel 或 DistributedDataParallel 包装的模型，
    则返回 model.module。
    
    参数:
        model: 可能被包装的模型
    
    返回:
        实际的模型对象
    """
    if hasattr(model, 'module'):
        return model.module
    return model


def create_pgd_attacker(eps, normalizer, cfg=None, num_iters=None, num_restarts=None):
    """
    创建 PGD 攻击器
    
    参数:
        eps: 扰动幅度（像素值，会自动除以 255）
        normalizer: ImageNormalizer 实例
        cfg: 配置对象，用于读取 PGD_NUM_ITERS
        num_iters: 迭代次数（如果指定，则覆盖 cfg 中的值）
        num_restarts: 随机初始化次数
    
    返回:
        PGD 攻击器实例
    """
    from attack.attackFeature import PGD
    
    # 确定迭代次数
    if num_iters is None:
        if cfg and hasattr(cfg.DATASET, 'PGD_NUM_ITERS'):
            num_iters = cfg.DATASET.PGD_NUM_ITERS
        else:
            num_iters = 40  # 默认值
    
    # 确定重启次数
    if num_restarts is None:
        if cfg and hasattr(cfg.DATASET, 'PGD_NUM_RESTARTS'):
            num_restarts = cfg.DATASET.PGD_NUM_RESTARTS
        else:
            num_restarts = 1  # 默认值
    
    return PGD(
        eps / 255., 
        preprocess=normalizer.get_transform(), 
        num_iters=num_iters,
        num_restarts=num_restarts
    )


def create_surrogate_model(clip_model, device, prec='fp16'):
    """
    创建用于对抗攻击的代理模型
    
    参数:
        clip_model: 加载的 CLIP 模型
        device: 设备
        prec: 精度 ('fp16', 'fp32', 'amp')
    
    返回:
        代理模型（eval 模式）
    """
    # 获取视觉编码器
    visual = get_model(clip_model.visual)
    
    # 创建代理模型
    surrogate = ClipModel(model=visual, num_classes=2)
    surrogate = surrogate.eval().to(device)
    
    return surrogate
