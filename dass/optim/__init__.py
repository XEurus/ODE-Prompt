"""
DASS 优化器模块

本模块导出优化器和学习率调度器的构建函数。

导出函数：
    - build_optimizer: 构建 PyTorch 优化器
    - build_lr_scheduler: 构建学习率调度器

支持的优化器：
    - SGD: 随机梯度下降
    - Adam: Adam 优化器
    - AdamW: 带权重衰减的 Adam
    - RAdam: Rectified Adam

支持的调度策略：
    - single_step: 单步衰减
    - multi_step: 多步衰减
    - cosine: 余弦退火

使用示例：
    >>> from dass.optim import build_optimizer, build_lr_scheduler
    >>> optimizer = build_optimizer(model, cfg.OPTIM)
    >>> scheduler = build_lr_scheduler(optimizer, cfg.OPTIM)
"""

from .optimizer import build_optimizer
from .lr_scheduler import build_lr_scheduler
