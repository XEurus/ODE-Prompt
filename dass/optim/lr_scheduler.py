"""
学习率调度器模块

本模块提供了多种学习率调度策略，支持预热（warmup）机制。
修改自 https://github.com/KaiyangZhou/deep-person-reid

支持的调度策略：
    - single_step: 单步衰减（在指定 epoch 衰减一次）
    - multi_step: 多步衰减（在多个 epoch 衰减）
    - cosine: 余弦退火

支持的预热策略：
    - constant: 常数预热（使用固定小学习率）
    - linear: 线性预热（从小学习率线性增加到目标学习率）

使用示例：
    >>> from dass.optim import build_lr_scheduler
    >>> scheduler = build_lr_scheduler(optimizer, cfg.OPTIM)
    >>> for epoch in range(epochs):
    >>>     train(...)
    >>>     scheduler.step()
"""

import torch
import warnings
from torch.optim.lr_scheduler import _LRScheduler

# 可用的调度策略
AVAI_SCHEDS = ["single_step", "multi_step", "cosine", "plateau"]


class _BaseWarmupScheduler(_LRScheduler):
    """
    预热调度器基类
    
    实现预热阶段和主调度器之间的切换逻辑。
    
    参数：
        optimizer: 优化器
        successor: 主调度器（预热结束后使用）
        warmup_epoch: 预热 epoch 数
        last_epoch: 上一个 epoch 编号
        verbose: 是否打印调试信息
    
    工作原理：
        - 前 warmup_epoch 个 epoch: 使用预热策略
        - 之后: 切换到 successor 调度器
    """

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """获取当前学习率（子类实现）"""
        raise NotImplementedError

    def step(self, epoch=None):
        """
        更新学习率
        
        在预热阶段结束后，将控制权交给 successor。
        """
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):
    """
    常数预热调度器
    
    在预热阶段使用固定的小学习率。
    
    参数：
        optimizer: 优化器
        successor: 主调度器
        warmup_epoch: 预热 epoch 数
        cons_lr: 预热阶段使用的常数学习率
    
    学习率曲线：
        epoch < warmup_epoch: lr = cons_lr
        epoch >= warmup_epoch: lr = successor.get_lr()
    """

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        """返回当前学习率"""
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):
    """
    线性预热调度器
    
    在预热阶段从最小学习率线性增加到目标学习率。
    
    参数：
        optimizer: 优化器
        successor: 主调度器
        warmup_epoch: 预热 epoch 数
        min_lr: 预热起始学习率
    
    学习率曲线：
        epoch = 0: lr = min_lr
        epoch < warmup_epoch: lr = base_lr * epoch / warmup_epoch
        epoch >= warmup_epoch: lr = successor.get_lr()
    
    优势：
        - 避免训练初期学习率过大导致的不稳定
        - 平滑过渡到正常训练阶段
    """

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        """返回当前学习率"""
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        # 线性插值
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]


def build_lr_scheduler(optimizer, optim_cfg):
    """
    构建学习率调度器
    
    根据配置创建适当的调度器，支持预热机制。
    
    参数：
        optimizer: PyTorch 优化器
        optim_cfg: 优化配置节点，包含：
            - LR_SCHEDULER: 调度策略名称
            - STEPSIZE: 衰减步长（可以是 int 或 list）
            - GAMMA: 衰减系数
            - MAX_EPOCH: 最大 epoch 数
            - WARMUP_EPOCH: 预热 epoch 数
            - WARMUP_TYPE: 预热类型 ("constant" 或 "linear")
            - WARMUP_CONS_LR: 常数预热的学习率
            - WARMUP_MIN_LR: 线性预热的起始学习率
            - WARMUP_RECOUNT: 预热后是否重置 epoch 计数
    
    返回：
        scheduler: 学习率调度器
    
    示例配置：
        optim_cfg.LR_SCHEDULER = "cosine"
        optim_cfg.MAX_EPOCH = 100
        optim_cfg.WARMUP_EPOCH = 5
        optim_cfg.WARMUP_TYPE = "linear"
        optim_cfg.WARMUP_MIN_LR = 1e-6
    """
    lr_scheduler = optim_cfg.LR_SCHEDULER
    stepsize = optim_cfg.STEPSIZE
    gamma = optim_cfg.GAMMA
    max_epoch = optim_cfg.MAX_EPOCH

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    # 单步衰减调度器
    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                f"be an integer, but got {type(stepsize)}"
            )

        if stepsize <= 0:
            stepsize = max_epoch

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    # 多步衰减调度器
    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                f"be a list, but got {type(stepsize)}"
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    # 余弦退火调度器
    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )

    # 验证集损失驱动的调度器（ReduceLROnPlateau）
    elif lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=optim_cfg.PLATEAU_MODE,
            factor=optim_cfg.PLATEAU_FACTOR,
            patience=optim_cfg.PLATEAU_PATIENCE,
            threshold=optim_cfg.PLATEAU_THRESHOLD,
            min_lr=optim_cfg.PLATEAU_MIN_LR,
        )

    # 添加预热
    if optim_cfg.WARMUP_EPOCH > 0:
        if lr_scheduler == "plateau":
            warnings.warn(
                "Warmup is not applied to ReduceLROnPlateau; set WARMUP_EPOCH<=0."
            )
            return scheduler
        if not optim_cfg.WARMUP_RECOUNT:
            scheduler.last_epoch = optim_cfg.WARMUP_EPOCH

        if optim_cfg.WARMUP_TYPE == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_CONS_LR
            )

        elif optim_cfg.WARMUP_TYPE == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_MIN_LR
            )

        else:
            raise ValueError

    return scheduler
