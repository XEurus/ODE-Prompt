"""
训练指标日志记录器模块

本模块提供了用于训练过程中指标记录和可视化的工具类。
支持滑动窗口平均、全局平均、分布式同步等功能。

主要类：
    - SmoothedValue: 单个指标的平滑值追踪器
    - MetricLogger: 多指标日志管理器

功能特点：
    - 滑动窗口平均（减少噪声）
    - 全局平均（整体统计）
    - 分布式训练支持（多 GPU 同步）
    - ETA 估计和进度显示

使用示例：
    >>> logger = MetricLogger(delimiter="  ")
    >>> for batch in logger.log_every(dataloader, print_freq=100, header="Train"):
    >>>     loss = train_step(batch)
    >>>     logger.update(loss=loss, acc=accuracy)
    >>> print(logger.global_avg())
"""

import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """
    平滑值追踪器
    
    追踪一系列值并提供滑动窗口平均和全局平均。
    适用于训练过程中的损失、准确率等指标的记录。
    
    参数：
        window_size: 滑动窗口大小（默认 20）
        fmt: 输出格式字符串
    
    属性：
        deque: 滑动窗口数据
        total: 累计总和
        count: 累计计数
    
    示例：
        >>> sv = SmoothedValue(window_size=10)
        >>> for loss in losses:
        >>>     sv.update(loss)
        >>> print(f"Avg: {sv.avg}, Global: {sv.global_avg}")
    """

    def __init__(self, window_size=20, fmt=None):
        """初始化平滑值追踪器"""
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        更新值
        
        参数：
            value: 新的值
            n: 样本数量（用于加权平均）
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        在分布式进程间同步统计量
        
        警告：不会同步 deque（滑动窗口），只同步 count 和 total。
        适用于多 GPU 训练时聚合各进程的统计量。
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """滑动窗口中位数"""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """滑动窗口平均值"""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """全局平均值（所有历史数据）"""
        return self.total / self.count

    @property
    def max(self):
        """滑动窗口最大值"""
        return max(self.deque)

    @property
    def value(self):
        """最新值"""
        return self.deque[-1]

    def __str__(self):
        """格式化输出"""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """
    多指标日志管理器
    
    管理多个指标的记录和输出，支持迭代器包装和进度显示。
    
    参数：
        delimiter: 输出分隔符（默认 "\t"）
    
    属性：
        meters: 指标名到 SmoothedValue 的映射
    
    使用示例：
        >>> logger = MetricLogger()
        >>> logger.update(loss=0.5, acc=0.95)
        >>> print(logger)
        >>> 
        >>> # 使用迭代器包装
        >>> for batch in logger.log_every(dataloader, 100, "Epoch 1"):
        >>>     ...
    """
    
    def __init__(self, delimiter="\t"):
        """初始化日志管理器"""
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        更新指标
        
        参数：
            **kwargs: 指标名=值的键值对
        
        示例：
            >>> logger.update(loss=0.5, acc=0.95)
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """支持通过属性访问指标"""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """格式化输出所有指标"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        """输出所有指标的全局平均值"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """同步所有指标（分布式训练）"""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """添加自定义指标"""
        self.meters[name] = meter

    def clear(self):
        """清空所有指标"""
        self.meters.clear()

    def log_every(self, iterable, print_freq, header=None):
        """
        带日志的迭代器包装器
        
        在迭代过程中定期输出进度信息，包括：
        - 当前进度
        - 预计剩余时间 (ETA)
        - 指标统计
        - 迭代时间
        - 数据加载时间
        - GPU 显存使用（如果可用）
        
        参数：
            iterable: 要迭代的对象
            print_freq: 打印频率
            header: 日志前缀
        
        Yields：
            迭代器中的元素
        
        示例：
            >>> for batch in logger.log_every(dataloader, 100, "Train"):
            >>>     loss = train_step(batch)
            >>>     logger.update(loss=loss)
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        
        # 构建日志格式
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 计算预计剩余时间
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        
        # 输出总时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def is_dist_avail_and_initialized():
    """
    检查分布式训练是否可用并已初始化
    
    返回：
        bool: 是否处于分布式训练模式
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
