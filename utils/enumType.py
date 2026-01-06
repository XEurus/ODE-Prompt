"""
枚举类型定义模块

本模块定义了项目中使用的枚举类型。

枚举类型：
    - NormType: 范数类型（用于对抗攻击）
"""

from enum import Enum


class NormType(Enum):
    """
    范数类型枚举
    
    用于指定对抗攻击中使用的范数约束类型。
    
    值：
        Linf (0): L∞ 范数（最大值约束）
            - 约束: ||δ||_∞ ≤ ε
            - 特点: 每个像素的扰动不超过 ε
            
        L2 (1): L2 范数（欧几里得距离约束）
            - 约束: ||δ||_2 ≤ ε
            - 特点: 整体扰动的能量不超过 ε
    
    使用示例：
        >>> norm_type = NormType.Linf
        >>> if norm_type == NormType.Linf:
        >>>     delta = torch.clamp(delta, -eps, eps)
        >>> elif norm_type == NormType.L2:
        >>>     delta = clamp_by_l2(delta, eps)
    """
    Linf = 0
    L2 = 1
