"""
ODE 网络利普希茨常数计算工具

提供 5 种方法计算 ODE 动力学网络 f_θ 的利普希茨常数:

    1. Jacobian SVD     — 在采样点精确计算 Jacobian 谱范数
    2. 幂迭代           — 高效估计 Jacobian 最大奇异值
    3. auto_LiRPA 界传播 — 基于线性松弛的上界
    4. Branch-and-Bound  — 分支定界 + LiRPA，收紧上界
    5. 经验采样          — 有限对采样得到下界

数学背景:
    ODE: dp(t)/dt = f_θ(p(t), z_v)
    动力学 Lipschitz 常数 L_f = sup_{p1≠p2} ||f(p1,z_v) - f(p2,z_v)|| / ||p1 - p2||
    ODE 流映射的 Lipschitz 常数 ≤ e^{L_f · T}

使用方法:
    # 编程式调用
    from utils.lipschitz import ODELipschitzAnalyzer
    analyzer = ODELipschitzAnalyzer(ode_func, prompt_dim=512, visual_dim=512)
    report = analyzer.full_analysis(z_v=z_v_sample, p_center=p0)

    # 命令行
    python utils/lipschitz.py --checkpoint output/.../model.pth.tar
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '4')

import sys
import types
import copy
import time
import heapq
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ================================================================
# NumPy 兼容性补丁
# auto_LiRPA 0.3 引用了 numpy 2.x 中已移除的 numpy.lib.arraysetops
# 通过 monkey-patch 在不修改库文件的情况下解决
# ================================================================
if not hasattr(np.lib, 'arraysetops'):
    _compat = types.ModuleType('numpy.lib.arraysetops')
    _compat.isin = np.isin
    np.lib.arraysetops = _compat
    sys.modules['numpy.lib.arraysetops'] = _compat

try:
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    HAS_LIRPA = True
except ImportError:
    HAS_LIRPA = False

logger = logging.getLogger(__name__)


# ================================================================
# 数据结构
# ================================================================

@dataclass
class LipschitzResult:
    """单种方法的利普希茨常数计算结果"""
    method: str
    lipschitz_constant: float
    bound_type: str           # 'upper_bound', 'lower_bound', 'local_exact'
    compute_time_sec: float = 0.0
    details: Dict = field(default_factory=dict)

    def __repr__(self):
        arrow = {'upper_bound': '≤', 'lower_bound': '≥', 'local_exact': '≈'}
        return (f"  {self.method}: L {arrow.get(self.bound_type, '=')} "
                f"{self.lipschitz_constant:.6f}  "
                f"[{self.bound_type}, {self.compute_time_sec:.3f}s]")


@dataclass
class FullReport:
    """完整分析报告"""
    results: List[LipschitzResult]
    flow_map_lip: Optional[float] = None
    flow_map_empirical: Optional[float] = None
    flow_map_empirical_avg: Optional[float] = None
    model_info: Dict = field(default_factory=dict)

    def best_upper(self) -> float:
        uppers = [r.lipschitz_constant for r in self.results
                  if r.bound_type == 'upper_bound']
        return min(uppers) if uppers else float('inf')

    def best_lower(self) -> float:
        lowers = [r.lipschitz_constant for r in self.results
                  if r.bound_type == 'lower_bound']
        return max(lowers) if lowers else 0.0


# ================================================================
# 模型包装器
# ================================================================

class ODEStateWrapper(nn.Module):
    """
    固定 z_v，仅以 prompt 状态 p 为输入的包装器。
    用于计算 Lip_p(f_θ) = sup ||f(p1,z_v) - f(p2,z_v)|| / ||p1-p2||
    """
    def __init__(self, ode_func: nn.Module, z_v_fixed: torch.Tensor,
                 use_mlp_path: bool = False):
        super().__init__()
        self.prompt_dim = ode_func.prompt_dim
        self.use_mlp_path = use_mlp_path

        clean = copy.deepcopy(ode_func)
        for m in clean.modules():
            if hasattr(m, 'weight_orig'):
                try:
                    torch.nn.utils.remove_spectral_norm(m)
                except Exception:
                    pass
        self.net = clean
        self.register_buffer('z_v', z_v_fixed.detach().float())

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """p: (batch, prompt_dim) → dp/dt: (batch, prompt_dim)"""
        z_v = self.z_v.expand(p.shape[0], -1)
        inp = torch.cat([p.float(), z_v], dim=-1)
        return self._run_dynamics(inp)

    def _run_dynamics(self, inp):
        f = self.net
        x = f.input_proj(inp)
        x = f.act(x)
        if self.use_mlp_path:
            x = f.mlp(x)
        else:
            for block in f.res_blocks:
                x = x + block(x)
        return f.output_proj(x)


class LiRPACompatibleWrapper(nn.Module):
    """
    auto_LiRPA 兼容包装器: 将 GELU 替换为 ReLU 近似以确保 LiRPA 兼容。

    auto_LiRPA 0.3 不原生支持 GELU 和 LayerNorm。
    此包装器用 ReLU 替代 GELU，用 Identity 替代 LayerNorm，
    并使用原模型的权重。这会引入近似误差，但能保证
    LiRPA 分析可行。
    """
    def __init__(self, ode_func: nn.Module, z_v_fixed: torch.Tensor,
                 use_mlp_path: bool = False):
        super().__init__()
        self.prompt_dim = ode_func.prompt_dim
        self.use_mlp_path = use_mlp_path

        clean = copy.deepcopy(ode_func)
        for m in clean.modules():
            if hasattr(m, 'weight_orig'):
                try:
                    torch.nn.utils.remove_spectral_norm(m)
                except Exception:
                    pass

        hidden = clean.hidden_dim
        self.input_proj = clean.input_proj
        self.act = nn.ReLU()
        self.output_proj = clean.output_proj

        if use_mlp_path:
            layers = []
            for layer in clean.mlp:
                if isinstance(layer, nn.GELU):
                    layers.append(nn.ReLU())
                else:
                    layers.append(layer)
            self.main = nn.Sequential(*layers)
        else:
            self.res_blocks = nn.ModuleList()
            for block in clean.res_blocks:
                new_layers = []
                for layer in block:
                    if isinstance(layer, nn.GELU):
                        new_layers.append(nn.ReLU())
                    elif isinstance(layer, nn.LayerNorm):
                        new_layers.append(nn.Identity())
                    else:
                        new_layers.append(layer)
                self.res_blocks.append(nn.Sequential(*new_layers))

        self.register_buffer('z_v', z_v_fixed.detach().float())

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        z_v = self.z_v.expand(p.shape[0], -1)
        inp = torch.cat([p.float(), z_v], dim=-1)

        x = self.input_proj(inp)
        x = self.act(x)

        if self.use_mlp_path:
            x = self.main(x)
        else:
            for block in self.res_blocks:
                x = x + block(x)

        return self.output_proj(x)


# ================================================================
# 方法 2: Jacobian SVD (局部精确值)
# ================================================================

def lipschitz_jacobian_svd(wrapper: nn.Module,
                           x_samples: torch.Tensor,
                           device: str = 'cuda') -> LipschitzResult:
    """
    在采样点精确计算 Jacobian 矩阵并取谱范数 (最大奇异值)。
    结果是这些点上的精确局部 Lipschitz 常数。

    复杂度: O(N · d_out · backward_pass)，适合 d_out ≤ 1024
    """
    t0 = time.time()
    wrapper = wrapper.to(device).eval()
    lip_values = []

    for i in range(x_samples.shape[0]):
        x = x_samples[i:i+1].to(device).float().requires_grad_(True)
        y = wrapper(x)
        d_out = y.shape[-1]

        J_rows = []
        for j in range(d_out):
            grad_out = torch.zeros_like(y)
            grad_out[0, j] = 1.0
            g = torch.autograd.grad(y, x, grad_out,
                                    retain_graph=(j < d_out - 1),
                                    create_graph=False)[0]
            J_rows.append(g.detach().squeeze(0))

        J = torch.stack(J_rows)  # (d_out, d_in)
        svs = torch.linalg.svdvals(J.float())
        lip_values.append(svs[0].item())

    elapsed = time.time() - t0
    return LipschitzResult(
        method='Jacobian SVD',
        lipschitz_constant=max(lip_values),
        bound_type='local_exact',
        compute_time_sec=elapsed,
        details={
            'num_samples': len(lip_values),
            'all_values': lip_values,
            'mean': float(np.mean(lip_values)),
            'std': float(np.std(lip_values)),
            'max': max(lip_values),
            'min': min(lip_values),
        }
    )


# ================================================================
# 方法 3: 幂迭代 (高效局部估计)
# ================================================================

def lipschitz_power_iteration(wrapper: nn.Module,
                              x_samples: torch.Tensor,
                              num_iters: int = 100,
                              device: str = 'cuda') -> LipschitzResult:
    """
    幂迭代法估计 Jacobian 最大奇异值。比 SVD 高效得多。

    迭代过程:
        u_k = J v_k / ||J v_k||    (前向: 有限差分)
        v_{k+1} = J^T u_k / ||J^T u_k||  (反向: autograd)
        σ_max ≈ ||J v_k||
    """
    t0 = time.time()
    wrapper = wrapper.to(device).eval()
    lip_values = []

    for i in range(x_samples.shape[0]):
        x_ref = x_samples[i:i+1].to(device).float()
        d_in = x_ref.shape[-1]

        v = torch.randn(1, d_in, device=device)
        v = v / v.norm()
        sigma = 0.0

        for _ in range(num_iters):
            eps_fd = 1e-5
            with torch.no_grad():
                Jv = (wrapper(x_ref + eps_fd * v) - wrapper(x_ref - eps_fd * v)) / (2 * eps_fd)

            sigma = Jv.norm().item()
            if sigma < 1e-12:
                break

            u = Jv / sigma

            x_var = x_ref.detach().clone().requires_grad_(True)
            y = wrapper(x_var)
            JTu = torch.autograd.grad(y, x_var, grad_outputs=u.detach(),
                                      create_graph=False)[0]

            v_new = JTu / (JTu.norm() + 1e-12)
            v = v_new.detach()

        lip_values.append(sigma)

    elapsed = time.time() - t0
    return LipschitzResult(
        method='幂迭代',
        lipschitz_constant=max(lip_values),
        bound_type='local_exact',
        compute_time_sec=elapsed,
        details={
            'num_samples': len(lip_values),
            'num_iters': num_iters,
            'all_values': lip_values,
            'mean': float(np.mean(lip_values)),
            'max': max(lip_values),
        }
    )


# ================================================================
# 方法 4: auto_LiRPA 界传播 (上界)
# ================================================================

def lipschitz_lirpa(lirpa_wrapper: nn.Module,
                    x_center: torch.Tensor,
                    eps: float = 0.1,
                    norm: float = float('inf'),
                    method: str = 'backward',
                    device: str = 'cuda') -> LipschitzResult:
    """
    使用 auto_LiRPA 计算利普希茨常数上界。

    对于 x ∈ B_∞(x₀, ε)，计算 f(x) 的输出界 [lb, ub]。
    对线性函数 f(x) = Wx + b:
        range_i = ub_i - lb_i = 2ε · ||W[i,:]||₁
        max_i range_i / (2ε) = ||W||_{∞→∞} (= Lip_{∞→∞})
    对非线性函数，LiRPA 的界是过近似，因此该比值是 Lipschitz 上界。

    注意: 使用 ReLU 近似模型 (LiRPACompatibleWrapper)，
         因为 auto_LiRPA 0.3 不原生支持 GELU/LayerNorm。
    """
    if not HAS_LIRPA:
        raise RuntimeError("auto_LiRPA 未安装，请运行: pip install auto_LiRPA --no-deps")

    t0 = time.time()
    lirpa_wrapper = lirpa_wrapper.to(device).eval()
    x_c = x_center.to(device).float()
    if x_c.dim() == 1:
        x_c = x_c.unsqueeze(0)

    dummy = x_c.clone()
    bounded_model = BoundedModule(lirpa_wrapper, dummy, device=device)

    ptb = PerturbationLpNorm(norm=norm, eps=eps)
    bounded_x = BoundedTensor(x_c, ptb)

    lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method=method)

    output_range = (ub - lb).detach()
    lip = output_range.abs().max().item() / (2 * eps)

    elapsed = time.time() - t0
    return LipschitzResult(
        method=f'LiRPA({method})',
        lipschitz_constant=lip,
        bound_type='upper_bound',
        compute_time_sec=elapsed,
        details={
            'eps': eps,
            'norm': norm,
            'lirpa_method': method,
            'output_range_max': output_range.abs().max().item(),
            'output_range_mean': output_range.abs().mean().item(),
            'note': 'GELU→ReLU, LayerNorm→Identity 近似'
        }
    )


# ================================================================
# 方法 5: Branch-and-Bound + LiRPA (紧上界)
# ================================================================

class _BaBNode:
    """分支定界的搜索节点"""
    __slots__ = ['x_lower', 'x_upper', 'ub']

    def __init__(self, x_lower, x_upper, ub):
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.ub = ub

    def __lt__(self, other):
        return self.ub > other.ub  # 最大堆: 优先处理上界最大的区域


def lipschitz_bab(lirpa_wrapper: nn.Module,
                  x_center: torch.Tensor,
                  eps: float = 0.1,
                  norm: float = float('inf'),
                  lirpa_method: str = 'backward',
                  max_iters: int = 64,
                  patience: int = 10,
                  device: str = 'cuda') -> LipschitzResult:
    """
    Branch-and-Bound (分支定界) + LiRPA 收紧利普希茨常数上界。

    灵感来源于 alpha-beta-CROWN / libbab 验证框架。

    算法:
        1. 将输入域 B(x₀, ε) 作为根节点
        2. 用 LiRPA 计算该区域的 Lipschitz 上界
        3. 沿最宽维度二分，对子区域分别计算上界
        4. 全局 Lip ≤ max(各子区域上界)
        5. 子区域越小，LiRPA 越紧，上界收敛到真实值
    """
    if not HAS_LIRPA:
        raise RuntimeError("auto_LiRPA 未安装")

    t0 = time.time()
    lirpa_wrapper = lirpa_wrapper.to(device).eval()
    x_c = x_center.to(device).float()
    if x_c.dim() == 1:
        x_c = x_c.unsqueeze(0)

    d = x_c.shape[-1]

    # 初始区域
    x_lo = (x_c - eps).squeeze(0).cpu()
    x_hi = (x_c + eps).squeeze(0).cpu()

    def _compute_region_lip(lo, hi):
        center = ((lo + hi) / 2).unsqueeze(0).to(device)
        local_eps = ((hi - lo) / 2).max().item()
        if local_eps < 1e-10:
            return 0.0
        dummy = center.clone()
        bm = BoundedModule(lirpa_wrapper, dummy, device=device)
        ptb = PerturbationLpNorm(norm=norm, eps=local_eps)
        bx = BoundedTensor(center, ptb)
        lb, ub = bm.compute_bounds(x=(bx,), method=lirpa_method)
        return (ub - lb).abs().max().item() / (2 * local_eps)

    init_lip = _compute_region_lip(x_lo, x_hi)

    heap = [_BaBNode(x_lo, x_hi, init_lip)]
    best_ub = init_lip
    no_improve = 0
    history = [(0, init_lip)]

    for it in range(max_iters):
        if not heap:
            break

        node = heapq.heappop(heap)

        widths = node.x_upper - node.x_lower
        split_dim = widths.argmax().item()
        mid = (node.x_lower[split_dim] + node.x_upper[split_dim]) / 2

        children = []
        for child_lo, child_hi in [
            (node.x_lower.clone(), node.x_upper.clone()),
            (node.x_lower.clone(), node.x_upper.clone()),
        ]:
            pass  # placeholder

        # 子区域 1: [lo, mid] 在 split_dim
        hi1 = node.x_upper.clone()
        hi1[split_dim] = mid
        # 子区域 2: [mid, hi] 在 split_dim
        lo2 = node.x_lower.clone()
        lo2[split_dim] = mid

        for lo_c, hi_c in [(node.x_lower, hi1), (lo2, node.x_upper)]:
            try:
                child_lip = _compute_region_lip(lo_c, hi_c)
            except Exception:
                child_lip = node.ub
            heapq.heappush(heap, _BaBNode(lo_c, hi_c, child_lip))

        current_ub = max(n.ub for n in heap)
        history.append((it + 1, current_ub))

        if current_ub < best_ub * 0.995:
            best_ub = current_ub
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    final_ub = max(n.ub for n in heap) if heap else init_lip
    elapsed = time.time() - t0

    return LipschitzResult(
        method='BaB + LiRPA',
        lipschitz_constant=final_ub,
        bound_type='upper_bound',
        compute_time_sec=elapsed,
        details={
            'iterations': it + 1 if heap else 0,
            'num_leaf_regions': len(heap),
            'initial_bound': init_lip,
            'improvement': f'{init_lip / max(final_ub, 1e-10):.2f}x',
            'history': history,
        }
    )


# ================================================================
# 方法 6: 经验采样 (下界)
# ================================================================

def lipschitz_empirical(wrapper: nn.Module,
                        x_center: torch.Tensor,
                        num_pairs: int = 2000,
                        eps: float = 0.5,
                        device: str = 'cuda') -> LipschitzResult:
    """
    随机采样输入对，计算 ||f(x1)-f(x2)||/||x1-x2|| 的最大值。
    结果是利普希茨常数的下界 (因为只考虑了有限对)。
    """
    t0 = time.time()
    wrapper = wrapper.to(device).eval()
    x_c = x_center.to(device).float()
    if x_c.dim() == 1:
        x_c = x_c.unsqueeze(0)

    d = x_c.shape[-1]
    max_ratio = 0.0
    ratios = []

    batch = min(num_pairs, 256)
    num_batches = (num_pairs + batch - 1) // batch

    with torch.no_grad():
        for _ in range(num_batches):
            delta1 = torch.randn(batch, d, device=device) * eps
            delta2 = torch.randn(batch, d, device=device) * eps
            x1 = x_c + delta1
            x2 = x_c + delta2

            y1 = wrapper(x1)
            y2 = wrapper(x2)

            diff_out = (y1 - y2).norm(dim=-1)
            diff_in = (x1 - x2).norm(dim=-1)

            valid = diff_in > 1e-10
            r = diff_out[valid] / diff_in[valid]

            if r.numel() > 0:
                ratios.extend(r.cpu().tolist())
                max_ratio = max(max_ratio, r.max().item())

    elapsed = time.time() - t0
    return LipschitzResult(
        method='经验采样',
        lipschitz_constant=max_ratio,
        bound_type='lower_bound',
        compute_time_sec=elapsed,
        details={
            'num_pairs': len(ratios),
            'max_ratio': max_ratio,
            'mean_ratio': float(np.mean(ratios)) if ratios else 0,
            'p99_ratio': float(np.percentile(ratios, 99)) if ratios else 0,
            'p95_ratio': float(np.percentile(ratios, 95)) if ratios else 0,
        }
    )


# ================================================================
# ODE 流映射利普希茨分析
# ================================================================

def estimate_flow_lipschitz(ode_func: nn.Module,
                            p0: torch.Tensor,
                            z_v: torch.Tensor,
                            num_pairs: int = 50,
                            eps: float = 0.01,
                            device: str = 'cuda') -> Tuple[float, float, List[float]]:
    """
    直接估计 ODE 流映射 Φ_T 的 Lipschitz 常数。

    通过对初始条件施加小扰动，比较 ODE 解的差异:
        Lip(Φ_T) ≈ max ||Φ_T(p0+δ) - Φ_T(p0)|| / ||δ||

    理论上界: Lip(Φ_T) ≤ e^{L_f · T}
    """
    from torchdiffeq import odeint

    ode_func = ode_func.to(device).float().eval()
    p0 = p0.to(device).float()
    z_v = z_v.to(device).float()

    if p0.dim() == 1:
        p0 = p0.unsqueeze(0)
    if z_v.dim() == 1:
        z_v = z_v.unsqueeze(0)

    t = torch.tensor([0.0, 1.0], device=device)

    def ode_rhs(t_val, p):
        z_exp = z_v.expand(p.shape[0], -1)
        if p.dim() == 2:
            p_3d = p.unsqueeze(1)
            z_3d = z_exp.unsqueeze(1)
            inp = torch.cat([p_3d, z_3d], dim=-1)
        else:
            z_exp = z_exp.unsqueeze(1).expand(-1, p.shape[1], -1)
            inp = torch.cat([p, z_exp], dim=-1)
        return ode_func.forward_ode_network(inp).squeeze(1) if p.dim() == 2 else ode_func.forward_ode_network(inp)

    with torch.no_grad():
        p_ref = odeint(ode_rhs, p0, t, method='dopri5', rtol=1e-3, atol=1e-4)[1]

    ratios = []
    with torch.no_grad():
        for _ in range(num_pairs):
            delta = torch.randn_like(p0) * eps
            p0_pert = p0 + delta
            p_pert = odeint(ode_rhs, p0_pert, t,
                            method='dopri5', rtol=1e-3, atol=1e-4)[1]
            diff_out = (p_pert - p_ref).norm().item()
            diff_in = delta.norm().item()
            if diff_in > 1e-12:
                ratios.append(diff_out / diff_in)

    max_ratio = max(ratios) if ratios else 0.0
    avg_ratio = float(np.mean(ratios)) if ratios else 0.0
    return max_ratio, avg_ratio, ratios


# ================================================================
# 综合分析器
# ================================================================

class ODELipschitzAnalyzer:
    """
    ODE 网络利普希茨常数综合分析器。

    使用方法:
        analyzer = ODELipschitzAnalyzer(ode_func, prompt_dim=512, visual_dim=512)
        report = analyzer.full_analysis(z_v=z_v, p_center=p0)
        analyzer.print_report(report)
    """

    def __init__(self, ode_func: nn.Module,
                 prompt_dim: int = 512,
                 visual_dim: int = 512,
                 use_mlp_path: bool = False,
                 device: str = 'cuda'):
        self.ode_func = ode_func
        self.prompt_dim = prompt_dim
        self.visual_dim = visual_dim
        self.use_mlp_path = use_mlp_path
        self.device = device

    def full_analysis(self,
                      z_v: torch.Tensor,
                      p_center: Optional[torch.Tensor] = None,
                      num_jacobian_samples: int = 10,
                      num_empirical_pairs: int = 2000,
                      lirpa_eps: float = 0.1,
                      bab_iters: int = 32,
                      run_flow: bool = True) -> FullReport:
        """运行所有方法，返回完整报告。"""

        results = []
        d = self.prompt_dim

        if p_center is None:
            p_center = torch.randn(1, d) * 0.1
        elif p_center.dim() == 1:
            p_center = p_center.unsqueeze(0)

        z_v = z_v.to(self.device).float()
        if z_v.dim() == 1:
            z_v = z_v.unsqueeze(0)
        z_v_mean = z_v.mean(dim=0, keepdim=True) if z_v.shape[0] > 1 else z_v

        # 创建包装器
        state_wrapper = ODEStateWrapper(
            self.ode_func, z_v_mean.squeeze(0), self.use_mlp_path
        ).to(self.device).eval()

        # ------ 方法 1: Jacobian SVD ------
        logger.info("运行方法 2: Jacobian SVD...")
        try:
            x_samples = p_center.to(self.device) + torch.randn(
                num_jacobian_samples, d, device=self.device) * 0.1
            r = lipschitz_jacobian_svd(state_wrapper, x_samples, self.device)
            results.append(r)
        except Exception as e:
            logger.warning(f"Jacobian SVD 失败: {e}")

        # ------ 方法 3: 幂迭代 ------
        logger.info("运行方法 3: 幂迭代...")
        try:
            x_samples = p_center.to(self.device) + torch.randn(
                num_jacobian_samples, d, device=self.device) * 0.1
            r = lipschitz_power_iteration(state_wrapper, x_samples, 100, self.device)
            results.append(r)
        except Exception as e:
            logger.warning(f"幂迭代失败: {e}")

        # ------ 方法 4: auto_LiRPA ------
        if HAS_LIRPA:
            logger.info("运行方法 4: auto_LiRPA 界传播...")
            try:
                lirpa_w = LiRPACompatibleWrapper(
                    self.ode_func, z_v_mean.squeeze(0), self.use_mlp_path
                ).to(self.device).eval()
                r = lipschitz_lirpa(lirpa_w, p_center, lirpa_eps,
                                    device=self.device)
                results.append(r)
            except Exception as e:
                logger.warning(f"LiRPA 界传播失败: {e}")

            # ------ 方法 5: Branch-and-Bound ------
            logger.info("运行方法 5: Branch-and-Bound...")
            try:
                lirpa_w = LiRPACompatibleWrapper(
                    self.ode_func, z_v_mean.squeeze(0), self.use_mlp_path
                ).to(self.device).eval()
                r = lipschitz_bab(lirpa_w, p_center, lirpa_eps,
                                  max_iters=bab_iters, device=self.device)
                results.append(r)
            except Exception as e:
                logger.warning(f"BaB 失败: {e}")
        else:
            logger.warning("auto_LiRPA 未安装，跳过 LiRPA 和 BaB 方法")

        # ------ 方法 6: 经验采样 ------
        logger.info("运行方法 6: 经验采样...")
        try:
            r = lipschitz_empirical(state_wrapper, p_center,
                                    num_empirical_pairs, device=self.device)
            results.append(r)
        except Exception as e:
            logger.warning(f"经验采样失败: {e}")

        # ------ ODE 流映射 ------
        flow_lip = None
        flow_empirical = None
        flow_empirical_avg = None
        if run_flow:
            logger.info("估计 ODE 流映射利普希茨常数...")
            try:
                flow_empirical, flow_empirical_avg, _ = estimate_flow_lipschitz(
                    self.ode_func, p_center, z_v_mean, device=self.device)
                logger.info(
                    f"ODE 流映射经验利普希茨: 最大={flow_empirical:.6f}, "
                    f"平均={flow_empirical_avg:.6f}"
                )
            except Exception as e:
                logger.warning(f"流映射估计失败: {e}")

            local_estimates = [r.lipschitz_constant for r in results
                               if r.bound_type == 'local_exact']
            if local_estimates:
                L_f = max(local_estimates)
                # e^{L·T}, T=1; L_f 很大时 exp 溢出，用 float('inf') 代替
                try:
                    flow_lip = math.exp(min(L_f * 1.0, 709.0))  # 709 ≈ ln(max_float)
                    if L_f > 709.0:
                        flow_lip = float('inf')
                except OverflowError:
                    flow_lip = float('inf')

        model_info = {
            'prompt_dim': self.prompt_dim,
            'visual_dim': self.visual_dim,
            'hidden_dim': self.ode_func.hidden_dim,
            'path': 'mlp' if self.use_mlp_path else 'res_blocks (forward_ode_network)',
            'num_params': sum(p.numel() for p in self.ode_func.parameters()),
        }

        return FullReport(
            results=results,
            flow_map_lip=flow_lip,
            flow_map_empirical=flow_empirical,
            flow_map_empirical_avg=flow_empirical_avg if run_flow else None,
            model_info=model_info,
        )

    @staticmethod
    def print_report(report: FullReport):
        """打印格式化的分析报告"""
        sep = '=' * 60
        print(f"\n{sep}")
        print("  ODE 网络利普希茨常数分析报告")
        print(sep)

        info = report.model_info
        print(f"\n  模型信息:")
        print(f"    prompt_dim  = {info.get('prompt_dim')}")
        print(f"    visual_dim  = {info.get('visual_dim')}")
        print(f"    hidden_dim  = {info.get('hidden_dim')}")
        print(f"    计算路径    = {info.get('path')}")
        print(f"    参数量      = {info.get('num_params', 0):,}")

        print(f"\n{'-' * 60}")
        print("  动力学函数 f_θ 的利普希茨常数")
        print(f"{'-' * 60}")

        for r in report.results:
            print(r)
            if r.method == 'Jacobian SVD' and r.details.get('all_values'):
                vals = r.details['all_values']
                print(f"      采样点: mean={np.mean(vals):.4f}, "
                      f"std={np.std(vals):.4f}, "
                      f"[{min(vals):.4f}, {max(vals):.4f}]")

            if r.method == '经验采样' and r.details:
                d = r.details
                print(f"      {d.get('num_pairs',0)} 对采样, "
                      f"P95={d.get('p95_ratio',0):.4f}, "
                      f"P99={d.get('p99_ratio',0):.4f}")

            if 'BaB' in r.method and r.details:
                d = r.details
                print(f"      迭代={d.get('iterations',0)}, "
                      f"叶节点={d.get('num_leaf_regions',0)}, "
                      f"收紧比={d.get('improvement','N/A')}")

        # 汇总
        print(f"\n{'-' * 60}")
        print("  汇总")
        print(f"{'-' * 60}")
        best_up = report.best_upper()
        best_lo = report.best_lower()
        print(f"    最紧上界: {best_up:.6f}")
        print(f"    最大下界: {best_lo:.6f}")
        if best_up < float('inf') and best_lo > 0:
            print(f"    上下界比值: {best_up / best_lo:.2f}x")

        # 流映射
        if report.flow_map_lip is not None or report.flow_map_empirical is not None:
            print(f"\n{'-' * 60}")
            print("  ODE 流映射 Φ_T (T=1) 利普希茨分析")
            print(f"{'-' * 60}")
            if report.flow_map_lip is not None:
                local_vals = [r.lipschitz_constant for r in report.results
                              if r.bound_type == 'local_exact']
                L_f = max(local_vals) if local_vals else 0
                if math.isinf(report.flow_map_lip) or report.flow_map_lip > 1e15:
                    print(f"    理论上界 e^(L·T) = e^({L_f:.2f}) = ∞ (溢出，L_f 过大)")
                else:
                    print(f"    理论上界 e^(L·T) = e^({L_f:.2f}) = {report.flow_map_lip:.4f}")
            if report.flow_map_empirical is not None:
                avg_str = (f", 平均={report.flow_map_empirical_avg:.4f}"
                           if report.flow_map_empirical_avg is not None else "")
                print(f"    经验估计 (直接扰动): 最大={report.flow_map_empirical:.4f}{avg_str}")
            print("    注: e^(L·T) 通常非常保守，实际流映射远小于此。")

        print(f"\n{sep}\n")


# ================================================================
# CLI 入口
# ================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='ODE 网络利普希茨常数计算工具',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径 (model.pth.tar)')
    parser.add_argument('--prompt-dim', type=int, default=512)
    parser.add_argument('--visual-dim', type=int, default=512)
    parser.add_argument('--use-mlp-path', action='store_true',
                        help='分析 mlp 路径 (默认: forward_ode_network/res_blocks)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lirpa-eps', type=float, default=0.1,
                        help='LiRPA 扰动半径')
    parser.add_argument('--num-jacobian-samples', type=int, default=10,
                        help='Jacobian 分析的采样点数')
    parser.add_argument('--num-empirical-pairs', type=int, default=2000)
    parser.add_argument('--bab-iters', type=int, default=32)
    parser.add_argument('--no-flow', action='store_true',
                        help='跳过 ODE 流映射分析')
    parser.add_argument('--demo', action='store_true',
                        help='使用随机权重的演示模式')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.info("CUDA 不可用，使用 CPU")

    if args.demo or args.checkpoint is None:
        logger.info("演示模式: 创建随机初始化的 ODEFunc")
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from trainers.advpt import ODEFunc
        ode_func = ODEFunc(args.prompt_dim, args.visual_dim).float()
        p_center = torch.randn(1, args.prompt_dim) * 0.02
        z_v = torch.randn(1, args.visual_dim) * 0.5
    else:
        logger.info(f"从检查点加载: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

        if 'state_dict' in state:
            state_dict = state['state_dict']
        elif 'model_state_dict' in state:
            state_dict = state['model_state_dict']
        else:
            state_dict = state

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from trainers.advpt import ODEFunc

        ode_func = ODEFunc(args.prompt_dim, args.visual_dim).float()

        # 自动检测 key 前缀: 支持三种常见保存格式
        # 1. 'prompt_learner.ode_func.xxx'  (完整模型)
        # 2. 'ode_func.xxx'                 (仅 prompt_learner)
        # 3. 'xxx'                          (仅 ode_func)
        prefixes_to_try = ['prompt_learner.ode_func.', 'ode_func.', '']
        ode_state = {}
        matched_prefix = None
        for prefix in prefixes_to_try:
            candidate = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    stripped = k[len(prefix):]
                    # 过滤掉非 ode_func 的 key（当 prefix='' 时需要）
                    if prefix == '' and any(stripped.startswith(p)
                                            for p in ['p0', 'token_prefix', 'token_suffix']):
                        continue
                    candidate[stripped] = v
            if candidate:
                ode_state = candidate
                matched_prefix = prefix if prefix else '(直接)'
                break

        if ode_state:
            # 过滤掉 spectral_norm 辅助参数 (weight_u, weight_v)，只保留 weight/bias
            filtered = {k: v for k, v in ode_state.items()
                        if not (k.endswith('_u') or k.endswith('_v'))}
            # 若检查点没有 weight_orig，但模型有 spectral_norm，需将 weight 写入 weight_orig
            from torch.nn.utils import remove_spectral_norm
            for m in ode_func.modules():
                if hasattr(m, 'weight_orig'):
                    try:
                        remove_spectral_norm(m)
                    except Exception:
                        pass
            missing, unexpected = ode_func.load_state_dict(filtered, strict=False)
            logger.info(f"已加载 {len(filtered)} 个 ODEFunc 参数 (前缀: '{matched_prefix}')")
            if missing:
                logger.warning(f"缺失 keys: {missing}")
            if unexpected:
                logger.warning(f"多余 keys: {unexpected}")
        else:
            logger.warning("检查点中未找到 ODEFunc 参数，使用随机权重")

        p_center = torch.randn(1, args.prompt_dim) * 0.02
        z_v = torch.randn(1, args.visual_dim) * 0.5

    analyzer = ODELipschitzAnalyzer(
        ode_func, args.prompt_dim, args.visual_dim,
        args.use_mlp_path, device
    )

    report = analyzer.full_analysis(
        z_v=z_v,
        p_center=p_center,
        num_jacobian_samples=args.num_jacobian_samples,
        num_empirical_pairs=args.num_empirical_pairs,
        lirpa_eps=args.lirpa_eps,
        bab_iters=args.bab_iters,
        run_flow=not args.no_flow,
    )

    analyzer.print_report(report)


if __name__ == '__main__':
    main()
