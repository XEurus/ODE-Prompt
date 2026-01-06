"""
神经网络 Hook 工具模块

本模块提供了用于提取神经网络中间层特征的 Hook 工具类。
在对抗攻击和可解释性研究中，经常需要访问模型的中间表示。

主要类：
    - SingleModelHook: 在 GPU 上收集中间特征
    - SingleModelHookCpu: 将中间特征转移到 CPU（节省显存）

使用场景：
    1. 特征可视化
    2. 对抗样本分析
    3. 梯度分析
    4. 中间表示提取

使用示例：
    >>> model = torchvision.models.resnet50(pretrained=True)
    >>> hook = SingleModelHook(model, 'layer4', use_inp=False)
    >>> output = model(images)
    >>> features = hook.get_hooked_value()  # 获取 layer4 的输出
    >>> hook.clear()  # 清理缓存
    >>> hook.remove()  # 移除 hook
"""

import torch
import gc


class SingleModelHook():
    """
    单模型前向/反向 Hook 类（GPU 版本）
    
    用于拦截和收集神经网络指定层的输入或输出。
    收集的张量保存在 GPU 上，适合需要进一步处理的场景。
    
    参数：
        model: PyTorch 模型
        name: 目标层的名称（如 'layer4', 'fc'）
        use_inp: 是否收集输入（True）还是输出（False）
        forward: 是否使用前向 hook（True）还是反向 hook（False）
    
    属性：
        _hooked_value: 收集的张量，多次前向传播会自动拼接
    """
    
    def __init__(self, model, name, use_inp=False, forward=True):
        """
        初始化 Hook
        
        查找模型中指定名称的层，并注册 hook 函数。
        """
        self.model = model
        self.name = name
        
        # 构建层名到模块的映射
        _dict = {n: m for n, m in self.model.named_modules()}
        if self.name not in _dict.keys():
            raise NameError(f"No such name ({self.name}) in the model")

        self._module = _dict[self.name]
        self._hooked_value = None
        self.use_inp = use_inp
        self.forward = forward

        self._register_hook()

    def clear(self):
        """
        清除已收集的张量
        
        在开始新的收集周期前调用，避免内存累积。
        """
        self._hooked_value = None

    def remove(self):
        """
        移除 hook
        
        当不再需要收集特征时调用，释放资源。
        """
        self.handle.remove()

    def _register_hook(self):
        """
        注册 hook 函数
        
        根据 use_inp 和 forward 参数选择合适的 hook 类型：
        - use_inp=True: 收集层的输入
        - use_inp=False: 收集层的输出
        - forward=True: 前向传播 hook
        - forward=False: 反向传播 hook
        """
        if self.use_inp:
            # 收集输入
            def hook(_, inp, __):
                if self._hooked_value is None:
                    self._hooked_value = inp[0]
                else:
                    # 多个 batch 自动拼接
                    self._hooked_value = torch.cat((self._hooked_value, inp[0]), dim=0)
        else:
            # 收集输出
            def hook(_, __, output):
                if self._hooked_value is None:
                    self._hooked_value = output
                else:
                    self._hooked_value = torch.cat((self._hooked_value, output), dim=0)

        if self.forward:
            self.handle = self._module.register_forward_hook(hook)
        else:
            self.handle = self._module.register_full_backward_hook(hook)

    def get_hooked_value(self):
        """
        获取收集的张量
        
        返回：
            收集的张量，形状 (total_samples, ...)
        """
        return self._hooked_value


class SingleModelHookCpu():
    """
    单模型前向/反向 Hook 类（CPU 版本）
    
    与 SingleModelHook 类似，但将收集的张量转移到 CPU。
    适合处理大规模数据或显存受限的场景。
    
    参数：
        model: PyTorch 模型
        name: 目标层的名称
        use_inp: 是否收集输入
        forward: 是否使用前向 hook
    
    优势：
        - 节省 GPU 显存
        - 适合收集大量中间特征
        - 适合离线分析
    """
    
    def __init__(self, model, name, use_inp=False, forward=True):
        """初始化 CPU 版本的 Hook"""
        self.model = model
        self.name = name
        _dict = {n: m for n, m in self.model.named_modules()}
        if self.name not in _dict.keys():
            raise NameError(f"No such name ({self.name}) in the model")

        self._module = _dict[self.name]
        self._hooked_value = None
        self.use_inp = use_inp
        self.forward = forward

        self._register_hook()

    def clear(self):
        """清除已收集的张量"""
        self._hooked_value = None

    def _register_hook(self):
        """
        注册 hook 函数（CPU 版本）
        
        与 GPU 版本的区别是，收集的张量会立即转移到 CPU。
        """
        if self.use_inp:
            def hook(_, inp, __):
                if self._hooked_value is None:
                    self._hooked_value = inp[0].cpu()
                else:
                    self._hooked_value = torch.cat((self._hooked_value, inp[0].cpu()), dim=0)
        else:
            def hook(_, __, output):
                if self._hooked_value is None:
                    self._hooked_value = output.cpu()
                else:
                    self._hooked_value = torch.cat((self._hooked_value, output.cpu()), dim=0)

        if self.forward:
            self.handle = self._module.register_forward_hook(hook)
        else:
            self.handle = self._module.register_full_backward_hook(hook)

    def get_hooked_value(self):
        """获取收集的张量（位于 CPU）"""
        return self._hooked_value

    def remove(self):
        """移除 hook"""
        self.handle.remove()
