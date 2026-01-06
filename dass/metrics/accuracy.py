"""
准确率计算模块

本模块提供了计算分类准确率的函数。
支持 Top-K 准确率计算。

主要函数：
    - compute_accuracy: 计算 Top-K 准确率

使用示例：
    >>> output = model(images)  # [batch, num_classes]
    >>> target = labels         # [batch]
    >>> top1, top5 = compute_accuracy(output, target, topk=(1, 5))
    >>> print(f"Top-1: {top1.item():.2f}%, Top-5: {top5.item():.2f}%")
"""


def compute_accuracy(output, target, topk=(1, )):
    """
    计算 Top-K 准确率
    
    对于每个样本，检查其真实标签是否在预测的前 K 个类别中。
    
    参数：
        output: 预测分数矩阵，形状 (batch_size, num_classes)
                可以是张量或包含张量的元组/列表
        target: 真实标签，形状 (batch_size,)
        topk: 要计算的 K 值元组
              例如 topk=(1, 5) 表示计算 Top-1 和 Top-5 准确率
    
    返回：
        list: 各 K 值对应的准确率（百分比形式）
    
    示例：
        >>> output = torch.randn(32, 1000)  # 32 样本，1000 类
        >>> target = torch.randint(0, 1000, (32,))
        >>> 
        >>> # 计算 Top-1 准确率
        >>> top1 = compute_accuracy(output, target, topk=(1,))
        >>> 
        >>> # 计算 Top-1 和 Top-5 准确率
        >>> top1, top5 = compute_accuracy(output, target, topk=(1, 5))
    
    算法：
        1. 获取每个样本预测分数最高的 K 个类别
        2. 检查真实标签是否在这 K 个类别中
        3. 统计正确预测的比例
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # 处理元组/列表输入（某些模型返回多个输出）
    if isinstance(output, (tuple, list)):
        output = output[0]

    # 获取 Top-K 预测
    # pred: 形状 (batch_size, maxk)，每行是该样本的 Top-maxk 类别索引
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # 转置为 (maxk, batch_size)
    
    # 比较预测和真实标签
    # correct: 布尔矩阵，形状 (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # 计算各 K 值的准确率
    res = []
    for k in topk:
        # 前 k 行中任意一行正确即算正确
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res
