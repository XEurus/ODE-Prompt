"""
模型评估器模块

本模块提供了用于评估模型性能的评估器类。
支持分类任务的各种评估指标。

主要类：
    - EvaluatorBase: 评估器基类
    - Classification: 分类任务评估器

评估指标：
    - accuracy: 准确率
    - error_rate: 错误率
    - macro_f1: 宏平均 F1 分数
    - perclass_accuracy: 每类准确率的平均值
    - confusion_matrix: 混淆矩阵

使用示例：
    >>> evaluator = Classification(cfg, lab2cname)
    >>> evaluator.reset()
    >>> for batch in dataloader:
    >>>     output = model(batch['img'])
    >>>     evaluator.process(output, batch['label'])
    >>> results = evaluator.evaluate()
"""

import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """
    评估器基类
    
    定义评估器的基本接口。
    
    参数：
        cfg: 配置对象
    
    方法：
        reset(): 重置评估状态
        process(mo, gt): 处理一批预测和真实标签
        evaluate(): 计算并返回评估结果
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        """重置评估状态"""
        raise NotImplementedError

    def process(self, mo, gt):
        """处理模型输出和真实标签"""
        raise NotImplementedError

    def evaluate(self):
        """计算评估指标"""
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """
    分类任务评估器
    
    计算分类任务的各种评估指标。
    
    参数：
        cfg: 配置对象
        lab2cname: 标签到类别名的映射字典
    
    属性：
        _correct: 正确预测的数量
        _total: 总样本数量
        _y_true: 真实标签列表
        _y_pred: 预测标签列表
        _per_class_res: 每类预测结果
    
    支持的指标：
        - accuracy: 整体准确率
        - error_rate: 错误率 (100 - accuracy)
        - macro_f1: 宏平均 F1 分数
        - perclass_accuracy: 每类准确率平均（如果启用）
    """

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        
        # 如果需要每类结果，必须提供类别名映射
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        """重置评估状态"""
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        """
        处理一批模型输出
        
        参数：
            mo: 模型输出，形状 [batch, num_classes]
            gt: 真实标签，形状 [batch]
        """
        # 获取预测类别（最大 logit 的索引）
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        # 记录用于计算 F1 分数
        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        # 记录每类结果
        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        """
        计算并返回评估结果
        
        返回：
            results: 有序字典，包含各种评估指标
        
        输出格式：
            => result
            * total: 1000
            * correct: 850
            * accuracy: 85.0%
            * error: 15.0%
            * macro_f1: 83.5%
        """
        results = OrderedDict()
        
        # 计算基本指标
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # 第一个值将被 trainer.test() 返回
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        # 打印结果
        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        # 每类结果
        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        # 混淆矩阵
        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
