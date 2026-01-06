"""
Oxford-IIIT Pet 数据集加载器

数据集简介：
    - 包含 37 种不同品种的宠物图像
    - 共约 7,400 张图像（每类约 200 张）
    - 任务：细粒度宠物品种分类

数据集结构：
    oxford_pets/
    ├── images/                    # 所有图像文件
    │   ├── Abyssinian_1.jpg
    │   ├── ...
    ├── annotations/               # 标注文件
    │   ├── trainval.txt
    │   ├── test.txt
    │   ├── ...
    ├── split_zhou_OxfordPets.json # 标准划分（Zhou et al.）
    └── split_fewshot/             # Few-shot 划分缓存

支持的功能：
    - 标准划分（train/val/test）
    - Few-shot 学习（按 NUM_SHOTS 采样）
    - 子采样类别（base/new/all）

使用方式：
    在配置中设置：
        cfg.DATASET.NAME = "OxfordPets"
        cfg.DATASET.ROOT = "/path/to/data"
        cfg.DATASET.NUM_SHOTS = 16  # Few-shot 设置
"""

import os
import pickle
import math
import random
from collections import defaultdict

from dass.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dass.utils import read_json, write_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):
    """
    Oxford-IIIT Pet 数据集类
    
    继承自 DatasetBase，提供标准的数据加载接口。
    支持全监督、Few-shot 和类别子采样等多种设置。
    """

    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        """
        初始化数据集
        
        参数：
            cfg: 配置对象，包含数据集相关设置
        
        流程：
            1. 设置数据集路径
            2. 加载或创建数据划分
            3. 处理 Few-shot 采样
            4. 处理类别子采样
        """
        # 设置数据集路径
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # 加载或创建数据划分
        if os.path.exists(self.split_path):
            # 使用预定义的标准划分
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            # 从原始标注文件创建划分
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        # Few-shot 采样
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                # 加载缓存的 Few-shot 数据
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                # 生成新的 Few-shot 数据并缓存
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # 类别子采样（用于 base-to-new 泛化实验）
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split_file):
        """
        从标注文件读取数据
        
        参数：
            split_file: 标注文件名（如 "trainval.txt"）
        
        返回：
            items: Datum 对象列表
        
        标注文件格式：
            每行: <图像名> <标签> <物种> <品种ID>
            例如: Abyssinian_1 1 1 1
        """
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                # 从图像名提取品种名（去除末尾的数字ID）
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # 转换为 0-based 索引
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        """
        将 trainval 集划分为训练集和验证集
        
        参数：
            trainval: 原始 trainval 数据列表
            p_val: 验证集比例（默认 20%）
        
        返回：
            train: 训练集数据列表
            val: 验证集数据列表
        
        策略：
            - 分层采样：确保每个类别都有验证样本
            - 随机打乱后按比例划分
        """
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        
        # 按类别组织样本
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        """
        保存数据划分到 JSON 文件
        
        参数：
            train, val, test: 数据列表
            filepath: 保存路径
            path_prefix: 图像路径前缀（用于相对路径转换）
        """
        def _extract(items):
            """提取并转换为相对路径"""
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        """
        从 JSON 文件读取数据划分
        
        参数：
            filepath: 划分文件路径
            path_prefix: 图像路径前缀
        
        返回：
            train, val, test: 数据列表
        """
        def _convert(items):
            """转换为 Datum 对象"""
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """
        类别子采样（用于 base-to-new 泛化实验）
        
        将类别分为两组：
            - base: 前一半类别，用于训练
            - new: 后一半类别，用于测试泛化能力
        
        参数：
            args: 数据集列表（如 train, val, test）
            subsample: 子采样模式
                - "all": 使用所有类别
                - "base": 只使用前半部分类别
                - "new": 只使用后半部分类别
        
        返回：
            重新标记后的数据集列表
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        # 获取所有类别标签
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        
        # 将类别分为两半
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # 前半部分
        else:
            selected = labels[m:]  # 后半部分
        
        # 创建重新标记映射
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        # 过滤并重新标记
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output
