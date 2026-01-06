"""
ImageNet 数据集加载器

数据集简介：
    - ImageNet-1K：包含 1000 个类别
    - 约 120 万张训练图像，5 万张验证图像
    - 标准的大规模图像分类基准

数据集结构：
    imagenet/
    ├── images/
    │   ├── train/           # 训练集（按类别文件夹组织）
    │   │   ├── n01440764/   # WordNet synset ID
    │   │   │   ├── n01440764_10026.JPEG
    │   │   │   └── ...
    │   │   └── ...
    │   └── val/             # 验证集（同上结构）
    ├── classnames.txt       # 类别名称映射文件
    ├── preprocessed.pkl     # 预处理数据缓存
    └── split_fewshot/       # Few-shot 划分缓存

使用方式：
    cfg.DATASET.NAME = "ImageNet"
    cfg.DATASET.ROOT = "/path/to/data"
    cfg.DATASET.NUM_SHOTS = 16  # Few-shot 设置
"""

import os
import pickle
from collections import OrderedDict

from dass.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dass.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):
    """
    ImageNet-1K 数据集类
    
    特点：
        - 大规模：1000 类，百万级图像
        - 标准划分：使用验证集作为测试集
        - 支持 Few-shot 学习设置
    """

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        """
        初始化 ImageNet 数据集
        
        参数：
            cfg: 配置对象
        
        流程：
            1. 加载或预处理数据
            2. 处理 Few-shot 采样（如果指定）
            3. 处理类别子采样
        """
        # 设置路径
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # 加载或创建预处理数据
        if os.path.exists(self.preprocessed):
            # 使用缓存的预处理数据（加速加载）
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            # 首次运行：读取并预处理数据
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # 遵循标准实践：在验证集上评估
            test = self.read_data(classnames, "val")

            # 缓存预处理结果
            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Few-shot 采样
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # 类别子采样（复用 OxfordPets 的方法）
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """
        读取类别名称映射
        
        参数：
            text_file: 类别名称文件路径
        
        返回：
            classnames: 有序字典 {文件夹名: 类别名}
        
        文件格式：
            每行: <文件夹名> <类别名>
            例如: n01440764 tench
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        """
        读取数据集
        
        参数：
            classnames: 类别名称映射
            split_dir: 数据划分目录名（"train" 或 "val"）
        
        返回：
            items: Datum 对象列表
        
        目录结构：
            split_dir/
            ├── n01440764/
            │   ├── image1.JPEG
            │   └── ...
            └── ...
        """
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
