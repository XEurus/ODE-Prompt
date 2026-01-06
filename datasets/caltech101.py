"""
Caltech-101 数据集加载器

数据集简介：
    - 包含 101 个物体类别 + 1 个背景类别
    - 每类约 40-800 张图像（共约 9,000 张）
    - 任务：通用物体识别

数据集结构：
    caltech-101/
    ├── 101_ObjectCategories/      # 图像目录
    │   ├── accordion/
    │   │   ├── image_0001.jpg
    │   │   └── ...
    │   ├── airplanes/
    │   └── ...
    ├── split_zhou_Caltech101.json # 标准划分
    └── split_fewshot/             # Few-shot 划分缓存

特殊处理：
    - 忽略的类别：BACKGROUND_Google, Faces_easy
    - 类别名重映射：airplanes -> airplane, Faces -> face 等

使用方式：
    cfg.DATASET.NAME = "Caltech101"
    cfg.DATASET.ROOT = "/path/to/data"
"""

import os
import pickle

from dass.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dass.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

# 忽略的类别（背景和简单人脸）
IGNORED = ["BACKGROUND_Google", "Faces_easy"]

# 类别名重映射（统一命名）
NEW_CNAMES = {
    "airplanes": "airplane",  # 复数 -> 单数
    "Faces": "face",          # 大写 -> 小写
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):
    """
    Caltech-101 数据集类
    
    特点：
        - 经典的小规模图像分类数据集
        - 类别分布不均衡
        - 支持 Few-shot 学习设置
    """

    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        """
        初始化 Caltech-101 数据集
        
        参数：
            cfg: 配置对象
        
        流程：
            1. 加载或创建数据划分
            2. 处理 Few-shot 采样
            3. 处理类别子采样
        """
        # 设置路径
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # 加载或创建数据划分
        if os.path.exists(self.split_path):
            # 使用预定义划分
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            # 创建新划分（复用 DTD 的读取和划分方法）
            train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # Few-shot 采样
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # 类别子采样
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
