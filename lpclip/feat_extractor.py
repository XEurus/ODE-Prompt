"""
CLIP 特征提取器

本脚本用于从 CLIP 模型中提取图像特征，并保存为 .npz 文件。
提取的特征可用于线性探针 (Linear Probe) 和其他下游任务。

功能：
    - 加载预训练 CLIP 模型（默认 RN50）
    - 遍历数据集提取视觉特征
    - 将特征和标签保存为 NumPy 格式

使用示例：
    python feat_extractor.py \
        --root /path/to/data \
        --output-dir ./clip_feat \
        --dataset-config-file configs/datasets/oxford_pets.yaml \
        --split train

输出文件：
    {output_dir}/{dataset_name}/{split}.npz
    包含：
        - feature_list: 特征数组，形状 (N, feature_dim)
        - label_list: 标签数组，形状 (N,)
"""

import os, argparse
import numpy as np
import torch
import sys

sys.path.append(os.path.abspath(".."))

# 数据集导入
from datasets.oxford_pets import OxfordPets
from datasets.oxford_flowers import OxfordFlowers
from datasets.fgvc_aircraft import FGVCAircraft
from datasets.dtd import DescribableTextures
from datasets.eurosat import EuroSAT
from datasets.stanford_cars import StanfordCars
from datasets.food101 import Food101
from datasets.sun397 import SUN397
from datasets.caltech101 import Caltech101
from datasets.ucf101 import UCF101
from datasets.imagenet import ImageNet
from datasets.imagenetv2 import ImageNetV2
from datasets.imagenet_sketch import ImageNetSketch
from datasets.imagenet_a import ImageNetA
from datasets.imagenet_r import ImageNetR

from dass.utils import setup_logger, set_random_seed, collect_env_info
from dass.config import get_cfg_default
from dass.data.transforms import build_transform
from dass.data import DatasetWrapper

import clip


def print_args(args, cfg):
    """打印命令行参数和配置信息"""
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    """根据命令行参数重置配置"""
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    扩展默认配置
    
    添加自定义训练器相关的配置项。
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.OURS = CN()
    cfg.TRAINER.OURS.N_CTX = 10       # 上下文向量数量
    cfg.TRAINER.OURS.CSC = False      # 类别特定上下文
    cfg.TRAINER.OURS.CTX_INIT = ""    # 上下文初始化词语
    cfg.TRAINER.OURS.WEIGHT_U = 0.1   # 无监督损失权重


def setup_cfg(args):
    """设置完整配置"""
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 按优先级加载配置
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    reset_cfg(cfg, args)

    cfg.freeze()

    return cfg


def main(args):
    """
    主函数 - 执行特征提取
    
    流程：
        1. 设置配置和随机种子
        2. 加载数据集和数据加载器
        3. 加载 CLIP 模型
        4. 遍历数据提取特征
        5. 保存特征和标签
    """
    cfg = setup_cfg(args)
    
    # 设置随机种子
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    # 启用 cuDNN benchmark
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    # ========================================
    # 设置数据加载器
    # ========================================
    # 通过 eval 动态加载数据集类
    dataset = eval(cfg.DATASET.NAME)(cfg)

    # 选择数据划分
    if args.split == "train":
        dataset_input = dataset.train_x
    elif args.split == "val":
        dataset_input = dataset.val
    else:
        dataset_input = dataset.test

    # 构建数据加载器（使用测试时变换，不使用数据增强）
    tfm_train = build_transform(cfg, is_train=False)
    data_loader = torch.utils.data.DataLoader(
        DatasetWrapper(cfg, dataset_input, transform=tfm_train, is_train=False),
        batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        sampler=None,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )

    # ========================================
    # 设置 CLIP 模型
    # ========================================
    clip_model, _ = clip.load("RN50", "cuda", jit=False)
    clip_model.eval()
    
    # ========================================
    # 特征提取
    # ========================================
    feature_list = []
    label_list = []
    train_dataiter = iter(data_loader)
    
    for train_step in range(1, len(train_dataiter) + 1):
        batch = next(train_dataiter)
        data = batch["img"].cuda()
        
        # 提取视觉特征
        feature = clip_model.visual(data)
        feature = feature.cpu()
        
        # 收集特征和标签
        for idx in range(len(data)):
            feature_list.append(feature[idx].tolist())
        label_list.extend(batch["label"].tolist())
    
    # ========================================
    # 保存特征
    # ========================================
    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME)
    os.makedirs(save_dir, exist_ok=True)
    save_filename = f"{args.split}"
    np.savez(
        os.path.join(save_dir, save_filename),
        feature_list=feature_list,
        label_list=label_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="数据集根目录")
    parser.add_argument("--output-dir", type=str, default="", help="输出目录")
    parser.add_argument("--config-file", type=str, default="", help="配置文件路径")
    parser.add_argument("--dataset-config-file", type=str, default="",
                        help="数据集配置文件路径")
    parser.add_argument("--num-shot", type=int, default=1, help="Few-shot 样本数")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"],
                        help="数据划分")
    parser.add_argument("--trainer", type=str, default="", help="训练器名称")
    parser.add_argument("--backbone", type=str, default="", help="骨干网络名称")
    parser.add_argument("--head", type=str, default="", help="分类头名称")
    parser.add_argument("--seed", type=int, default=-1, help="随机种子")
    parser.add_argument("--eval-only", action="store_true", help="仅评估模式")
    args = parser.parse_args()
    main(args)
