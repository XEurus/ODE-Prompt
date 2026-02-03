"""
诊断脚本：检查不同 DataManager 实例之间的数据对齐问题

主要检查：
1. 多个 DataManager 实例加载的数据是否顺序一致
2. val_loader 和 val_loader_notransform 的数据顺序是否一致
3. pkl 文件与 data_loader 的索引对齐是否正确
"""

import sys
import os
sys.path.insert(0, '/home/dji/Project/ODE-Prompt/ODE-Adversarial-Prompt-Tuning')

import torch
from dass.data.data_manager import DataManager, build_data_loader
from dass.config import get_cfg_default

# 注册数据集
import datasets.oxford_pets

def get_paths_from_loader(data_loader, max_batches=None):
    """从 data_loader 中提取所有样本的路径和标签"""
    paths = []
    labels = []
    for batch_idx, batch in enumerate(data_loader):
        if max_batches and batch_idx >= max_batches:
            break
        paths.extend(batch['impath'])
        labels.extend(batch['label'].tolist())
    return paths, labels


def main():
    # 加载配置
    cfg = get_cfg_default()
    cfg.DATASET.ROOT = '/home/dji/Project/ODE-Prompt/ODE-Adversarial-Prompt-Tuning/Data'
    cfg.DATASET.NAME = 'OxfordPets'
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 32
    cfg.DATALOADER.TEST.BATCH_SIZE = 32
    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.INTERPOLATION = 'bilinear'
    cfg.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
    cfg.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
    cfg.INPUT.TRANSFORMS = ['random_resized_crop', 'random_flip', 'normalize']
    cfg.VERBOSE = False
    cfg.DATASET.SUBSAMPLE_CLASSES = 'all'
    cfg.DATASET.NUM_SHOTS = -1
    cfg.SEED = 1
    
    batch_size = 32
    
    print("=" * 70)
    print("诊断：检查不同 DataManager 实例之间的数据对齐")
    print("=" * 70)
    
    # 创建多个 DataManager 实例（模拟 build_data_loader 中的行为）
    print("\n[1] 创建主 DataManager (dm)...")
    dm = DataManager(cfg, batch_size)
    
    print("\n[2] 创建 notransform DataManager (dm_notransform)...")
    dm_notransform = DataManager(cfg, batch_size, adv='notransform_noshuffle')
    
    # 比较验证集数据源
    print("\n" + "=" * 70)
    print("检查 val_loader vs val_loader_notransform 数据源")
    print("=" * 70)
    
    val_source_1 = dm.val_loader.dataset.data_source
    val_source_2 = dm_notransform.val_loader.dataset.data_source
    
    print(f"dm.val_loader 数据源长度: {len(val_source_1)}")
    print(f"dm_notransform.val_loader 数据源长度: {len(val_source_2)}")
    
    # 检查数据源是否相同
    if len(val_source_1) == len(val_source_2):
        mismatch_count = 0
        mismatch_examples = []
        for i in range(len(val_source_1)):
            if val_source_1[i].impath != val_source_2[i].impath:
                mismatch_count += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append((i, val_source_1[i].impath, val_source_2[i].impath))
            if val_source_1[i].label != val_source_2[i].label:
                mismatch_count += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append((i, f"label: {val_source_1[i].label}", f"label: {val_source_2[i].label}"))
        
        if mismatch_count == 0:
            print("✅ 数据源完全一致")
        else:
            print(f"❌ 数据源不一致! 发现 {mismatch_count} 处差异")
            for idx, v1, v2 in mismatch_examples:
                print(f"   索引 {idx}: {v1} vs {v2}")
    else:
        print("❌ 数据源长度不一致!")
    
    # 比较 DataLoader 迭代顺序
    print("\n" + "=" * 70)
    print("检查 DataLoader 迭代顺序")
    print("=" * 70)
    
    print("\n从 dm.val_loader 提取路径...")
    paths_1, labels_1 = get_paths_from_loader(dm.val_loader)
    
    print(f"从 dm_notransform.val_loader 提取路径...")
    paths_2, labels_2 = get_paths_from_loader(dm_notransform.val_loader)
    
    print(f"\ndm.val_loader 样本数: {len(paths_1)}")
    print(f"dm_notransform.val_loader 样本数: {len(paths_2)}")
    
    if len(paths_1) == len(paths_2):
        path_mismatch = sum(1 for p1, p2 in zip(paths_1, paths_2) if p1 != p2)
        label_mismatch = sum(1 for l1, l2 in zip(labels_1, labels_2) if l1 != l2)
        
        if path_mismatch == 0 and label_mismatch == 0:
            print("✅ DataLoader 迭代顺序完全一致")
        else:
            print(f"❌ DataLoader 迭代顺序不一致!")
            print(f"   路径不匹配数: {path_mismatch}")
            print(f"   标签不匹配数: {label_mismatch}")
            
            # 打印一些不匹配的例子
            print("\n   不匹配示例 (前5个):")
            count = 0
            for i, (p1, p2, l1, l2) in enumerate(zip(paths_1, paths_2, labels_1, labels_2)):
                if p1 != p2 or l1 != l2:
                    print(f"   索引 {i}: path1={os.path.basename(p1)}, path2={os.path.basename(p2)}, label1={l1}, label2={l2}")
                    count += 1
                    if count >= 5:
                        break
    else:
        print("❌ DataLoader 样本数不一致!")
    
    # 检查 pkl 文件与 val_loader 的对齐
    print("\n" + "=" * 70)
    print("检查 pkl 文件与 val_loader 的对齐")
    print("=" * 70)
    
    val_pkl_path = '/home/dji/Project/ODE-Prompt/ODE-Adversarial-Prompt-Tuning/pkl_data/OxfordPets_ViT-B_16_val_v2.pkl'
    if os.path.exists(val_pkl_path):
        val_pkl = torch.load(val_pkl_path, map_location='cpu')
        print(f"val_pkl 形状: {val_pkl.shape}")
        print(f"dm.val_loader 样本数: {len(paths_1)}")
        print(f"dm_notransform.val_loader 样本数: {len(paths_2)}")
        
        if val_pkl.shape[0] == len(paths_1) == len(paths_2):
            print("✅ 数量一致")
        else:
            print("❌ 数量不一致!")
    else:
        print(f"⚠️ val_pkl 文件不存在: {val_pkl_path}")
    
    # 检查训练集
    print("\n" + "=" * 70)
    print("检查训练集 DataLoader")
    print("=" * 70)
    
    train_source_1 = dm.train_loader_x.dataset.data_source
    train_source_2 = dm_notransform.train_loader_x.dataset.data_source
    
    print(f"dm.train_loader_x 数据源长度: {len(train_source_1)}")
    print(f"dm_notransform.train_loader_x 数据源长度: {len(train_source_2)}")
    
    # 检查数据源是否相同
    if len(train_source_1) == len(train_source_2):
        mismatch_count = 0
        for i in range(len(train_source_1)):
            if train_source_1[i].impath != train_source_2[i].impath or train_source_1[i].label != train_source_2[i].label:
                mismatch_count += 1
        
        if mismatch_count == 0:
            print("✅ 训练集数据源完全一致")
        else:
            print(f"❌ 训练集数据源不一致! 发现 {mismatch_count} 处差异")
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
