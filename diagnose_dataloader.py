"""
诊断脚本：检测训练和测试数据加载器的问题

运行方式：
    python diagnose_dataloader.py

这个脚本会检测以下问题：
1. 两个 DataManager 实例的数据顺序是否一致
2. train_pkl 中的 embedding 和 batch label 是否对应
3. image_encoder 输出是否与 train_pkl 中的 embedding 一致
"""

import torch
import os
import argparse
from yacs.config import CfgNode as CN

# 导入项目模块
from dass.utils import setup_logger, set_random_seed
from dass.config import get_cfg_default
from dass.data import DataManager

import datasets.oxford_pets
import trainers.advpt

from clip import clip
from trainers.advpt import load_clip_to_cpu


def extend_cfg(cfg):
    """扩展配置"""
    cfg.TRAINER.ADV = CN()
    cfg.TRAINER.ADV.N_CTX = 32
    cfg.TRAINER.ADV.CSC = False
    cfg.TRAINER.ADV.CTX_INIT = ""
    cfg.TRAINER.ADV.PREC = "fp16"
    cfg.TRAINER.ADV.CLASS_TOKEN_POSITION = "end"
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATALOADER.TRAIN_X.BATCH_EMBEDDING_SIZE = 256
    cfg.DATASET.TRAIN_EPS = 16
    cfg.DATASET.TEST_EPS = 16
    cfg.DATASET.PGD_NUM_ITERS = 40
    cfg.MODEL.FILE_PREFIX = "model"


def diagnose():
    print("=" * 80)
    print("开始诊断数据加载器问题...")
    print("=" * 80)
    
    # 加载配置
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file("configs/datasets/oxford_pets.yaml")
    cfg.merge_from_file("configs/trainers/AdvPT/vit_b16.yaml")
    cfg.defrost()
    cfg.DATASET.ROOT = "/home/dji/Project/ODE-Prompt/ODE-Adversarial-Prompt-Tuning/Data"
    cfg.freeze()
    
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    else:
        set_random_seed(1)
    
    batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
    
    # ========================================================================
    # 测试 1: 检查两个 DataManager 实例的数据顺序是否一致
    # ========================================================================
    print("\n" + "=" * 80)
    print("[测试 1] 检查两个 DataManager 实例的数据顺序一致性")
    print("=" * 80)
    
    # 创建第一个 DataManager (notransform_noshuffle)
    dm1 = DataManager(cfg, batch_size, adv='notransform_noshuffle')
    data_source_1 = dm1.train_loader_x.dataset.data_source
    
    # 创建第二个 DataManager (noshuffle)
    dm2 = DataManager(cfg, batch_size, adv='noshuffle')
    data_source_2 = dm2.train_loader_x.dataset.data_source
    
    # 比较数据源
    print(f"DataManager 1 (notransform_noshuffle) 数据量: {len(data_source_1)}")
    print(f"DataManager 2 (noshuffle) 数据量: {len(data_source_2)}")
    
    if len(data_source_1) != len(data_source_2):
        print("❌ 错误: 两个 DataManager 的数据量不一致!")
    else:
        # 检查前 10 个样本的路径和标签是否一致
        mismatch_count = 0
        for i in range(min(len(data_source_1), 100)):
            if data_source_1[i].impath != data_source_2[i].impath:
                mismatch_count += 1
                if mismatch_count <= 5:
                    print(f"  样本 {i} 不匹配:")
                    print(f"    DM1: {data_source_1[i].impath}")
                    print(f"    DM2: {data_source_2[i].impath}")
        
        if mismatch_count > 0:
            print(f"❌ 错误: 前 100 个样本中有 {mismatch_count} 个不匹配!")
            print("   这会导致训练时 label 和 embedding 不对应!")
        else:
            print("✅ 前 100 个样本顺序一致")
            
    # ========================================================================
    # 测试 2: 检查 test_loader 的顺序稳定性
    # ========================================================================
    print("\n" + "=" * 80)
    print("[测试 2] 检查 test_loader 的顺序稳定性")
    print("=" * 80)
    
    # 两次迭代 test_loader，检查顺序是否一致
    test_loader = dm1.test_loader
    
    # 第一次迭代
    first_iter_paths = []
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 3:
            break
        first_iter_paths.extend(batch['impath'][:3])
    
    # 第二次迭代
    second_iter_paths = []
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 3:
            break
        second_iter_paths.extend(batch['impath'][:3])
    
    if first_iter_paths == second_iter_paths:
        print("✅ test_loader 两次迭代顺序一致")
    else:
        print("❌ 错误: test_loader 两次迭代顺序不一致!")
        print(f"  第一次: {first_iter_paths[:3]}")
        print(f"  第二次: {second_iter_paths[:3]}")
    
    # ========================================================================
    # 测试 3: 检查 train_pkl 与当前模型 image_encoder 输出的一致性
    # ========================================================================
    print("\n" + "=" * 80)
    print("[测试 3] 检查 train_pkl 与 image_encoder 输出的一致性")
    print("=" * 80)
    
    pkl_path = './pkl_data/OxfordPets_ViT-B_16.pkl'
    if not os.path.exists(pkl_path):
        print(f"⚠️ train_pkl 文件不存在: {pkl_path}")
        print("  请先运行训练以生成 train_pkl")
    else:
        train_pkl = torch.load(pkl_path, weights_only=False)
        print(f"train_pkl 形状: {train_pkl.shape}")
        print(f"train_pkl dtype: {train_pkl.dtype}")
        
        # 加载 CLIP 模型 (使用与 before_adv_train 相同的方式)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 方式 1: clip.load (before_adv_train 使用)
        clip_model_1, _ = clip.load(cfg.MODEL.BACKBONE.NAME, device='cpu')
        if cfg.TRAINER.ADV.PREC == "fp32" or cfg.TRAINER.ADV.PREC == "amp":
            clip_model_1.float()
        clip_model_1 = clip_model_1.to(device)
        
        # 方式 2: load_clip_to_cpu (CustomCLIP 使用)
        clip_model_2 = load_clip_to_cpu(cfg)
        if cfg.TRAINER.ADV.PREC == "fp32" or cfg.TRAINER.ADV.PREC == "amp":
            clip_model_2.float()
        clip_model_2 = clip_model_2.to(device)
        
        # 取一个测试图像
        train_loader = dm1.train_loader_x
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 1:
                break
            images = batch['img'].to(device)
            labels = batch['label']
            
            print(f"\n测试 batch {batch_idx}:")
            print(f"  图像形状: {images.shape}")
            print(f"  标签: {labels[:5].tolist()}")
            
            with torch.no_grad():
                # 方式 1 的输出
                emb1 = clip_model_1.encode_image(images)
                # 方式 2 的输出 (直接用 visual)
                emb2 = clip_model_2.visual(images.type(clip_model_2.dtype))
            
            # 比较两种方式
            diff_12 = (emb1 - emb2).abs().mean().item()
            print(f"  clip.load vs load_clip_to_cpu 差异: {diff_12:.8f}")
            
            if diff_12 < 1e-5:
                print("  ✅ 两种加载方式的输出一致")
            else:
                print("  ❌ 警告: 两种加载方式的输出有差异!")
                
            # 比较 train_pkl 中的 embedding
            # 注意：train_pkl 是用 notransform_noshuffle loader 生成的
            # 这里用的是 standard loader，图像可能有归一化差异
            pkl_emb = train_pkl[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            if pkl_emb.shape[0] == emb1.shape[0]:
                diff_pkl = (emb1.cpu() - pkl_emb).abs().mean().item()
                print(f"  train_pkl vs 当前计算的 embedding 差异: {diff_pkl:.6f}")
                
                if diff_pkl < 1e-3:
                    print("  ✅ train_pkl 与当前计算一致")
                else:
                    print("  ❌ 警告: train_pkl 与当前计算有较大差异!")
                    print("     可能原因：图像预处理（归一化）不一致")
                    
                    # 进一步检查：打印前 5 个样本的 embedding 统计
                    print(f"\n  pkl embedding 统计 (前5个样本):")
                    for i in range(min(5, pkl_emb.shape[0])):
                        print(f"    样本 {i}: mean={pkl_emb[i].mean():.4f}, std={pkl_emb[i].std():.4f}")
                    
                    print(f"\n  当前计算的 embedding 统计 (前5个样本):")
                    for i in range(min(5, emb1.shape[0])):
                        print(f"    样本 {i}: mean={emb1[i].cpu().mean():.4f}, std={emb1[i].cpu().std():.4f}")
    
    # ========================================================================
    # 测试 4: 检查训练循环中的 label-embedding 对应关系
    # ========================================================================
    print("\n" + "=" * 80)
    print("[测试 4] 模拟训练循环中的 label-embedding 对应关系")
    print("=" * 80)
    
    if os.path.exists(pkl_path):
        train_pkl = torch.load(pkl_path, weights_only=False).to('cpu')
        
        # 模拟 run_epoch_adv 的打乱操作
        from torch import randperm
        
        # 获取 noshuffle loader
        dm_noshuffle = DataManager(cfg, batch_size, adv='noshuffle')
        train_loader_noshuffle = dm_noshuffle.train_loader_x
        
        # 模拟打乱
        length = randperm(len(train_loader_noshuffle.dataset.data_source)).tolist()
        original_data_source = train_loader_noshuffle.dataset.data_source.copy()
        train_loader_noshuffle.dataset.data_source = [original_data_source[i] for i in length]
        shuffled_train_pkl = train_pkl[torch.LongTensor(length)]
        
        print(f"打乱后的数据量: {len(train_loader_noshuffle.dataset.data_source)}")
        print(f"打乱后的 train_pkl 形状: {shuffled_train_pkl.shape}")
        
        # 检查打乱后的第一个 batch
        for batch_idx, batch in enumerate(train_loader_noshuffle):
            if batch_idx >= 1:
                break
                
            labels = batch['label']
            impaths = batch['impath']
            
            # 对应的 train_pkl embedding
            emb_batch = shuffled_train_pkl[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            
            print(f"\n第一个 batch:")
            print(f"  标签: {labels[:5].tolist()}")
            print(f"  图像路径 (前3个):")
            for p in impaths[:3]:
                print(f"    {p}")
            print(f"  对应的 embedding 形状: {emb_batch.shape}")
            
            # 注意：这里无法直接验证 label 和 embedding 是否对应
            # 因为 train_pkl 是用 notransform_noshuffle 生成的
            # 需要确保两个 data_source 的初始顺序一致
            
            print("\n⚠️ 注意: 如果测试 1 显示顺序一致，则 label-embedding 对应应该正确")
            print("   如果测试 1 显示顺序不一致，则这是导致训练效果与测试差距大的原因!")
    
    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("诊断总结")
    print("=" * 80)
    print("""
可能导致训练效果(100%)与测试效果(30-40%)差距大的原因:

1. 【关键】训练时使用预计算的 embedding (train_pkl)，跳过了 image_encoder
   测试时使用完整的图像通过 image_encoder 计算 embedding
   → 解决方案: 确保 train_pkl 的生成方式与测试时 image_encoder 完全一致

2. 【关键】两个 DataManager 实例的数据顺序可能不同
   → train_pkl 和 batch['label'] 可能不对应
   → 解决方案: 使用同一个 DataManager 实例，或验证顺序一致性

3. 【可能】test_pkl 和 test_loader 的顺序不一致
   → 测试时对抗样本和标签不对应
   → 解决方案: 在 test_pkl 中同时保存 label

4. 【可能】CLIP 模型加载方式不一致 (clip.load vs load_clip_to_cpu)
   → 解决方案: 统一使用同一个模型实例
""")


if __name__ == "__main__":
    diagnose()
