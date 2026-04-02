#!/usr/bin/env python3
"""检测 pkl 文件结构，显示形状和大小信息"""
import pickle
import torch
import sys
import os

def inspect_pkl(pkl_path):
    """检查单个 pkl 文件的结构"""
    print(f"\n{'='*60}")
    print(f"文件: {pkl_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(pkl_path):
        print(f"文件不存在: {pkl_path}")
        return
    
    file_size = os.path.getsize(pkl_path)
    print(f"文件大小: {file_size / (1024**3):.2f} GB ({file_size:,} bytes)")
    
    try:
        # 使用 torch.load 读取 PyTorch 保存的文件
        data = torch.load(pkl_path, map_location='cpu')
        
        if isinstance(data, torch.Tensor):
            print(f"类型: torch.Tensor")
            print(f"形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
            print(f"设备: {data.device}")
            print(f"内存占用: {data.element_size() * data.nelement() / (1024**3):.2f} GB")
        elif isinstance(data, dict):
            print(f"类型: dict，包含 {len(data)} 个键")
            for k, v in list(data.items())[:5]:  # 只显示前5个
                if isinstance(v, torch.Tensor):
                    print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  - {k}: type={type(v)}")
        elif isinstance(data, (list, tuple)):
            print(f"类型: {type(data).__name__}，长度: {len(data)}")
            for i, v in enumerate(data[:3]):  # 只显示前3个
                if isinstance(v, torch.Tensor):
                    print(f"  [{i}]: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  [{i}]: type={type(v)}")
        else:
            print(f"类型: {type(data)}")
            print(f"内容: {data}")
            
    except Exception as e:
        print(f"读取错误: {e}")

if __name__ == "__main__":
    # 对比两个目录的pkl文件
    import glob
    
    print("\n" + "="*60)
    print("检测 pkl_data (原始单重启)")
    print("="*60)
    
    for pkl_file in sorted(glob.glob("/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/pkl_data/*.pkl")):
        inspect_pkl(pkl_file)
    
    print("\n" + "="*60)
    print("检测 pkl_data_mix (多重启)")
    print("="*60)
    
    for pkl_file in sorted(glob.glob("/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/pkl_data_mix/*.pkl")):
        inspect_pkl(pkl_file)
