import torch
import os
import numpy as np

data_dir = '/home/dji/Project/ODE-Prompt/ODE-Adversarial-Prompt-Tuning/pkl_data'
files = [
    'OxfordPets_ViT-B_16_PGD.pkl',
    'OxfordPets_ViT-B_16_v2.pkl',
    'OxfordPets_ViT-B_16_val_v2.pkl'
]

for f in files:
    path = os.path.join(data_dir, f)
    print(f"Loading {f}...")
    try:
        data = torch.load(path, map_location='cpu')
        
        if isinstance(data, torch.Tensor):
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Min: {data.min().item():.4f}")
            print(f"  Max: {data.max().item():.4f}")
            print(f"  Mean: {data.mean().item():.4f}")
            print(f"  Std: {data.std().item():.4f}")
        else:
            print(f"  Type: {type(data)}")
            
    except Exception as e:
        print(f"Error loading {f}: {e}")
    print("-" * 50)
