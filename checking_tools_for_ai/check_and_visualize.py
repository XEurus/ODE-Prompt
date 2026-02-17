import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Paths
data_dir = '/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/pkl_data'
split_path = '/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/Data/oxford_pets/split_zhou_OxfordPets.json'
output_plot = 'distribution_plot.png'

# Load Split Info
print("Loading split info...")
with open(split_path, 'r') as f:
    split_data = json.load(f)

split_counts = {
    'train': len(split_data['train']),
    'val': len(split_data['val']),
    'test': len(split_data['test'])
}
print(f"Split counts from JSON: {split_counts}")

# Load Pickle Data
files = {
    'train': 'OxfordPets_ViT-B_16_v2.pkl',
    'val': 'OxfordPets_ViT-B_16_val_v2.pkl',
    'test': 'OxfordPets_ViT-B_16_PGD.pkl'
}

data_tensors = {}

print("\nLoading pickle files...")
for key, filename in files.items():
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        try:
            data = torch.load(path, map_location='cpu')
            data_tensors[key] = data
            print(f"Loaded {key}: {filename} | Shape: {data.shape}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    else:
        print(f"File not found: {path}")

# Check Consistency
print("\nChecking consistency...")
consistent = True
for key in ['train', 'val', 'test']:
    if key in data_tensors:
        pkl_count = data_tensors[key].shape[0]
        json_count = split_counts[key]
        if pkl_count == json_count:
            print(f"[OK] {key} set count matches: {pkl_count}")
        else:
            print(f"[FAIL] {key} set count MISMATCH: pkl={pkl_count}, json={json_count}")
            consistent = False
    else:
        print(f"[WARN] {key} data missing from pickle files")

if consistent:
    print("\nAll dataset counts match perfectly!")
else:
    print("\nThere are mismatches in dataset counts.")

# Visualization
print("\nGenerating visualization for Train and Val embeddings...")
if 'train' in data_tensors and 'val' in data_tensors:
    train_data = data_tensors['train'].numpy()
    val_data = data_tensors['val'].numpy()
    
    # Combine for dimensionality reduction
    combined_data = np.vstack([train_data, val_data])
    labels = np.array(['Train'] * len(train_data) + ['Val'] * len(val_data))
    
    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_data)
    
    # t-SNE (subset if data is too large for speed, but here it's small enough)
    print("Running t-SNE...")
    # Using a smaller perplexity since validation set is small (736)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(combined_data)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # PCA Plot
    for label in ['Train', 'Val']:
        mask = labels == label
        alpha = 0.5 if label == 'Train' else 0.8
        axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1], label=label, alpha=alpha, s=10)
    axes[0].set_title('PCA of Embeddings')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE Plot
    for label in ['Train', 'Val']:
        mask = labels == label
        alpha = 0.5 if label == 'Train' else 0.8
        axes[1].scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=label, alpha=alpha, s=10)
    axes[1].set_title('t-SNE of Embeddings')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
else:
    print("Skipping visualization: Train or Val data missing.")
