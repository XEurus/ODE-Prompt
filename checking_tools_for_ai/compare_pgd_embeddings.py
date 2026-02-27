import os
import sys
# 添加项目根目录到路径以使用本地clip模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import clip.clip as clip
from utils.adv_utils import ImageNormalizer

# Paths
BASE_DIR = '/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning'
PKL_DIR = os.path.join(BASE_DIR, 'pkl_data_mix')
CLIP_WEIGHTS = os.path.join(BASE_DIR, 'clip', 'ViT-B-16.pt')
OUTPUT_PLOT = os.path.join(BASE_DIR, 'distribution_plot_pgd_vs_train_val.png')

FILES = {
    'train': 'OxfordPets_ViT-B_16_v2.pkl',
    'val': 'OxfordPets_ViT-B_16_val_v2.pkl',
    'pgd_images': 'OxfordPets_ViT-B_16_PGD.pkl'
}

# Sampling for visualization (set to None to use all samples)
MAX_SAMPLES_PER_SPLIT = 2000
RANDOM_SEED = 42


def maybe_subsample(data, labels, max_samples, seed=42):
    if max_samples is None or len(data) <= max_samples:
        return data, labels
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=max_samples, replace=False)
    return data[indices], labels[indices]


def load_clip_model(device):
    if not os.path.isfile(CLIP_WEIGHTS):
        raise FileNotFoundError(f"CLIP weights not found at {CLIP_WEIGHTS}")
    model, _ = clip.load(CLIP_WEIGHTS, device=device)
    model.eval()
    return model


def encode_pgd_images(pgd_tensor, model, device):
    pgd_min = pgd_tensor.min().item()
    pgd_max = pgd_tensor.max().item()
    print(f"PGD image tensor range: min={pgd_min:.4f}, max={pgd_max:.4f}")

    # Determine if normalization is needed
    needs_normalize = (pgd_min >= -0.1) and (pgd_max <= 1.2)
    if needs_normalize:
        print("PGD images appear to be in pixel space [0,1]; applying CLIP normalization...")
        normalizer = ImageNormalizer(device=device)
    else:
        print("PGD images appear to be already CLIP-normalized; skipping normalization...")
        normalizer = None

    batch_size = 64
    embeddings = []

    with torch.no_grad():
        for start in range(0, pgd_tensor.shape[0], batch_size):
            end = min(start + batch_size, pgd_tensor.shape[0])
            batch = pgd_tensor[start:end].to(device)
            if normalizer is not None:
                batch = normalizer.normalize(batch)
            batch = batch.type(model.dtype)
            emb = model.encode_image(batch)
            embeddings.append(emb.cpu().float())

    return torch.cat(embeddings, dim=0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load embeddings and PGD images
    print("Loading train/val embeddings and PGD images...")
    train_emb = torch.load(os.path.join(PKL_DIR, FILES['train']), map_location='cpu')
    val_emb = torch.load(os.path.join(PKL_DIR, FILES['val']), map_location='cpu')
    pgd_images = torch.load(os.path.join(PKL_DIR, FILES['pgd_images']), map_location='cpu')

    print(f"Train embeddings shape: {train_emb.shape}")
    print(f"Val embeddings shape: {val_emb.shape}")
    print(f"PGD images shape: {pgd_images.shape}")

    # Encode PGD images to embeddings
    print("Loading CLIP model and encoding PGD images...")
    clip_model = load_clip_model(device)
    pgd_emb = encode_pgd_images(pgd_images, clip_model, device)
    print(f"PGD embeddings shape: {pgd_emb.shape}")

    # Prepare data for visualization
    train_np = train_emb.numpy()
    val_np = val_emb.numpy()
    pgd_np = pgd_emb.numpy()

    labels = (
        ['Train'] * len(train_np)
        + ['Val'] * len(val_np)
        + ['PGD'] * len(pgd_np)
    )
    combined = np.vstack([train_np, val_np, pgd_np])
    labels = np.array(labels)

    # Subsample for faster visualization
    combined, labels = maybe_subsample(combined, labels, MAX_SAMPLES_PER_SPLIT * 3, seed=RANDOM_SEED)

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined)

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED)
    tsne_result = tsne.fit_transform(combined)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    color_map = {'Train': '#4C78A8', 'Val': '#F58518', 'PGD': '#54A24B'}
    alpha_map = {'Train': 0.5, 'Val': 0.8, 'PGD': 0.7}

    for label in ['Train', 'Val', 'PGD']:
        mask = labels == label
        axes[0].scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            label=label,
            alpha=alpha_map[label],
            s=10,
            c=color_map[label],
        )
    axes[0].set_title('PCA: Train vs Val vs PGD')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for label in ['Train', 'Val', 'PGD']:
        mask = labels == label
        axes[1].scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            label=label,
            alpha=alpha_map[label],
            s=10,
            c=color_map[label],
        )
    axes[1].set_title('t-SNE: Train vs Val vs PGD')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Saved plot to {OUTPUT_PLOT}")


if __name__ == '__main__':
    main()
