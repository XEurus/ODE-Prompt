import sys
import os
# Insert parent directory to sys.path to prioritize local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import clip
from utils.adv_utils import ImageNormalizer

# Paths
data_dir = '/home/dji/Project/ODE-Prompt/ODE-Adversarial-Prompt-Tuning/pkl_data_40_16'
split_path = '/home/dji/Project/ODE-Prompt/ODE-Adversarial-Prompt-Tuning/Data/oxford_pets/split_zhou_OxfordPets.json'
output_plot = 'distribution_plot.png'
clip_backbone = 'ViT-B/16'
clip_download_root = '/home/dji/Project/ODE-Prompt/ODE-Adversarial-Prompt-Tuning/clip'
encoder_batch_size = 32 if torch.cuda.is_available() else 32

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
    'test': 'OxfordPets_ViT-B_16_PGD.pkl',
    'clean': 'OxfordPets_ViT-B_16_clean.pkl'
}

data_tensors = {}


_encoder_state = {}


def _get_encoder_state():
    if _encoder_state:
        return _encoder_state

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading CLIP image encoder ({clip_backbone}) on {device}...")
    model, _ = clip.load(clip_backbone, device=device, download_root=clip_download_root)
    model.eval()
    _encoder_state.update(
        {
            'device': device,
            'dtype': model.dtype,
            'image_encoder': model.visual,
            'normalizer': ImageNormalizer(device=device),
        }
    )
    return _encoder_state


def _encode_images(tensor: torch.Tensor, name: str) -> np.ndarray:
    state = _get_encoder_state()
    image_encoder = state['image_encoder']
    dtype = state['dtype']
    device = state['device']
    normalizer = state['normalizer']

    images = tensor
    if images.dim() == 3:
        images = images.unsqueeze(0)

    images = images.float()
    img_min = images.min().item()
    img_max = images.max().item()
    if img_min >= 0.0 and img_max <= 1.0:
        images = normalizer.normalize(images)

    num_samples = images.shape[0]
    embedding_dim = image_encoder.output_dim
    embeddings = torch.empty(size=[num_samples, embedding_dim])

    image_encoder.eval()
    with torch.no_grad():
        for start in range(0, num_samples, encoder_batch_size):
            end = min(start + encoder_batch_size, num_samples)
            batch = images[start:end].to(device)
            embedding = image_encoder(batch.type(dtype))
            embeddings[start:end] = embedding.cpu().float()
            if (start // encoder_batch_size + 1) % 20 == 0:
                print(f"  Encoded {name}: {end}/{num_samples}")

    print(f"Encoded {name} images to embeddings: {embeddings.shape}")
    return embeddings.numpy()


def to_feature_array(tensor: torch.Tensor, name: str) -> np.ndarray:
    """Convert tensor to 2D feature array for plotting.

    If input is images, encode with CLIP image encoder (test path).
    """

    if tensor.dim() > 2:
        return _encode_images(tensor, name)
    return tensor.reshape(tensor.shape[0], -1).cpu().numpy()

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
for key in ['train', 'val', 'test', 'clean']:
    if key in data_tensors:
        pkl_count = data_tensors[key].shape[0]
        json_count = split_counts.get(key)
        if json_count is None:
            print(f"[WARN] {key} count not found in split JSON; found {pkl_count} in pickle")
            continue
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
print("\nGenerating visualization for Train, Val, Clean, and Test embeddings...")
required_keys = {'train', 'val', 'clean', 'test'}
if required_keys.issubset(data_tensors.keys()):
    train_data = to_feature_array(data_tensors['train'], 'train')
    val_data = to_feature_array(data_tensors['val'], 'val')
    clean_data = to_feature_array(data_tensors['clean'], 'clean')
    test_data = to_feature_array(data_tensors['test'], 'test')

    # Combine for dimensionality reduction
    combined_data = np.vstack([train_data, val_data, clean_data, test_data])
    labels = np.array(
        ['Train'] * len(train_data) +
        ['Val'] * len(val_data) +
        ['Clean'] * len(clean_data) +
        ['Test'] * len(test_data)
    )

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_data)

    # t-SNE (subset if data is too large for speed, but here it's small enough)
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(combined_data)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PCA Plot
    for label in ['Train', 'Val', 'Clean', 'Test']:
        mask = labels == label
        alpha = 0.5 if label == 'Train' else 0.8
        axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1], label=label, alpha=alpha, s=10)
    axes[0].set_title('PCA of Embeddings')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # t-SNE Plot
    for label in ['Train', 'Val', 'Clean', 'Test']:
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
    missing = required_keys - data_tensors.keys()
    print(f"Skipping visualization: missing data for {', '.join(sorted(missing))}.")
