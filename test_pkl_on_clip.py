"""
测试各 pkl 中的 embedding 在原版 CLIP 零样本分类器上的准确率
用于验证不同攻击方式生成的对抗 embedding 的实际攻击效果
"""
import torch
import clip
from dass.engine import build_trainer
from dass.utils import set_random_seed, setup_logger
from train import setup_cfg

device = 'cuda'

class Args:
    prompt_learner_dir = './8_adamw-pla_1e4_resnet_10/prompt_learner'
    root = '/autodl-fs/data/Data'
    output_dir = '/tmp/pkl_test'
    resume = ''; seed = 1; trainer = 'AdvPT'
    backbone = ''; head = ''
    model_file = 'resnet_model.pth.tar'
    config_file = 'configs/trainers/AdvPT/vit_b16/oxford_pets.yaml'
    dataset_config_file = 'configs/datasets/oxford_pets.yaml'
    path = './pkl_whitebox_pgd5_eps5'
    white_attack = 'PGD'; black_attack = ''; mode = 'test-only'
    csv_path = ''; plot_path = ''; note = 'None'
    opts = ['TRAINER.ADV.ODE_NETWORK_TYPE', 'resnet',
            'TRAINER.ADV.N_CTX', '32',
            'TRAINER.ADV.CLASS_TOKEN_POSITION', 'end',
            'TRAINER.ADV.CSC', 'False']

cfg = setup_cfg(Args())
set_random_seed(cfg.SEED)
setup_logger(cfg.OUTPUT_DIR)
trainer = build_trainer(cfg)

classnames = [trainer.lab2cname[i] for i in range(len(trainer.lab2cname))]
print(f"数据集: OxfordPets, 类别数: {len(classnames)}")

clip_model, _ = clip.load('ViT-B/16', device=device)
clip_model.float().eval()

with torch.no_grad():
    prompts = [f"a photo of a {c.replace('_',' ')}." for c in classnames]
    tokens = clip.tokenize(prompts).to(device)
    text_features = clip_model.encode_text(tokens).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

logit_scale = clip_model.logit_scale.exp().detach()
print(f"文本特征: {text_features.shape}, logit_scale: {logit_scale.item():.2f}")


def eval_pkl(pkl_path, labels, name):
    """在原版 CLIP 零样本分类头上测试 embedding 准确率"""
    emb = torch.load(pkl_path, weights_only=False)
    if emb.dim() == 3:
        N, R, D = emb.shape
        print(f"\n{'='*60}")
        print(f"[{name}]  shape={emb.shape} (含 {R} restarts)")
        emb_flat = emb.view(-1, D)
        labels_flat = labels.repeat_interleave(R)
    else:
        print(f"\n{'='*60}")
        print(f"[{name}]  shape={emb.shape}")
        emb_flat = emb
        labels_flat = labels

    emb_flat = emb_flat.to(device).float()
    emb_flat = emb_flat / emb_flat.norm(dim=-1, keepdim=True)

    logits = logit_scale * emb_flat @ text_features.T
    preds = logits.argmax(dim=-1)
    correct = (preds == labels_flat.to(device)).sum().item()
    total = labels_flat.shape[0]
    acc = 100. * correct / total
    print(f"  准确率: {correct}/{total} = {acc:.2f}%")

    if emb.dim() == 3:
        for r in range(R):
            emb_r = emb[:, r, :].to(device).float()
            emb_r = emb_r / emb_r.norm(dim=-1, keepdim=True)
            logits_r = logit_scale * emb_r @ text_features.T
            preds_r = logits_r.argmax(dim=-1)
            correct_r = (preds_r == labels.to(device)).sum().item()
            acc_r = 100. * correct_r / labels.shape[0]
            if r < 3 or r == R - 1:
                print(f"  restart {r}: {acc_r:.2f}%")
            elif r == 3:
                print(f"  ...")

    return acc


print("\n" + "="*60)
print("获取标签...")
print("="*60)

train_loader = trainer.train_loader_x_notransform_noshuffle
train_labels = torch.cat([b['label'] for b in train_loader])
print(f"训练集标签: {train_labels.shape}")

val_loader = trainer.val_loader_notransform
val_labels = torch.cat([b['label'] for b in val_loader])
print(f"验证集标签: {val_labels.shape}")

test_loader = trainer.test_loader_notransform
test_labels = torch.cat([b['label'] for b in test_loader])
print(f"测试集标签: {test_labels.shape}")

#PKL_DIR = "./pkl_whitebox_pgd5_eps5"
PKL_DIR = "./pkl_data_mix_5"

eval_pkl(f"{PKL_DIR}/OxfordPets_ViT-B_16_v2.pkl",
         train_labels, "训练集对抗 (PGD5 eps=5/255 whitebox, 10 restarts)")

eval_pkl(f"{PKL_DIR}/OxfordPets_ViT-B_16_val_v2.pkl",
         val_labels, "验证集对抗 (PGD5 eps=5/255 whitebox)")

# eval_pkl(f"{PKL_DIR}/OxfordPets_ViT-B_16_test_v2_whitebox.pkl",
#          test_labels, "测试集对抗 (PGD100 eps=1/255 whitebox)")

print(f"\n{'='*60}")

# clean_path = f"{PKL_DIR}/OxfordPets_ViT-B_16_clean.pkl"
# import os
# if os.path.isfile(clean_path):
#     eval_pkl(clean_path, train_labels, "训练集干净 (无攻击)")
# else:
#     print(f"\n[跳过] 干净 pkl 不存在: {clean_path}")

print(f"\n{'='*60}")
print("测试完成")
print(f"{'='*60}")
