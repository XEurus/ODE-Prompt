"""
白盒 PGD100 eps=1/255 攻击原版 CLIP 零样本分类
与论文 NAP-Tuning 的评测标准对齐
"""
import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm
from dass.engine import build_trainer
from dass.utils import set_random_seed, setup_logger
from train import setup_cfg
from utils.adv_utils import ImageNormalizer

device = 'cuda'

class Args:
    prompt_learner_dir = './8_adamw-pla_1e4_resnet_10/prompt_learner'
    root = '/autodl-fs/data/Data'
    output_dir = '/tmp/whitebox_test'
    resume = ''; seed = 1; trainer = 'AdvPT'
    backbone = ''; head = ''
    model_file = 'resnet_model.pth.tar'
    config_file = 'configs/trainers/AdvPT/vit_b16/oxford_pets.yaml'
    dataset_config_file = 'configs/datasets/oxford_pets.yaml'
    path = './pkl_pgd100_eps1'
    white_attack = 'PGD'; black_attack = ''; mode = 'test-only'
    csv_path = ''; plot_path = ''; note = 'None'
    opts = ['TRAINER.ADV.ODE_NETWORK_TYPE', 'resnet',
            'TRAINER.ADV.N_CTX', '32',
            'TRAINER.ADV.CLASS_TOKEN_POSITION', 'end',
            'TRAINER.ADV.CSC', 'False',
            'DATASET.TEST_EPS', '1',
            'DATASET.Test_PGD_NUM_ITERS', '100']

cfg = setup_cfg(Args())
set_random_seed(cfg.SEED)
setup_logger(cfg.OUTPUT_DIR)

trainer = build_trainer(cfg)
classnames = [trainer.lab2cname[i] for i in range(len(trainer.lab2cname))]

clip_model, _ = clip.load('ViT-B/16', device=device)
clip_model.float().eval()

normalizer = ImageNormalizer(device=device)

with torch.no_grad():
    prompts = [f"a photo of a {c.replace('_',' ')}." for c in classnames]
    tokens = clip.tokenize(prompts).to(device)
    text_features = clip_model.encode_text(tokens).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print(f"类别数: {len(classnames)}, 文本特征: {text_features.shape}", flush=True)

eps = 1.0 / 255.0
num_iters = 100
alpha = eps / num_iters * 2.5

logit_scale = clip_model.logit_scale.exp().detach()

def whitebox_pgd_attack(images, labels):
    images = images.clone().detach()
    delta = torch.zeros_like(images).uniform_(-eps, eps).to(device)
    delta = torch.clamp(images + delta, 0, 1) - images

    for _ in range(num_iters):
        delta.requires_grad_(True)
        adv_images = images + delta
        adv_norm = normalizer.normalize(adv_images)
        image_features = clip_model.encode_image(adv_norm).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.T
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        grad = delta.grad.detach().sign()
        delta = (delta.detach() + alpha * grad)
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(images + delta, 0, 1) - images

    return (images + delta.detach()).clamp(0, 1)

data_loader = trainer.test_loader_notransform
correct_clean = 0
correct_adv = 0
total = 0

print(f"\n{'='*60}", flush=True)
print(f"白盒 PGD: eps={eps:.6f} ({eps*255:.0f}/255), iters={num_iters}, alpha={alpha:.6f}", flush=True)
print(f"{'='*60}", flush=True)

for batch_idx, batch in enumerate(data_loader):
    images = batch['img'].to(device).float()
    labels = batch['label'].to(device)
    bs = labels.shape[0]

    with torch.no_grad():
        clean_norm = normalizer.normalize(images)
        clean_feats = clip_model.encode_image(clean_norm).float()
        clean_feats = clean_feats / clean_feats.norm(dim=-1, keepdim=True)
        clean_logits = logit_scale * clean_feats @ text_features.T
        correct_clean += (clean_logits.argmax(dim=-1) == labels).sum().item()

    adv_images = whitebox_pgd_attack(images, labels)

    with torch.no_grad():
        adv_norm = normalizer.normalize(adv_images)
        adv_feats = clip_model.encode_image(adv_norm).float()
        adv_feats = adv_feats / adv_feats.norm(dim=-1, keepdim=True)
        adv_logits = logit_scale * adv_feats @ text_features.T
        correct_adv += (adv_logits.argmax(dim=-1) == labels).sum().item()

    total += bs
    print(f"  [{batch_idx+1}/{len(data_loader)}] "
          f"clean={100.*correct_clean/total:.2f}% "
          f"adv={100.*correct_adv/total:.2f}%", flush=True)

clean_acc = 100. * correct_clean / total
adv_acc = 100. * correct_adv / total

print(f"\n{'='*60}", flush=True)
print(f"原版 CLIP ViT-B/16 — 白盒 PGD100 eps={eps*255:.0f}/255:", flush=True)
print(f"  干净准确率:  {clean_acc:.2f}%", flush=True)
print(f"  对抗准确率:  {adv_acc:.2f}%", flush=True)
print(f"  下降幅度:    {clean_acc - adv_acc:.2f}%", flush=True)
print(f"{'='*60}", flush=True)
