#!/usr/bin/env python3
"""
对比测试：原版 ZeroShot CLIP ("a photo of [cls]") vs ODE-Prompt 训练版本

用法示例:
  python compare_baseline_vs_ode.py \
    --config-file configs/trainers/AdvPT/vit_b16/oxford_pets.yaml \
    --dataset-config-file configs/datasets/oxford_pets.yaml \
    --model-dir output/oxford_pets/AdvPT/vit_b16/adv/7_PGD40_16_mix-loss06_adamw-plateau_1e3_10 \
    --model-file model-best.pth.tar \
    --pkl-path ./pkl_data_mix_5 \
    --white-attack PGD \
    --black-attack RAP \
    TRAINER.ADV.N_CTX 32 TRAINER.ADV.CLASS_TOKEN_POSITION end TRAINER.ADV.CSC False
"""

import argparse
import sys
import os
from copy import deepcopy
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train import setup_cfg, build_trainer
from dass.utils import set_random_seed
from utils.adv_utils import ImageNormalizer


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Baseline vs ODE-Prompt robustness comparison")
    p.add_argument("--root",               type=str, default="./Data")
    p.add_argument("--output-dir",         type=str, default="")
    p.add_argument("--resume",             type=str, default="")
    p.add_argument("--seed",               type=int, default=1)
    p.add_argument("--trainer",            type=str, default="AdvPT")
    p.add_argument("--backbone",           type=str, default="")
    p.add_argument("--head",               type=str, default="")
    p.add_argument("--model-file",         type=str, default="")
    p.add_argument("--config-file",        type=str, required=True)
    p.add_argument("--dataset-config-file",type=str, required=True)
    p.add_argument("--model-dir",          type=str, default="",
                   help="目录，包含 prompt_learner/model-*.pth.tar；为空则跳过 ODE 评估")
    p.add_argument("--pkl-path",           type=str, default="./pkl_data_mix_5")
    p.add_argument("--white-attack",       type=str, default="PGD")
    p.add_argument("--black-attack",       type=str, default="RAP")
    p.add_argument("--baseline-prompt",    type=str, default="a photo of",
                   help="原版 ZeroShot 使用的提示词前缀")
    p.add_argument("--note",               type=str, default="None")
    p.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_clean(trainer):
    return float(trainer.test(split="test"))


@torch.no_grad()
def eval_adv(trainer, pkl_tensor, normalizer=None):
    """
    使用 pkl_tensor（已归一化）评估对抗准确率。
    若传入 normalizer，则做 clamp_perturbation；否则直接推理。
    """
    trainer.set_model_mode("eval")
    trainer.evaluator.reset()
    data_loader = trainer.test_loader

    if normalizer is None:
        normalizer = ImageNormalizer(device=trainer.device)
    normalizer.to(trainer.device)
    eps = trainer.cfg.DATASET.TEST_EPS / 255.0

    for batch_idx, batch in enumerate(tqdm(data_loader, leave=False)):
        inp, label = trainer.parse_batch_test(batch)
        s = batch_idx * data_loader.batch_size
        e = s + inp.shape[0]
        inp_adv = pkl_tensor[s:e].to(inp.device)
        inp_adv = normalizer.clamp_perturbation(inp_adv, inp, eps)
        output = trainer.model_inference(inp_adv)
        trainer.evaluator.process(output, label.to(inp.device))

    results = trainer.evaluator.evaluate()
    return list(results.values())[0]


def load_pkl_normalized(pkl_path: str) -> torch.Tensor:
    """加载原始像素空间 pkl 并 CLIP 归一化。"""
    norm = ImageNormalizer(device="cpu")
    raw = torch.load(pkl_path, weights_only=False)
    return norm.normalize(raw)


def print_row(label, clean, pgd, rap):
    clean_s = f"{clean:.2f}%" if clean is not None else "  N/A  "
    pgd_s   = f"{pgd:.2f}%"   if pgd   is not None else "  N/A  "
    rap_s   = f"{rap:.2f}%"   if rap   is not None else "  N/A  "
    print(f"  {label:<40s}  Clean={clean_s}  PGD={pgd_s}  RAP={rap_s}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    pkl_root = Path(args.pkl_path).resolve()
    dataset_name = cfg.DATASET.NAME
    backbone_name = cfg.MODEL.BACKBONE.NAME.replace("/", "_")

    white_pkl_path = str(pkl_root / f"{dataset_name}_{backbone_name}_{args.white_attack}.pkl")
    black_pkl_path = str(pkl_root / f"{dataset_name}_{args.black_attack}.pkl")

    has_white = Path(white_pkl_path).exists()
    has_black = args.black_attack and Path(black_pkl_path).exists()

    if not has_white:
        raise FileNotFoundError(f"White-box pkl not found: {white_pkl_path}")

    print(f"\nLoading white-box pkl  : {white_pkl_path}")
    white_pkl = torch.load(white_pkl_path, weights_only=False)
    # white_pkl is already CLIP-normalized (saved by before_adv_test)

    black_pkl = None
    if has_black:
        print(f"Loading black-box pkl  : {black_pkl_path}")
        black_pkl = load_pkl_normalized(black_pkl_path)
    else:
        print(f"[Warning] Black-box pkl not found or --black-attack not set; skipping RAP")

    normalizer = ImageNormalizer(device="cpu")  # will be moved to device in eval_adv

    results = {}

    # ──────────────────────────────────────────────────────────────────────────
    # (A) ZeroShot CLIP baseline: "a photo of [classname]", untrained ODE
    # ──────────────────────────────────────────────────────────────────────────
    baseline_prompt = args.baseline_prompt
    n_ctx_baseline  = len(baseline_prompt.split())

    print(f"\n{'='*70}")
    print(f"[A] Baseline: ZeroShot CLIP  (\"{baseline_prompt} [classname]\")")
    print(f"{'='*70}")

    baseline_cfg = deepcopy(cfg)
    baseline_cfg.defrost()
    baseline_cfg.TRAINER.ADV.CTX_INIT = baseline_prompt
    baseline_cfg.TRAINER.ADV.N_CTX    = n_ctx_baseline
    baseline_cfg.freeze()

    baseline_trainer = build_trainer(baseline_cfg)

    print("[A] Clean accuracy ...")
    a_clean = eval_clean(baseline_trainer)
    print(f"    Clean = {a_clean:.2f}%")

    print("[A] White-box PGD ...")
    baseline_trainer.test_pkl = white_pkl
    baseline_trainer.normalizer = normalizer
    a_pgd = eval_adv(baseline_trainer, white_pkl, normalizer)
    print(f"    PGD   = {a_pgd:.2f}%")

    a_rap = None
    if has_black:
        print(f"[A] Black-box RAP ...")
        a_rap = eval_adv(baseline_trainer, black_pkl, normalizer)
        print(f"    RAP   = {a_rap:.2f}%")

    results["baseline"] = (a_clean, a_pgd, a_rap)
    del baseline_trainer

    # ──────────────────────────────────────────────────────────────────────────
    # (B) ODE-Prompt trained model(s)
    # ──────────────────────────────────────────────────────────────────────────
    if args.model_dir:
        model_dir  = Path(args.model_dir).resolve()
        model_file = args.model_file or "model-best.pth.tar"

        # list checkpoints to evaluate
        if model_file == "all":
            import re
            ckpt_dir = model_dir / "prompt_learner"
            ckpt_files = sorted(
                [p.name for p in ckpt_dir.iterdir() if ".pth.tar" in p.name],
                key=lambda n: (
                    int(re.search(r"-(\d+)$", n).group(1))
                    if re.search(r"-(\d+)$", n) else 10**9
                )
            )
        else:
            ckpt_files = [model_file]

        print(f"\n{'='*70}")
        print(f"[B] ODE-Prompt  (model_dir={model_dir})")
        print(f"{'='*70}")

        ode_trainer = build_trainer(cfg)
        # preload white pkl (already normalized) via before_adv_test path
        ode_trainer.before_adv_test(path=str(pkl_root), attack=args.white_attack)
        white_pkl_ode = ode_trainer.test_pkl   # already normalized by before_adv_test
        ode_normalizer = ode_trainer.normalizer

        for ckpt in ckpt_files:
            print(f"\n  -- checkpoint: {ckpt}")
            ode_trainer.load_model(str(model_dir), model_file=ckpt)

            print("  [B] Clean accuracy ...")
            b_clean = eval_clean(ode_trainer)
            print(f"      Clean = {b_clean:.2f}%")

            print("  [B] White-box PGD ...")
            b_pgd = eval_adv(ode_trainer, white_pkl_ode, ode_normalizer)
            print(f"      PGD   = {b_pgd:.2f}%")

            b_rap = None
            if has_black:
                print(f"  [B] Black-box RAP ...")
                b_rap = eval_adv(ode_trainer, black_pkl, ode_normalizer)
                print(f"      RAP   = {b_rap:.2f}%")

            results[f"ode_{ckpt}"] = (b_clean, b_pgd, b_rap)

    # ──────────────────────────────────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':<40s}  {'Clean':>8s}  {'PGD':>8s}  {'RAP':>8s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}")
    for key, (c, p, r) in results.items():
        c_s = f"{c:.2f}%" if c is not None else "  N/A  "
        p_s = f"{p:.2f}%" if p is not None else "  N/A  "
        r_s = f"{r:.2f}%" if r is not None else "  N/A  "
        print(f"  {key:<40s}  {c_s:>8s}  {p_s:>8s}  {r_s:>8s}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
