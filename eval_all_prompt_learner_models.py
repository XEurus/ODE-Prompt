#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dass.engine import build_trainer
from dass.utils import set_random_seed, setup_logger
from train import setup_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "对 prompt_learner 目录下所有 checkpoint 做完整评估，"
            "并尽量复用 train.py 的评估 pipeline"
        )
    )
    parser.add_argument(
        "--prompt-learner-dir",
        type=str,
        default=(
            "/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/output/ucf101/AdvPT/"
            "vit_b16/adv/5_PGD40_16_mix6_sgd_1e3_60/prompt_learner"
        ),
        help="包含模型文件的 prompt_learner 目录",
    )

    parser.add_argument("--root", type=str, default="/autodl-fs/data/Data")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trainer", type=str, default="AdvPT")
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--model-file", type=str, default="resnet_model.pth.tar")
    parser.add_argument("--config-file", type=str, default="configs/trainers/AdvPT/vit_b16.yaml")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/ucf101.yaml")
    parser.add_argument(
        "--path",
        type=str,
        default="/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/pkl_data",
        help="pkl 数据目录（与 train.py --path 保持一致）",
    )
    parser.add_argument("--white-attack", type=str, default="PGD", help="白盒攻击类型，默认 PGD")
    parser.add_argument("--black-attack", type=str, default="", help="黑盒攻击类型（如 RAP），留空则跳过黑盒评估")

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "test-only"],
        help="评估模式: full=完整评估(clean/test/train/val), test-only=仅测试集+绘图",
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default="",
        help="结果 CSV 输出路径；默认写入 prompt_learner/eval_all_models_clean_test_train_val.csv",
    )

    parser.add_argument(
        "--plot-path",
        type=str,
        default="",
        help="曲线图输出路径 (test-only 模式); 默认写入 prompt_learner/eval_test_only_curve.png",
    )

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--note", type=str, default="None", help="training note/remark for experiment tracking")
    return parser.parse_args()


def list_model_files(prompt_learner_dir: Path):
    model_files = []
    for path in prompt_learner_dir.iterdir():
        if path.is_file() and ".pth.tar" in path.name:
            model_files.append(path)

    if not model_files:
        raise RuntimeError(f"在 {prompt_learner_dir} 中未找到任何 .pth.tar 模型文件")

    def sort_key(path: Path):
        name = path.name
        m = re.search(r"\.pth\.tar-(\d+)$", name)
        if m:
            return (0, int(m.group(1)), name)
        if "best" in name:
            return (1, 10**9, name)
        return (2, 10**9, name)

    model_files.sort(key=sort_key)
    return model_files


@torch.no_grad()
def eval_adv_embedding_full(trainer, embedding_pkl, data_loader, split_name):
    if embedding_pkl is None:
        raise RuntimeError(f"{split_name} 对应的 embedding_pkl 为 None")
    if data_loader is None:
        raise RuntimeError(f"{split_name} 对应的数据加载器为 None")
    if not hasattr(trainer, "_eval_adv_embedding"):
        raise RuntimeError("当前 trainer 不支持 _eval_adv_embedding，无法评估 embedding pkl")

    print(f"Evaluate adversarial embedding on the *{split_name}* set (all batches)")
    return float(trainer._eval_adv_embedding(embedding_pkl, data_loader, max_batches=None))


def get_required_pkl_paths(cfg, pkl_root: Path, attack: str):
    dataset_name = cfg.DATASET.NAME
    backbone_name = cfg.MODEL.BACKBONE.NAME.replace("/", "_")
    # PGD_whitebox 的 test embedding 由 before_adv_test 在线生成/缓存，
    # 不依赖旧的全图 pkl；这里回退到检查基础 PGD.pkl（仅用于兼容断言）
    base_attack = attack.replace('_whitebox', '').replace('_adaptive', '')
    return {
        "clean": pkl_root / f"{dataset_name}_{backbone_name}_clean.pkl",
        "test": pkl_root / f"{dataset_name}_{backbone_name}_{base_attack}.pkl",
        "train": pkl_root / f"{dataset_name}_{backbone_name}_v2.pkl",
        "val": pkl_root / f"{dataset_name}_{backbone_name}_val_v2.pkl",
    }


def get_black_pkl_path(cfg, pkl_root: Path, black_attack: str):
    dataset_name = cfg.DATASET.NAME
    return pkl_root / f"{dataset_name}_{black_attack}.pkl"


def assert_required_pkl_paths(required_paths):
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "以下必需 pkl 文件不存在，请先准备完整数据：\n- " + "\n- ".join(missing)
        )


def eval_test_only_mode(args, prompt_learner_dir, model_files, model_dir, trainer, has_black):
    """test-only 模式: 仅评估测试集并绘制曲线"""
    import re

    csv_path = args.csv_path
    if not csv_path:
        csv_path = str(prompt_learner_dir / "eval_test_only.csv")

    plot_path = args.plot_path
    if not plot_path:
        plot_path = str(prompt_learner_dir / "eval_test_only_curve.png")

    # 在循环外预加载两个 pkl，避免循环内反复切换 self.test_pkl 造成状态污染
    white_pkl = trainer.test_pkl  # 已由 before_adv_test 加载并归一化
    black_pkl = None
    from utils.adv_utils import ImageNormalizer
    pkl_root = Path(args.path).resolve()
    if has_black:
        # 复制 before_black_test 的预处理逻辑：加载原始像素空间 pkl 并归一化
        black_pkl_path = pkl_root / f"{trainer.cfg.DATASET.NAME}_{args.black_attack}.pkl"
        _norm = ImageNormalizer(device='cpu')
        _raw = torch.load(str(black_pkl_path), weights_only=False)
        black_pkl = _norm.normalize(_raw)
        print(f"[Preloaded] black pkl: {black_pkl_path}, shape={black_pkl.shape}")

    rows = []
    step_numbers = []
    robust_accs = []
    black_accs = []

    # -----------------------------------------------------------------------
    # Baseline row (step 0): 使用 "a photo of [classname]" 提示词，不加载任何
    # 训练权重，代表未训练时的初始性能
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Evaluating: BASELINE (a photo of [classname], untrained ODE)")
    print("=" * 80)
    baseline_row = {
        "model_file": "baseline_a_photo_of",
        "robust_accuracy": "",
        "black_accuracy": "",
        "status": "ok",
        "error": "",
    }
    try:
        from dass.engine import build_trainer as _build_trainer
        from copy import deepcopy
        baseline_cfg = deepcopy(trainer.cfg)
        baseline_cfg.defrost()
        baseline_cfg.TRAINER.ADV.CTX_INIT = "a photo of"
        baseline_cfg.TRAINER.ADV.N_CTX = 3
        baseline_cfg.freeze()
        baseline_trainer = _build_trainer(baseline_cfg)
        baseline_trainer.test_pkl = white_pkl
        baseline_trainer.normalizer = trainer.normalizer
        print("[Baseline] Evaluating white-box PGD", flush=True)
        b_white = float(baseline_trainer.test_adv(split="test"))
        baseline_row["robust_accuracy"] = f"{b_white:.4f}"
        if has_black:
            baseline_trainer.test_pkl = black_pkl
            print(f"[Baseline] Evaluating black-box {args.black_attack}", flush=True)
            b_black = float(baseline_trainer.test_adv(split="test"))
            baseline_row["black_accuracy"] = f"{b_black:.4f}"
            black_accs.append(b_black)
        del baseline_trainer
        step_numbers.append(0)
        robust_accs.append(b_white)
        print(f"[Baseline Done] robust={baseline_row['robust_accuracy']}% | black={baseline_row['black_accuracy'] or 'N/A'}%")
    except Exception as exc:
        baseline_row["status"] = "error"
        baseline_row["error"] = str(exc)
        print(f"[Baseline Error] {exc}")
    rows.append(baseline_row)

    for model_path in model_files:
        model_file = model_path.name
        print("\n" + "=" * 80)
        print(f"Evaluating: {model_file}")
        print("=" * 80)

        row = {
            "model_file": model_file,
            "robust_accuracy": "",
            "black_accuracy": "",
            "status": "ok",
            "error": "",
        }

        try:
            trainer.load_model(model_dir, model_file=model_file)

            print("[1/2] Evaluating: test set (white-box adversarial images)")
            trainer.test_pkl = white_pkl
            test_acc = float(trainer.test_adv(split="test"))
            row["robust_accuracy"] = f"{test_acc:.4f}"

            if has_black:
                print(f"[2/2] Evaluating: test set (black-box {args.black_attack} images)")
                trainer.test_pkl = black_pkl
                black_acc = float(trainer.test_adv(split="test"))
                row["black_accuracy"] = f"{black_acc:.4f}"
                black_accs.append(black_acc)
            else:
                print("[2/2] Skipping: black-box attack (--black-attack not specified)")

            m = re.search(r"\.pth\.tar-(\d+)$", model_file)
            step_num = int(m.group(1)) if m else len(step_numbers)
            step_numbers.append(step_num)
            robust_accs.append(test_acc)

            print(f"\n[Done] {model_file} | robust={row['robust_accuracy']}% | black={row['black_accuracy'] or 'N/A'}%")
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            print(f"[Error] {model_file}: {exc}")

        rows.append(row)

    # 保存 CSV
    csv_parent = Path(csv_path).resolve().parent
    csv_parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_file", "robust_accuracy", "black_accuracy", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # 绘制曲线
    if step_numbers:
        plt.figure(figsize=(10, 6))
        plt.plot(step_numbers, robust_accs, "r-s", label=f"White-box Robust Acc ({args.white_attack})", linewidth=2, markersize=6)
        if has_black and black_accs:
            plt.plot(step_numbers[:len(black_accs)], black_accs, "b-^", label=f"Black-box Robust Acc ({args.black_attack})", linewidth=2, markersize=6)
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title("Test Set: Robust Accuracy", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nCurve saved to: {plot_path}")

    print("\n" + "=" * 80)
    print("test-only 评估完成:")
    for row in rows:
        print(
            f"- {row['model_file']}: robust={row['robust_accuracy'] or 'N/A'}, "
            f"black={row['black_accuracy'] or 'N/A'}, status={row['status']}"
        )
    print(f"CSV 已保存到: {csv_path}")


def main():
    args = parse_args()

    prompt_learner_dir = Path(args.prompt_learner_dir).resolve()
    if not prompt_learner_dir.exists() or not prompt_learner_dir.is_dir():
        raise FileNotFoundError(f"prompt_learner 目录不存在: {prompt_learner_dir}")

    exp_dir = prompt_learner_dir.parent
    if not args.output_dir:
        args.output_dir = str(exp_dir)

    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    trainer = build_trainer(cfg)

    pkl_root = Path(args.path).resolve()
    required_pkl_paths = get_required_pkl_paths(cfg, pkl_root, args.white_attack)
    assert_required_pkl_paths(required_pkl_paths)

    # 检查黑盒 pkl 是否存在
    has_black = False
    if args.black_attack:
        black_pkl_path = get_black_pkl_path(cfg, pkl_root, args.black_attack)
        if black_pkl_path.exists():
            has_black = True
            print(f"- black ({args.black_attack}): {black_pkl_path}")
        else:
            print(f"[Warning] 黑盒 pkl 不存在: {black_pkl_path}，跳过黑盒评估")
            print(f"  请先运行: python black.py --dataset {cfg.DATASET.NAME} --path {args.path}")

    print("Using pkl files:")
    for key in ["clean", "test", "train", "val"]:
        print(f"- {key}: {required_pkl_paths[key]}")

    # 预加载四类 pkl，严格沿用现有训练代码中的 before_* 入口
    trainer.before_clean_train(path=str(pkl_root))
    trainer.before_adv_test(path=str(pkl_root), attack=args.white_attack)
    trainer.before_adv_train(path=str(pkl_root), attack=args.white_attack)
    trainer.before_adv_val(path=str(pkl_root), attack=args.white_attack)

    model_files = list_model_files(prompt_learner_dir)
    model_dir = str(exp_dir)

    if args.mode == "test-only":
        return eval_test_only_mode(args, prompt_learner_dir, model_files, model_dir, trainer, has_black)

    # Full mode (original logic)
    csv_path = args.csv_path
    if not csv_path:
        csv_path = str(prompt_learner_dir / "eval_all_models_clean_test_train_val.csv")

    rows = []
    for model_path in model_files:
        model_file = model_path.name
        print("\n" + "=" * 80)
        print(f"Evaluating: {model_file}")
        print("=" * 80)

        row = {
            "model_file": model_file,
            "clean_accuracy": "",
            "test_accuracy": "",
            "black_accuracy": "",
            "train_accuracy": "",
            "val_accuracy": "",
            "clean_train_embedding_accuracy": "",
            "status": "ok",
            "error": "",
        }

        try:
            trainer.load_model(model_dir, model_file=model_file)

            n_steps = 5 + (1 if has_black else 0)
            print(f"[1/{n_steps}] Evaluating: test set (clean images - standard clean accuracy)")
            clean_acc = float(trainer.test())

            print(f"[2/{n_steps}] Evaluating: test set (white-box {args.white_attack} - robust accuracy)")
            test_acc = float(trainer.test_adv(split="test"))

            if has_black:
                print(f"[3/{n_steps}] Evaluating: test set (black-box {args.black_attack} - black-box robust accuracy)")
                trainer.before_black_test(args.path, args.black_attack)
                black_acc = float(trainer.test_adv(split="test"))
                row["black_accuracy"] = f"{black_acc:.4f}"
                # 恢复白盒 test_pkl 供后续 before_* 调用
                trainer.before_adv_test(path=str(pkl_root), attack=args.white_attack)
            else:
                print(f"[3/{n_steps}] Skipping: black-box attack (--black-attack not specified or pkl not found)")

            # train/val 使用预计算 embedding 完整评估（all batches）
            print(f"[{3 + (1 if has_black else 1)}/{n_steps}] Evaluating: train set (adversarial embeddings from _v2.pkl)")
            train_acc = eval_adv_embedding_full(
                trainer,
                trainer.train_pkl,
                trainer.train_loader_x_notransform_noshuffle,
                split_name="train",
            )

            val_acc = ""
            if trainer.val_loader is not None and getattr(trainer, "val_pkl", None) is not None:
                print(f"[{4 + (1 if has_black else 1)}/{n_steps}] Evaluating: val set (adversarial embeddings from _val_v2.pkl)")
                val_acc = f"{eval_adv_embedding_full(trainer, trainer.val_pkl, trainer.val_loader, split_name='val'):.4f}"
            else:
                print(f"[{4 + (1 if has_black else 1)}/{n_steps}] Skipping: val set (no val_loader or val_pkl available)")

            # clean.pkl（训练集 clean embedding）完整评估
            print(f"[{n_steps}/{n_steps}] Evaluating: train set (clean embeddings from _clean.pkl)")
            clean_train_embed_acc = eval_adv_embedding_full(
                trainer,
                trainer.clean_pkl,
                trainer.train_loader_x_notransform_noshuffle,
                split_name="clean_train_embedding",
            )

            row["clean_accuracy"] = f"{clean_acc:.4f}"
            row["test_accuracy"] = f"{test_acc:.4f}"
            row["train_accuracy"] = f"{train_acc:.4f}"
            row["val_accuracy"] = val_acc
            row["clean_train_embedding_accuracy"] = f"{clean_train_embed_acc:.4f}"

            print(
                f"\n[Done] {model_file} | Results Summary:\n"
                f"  - test (clean):         {row['clean_accuracy']}%\n"
                f"  - test (white-box adv): {row['test_accuracy']}%\n"
                f"  - test (black-box adv): {row['black_accuracy'] or 'N/A'}%\n"
                f"  - train (adv embed):    {row['train_accuracy']}%\n"
                f"  - val   (adv embed):    {row['val_accuracy'] or 'N/A'}%\n"
                f"  - train (clean embed):  {row['clean_train_embedding_accuracy']}%"
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            print(f"[Error] {model_file}: {exc}")

        rows.append(row)

    csv_parent = Path(csv_path).resolve().parent
    csv_parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_file",
                "clean_accuracy",
                "test_accuracy",
                "black_accuracy",
                "train_accuracy",
                "val_accuracy",
                "clean_train_embedding_accuracy",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 80)
    print("评估完成，结果如下：")
    for row in rows:
        print(
            f"- {row['model_file']}: clean={row['clean_accuracy'] or 'N/A'}, "
            f"test={row['test_accuracy'] or 'N/A'}, black={row['black_accuracy'] or 'N/A'}, "
            f"train={row['train_accuracy'] or 'N/A'}, val={row['val_accuracy'] or 'N/A'}, "
            f"clean_train_emb={row['clean_train_embedding_accuracy'] or 'N/A'}, status={row['status']}"
        )
    print(f"CSV 已保存到: {csv_path}")


if __name__ == "__main__":
    main()
