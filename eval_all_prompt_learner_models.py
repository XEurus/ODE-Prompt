#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

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

    parser.add_argument("--root", type=str, default="/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/Data")
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

    parser.add_argument(
        "--csv-path",
        type=str,
        default="",
        help="结果 CSV 输出路径；默认写入 prompt_learner/eval_all_models_clean_test_train_val.csv",
    )

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
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
    return {
        "clean": pkl_root / f"{dataset_name}_{backbone_name}_clean.pkl",
        "test": pkl_root / f"{dataset_name}_{backbone_name}_{attack}.pkl",
        "train": pkl_root / f"{dataset_name}_{backbone_name}_v2.pkl",
        "val": pkl_root / f"{dataset_name}_{backbone_name}_val_v2.pkl",
    }


def assert_required_pkl_paths(required_paths):
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "以下必需 pkl 文件不存在，请先准备完整数据：\n- " + "\n- ".join(missing)
        )


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
            "train_accuracy": "",
            "val_accuracy": "",
            "clean_train_embedding_accuracy": "",
            "status": "ok",
            "error": "",
        }

        try:
            trainer.load_model(model_dir, model_file=model_file)

            # clean/test 评估入口严格对齐 train.py eval_only 逻辑
            clean_acc = float(trainer.test())
            test_acc = float(trainer.test_adv(split="test"))

            # train/val 使用预计算 embedding 完整评估（all batches）
            train_acc = eval_adv_embedding_full(
                trainer,
                trainer.train_pkl,
                trainer.train_loader_x_notransform_noshuffle,
                split_name="train",
            )

            val_acc = ""
            if trainer.val_loader is not None and getattr(trainer, "val_pkl", None) is not None:
                val_acc = f"{eval_adv_embedding_full(trainer, trainer.val_pkl, trainer.val_loader, split_name='val'):.4f}"

            # clean.pkl（训练集 clean embedding）完整评估
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
                f"[Done] {model_file} | clean={row['clean_accuracy']} | test={row['test_accuracy']} "
                f"| train={row['train_accuracy']} | val={row['val_accuracy'] or 'N/A'} "
                f"| clean_train_emb={row['clean_train_embedding_accuracy']}"
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
            f"test={row['test_accuracy'] or 'N/A'}, train={row['train_accuracy'] or 'N/A'}, "
            f"val={row['val_accuracy'] or 'N/A'}, clean_train_emb={row['clean_train_embedding_accuracy'] or 'N/A'}, "
            f"status={row['status']}"
        )
    print(f"CSV 已保存到: {csv_path}")


if __name__ == "__main__":
    main()
