#!/bin/bash
# ==================== PGD100 eps=1 评估脚本 ====================
# 用已有模型 (8_adamw-pla_1e4_resnet_10) 在 PGD100 上限=1 条件下测试
# 流程：
#   1. 创建新 pkl 目录，复用已有 clean/train/val embedding
#   2. 自动生成 PGD100 eps=1 的测试集 (test_v2.pkl)
#   3. 调用 eval_all_prompt_learner_models.py 评估所有 checkpoint

GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

PYTHON="./dassl/bin/python"

DATASET=oxford_pets
BACKBONE=vit_b16
TRAINER=AdvPT

# 模型目录
EXP_DIR=./8_adamw-pla_1e4_resnet_10

# 原 pkl 目录（已有 clean/train/val embedding）
OLD_PKL_DIR=./pkl_data_mix_5

# 新 pkl 目录（仅重新生成测试集）
NEW_PKL_DIR=./pkl_pgd100_eps1

CONFIG_FILE=configs/trainers/${TRAINER}/${BACKBONE}/${DATASET}.yaml
DATASET_CONFIG=configs/datasets/${DATASET}.yaml

# ==================== 1. 准备新 pkl 目录 ====================
echo "==================== 准备 pkl 目录 ===================="
mkdir -p ${NEW_PKL_DIR}

# 链接已有的 clean/train/val pkl（跳过 test_v2.pkl，需重新生成）
for f in ${OLD_PKL_DIR}/*.pkl; do
    fname=$(basename "$f")
    if [ "$fname" = "OxfordPets_ViT-B_16_test_v2.pkl" ]; then
        echo "跳过: ${fname} (将用 PGD100 eps=1 重新生成)"
        continue
    fi
    ln -sf "$(realpath "$f")" "${NEW_PKL_DIR}/${fname}"
    echo "链接: ${fname}"
done

echo ""
echo "新 pkl 目录内容:"
ls -lh ${NEW_PKL_DIR}/
echo ""

# ==================== 2. 运行评估 ====================
echo "==================== 开始 PGD100 eps=1 评估 ===================="
echo "模型目录: ${EXP_DIR}"
echo "TEST_EPS=1, Test_PGD_NUM_ITERS=100"
echo ""

$PYTHON eval_all_prompt_learner_models.py \
  --prompt-learner-dir ${EXP_DIR}/prompt_learner \
  --path ${NEW_PKL_DIR} \
  --dataset-config-file ${DATASET_CONFIG} \
  --config-file ${CONFIG_FILE} \
  --white-attack PGD \
  --mode test-only \
  --model-file resnet_model.pth.tar \
  --csv-path ${EXP_DIR}/eval_pgd100_eps1.csv \
  --plot-path ${EXP_DIR}/eval_pgd100_eps1_curve.png \
  TRAINER.ADV.ODE_NETWORK_TYPE resnet \
  TRAINER.ADV.N_CTX 32 \
  TRAINER.ADV.CLASS_TOKEN_POSITION end \
  TRAINER.ADV.CSC False \
  DATASET.TEST_EPS 1 \
  DATASET.Test_PGD_NUM_ITERS 100

echo ""
echo "==================== 评估完成 ===================="
echo "结果 CSV: ${EXP_DIR}/eval_pgd100_eps1.csv"
echo "曲线图:   ${EXP_DIR}/eval_pgd100_eps1_curve.png"
