#!/bin/bash
# ==================== PGD100 白盒攻击评估脚本 ====================
# 用已有模型 (8_adamw-pla_1e4_resnet_10) 在白盒 PGD100 条件下测试
#
# 攻击模式说明：
#   PGD          — 特征扰动攻击（旧方式，用随机代理模型，攻击力弱）
#   PGD_whitebox — 白盒分类攻击（标准评测，直接最大化零样本分类 CE loss）
#   PGD_adaptive — 自适应攻击（梯度穿过完整 ODE 模型，最强攻击）
#
# 流程：
#   1. 创建新 pkl 目录，复用已有 clean/train/val embedding
#   2. 自动生成白盒 PGD 对抗测试集 (test_v2_whitebox.pkl)
#   3. 调用 eval_all_prompt_learner_models.py 评估所有 checkpoint

GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

PYTHON="./dassl/bin/python"

DATASET=oxford_pets
BACKBONE=vit_b16
TRAINER=AdvPT

# ==================== 可调参数 ====================
ATTACK_MODE=PGD_whitebox   # PGD | PGD_whitebox | PGD_adaptive
TEST_EPS=1                 # 扰动强度 (x/255)
PGD_ITERS=100              # PGD 迭代次数

# 模型目录
EXP_DIR=./8_adamw-pla_1e4_resnet_10

# 原 pkl 目录（已有 clean/train/val embedding）
OLD_PKL_DIR=./pkl_data_mix_5

# 新 pkl 目录
NEW_PKL_DIR=./pkl_pgd${PGD_ITERS}_eps${TEST_EPS}

CONFIG_FILE=configs/trainers/${TRAINER}/${BACKBONE}/${DATASET}.yaml
DATASET_CONFIG=configs/datasets/${DATASET}.yaml

CSV_NAME=eval_${ATTACK_MODE}_pgd${PGD_ITERS}_eps${TEST_EPS}

# ==================== 1. 准备新 pkl 目录 ====================
echo "==================== 准备 pkl 目录 ===================="
mkdir -p ${NEW_PKL_DIR}

for f in ${OLD_PKL_DIR}/*.pkl; do
    fname=$(basename "$f")
    # 跳过任何 test 相关的 pkl（由 before_adv_test 在线生成）
    if [[ "$fname" == *"test"* ]]; then
        echo "跳过: ${fname}"
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
echo "==================== 开始评估 ===================="
echo "攻击模式: ${ATTACK_MODE}"
echo "模型目录: ${EXP_DIR}"
echo "TEST_EPS=${TEST_EPS}, PGD_ITERS=${PGD_ITERS}"
echo ""

$PYTHON eval_all_prompt_learner_models.py \
  --prompt-learner-dir ${EXP_DIR}/prompt_learner \
  --path ${NEW_PKL_DIR} \
  --dataset-config-file ${DATASET_CONFIG} \
  --config-file ${CONFIG_FILE} \
  --white-attack ${ATTACK_MODE} \
  --mode test-only \
  --model-file resnet_model.pth.tar \
  --csv-path ${EXP_DIR}/${CSV_NAME}.csv \
  --plot-path ${EXP_DIR}/${CSV_NAME}_curve.png \
  TRAINER.ADV.ODE_NETWORK_TYPE resnet \
  TRAINER.ADV.N_CTX 32 \
  TRAINER.ADV.CLASS_TOKEN_POSITION end \
  TRAINER.ADV.CSC False \
  DATASET.TEST_EPS ${TEST_EPS} \
  DATASET.Test_PGD_NUM_ITERS ${PGD_ITERS}

echo ""
echo "==================== 评估完成 ===================="
echo "结果 CSV: ${EXP_DIR}/${CSV_NAME}.csv"
echo "曲线图:   ${EXP_DIR}/${CSV_NAME}_curve.png"
