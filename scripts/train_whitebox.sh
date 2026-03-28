#!/bin/bash
# ==================== 白盒 PGD 对抗训练脚本 ====================
# 使用标准白盒分类 PGD 攻击（直接最大化 CLIP 零样本交叉熵）
#
# 训练: PGD5  eps=5/255  (快速白盒攻击, 10 次 restart)
# 测试: PGD100 eps=1/255 (标准白盒评测)

GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

# ==================== 基本配置 ====================
ROOT="/autodl-fs/data/Data"
TRAINER=AdvPT
DATASET=oxford_pets
BACKBONE=vit_b16
SEED=1

# ==================== Prompt 配置 ====================
NCTX=32
CSC=False
CTX_INIT=""
CTP=end
PREC=fp16

# ==================== ODE 网络配置 ====================
ODE_NETWORK_TYPE="resnet"
ODE_T=1.0

# ==================== 攻击配置 ====================
ATTACK_MODE=PGD_whitebox     # PGD | PGD_whitebox
TRAIN_EPS=5                  # 训练扰动强度 (x/255)
TEST_EPS=1                   # 测试扰动强度 (x/255)
TRAIN_PGD_ITERS=5            # 训练 PGD 迭代次数
TEST_PGD_ITERS=100           # 测试 PGD 迭代次数

# ==================== 数据加载配置 ====================
NUM_SHOTS=16                 # Few-shot: 每类训练样本数 (-1=全部)

# ==================== 实验配置 ====================
exp_name="10_whitebox_16shot_${ODE_NETWORK_TYPE}_pgd${TRAIN_PGD_ITERS}_eps${TRAIN_EPS}_adamw"
TRAINING_NOTE="白盒分类PGD训练: train=PGD${TRAIN_PGD_ITERS}-${TRAIN_EPS}/255, test=PGD${TEST_PGD_ITERS}-${TEST_EPS}/255"

# 模型文件
MODEL_FILE_PREFIX="${ODE_NETWORK_TYPE}_${BACKBONE}"
MODEL_FILE=${MODEL_FILE_PREFIX}.pth.tar
Best_Model=${MODEL_FILE_PREFIX}-best.pth.tar

# pkl 数据目录（白盒攻击生成的 embedding 单独存放）
PKL_DIR=./pkl_whitebox_pgd${TRAIN_PGD_ITERS}_eps${TRAIN_EPS}_${NUM_SHOTS}shot

D=$ROOT
DIR=/autodl-fs/data/output/${DATASET}/${TRAINER}/${BACKBONE}/adv/${exp_name}
TENSORBOARD_DIR="/autodl-fs/data/output/${DATASET}/${TRAINER}/${BACKBONE}/"
PYTHON="./dassl/bin/python"

# 确保输出目录存在
mkdir -p ${DIR}
mkdir -p ${PKL_DIR}

# 定义日志文件
LATEST_LOG=$(ls -t ${DIR}/log.txt-* 2>/dev/null | head -n 1)
if [ -n "${LATEST_LOG}" ]; then
    LOG_FILE="${LATEST_LOG}"
else
    LOG_FILE="${DIR}/log.txt"
fi

echo "======================================================================"
echo "白盒 PGD 对抗训练"
echo "  攻击模式:  ${ATTACK_MODE}"
echo "  训练攻击:  PGD${TRAIN_PGD_ITERS} eps=${TRAIN_EPS}/255"
echo "  测试攻击:  PGD${TEST_PGD_ITERS} eps=${TEST_EPS}/255"
echo "  实验名称:  ${exp_name}"
echo "  输出目录:  ${DIR}"
echo "  PKL目录:   ${PKL_DIR}"
echo "  日志文件:  ${LOG_FILE}"
echo "======================================================================"

$PYTHON train.py \
--root ${D} \
--adv-training \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${BACKBONE}/${DATASET}.yaml \
--output-dir ${DIR} \
--model-dir ${DIR} \
--model-file ${MODEL_FILE} \
--path ${PKL_DIR} \
--white-attack ${ATTACK_MODE} \
--note "${TRAINING_NOTE}" \
TRAINER.ADV.N_CTX ${NCTX} \
TRAINER.ADV.CSC ${CSC} \
TRAINER.ADV.CTX_INIT "${CTX_INIT}" \
TRAINER.ADV.PREC ${PREC} \
TRAINER.ADV.CLASS_TOKEN_POSITION ${CTP} \
TRAINER.ADV.ODE_NETWORK_TYPE ${ODE_NETWORK_TYPE} \
TRAINER.ADV.ODE_T ${ODE_T} \
DATASET.NUM_SHOTS ${NUM_SHOTS} \
DATASET.TRAIN_EPS ${TRAIN_EPS} \
DATASET.TEST_EPS ${TEST_EPS} \
DATASET.Train_PGD_NUM_ITERS ${TRAIN_PGD_ITERS} \
DATASET.Test_PGD_NUM_ITERS ${TEST_PGD_ITERS} \
MODEL.FILE_PREFIX ${MODEL_FILE_PREFIX} \
TRAIN.TENSORBOARD_DIR "${TENSORBOARD_DIR}" 2>&1 | tee -a ${LOG_FILE}

echo ""
echo "======================================================================"
echo "训练完成"
echo "======================================================================"

# 发送邮件通知
echo "发送训练完成邮件..."
$PYTHON utils/email_sender.py \
    --exp_name "${exp_name}" \
    --log_file ${LOG_FILE}

sleep 300
shutdown -h now