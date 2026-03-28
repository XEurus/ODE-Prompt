#!/bin/bash
# ==================== 实时白盒对抗训练脚本 ====================
# 支持两种训练模式：
#   1. Bank 模式 (--adv-training): 预计算对抗 embedding，训练时直接使用
#   2. 实时模式 (--realtime-adv): 每个 batch 实时 PGD，攻击包括 ODE 网络
#
# 实时模式特点：
#   - 每个 batch 都对当前模型进行 PGD 攻击
#   - 攻击目标包括正在训练的 ODE 网络
#   - 能生成更强的对抗样本，但训练速度较慢

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

# ==================== 训练模式配置 ====================
# 训练模式: bank | realtime
#   - bank: 使用预计算的 embedding bank（快，但攻击不包括 ODE）
#   - realtime: 每 batch 实时 PGD（慢，但攻击包括 ODE）
TRAINING_MODE="realtime"     # bank | realtime

# ==================== 攻击配置 ====================
ATTACK_MODE=PGD_whitebox     # PGD | PGD_whitebox
TRAIN_EPS=8                  # 训练扰动强度 (x/255)
TEST_EPS=1                   # 测试扰动强度 (x/255)
TRAIN_PGD_ITERS=5            # Bank模式: 训练 PGD 迭代次数
TEST_PGD_ITERS=100           # 测试 PGD 迭代次数

# ==================== 实时对抗训练配置 ====================
REALTIME_PGD_ITERS=5        # 实时模式: 每 batch 的 PGD 迭代次数
REALTIME_LOSS_MIX=True       # 是否混合干净损失 (0.4*adv + 0.6*clean)
# 验证集攻击：硬编码为 PGD100 eps=1/255（与最终测试一致）
# 测试集：跳过（实时攻击太慢）

# ==================== 数据加载配置 ====================
NUM_SHOTS=16                 # Few-shot: 每类训练样本数 (-1=全部)
TRAIN_BATCH_SIZE=4           # 实时模式建议 4（显存压力大），bank 模式可用 8

# ==================== 实验配置 ====================
if [ "$TRAINING_MODE" = "realtime" ]; then
    exp_name="11_realtime_${NUM_SHOTS}shot_${ODE_NETWORK_TYPE}_pgd${REALTIME_PGD_ITERS}_eps${TRAIN_EPS}"
    TRAINING_NOTE="实时对抗训练: train=realtime-PGD${REALTIME_PGD_ITERS}-${TRAIN_EPS}/255, test=PGD${TEST_PGD_ITERS}-${TEST_EPS}/255"
else
    exp_name="10_bank_${NUM_SHOTS}shot_${ODE_NETWORK_TYPE}_pgd${TRAIN_PGD_ITERS}_eps${TRAIN_EPS}"
    TRAINING_NOTE="Bank对抗训练: train=PGD${TRAIN_PGD_ITERS}-${TRAIN_EPS}/255, test=PGD${TEST_PGD_ITERS}-${TEST_EPS}/255"
fi

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
echo "对抗训练 - ${TRAINING_MODE} 模式"
echo "======================================================================"
echo "  训练模式:  ${TRAINING_MODE}"
echo "  攻击模式:  ${ATTACK_MODE}"
if [ "$TRAINING_MODE" = "realtime" ]; then
    echo "  训练PGD:   ${REALTIME_PGD_ITERS} iters, eps=${TRAIN_EPS}/255"
    echo "  验证PGD:   100 iters, eps=1/255 (与最终测试一致)"
    echo "  测试集:   SKIPPED (实时攻击太慢)"
    echo "  损失混合:  ${REALTIME_LOSS_MIX}"
else
    echo "  训练攻击:  PGD${TRAIN_PGD_ITERS} eps=${TRAIN_EPS}/255"
    echo "  测试攻击:  PGD${TEST_PGD_ITERS} eps=${TEST_EPS}/255"
fi
echo "  实验名称:  ${exp_name}"
echo "  输出目录:  ${DIR}"
echo "  PKL目录:   ${PKL_DIR}"
echo "  日志文件:  ${LOG_FILE}"
echo "======================================================================"

# 根据训练模式选择参数
if [ "$TRAINING_MODE" = "realtime" ]; then
    # 实时对抗训练模式
    $PYTHON train.py \
    --root ${D} \
    --realtime-adv \
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
    TRAINER.ADV.REALTIME_PGD_ITERS ${REALTIME_PGD_ITERS} \
    TRAINER.ADV.REALTIME_LOSS_MIX ${REALTIME_LOSS_MIX} \
    DATASET.NUM_SHOTS ${NUM_SHOTS} \
    DATALOADER.TRAIN_X.BATCH_SIZE ${TRAIN_BATCH_SIZE} \
    DATASET.TRAIN_EPS ${TRAIN_EPS} \
    DATASET.TEST_EPS ${TEST_EPS} \
    DATASET.Test_PGD_NUM_ITERS ${TEST_PGD_ITERS} \
    MODEL.FILE_PREFIX ${MODEL_FILE_PREFIX} \
    TRAIN.TENSORBOARD_DIR "${TENSORBOARD_DIR}" 2>&1 | tee -a ${LOG_FILE}
else
    # Bank 对抗训练模式（预计算 embedding）
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
    DATALOADER.TRAIN_X.BATCH_SIZE ${TRAIN_BATCH_SIZE} \
    DATASET.TRAIN_EPS ${TRAIN_EPS} \
    DATASET.TEST_EPS ${TEST_EPS} \
    DATASET.Train_PGD_NUM_ITERS ${TRAIN_PGD_ITERS} \
    DATASET.Test_PGD_NUM_ITERS ${TEST_PGD_ITERS} \
    MODEL.FILE_PREFIX ${MODEL_FILE_PREFIX} \
    TRAIN.TENSORBOARD_DIR "${TENSORBOARD_DIR}" 2>&1 | tee -a ${LOG_FILE}
fi

echo ""
echo "======================================================================"
echo "训练完成"
echo "======================================================================"

# 发送邮件通知
echo "发送训练完成邮件..."
$PYTHON utils/email_sender.py \
    --exp_name "${exp_name}" \
    --log_file ${LOG_FILE}

# sleep 300
# shutdown -h now