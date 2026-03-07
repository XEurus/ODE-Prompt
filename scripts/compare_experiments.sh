#!/bin/bash
# ============================================================================
# 对比实验脚本
# 包含: MLP vs ResNet10 对照, CTP 消融, NCTX 消融
# 每个实验可指定不同 GPU 并行运行
# ============================================================================

set -e

# ==================== 全局配置 ====================
ROOT="/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/Data"
PYTHON="./dassl/bin/python"
BACKBONE=vit_b16
DATASET=oxford_pets
SEED=1
CSC=False
MODEL_FILE=resnet_model.pth.tar

# ==================== 辅助函数 ====================
run_train() {
    local GPU=$1
    local TRAINER=$2
    local DATASET=$3
    local BACKBONE=$4
    local CTP=$5
    local NCTX=$6
    local EXP_NAME=$7
    local EXTRA_OPTS="${8:-}"

    local DIR=./output/${DATASET}/${TRAINER}/${BACKBONE}/adv/${EXP_NAME}
    mkdir -p ${DIR}
    local LOG_FILE="${DIR}/log.txt"

    echo "========================================================================"
    echo "[GPU=${GPU}] ${TRAINER} | ${DATASET} | CTP=${CTP} | NCTX=${NCTX} | ${EXP_NAME}"
    echo "  -> 输出目录: ${DIR}"
    echo "  -> 日志文件: ${LOG_FILE}"
    echo "========================================================================"

    CUDA_VISIBLE_DEVICES=${GPU} $PYTHON train.py \
        --root ${ROOT} \
        --adv-training \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${BACKBONE}/${DATASET}.yaml \
        --output-dir ${DIR} \
        --model-dir ${DIR} \
        --model-file ${MODEL_FILE} \
        --note "${EXP_NAME}" \
        TRAINER.ADV.N_CTX ${NCTX} \
        TRAINER.ADV.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.ADV.CSC ${CSC} \
        ${EXTRA_OPTS} 2>&1 | tee -a ${LOG_FILE}
}

# ============================================================================
# 实验一: MLP vs ResNet10 对照实验
# 在相同超参数 (CTP=end, NCTX=32) 下对比 MLP 和 ResNet10
# ============================================================================
run_mlp_vs_resnet() {
    echo ""
    echo "########################################################################"
    echo "# 实验一: MLP vs ResNet10 对照实验"
    echo "########################################################################"
    echo ""

    # MLP 实验 (GPU 0)
    run_train 0 mlp ${DATASET} ${BACKBONE} end 32 "compare_mlp" &
    PID_MLP=$!

    # ResNet10 实验 (GPU 1)
    run_train 1 resnet10 ${DATASET} ${BACKBONE} end 32 "compare_resnet10" &
    PID_RESNET=$!

    echo "等待 MLP (PID=${PID_MLP}) 和 ResNet10 (PID=${PID_RESNET}) 完成..."
    wait ${PID_MLP}
    echo "MLP 实验完成"
    wait ${PID_RESNET}
    echo "ResNet10 实验完成"
}

# ============================================================================
# 实验二: CTP 消融实验 (Class Token Position)
# 对比 CTP=end vs CTP=middle，使用 AdvPT trainer
# ============================================================================
run_ctp_ablation() {
    echo ""
    echo "########################################################################"
    echo "# 实验二: CTP 消融实验 (end vs middle)"
    echo "########################################################################"
    echo ""

    # CTP=end (GPU 0)
    run_train 0 AdvPT ${DATASET} ${BACKBONE} end 32 "ablation_ctp_end" &
    PID_END=$!

    # CTP=middle (GPU 1)
    run_train 1 AdvPT ${DATASET} ${BACKBONE} middle 32 "ablation_ctp_middle" &
    PID_MID=$!

    echo "等待 CTP=end (PID=${PID_END}) 和 CTP=middle (PID=${PID_MID}) 完成..."
    wait ${PID_END}
    echo "CTP=end 实验完成"
    wait ${PID_MID}
    echo "CTP=middle 实验完成"
}

# ============================================================================
# 实验三: NCTX 消融实验 (Number of Context Tokens)
# 对比不同 NCTX 值: 4, 8, 16, 32，使用 AdvPT trainer
# ============================================================================
run_nctx_ablation() {
    echo ""
    echo "########################################################################"
    echo "# 实验三: NCTX 消融实验 (4 / 8 / 16 / 32)"
    echo "########################################################################"
    echo ""

    # NCTX=4 (GPU 0)
    run_train 0 AdvPT ${DATASET} ${BACKBONE} end 4 "ablation_nctx_4" &
    PID_4=$!

    # NCTX=8 (GPU 1)
    run_train 1 AdvPT ${DATASET} ${BACKBONE} end 8 "ablation_nctx_8" &
    PID_8=$!

    echo "等待 NCTX=4 (PID=${PID_4}) 和 NCTX=8 (PID=${PID_8}) 完成..."
    wait ${PID_4}
    echo "NCTX=4 实验完成"
    wait ${PID_8}
    echo "NCTX=8 实验完成"

    # NCTX=16 (GPU 0)
    run_train 0 AdvPT ${DATASET} ${BACKBONE} end 16 "ablation_nctx_16" &
    PID_16=$!

    # NCTX=32 (GPU 1)
    run_train 1 AdvPT ${DATASET} ${BACKBONE} end 32 "ablation_nctx_32" &
    PID_32=$!

    echo "等待 NCTX=16 (PID=${PID_16}) 和 NCTX=32 (PID=${PID_32}) 完成..."
    wait ${PID_16}
    echo "NCTX=16 实验完成"
    wait ${PID_32}
    echo "NCTX=32 实验完成"
}

# ============================================================================
# 主流程: 按顺序运行三组实验（每组内部并行）
# 可通过注释选择性运行某些实验
# ============================================================================
echo "============================================================"
echo " ODE-Prompt 对比实验"
echo " 数据集: ${DATASET} | 骨干网络: ${BACKBONE}"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# --- 第一组: MLP vs ResNet10 ---
run_mlp_vs_resnet

# --- 第二组: CTP 消融 ---
run_ctp_ablation

# --- 第三组: NCTX 消融 ---
run_nctx_ablation

echo ""
echo "============================================================"
echo " 所有对比实验完成!"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
