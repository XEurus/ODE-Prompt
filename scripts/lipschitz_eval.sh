#!/usr/bin/env bash
# ==============================================================================
# ODE 网络利普希茨常数批量评测脚本
# 用法: bash scripts/lipschitz_eval.sh
#
# 依次对 OUTPUT_DIR 下所有含 model-best.pth.tar 的实验运行分析，
# 结果同时打印到终端并保存到各实验目录下的 lipschitz_report.txt。
# ==============================================================================

GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

PYTHON="./dassl/bin/python"

# ==================== 模型参数 ====================
PROMPT_DIM=512
VISUAL_DIM=512
DEVICE=cuda

# ==================== 分析参数 ====================
NUM_JACOBIAN_SAMPLES=10   # Jacobian SVD / 幂迭代 采样点数
NUM_EMPIRICAL_PAIRS=2000  # 经验采样对数
BAB_ITERS=10               # Branch-and-Bound 迭代次数 (0 = 跳过，需要 auto_LiRPA)
NO_FLOW=""                # 留空 = 估计流映射; 填 "--no-flow" = 跳过流映射分析

# ==================== 搜索目录 ====================
OUTPUT_BASE="./output/oxford_pets/AdvPT/vit_b16/adv"

# 也可手动指定要测试的检查点列表（注释掉自动搜索部分并取消以下注释）:
# CHECKPOINTS=(
#   "output/oxford_pets/AdvPT/vit_b16/tensorboard/7_PGD40_16_mix-loss06_adamw-plateau_1e3/prompt_learner/model-best.pth.tar"
#   "output/oxford_pets/AdvPT/vit_b16/adv/7_PGD40_16_mix-loss06_adamw-plateau_1e3_10/prompt_learner/model-best.pth.tar"
#   "output/oxford_pets/AdvPT/vit_b16/adv/8_adamw-plateau_1e3_Lp_MLP_10/prompt_learner/model-best.pth.tar"
#   "output/oxford_pets/AdvPT/vit_b16/adv/8_adamw-plateau_1e3_Lp_MLP_10_SpectralAll/prompt_learner/model-best.pth.tar"
# )

# ==============================================================================
# 自动搜索所有 model-best.pth.tar
# ==============================================================================
mapfile -t CHECKPOINTS < <(
    find "${OUTPUT_BASE}" -name "model-best.pth.tar" | sort
)

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "[错误] 在 ${OUTPUT_BASE} 下未找到任何 model-best.pth.tar"
    exit 1
fi

echo "======================================================================"
echo "  ODE 利普希茨常数批量评测"
echo "  共找到 ${#CHECKPOINTS[@]} 个模型"
echo "======================================================================"

SUMMARY_FILE="./output/lipschitz_summary.txt"
echo "实验名称 | Jacobian SVD | 幂迭代 | 经验下界 | 流映射(实测)" > "${SUMMARY_FILE}"
echo "---------|-------------|--------|----------|-------------" >> "${SUMMARY_FILE}"

SUCCESS=0
FAILED=0

for CKPT in "${CHECKPOINTS[@]}"; do
    EXP_DIR=$(dirname "${CKPT}")           # .../prompt_learner
    EXP_DIR=$(dirname "${EXP_DIR}")        # .../7_PGD40_...
    EXP_NAME=$(basename "${EXP_DIR}")
    REPORT_FILE="${EXP_DIR}/lipschitz_report.txt"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "  实验: ${EXP_NAME}"
    echo "  检查点: ${CKPT}"
    echo "----------------------------------------------------------------------"

    # BaB 参数（若 BAB_ITERS=0 则传 --no-bab 不传 --bab-iters，让工具跳过）
    BAB_ARGS=""
    if [ "${BAB_ITERS}" -gt 0 ] 2>/dev/null; then
        BAB_ARGS="--bab-iters ${BAB_ITERS}"
    fi

    $PYTHON utils/lipschitz.py \
        --checkpoint "${CKPT}" \
        --prompt-dim  "${PROMPT_DIM}" \
        --visual-dim  "${VISUAL_DIM}" \
        --device      "${DEVICE}" \
        --num-jacobian-samples "${NUM_JACOBIAN_SAMPLES}" \
        --num-empirical-pairs  "${NUM_EMPIRICAL_PAIRS}" \
        ${BAB_ARGS} \
        ${NO_FLOW} \
        2>&1 | tee "${REPORT_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ ${EXIT_CODE} -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))

        # 从报告中提取关键数字追加到汇总表
        SVD_VAL=$(grep   "Jacobian SVD"  "${REPORT_FILE}" | grep -oP "L [≈≤≥] \K[0-9.]+" | head -1)
        PWR_VAL=$(grep   "幂迭代"        "${REPORT_FILE}" | grep -oP "L [≈≤≥] \K[0-9.]+" | head -1)
        EMP_VAL=$(grep   "经验采样"      "${REPORT_FILE}" | grep -oP "L [≈≤≥] \K[0-9.]+" | head -1)
        FLOW_VAL=$(grep  "经验估计"      "${REPORT_FILE}" | grep -oP "[0-9]+\.[0-9]+"      | head -1)

        printf "%-55s | %-12s | %-6s | %-8s | %s\n" \
            "${EXP_NAME}" \
            "${SVD_VAL:-N/A}" \
            "${PWR_VAL:-N/A}" \
            "${EMP_VAL:-N/A}" \
            "${FLOW_VAL:-N/A}" \
            >> "${SUMMARY_FILE}"
    else
        FAILED=$((FAILED + 1))
        echo "[警告] ${EXP_NAME} 分析失败 (exit=${EXIT_CODE})" >> "${SUMMARY_FILE}"
    fi
done

echo ""
echo "======================================================================"
echo "  批量评测完成: ${SUCCESS} 成功 / ${FAILED} 失败"
echo "  汇总表: ${SUMMARY_FILE}"
echo "======================================================================"
echo ""
cat "${SUMMARY_FILE}"
