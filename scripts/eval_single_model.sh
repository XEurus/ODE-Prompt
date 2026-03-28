#!/bin/bash
# ==================== 单模型评测脚本 ====================
# 测试一个训练好的 ODE-Prompt 模型的干净准确率和白盒对抗准确率
#
# 用法:
#   bash scripts/eval_single_model.sh
#
# 或者覆盖参数:
#   MODEL_PATH=/path/to/model.pth.tar \
#   PKL_DIR=./pkl_xxx \
#   TEST_EPS=1 \
#   PGD_ITERS=100 \
#   bash scripts/eval_single_model.sh

GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

PYTHON="./dassl/bin/python"

# ==================== 可调参数（可通过环境变量覆盖） ====================
# 模型文件的完整路径
MODEL_PATH=${MODEL_PATH:-"/autodl-fs/data/output/oxford_pets/AdvPT/vit_b16/adv/10_whitebox_16shot_resnet_pgd5_eps5_adamw/prompt_learner/model-best.pth.tar"}

# pkl 数据目录（需包含 test_v2_whitebox.pkl，或会自动生成）
PKL_DIR=${PKL_DIR:-"./pkl_whitebox_pgd5_eps5"}

# 攻击参数
ATTACK_MODE=${ATTACK_MODE:-"PGD_whitebox"}   # PGD | PGD_whitebox | PGD_adaptive
TEST_EPS=${TEST_EPS:-1}                       # 扰动强度 (x/255)
PGD_ITERS=${PGD_ITERS:-100}                   # PGD 迭代次数

# 数据集和模型配置
DATASET=${DATASET:-"oxford_pets"}
BACKBONE=${BACKBONE:-"vit_b16"}
TRAINER=${TRAINER:-"AdvPT"}
ODE_NETWORK=${ODE_NETWORK:-"resnet"}
N_CTX=${N_CTX:-32}

CONFIG_FILE="configs/trainers/${TRAINER}/${BACKBONE}/${DATASET}.yaml"
DATASET_CONFIG="configs/datasets/${DATASET}.yaml"

echo "======================================================================"
echo "  单模型评测"
echo "  模型:      ${MODEL_PATH}"
echo "  攻击模式:  ${ATTACK_MODE}"
echo "  测试攻击:  PGD${PGD_ITERS} eps=${TEST_EPS}/255"
echo "  PKL目录:   ${PKL_DIR}"
echo "======================================================================"

$PYTHON -c "
import torch, gc, sys, os
from dass.engine import build_trainer
from dass.utils import set_random_seed, setup_logger
from train import setup_cfg

MODEL_PATH = '${MODEL_PATH}'
PKL_DIR    = '${PKL_DIR}'
ATTACK     = '${ATTACK_MODE}'

class Args:
    prompt_learner_dir = ''
    root = '/autodl-fs/data/Data'
    output_dir = '/tmp/eval_single'
    resume = ''; seed = 1; trainer = '${TRAINER}'
    backbone = ''; head = ''
    model_file = ''
    config_file = '${CONFIG_FILE}'
    dataset_config_file = '${DATASET_CONFIG}'
    path = PKL_DIR
    white_attack = ATTACK; black_attack = ''; mode = 'test-only'
    csv_path = ''; plot_path = ''; note = ''
    opts = ['TRAINER.ADV.ODE_NETWORK_TYPE', '${ODE_NETWORK}',
            'TRAINER.ADV.N_CTX', '${N_CTX}',
            'TRAINER.ADV.CLASS_TOKEN_POSITION', 'end',
            'TRAINER.ADV.CSC', 'False',
            'DATASET.TEST_EPS', '${TEST_EPS}',
            'DATASET.Test_PGD_NUM_ITERS', '${PGD_ITERS}']

cfg = setup_cfg(Args())
set_random_seed(cfg.SEED)
setup_logger(cfg.OUTPUT_DIR)
trainer = build_trainer(cfg)

# ---- 加载模型 ----
model_dir = os.path.dirname(os.path.dirname(MODEL_PATH))
model_file = os.path.basename(MODEL_PATH)
print(f'\n加载模型: {MODEL_PATH}')
trainer.load_model(model_dir, model_file=model_file)

# ---- 1. 干净准确率 ----
print()
print('=' * 60)
print('[1/2] 干净准确率 (test set)')
print('=' * 60)
clean_acc = trainer.test()

# ---- 2. 白盒对抗准确率 ----
print()
print('=' * 60)
print(f'[2/2] 对抗准确率 — {ATTACK} PGD${PGD_ITERS} eps=${TEST_EPS}/255')
print('=' * 60)
trainer.before_adv_test(path=PKL_DIR, attack=ATTACK)
adv_acc = trainer.test_adv(split='test')

# ---- 汇总 ----
print()
print('=' * 60)
print(f'  模型:          {MODEL_PATH}')
print(f'  干净准确率:    {clean_acc:.2f}%')
print(f'  对抗准确率:    {adv_acc:.2f}%')
print(f'  准确率下降:    {clean_acc - adv_acc:.2f}%')
print(f'  攻击方式:      {ATTACK} PGD${PGD_ITERS} eps=${TEST_EPS}/255')
print('=' * 60)
" 2>/dev/null
