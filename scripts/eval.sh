
# ==================== 配置 ====================
GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

PYTHON="./dassl/bin/python"

DATASET=oxford_pets
BACKBONE=vit_b16
TRAINER=AdvPT
EXP_NAME=7_PGD40_16_mix-loss06_adamw-plateau_1e3_10

PROMPT_LEARNER_DIR=./output/${DATASET}/${TRAINER}/${BACKBONE}/adv/${EXP_NAME}/prompt_learner
PKL_PATH=./pkl_data_mix_5
CONFIG_FILE=configs/trainers/${TRAINER}/${BACKBONE}/${DATASET}.yaml
DATASET_CONFIG=configs/datasets/${DATASET}.yaml
# ==================== 生成 RAP 黑盒对抗样本 ====================
$PYTHON black.py --gpu 0 --dataset OxfordPets --path ${PKL_PATH}

# # ==================== 白盒评估 (test-only 模式) ====================
# echo "--------------------------------------------------------------------------------------"
# echo "[白盒] test-only 评估: ${EXP_NAME}"
# $PYTHON eval_all_prompt_learner_models.py \
#   --prompt-learner-dir ${PROMPT_LEARNER_DIR} \
#   --path ${PKL_PATH} \
#   --dataset-config-file ${DATASET_CONFIG} \
#   --config-file ${CONFIG_FILE} \
#   --white-attack PGD \
#   --mode test-only \
#   --plot-path ./output/${DATASET}/${TRAINER}/${BACKBONE}/adv/${EXP_NAME}/curve_white.png \
#   --csv-path ./output/${DATASET}/${TRAINER}/${BACKBONE}/adv/${EXP_NAME}/eval_white_only.csv

# ==================== 黑盒+白盒联合评估 (test-only 模式) ====================
# 需先运行 black.py 生成对抗样本:

# echo "--------------------------------------------------------------------------------------"
# echo "[黑盒+白盒] test-only 评估: ${EXP_NAME}"
$PYTHON eval_all_prompt_learner_models.py \
  --prompt-learner-dir ${PROMPT_LEARNER_DIR} \
  --path ${PKL_PATH} \
  --dataset-config-file ${DATASET_CONFIG} \
  --config-file ${CONFIG_FILE} \
  --white-attack PGD \
  --black-attack RAP \
  --mode test-only \
  --plot-path ./output/${DATASET}/${TRAINER}/${BACKBONE}/adv/${EXP_NAME}/curve_black_white.png \
  --csv-path ./output/${DATASET}/${TRAINER}/${BACKBONE}/adv/${EXP_NAME}/eval_black_white_only.csv

# ==================== 完整评估 (full 模式，含 train/val embedding) ====================
# echo "--------------------------------------------------------------------------------------"
# echo "[Full] 完整评估: ${EXP_NAME}"
# $PYTHON eval_all_prompt_learner_models.py \
#   --prompt-learner-dir ${PROMPT_LEARNER_DIR} \
#   --path ${PKL_PATH} \
#   --dataset-config-file ${DATASET_CONFIG} \
#   --config-file ${CONFIG_FILE} \
#   --white-attack PGD \
#   --black-attack RAP \
#   --mode full \
#   --csv-path ./output/${DATASET}/${TRAINER}/${BACKBONE}/adv/${EXP_NAME}/eval_full.csv

# ==================== 原版 vs ODE 对比评估 ====================
echo "--------------------------------------------------------------------------------------"
echo "[对比] ZeroShot CLIP baseline vs ODE-Prompt: ${EXP_NAME}"
$PYTHON compare_baseline_vs_ode.py \
  --config-file ${CONFIG_FILE} \
  --dataset-config-file ${DATASET_CONFIG} \
  --model-dir ./output/${DATASET}/${TRAINER}/${BACKBONE}/adv/${EXP_NAME} \
  --model-file model-best.pth.tar \
  --pkl-path ${PKL_PATH} \
  --white-attack PGD \
  --black-attack RAP \
  TRAINER.ADV.N_CTX 32 TRAINER.ADV.CLASS_TOKEN_POSITION end TRAINER.ADV.CSC False TRAIN.TENSORBOARD_DIR ''

$PYTHON utils/email_sender.py \
    --exp_name "${exp_name}"


sleep 300
shutdown -h now
echo "-------------------------------train2----------------------------------"
# DIR=./output/${DATASET}/${TRAINER}/${CFG}/adv/5_ucf101_PGD40_16_mix5_plateau
# PYTHON="./dassl/bin/python"
# # rn50, vit_b16, vit_l14
# CFG=vit_b16_2 # config file
# $PYTHON train.py \
# --root ${D} \
# --adv-training \
# --seed ${SEED} \
# --trainer ${TRAINER} \
# --dataset-config-file configs/datasets/${DATASET}.yaml \
# --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
# --output-dir ${DIR} \
# --model-dir ${DIR} \
# --adv-training \
# --model-file ${MODEL_FILE} \
# TRAINER.ADV.N_CTX ${NCTX} \
# TRAINER.ADV.CLASS_TOKEN_POSITION ${CTP} \
# TRAINER.ADV.CSC ${CSC}

# sleep 60
# shutdown -h now
# echo "--------------------------------------------------------------------------------------"
# echo "zero shot"
# TRAINER=ZeroshotCLIP
# $PYTHON train.py \
# --root ${D} \
# --trainer ${TRAINER} \
# --dataset-config-file configs/datasets/${DATASET}.yaml \
# --config-file configs/trainers/AdvPT/${CFG}.yaml \
# --output-dir output/${TRAINER}/${CFG}/${DATASET} \
# --model-dir /root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/output/oxford_pets/AdvPT/vit_b16/adv/prompt_learner/8_layer_resnet_model.pth.tar-100 \
# --eval-only

# $PYTHON train.py \
# --root ${D} \
# --trainer ${TRAINER} \
# --dataset-config-file configs/datasets/${DATASET}.yaml \
# --config-file configs/trainers/AdvPT/${CFG}.yaml \
# --output-dir ${DIR}/eval \
# --model-dir ${DIR} \
# --model-file ${Best_Model} \
# --eval-only