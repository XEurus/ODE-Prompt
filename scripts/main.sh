
# ==================== GPU 配置 ====================
# 指定使用的GPU编号，多卡服务器上可同时运行多个任务
# 例如: GPU=0 或 GPU=1 或 GPU="0,1" (多卡)
GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

# ==================== 基本配置 ====================
ROOT="/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/Data"
TRAINER=AdvPT
# oxford_flowers, oxford_pets, imagenet, food101, sun397, dtd, eurosat, ucf101
DATASET=oxford_pets
# rn50, vit_b16, vit_l14
BACKBONE=vit_b16  # 模型大类 (对应 configs/trainers/{TRAINER}/{BACKBONE}/ 文件夹)
CTP=end  # class token position (end or middle)
NCTX=32  # number of context tokens
#SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
MODEL_FILE=resnet_model.pth.tar
Best_Model=resnet_model-best.pth.tar
D=$ROOT
SEED=1
exp_name="7_PGD40_16_mix-loss06_adamw-plateau_1e3_10"
TRAINING_NOTE="使用10倍数据展平训练"

DIR=./output/${DATASET}/${TRAINER}/${BACKBONE}/adv/${exp_name}
TENSORBOARD_DIR="./output/${DATASET}/${TRAINER}/${BACKBONE}/adv/TensorBoard"
PYTHON="./dassl/bin/python"

# 确保输出目录存在
mkdir -p ${DIR}

# 定义日志文件（在 tee 使用之前）
LATEST_LOG=$(ls -t ${DIR}/log.txt-* 2>/dev/null | head -n 1)
if [ -n "${LATEST_LOG}" ]; then
    LOG_FILE="${LATEST_LOG}"
else
    LOG_FILE="${DIR}/log.txt"
fi

echo "--------------------------------------------------------------------------------------"
echo "日志文件: ${LOG_FILE}"

$PYTHON train.py \
--root ${D} \
--adv-training \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${BACKBONE}/${DATASET}.yaml \
--output-dir ${DIR} \
--model-dir ${DIR} \
--adv-training \
--model-file ${MODEL_FILE} \
--note "${TRAINING_NOTE}" \
TRAINER.ADV.N_CTX ${NCTX} \
TRAINER.ADV.CLASS_TOKEN_POSITION ${CTP} \
TRAINER.ADV.CSC ${CSC} \
TRAIN.TENSORBOARD_DIR "${TENSORBOARD_DIR}" 2>&1 | tee -a ${LOG_FILE}

# 发送邮件通知
echo "发送训练完成邮件..."
$PYTHON utils/email_sender.py \
    --exp_name "${exp_name}" \
    --log_file ${LOG_FILE}

sleep 300
shutdown -h now
# # echo "-------------------------------train2----------------------------------"
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