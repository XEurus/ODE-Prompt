# ==================== GPU 配置 ====================
GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

# ==================== 基本配置 ====================
ROOT="/autodl-fs/data/Data"
TRAINER=AdvPT
# oxford_flowers, oxford_pets, imagenet, food101, sun397, dtd, eurosat, ucf101
DATASET=oxford_pets
# rn50, vit_b16, vit_l14
BACKBONE=vit_b16
SEED=1 # 是否随机种子

# ==================== Prompt 配置 ====================
NCTX=32                    # 提示token数量
CSC=False                  # 类别特定上下文 (False/True)
CTX_INIT=""                # 初始化词语 (空="a photo of a")
CTP=end                    # 类别token位置 (end/middle/front)
PREC=fp32                  # 计算精度 (fp16/fp32/amp)

# ==================== ODE 网络配置 ====================
# 网络类型: mlp, mlp_spectral, resnet, resnet_spectral
ODE_NETWORK_TYPE="resnet"
ODE_T=1.0                  # ODE 时间范围 [0, T]

# ==================== 攻击配置 ====================
TRAIN_EPS=16               # 训练扰动强度 (eps/255)
TEST_EPS=16                # 测试扰动强度
TRAIN_PGD_ITERS=40         # 训练 PGD 迭代次数
TEST_PGD_ITERS=40          # 测试 PGD 迭代次数

# ==================== 数据加载配置 ====================
SUBSAMPLE_CLASSES="all"    # 子采样策略 (all/base/new)

# ==================== 模型文件配置 ====================
MODEL_FILE_PREFIX="${ODE_NETWORK_TYPE}_${BACKBONE}"  # 模型文件名前缀
MODEL_FILE=${MODEL_FILE_PREFIX}.pth.tar
Best_Model=${MODEL_FILE_PREFIX}-best.pth.tar

# ==================== 实验配置 ====================
exp_name="9_${ODE_NETWORK_TYPE}_T${ODE_T}_pgd${TRAIN_PGD_ITERS}-${TRAIN_EPS}"
TRAINING_NOTE="${ODE_NETWORK_TYPE} T=${ODE_T} PGD${TRAIN_PGD_ITERS}-${TRAIN_EPS}"

D=$ROOT
DIR=/autodl-fs/data/output/${DATASET}/${TRAINER}/${BACKBONE}/adv/${exp_name}
TENSORBOARD_DIR="/autodl-fs/data/output/${DATASET}/${TRAINER}/${BACKBONE}/"
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
--model-file ${MODEL_FILE} \
--note "${TRAINING_NOTE}" \
TRAINER.ADV.N_CTX ${NCTX} \
TRAINER.ADV.CSC ${CSC} \
TRAINER.ADV.CTX_INIT "${CTX_INIT}" \
TRAINER.ADV.PREC ${PREC} \
TRAINER.ADV.CLASS_TOKEN_POSITION ${CTP} \
TRAINER.ADV.ODE_NETWORK_TYPE ${ODE_NETWORK_TYPE} \
TRAINER.ADV.ODE_T ${ODE_T} \
DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
DATALOADER.TRAIN_X.BATCH_EMBEDDING_SIZE ${BATCH_EMBEDDING_SIZE} \
DATALOADER.TRAIN_X.BATCH_PGD_SIZE ${BATCH_PGD_SIZE} \
DATASET.TRAIN_EPS ${TRAIN_EPS} \
DATASET.TEST_EPS ${TEST_EPS} \
DATASET.Train_PGD_NUM_ITERS ${TRAIN_PGD_ITERS} \
DATASET.Test_PGD_NUM_ITERS ${TEST_PGD_ITERS} \
MODEL.FILE_PREFIX ${MODEL_FILE_PREFIX} \
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