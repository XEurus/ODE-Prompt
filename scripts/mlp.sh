#!/bin/bash

# custom config
ROOT="./Data"
TRAINER=mlp
# oxford_flowers, oxford_pets, imagenet, food101, sun397, dtd, eurosat, ucf101
DATASET=oxford_pets
# rn50, vit_b16, vit_l14
CFG=vit_b16 # config file
CTP=end  # class token position (end or middle)
NCTX=32  # number of context tokens
#SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
MODEL_FILE=resnet_model.pth.tar
Best_Model=resnet_model-best.pth.tar
D=$ROOT
SEED=1

DIR=./output/${DATASET}/${TRAINER}/${CFG}/adv/MLP4_5e-4_PGD40_16_mix6
PYTHON="./dassl/bin/python"
echo "--------------------------------------------------------------------------------------"
# Redirect all output to a fresh log each run (prevents duplicate history)
# LOG_FILE=terminal.log
# : > "$LOG_FILE"
# exec > >(tee "$LOG_FILE") 2>&1
echo "train1"

$PYTHON train.py \
--root ${D} \
--adv-training \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${DIR} \
--adv-training \
--model-file ${MODEL_FILE} \
TRAINER.ADV.N_CTX ${NCTX} \
TRAINER.ADV.CLASS_TOKEN_POSITION ${CTP} \
TRAINER.ADV.CSC ${CSC}

# echo "--------------------------------------------------------------------------------------"
# echo "zero shot"
# TRAINER=ZeroshotCLIP
# $PYTHON train.py \
# --root ${D} \
# --trainer ${TRAINER} \
# --dataset-config-file configs/datasets/${DATASET}.yaml \
# --config-file configs/trainers/AdvPT/${CFG}.yaml \
# --output-dir output/${TRAINER}/${CFG}/${DATASET} \
# --model-dir /home/dji/Project/ODE-Prompt/ODE-Adversarial-Prompt-Tuning/output/oxford_pets/AdvPT/vit_b16/adv/prompt_learner/8_layer_resnet_model.pth.tar-100 \
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

# echo "--------------------------------training 2---------------------------------------"
# CFG=vit_b16_2 # config file

# DIR=./output/${DATASET}/${TRAINER}/${CFG}/adv/4-4_resnet_PGD10_8_mix
# PYTHON="/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/dassl/bin/python"
# echo "--------------------------------------------------------------------------------------"
# # Redirect all output to a fresh log each run (prevents duplicate history)
# LOG_FILE=terminal.log
# : > "$LOG_FILE"
# exec > >(tee "$LOG_FILE") 2>&1
# echo "ceshi"

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