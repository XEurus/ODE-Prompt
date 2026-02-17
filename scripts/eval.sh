

PYTHON="./dassl/bin/python"
echo "--------------------------------------------------------------------------------------"
$PYTHON eval_all_prompt_learner_models.py \
  --prompt-learner-dir /root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/output/ucf101/AdvPT/vit_b16/adv/5_PGD40_16_mix6_sgd_1e3_60/prompt_learner \
  --path /root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/pkl_data \
  --dataset-config-file configs/datasets/ucf101.yaml \
  --config-file configs/trainers/AdvPT/vit_b16.yaml \
  --white-attack PGD

# sleep 60
# shutdown -h now
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