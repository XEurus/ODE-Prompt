# ODE-Adversarial-Prompt-Tuning

基于 ECCV 2024 论文 "Adversarial Prompt Tuning for Vision-Language Models" 的扩展实现。本项目将 ODE（常微分方程）动力学网络引入对抗提示学习框架，通过连续时间演化学习文本提示，增强视觉-语言模型（如 CLIP）的对抗鲁棒性。

## 环境配置

参考 [CoOp](https://github.com/KaiyangZhou/CoOp#how-to-install) 的安装指南配置环境。

## 数据准备

按照 [DATASETS.md](./DATASETS.md) 准备数据集到指定目录。

## 快速开始

### 训练

**Bank 模式**（预计算对抗嵌入，推荐）:
```bash
./scripts/main.sh
# 或使用 few-shot 训练脚本
./scripts/train_16shot.sh
```

**实时对抗训练模式**:
```bash
# 修改 scripts/train_16shot.sh 中的 TRAINING_MODE="realtime"
./scripts/train_16shot.sh
```

### 评估

**白盒攻击评估**:
```bash
python train.py \
  --root YOUR_DATA \
  --eval-only \
  --trainer AdvPT \
  --dataset-config-file configs/datasets/oxford_pets.yaml \
  --config-file configs/trainers/AdvPT/vit_b16/oxford_pets.yaml \
  --model-dir YOUR_MODEL_DIR \
  --white-attack PGD
```

**黑盒攻击评估**:
```bash
# 1. 生成黑盒对抗样本
python black.py --root YOUR_DATA --dataset OxfordPets --path ./pkl_data

# 2. 评估
python train.py \
  --root YOUR_DATA \
  --eval-black \
  --trainer AdvPT \
  --dataset-config-file configs/datasets/oxford_pets.yaml \
  --config-file configs/trainers/AdvPT/vit_b16/oxford_pets.yaml \
  --model-dir YOUR_MODEL_DIR \
  --black-attack RAP
```

## 主要参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--root` | 数据集根目录 | `/path/to/data` |
| `--adv-training` | 启用 Bank 对抗训练 | - |
| `--realtime-adv` | 启用实时对抗训练 | - |
| `--eval-only` | 白盒攻击评估模式 | - |
| `--eval-black` | 黑盒攻击评估模式 | - |
| `--white-attack` | 白盒攻击方法 | `PGD`, `FGSM` |
| `--black-attack` | 黑盒攻击方法 | `RAP` |
| `--path` | pkl 文件保存目录 | `./pkl_data` |

配置选项（通过 `KEY VALUE` 格式传递）：
```bash
TRAINER.ADV.N_CTX 32                    # 提示 token 数量
TRAINER.ADV.ODE_NETWORK_TYPE resnet     # ODE 网络类型
TRAINER.ADV.ODE_T 1.0                   # ODE 积分时间范围
DATASET.NUM_SHOTS 16                    # Few-shot 样本数
DATASET.TRAIN_EPS 16                    # 训练扰动强度 (x/255)
DATASET.TEST_EPS 16                     # 测试扰动强度
DATASET.Train_PGD_NUM_ITERS 40          # 训练 PGD 迭代次数
DATASET.Test_PGD_NUM_ITERS 100          # 测试 PGD 迭代次数
```

## 项目结构

```
├── train.py                 # 主训练脚本
├── black.py                 # 黑盒攻击生成
├── trainers/
│   ├── advpt.py            # ODE-Prompt 核心实现
│   └── zsclip.py           # 零样本 CLIP 基线
├── attack/
│   └── attackFeature.py    # PGD 攻击实现
├── configs/
│   ├── datasets/           # 数据集配置
│   └── trainers/AdvPT/     # 训练器配置（按骨干网络组织）
├── scripts/
│   ├── main.sh             # 标准对抗训练
│   ├── train_16shot.sh     # Few-shot 训练
│   └── eval.sh             # 评估脚本
└── utils/
    └── adv_utils.py        # 对抗训练工具
```

