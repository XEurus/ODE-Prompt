"""
ODE-Adversarial-Prompt-Tuning 主训练脚本

本脚本是 ODE-Prompt 框架的入口点，支持以下功能：
1. 基于 ODE 的对抗提示学习训练
2. 白盒对抗攻击评估（PGD等）
3. 黑盒对抗攻击评估（RAP等）
4. 常规干净样本准确率测试

核心组件：
- CLIP 模型作为视觉-语言基础模型
- ODE 动力学网络学习提示演化
- 对抗训练增强模型鲁棒性

使用示例：
    # 训练模式
    python train.py --root /path/to/data --trainer AdvPT --config-file configs/trainers/AdvPT/vit_b16.yaml
    
    # 评估模式（白盒攻击）
    python train.py --eval-only --model-dir /path/to/model --white-attack PGD
    
    # 评估模式（黑盒攻击）
    python train.py --eval-black --model-dir /path/to/model --black-attack RAP
"""

import argparse
import torch
import os

# DASS (Domain Adaptation / Semi-Supervised) 工具库
from dass.utils import setup_logger, set_random_seed, collect_env_info
from dass.config import get_cfg_default
from dass.engine import build_trainer

# ============================================================================
# 数据集导入 - 注册各种下游任务数据集
# 这些导入会触发数据集类的自动注册到 DASS 框架
# ============================================================================

# 细粒度分类数据集
import datasets.oxford_pets       # 宠物分类（37类）
import datasets.oxford_flowers    # 花卉分类（102类）
import datasets.fgvc_aircraft     # 飞机型号分类（100类）
import datasets.dtd               # 纹理分类（47类）
import datasets.eurosat           # 卫星图像分类（10类）
import datasets.stanford_cars     # 汽车型号分类（196类）
import datasets.food101           # 食物分类（101类）
import datasets.sun397            # 场景分类（397类）
import datasets.caltech101        # 通用物体分类（101类）
import datasets.ucf101            # 动作识别（101类）
import datasets.imagenet          # ImageNet-1K（1000类）

# ImageNet 分布偏移变体 - 用于评估域泛化能力
import datasets.imagenet_sketch   # ImageNet 素描版本
import datasets.imagenetv2        # ImageNet 验证集V2
import datasets.imagenet_a        # ImageNet 对抗样本版本
import datasets.imagenet_r        # ImageNet 渲染版本

# ============================================================================
# 训练器导入 - 注册训练算法
# ============================================================================
import trainers.advpt             # ODE-Prompt 对抗提示学习训练器
import trainers.zsclip            # 零样本 CLIP 基线

# 默认精度设置：fp16 用于加速训练，fp32 用于稳定性
prec = 'fp16'


def print_args(args, cfg, output_dir=None):
    """
    打印命令行参数和配置信息到日志和文件

    用于调试和实验记录，确保实验可复现性
    """
    # 构建输出字符串
    lines = []
    lines.append("=" * 60)
    lines.append("Arguments")
    lines.append("=" * 60)
    optkeys = sorted(args.__dict__.keys())
    for key in optkeys:
        lines.append("{}: {}".format(key, args.__dict__[key]))
    lines.append("=" * 60)
    lines.append("Config")
    lines.append("=" * 60)
    lines.append(str(cfg))
    lines.append("=" * 60)

    # 打印到控制台/日志（sys.stdout 已被重定向）
    for line in lines:
        print(line)

    # 同时保存到独立参数文件
    if output_dir is not None:
        import os.path as osp
        fpath = osp.join(output_dir, "args_config.txt")
        with open(fpath, "w") as f:
            f.write("\n".join(lines))


def reset_cfg(cfg, args):
    """
    根据命令行参数重置配置
    
    命令行参数优先级高于配置文件，此函数实现覆盖逻辑
    
    参数：
        cfg: YACS 配置节点对象（可变）
        args: 命令行参数对象
    
    覆盖的配置项：
        - DATASET.ROOT: 数据集根目录
        - OUTPUT_DIR: 输出目录
        - RESUME: 恢复训练的检查点路径
        - SEED: 随机种子
        - TRAINER.NAME: 训练器名称
        - MODEL.BACKBONE.NAME: 骨干网络名称
        - MODEL.HEAD.NAME: 分类头名称
    """
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.model_file:
        # 从文件名解析前缀
        # 假设格式如 "name.pth.tar-100" 或 "name.pth.tar"
        if ".pth.tar" in args.model_file:
            cfg.MODEL.FILE_PREFIX = args.model_file.split(".pth.tar")[0]
        else:
            cfg.MODEL.FILE_PREFIX = args.model_file

    if args.note:
        cfg.NOTE = args.note


def extend_cfg(cfg):
    """
    扩展默认配置，添加 ODE-Prompt 特有的配置项
    
    这些配置项控制 ODE-Prompt 的核心行为：
    - 提示长度和初始化
    - 对抗训练的扰动强度
    - 训练和测试的精度设置
    
    配置项说明：
        N_CTX: 上下文向量数量，即 p(t) 的序列长度
        CSC: 是否使用类别特定上下文（Class-Specific Context）
        CTX_INIT: 初始化词语，如 "a photo of a"
        PREC: 计算精度 (fp16/fp32/amp)
        CLASS_TOKEN_POSITION: 类别token位置 ('end'/'middle'/'front')
        TRAIN_EPS: 训练时对抗扰动强度（像素值，/255后使用）
        TEST_EPS: 测试时对抗扰动强度
    
    示例：
        >>> from yacs.config import CfgNode as CN
        >>> cfg.TRAINER.MY_MODEL = CN()
        >>> cfg.TRAINER.MY_MODEL.PARAM_A = 1.
    """
    from yacs.config import CfgNode as CN

    # ODE-Prompt 对抗训练配置
    cfg.TRAINER.ADV = CN()
    cfg.TRAINER.ADV.N_CTX = 32                    # 提示token数量（ODE状态维度）
    cfg.TRAINER.ADV.CSC = False                   # 是否使用类别特定上下文
    cfg.TRAINER.ADV.CTX_INIT = ""                 # 初始化词语（空则使用默认"a photo of a"）
    cfg.TRAINER.ADV.PREC = prec                   # 计算精度
    cfg.TRAINER.ADV.CLASS_TOKEN_POSITION = "end"  # 类别token位置
    
    # ODE 网络配置
    # 网络类型: mlp, mlp_spectral, resnet, resnet_spectral
    cfg.TRAINER.ADV.ODE_NETWORK_TYPE = "resnet"
    cfg.TRAINER.ADV.ODE_T = 1.0                   # ODE 时间范围终点 T（从 0 积分到 T）

    # 数据集和数据加载配置
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"         # 子采样策略：all/base/new
    cfg.DATALOADER.TRAIN_X.BATCH_EMBEDDING_SIZE = 256  # 嵌入bank的batch大小
    cfg.DATALOADER.TRAIN_X.BATCH_PGD_SIZE = 64    # PGD-bank生成时的batch大小（可以更大，充分利用显存）
    cfg.DATASET.TRAIN_EPS = 16                    # 训练扰动强度（16/255 ≈ 0.063）
    cfg.DATASET.TEST_EPS = 16                     # 测试扰动强度（16/255 ≈ 0.063）
    cfg.DATASET.Train_PGD_NUM_ITERS = 60                # PGD攻击迭代次数（训练和测试统一）
    cfg.DATASET.Test_PGD_NUM_ITERS = 60                # PGD攻击迭代次数（训练和测试统一）

    cfg.MODEL.FILE_PREFIX = "model"               # 模型文件名前缀
    cfg.NOTE = ""                                  # 训练备注



def setup_cfg(args):
    """
    设置完整配置
    
    配置加载优先级（从低到高）：
    1. 默认配置
    2. 数据集配置文件
    3. 方法配置文件
    4. 命令行参数
    5. 额外opts参数
    
    参数：
        args: 命令行参数对象
    
    返回：
        cfg: 冻结的配置对象（不可修改）
    """
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. 从数据集配置文件加载
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. 从方法配置文件加载
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. 从命令行参数覆盖
    reset_cfg(cfg, args)

    # 4. 从额外opts参数覆盖（如 TRAINER.ADV.N_CTX 64）
    cfg.merge_from_list(args.opts)

    # 冻结配置，防止运行时修改
    cfg.freeze()

    return cfg


def main(args):
    """主函数 - 协调训练和评估流程"""
    cfg = setup_cfg(args)
    
    # 设置随机种子
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # 创建 pkl 数据保存目录
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    
    if cfg.NOTE:
        print(f"[TRAINING NOTE] {cfg.NOTE}")

    # 调试信息
    print_args(args, cfg, cfg.OUTPUT_DIR)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    # 构建训练器（根据 cfg.TRAINER.NAME 自动选择）
    trainer = build_trainer(cfg)

    # ========================================================================
    # 白盒攻击评估模式
    # ========================================================================
    # print('-' * 60)
    # print('adaptive attack acc:')
    # trainer.test_adaptive_attack()
    # print('-' * 60)
    if args.eval_only:
        # 加载预训练模型
        args.eval_only = True
        trainer.load_model(args.model_dir, epoch=args.load_epoch, model_file=args.model_file)
        print(f"Model loaded from: {args.model_dir}")
        print('-' * 60)
        
        print('Clean accuracy:')
        trainer.test()
        print('-' * 60)
        
        print(f'Robust accuracy ({args.white_attack}):')
        trainer.before_adv_test(args.path, args.white_attack)
        trainer.test_adv()
        print('-' * 60)

        print('robust acc(PGD) - embedding path (unified with training):')
        trainer.generate_test_embedding(args.path)
        trainer.test_adv_embedding(split="test")
        print('-' * 60)

        # print('Adaptive attack accuracy:')
        # trainer.test_adaptive_attack()
        # print('-' * 60)
        return

    # ========== 黑盒攻击评估模式 ==========
    elif args.eval_black:
        trainer.load_model(args.model_dir, epoch=args.load_epoch, model_file=args.model_file)
        print(f"Model loaded from: {args.model_dir}")
        print('-' * 60)
        
        print('Clean accuracy:')
        trainer.test()
        print('-' * 60)
        
        print(f'Black-box robust accuracy ({args.black_attack}):')
        trainer.before_black_test(args.path, args.black_attack)
        trainer.test_adv()
        print('-' * 60)

        print(f'White-box robust accuracy ({args.white_attack}):')
        trainer.before_adv_test(args.path, args.white_attack)
        trainer.test_adv()
        print('-' * 60)

        print('Adaptive attack accuracy:')
        trainer.test_adaptive_attack()
        print('-' * 60)
        return

    # ========== 训练模式 ==========
    elif not args.no_train:
        if args.adv_training:
            print('=' * 60)
            print('Preparing adversarial training data...')
            print('=' * 60)
            
            attack_mode = args.white_attack
            print('\n[1/4] Generating/Loading training clean embeddings (for mixed training)...')
            trainer.before_clean_train(path=args.path)
            
            print(f'\n[2/4] Generating/Loading training adversarial embeddings ({attack_mode})...')
            trainer.before_adv_train(path=args.path, attack=attack_mode)
            
            print(f'\n[3/4] Generating/Loading validation adversarial embeddings ({attack_mode})...')
            trainer.before_adv_val(path=args.path, attack=attack_mode)
            
            print(f'\n[4/4] Generating/Loading test adversarial samples ({attack_mode})...')
            trainer.before_adv_test(path=args.path, attack=attack_mode)
            
            print('=' * 60)
            print('Starting adversarial training...')
            print('=' * 60 + '\n')
            
            trainer.train(path=args.path, adv_training=True)
        else:
            raise "error"
            # 标准训练模式
            trainer.train()
        
        # 训练完成后进行评估
        print('-' * 60)
        print('Clean accuracy:')
        trainer.test()
        print('-' * 60)
        # print('robust acc(RAP):')
        # trainer.before_black_test(args.path, args.black_attack)
        # trainer.test_adv()
        # print('-' * 60)

        print('robust acc(PGD) - image path:')
        trainer.test_adv()
        print('-' * 60)

        # print('robust acc(PGD) - embedding path (unified with training):')
        # trainer.generate_test_embedding(args.path)
        # trainer.test_adv_embedding(split="test")
        # print('-' * 60)

        # print('adaptive attack acc:')
        # trainer.test_adaptive_attack()
        # print('-' * 60)
        return



if __name__ == "__main__":
    # ==================== 默认参数配置（直接运行时使用） ====================
    # 修改这里的默认值即可直接 python train.py 调试，也兼容 sh 脚本传参覆盖
    DEFAULTS = {
        # 路径配置
        "root": "/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/Data",
        "output_dir": "./output/oxford_pets/AdvPT/vit_b16/adv",
        "path": "./pkl_data_mix_PGD5_1/",
        
        # 配置文件
        "config_file": "configs/trainers/AdvPT/vit_b16.yaml",
        "dataset_config_file": "configs/datasets/oxford_pets.yaml",
        
        # 模型设置
        "trainer": "AdvPT",
        "backbone": "",
        "head": "",
        "model_file": "2_layer_resnet_model.pth.tar-100",
        
        # 评估配置
        "model_dir": "./output/oxford_pets/AdvPT/vit_b16/adv",
        "load_epoch": None,
        
        # 攻击方法
        "black_attack": "RAP",
        "white_attack": "PGD",
        
        # 其他
        "resume": "",
        "seed": 1,
        "note": "",
        
        # === 布尔开关（直接运行时的默认值） ===
        "adv_training": True,   # 启用对抗训练
        "no_train": False,      # 不训练
        "eval_only": False,     # 白盒评估模式
        "eval_black": False,    # 黑盒评估模式
    }
    # ================================================================

    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument("--root", type=str, default=DEFAULTS["root"], help="path to dataset")
    parser.add_argument("--output-dir", type=str, default=DEFAULTS["output_dir"], help="output directory")
    parser.add_argument("--path", type=str, default=DEFAULTS["path"], help="directory of pkl")
    
    # 训练控制
    parser.add_argument("--adv-training", action="store_true", default=DEFAULTS["adv_training"], help="启用对抗训练")
    parser.add_argument("--no-adv-training", action="store_true", help="禁用对抗训练")
    parser.add_argument("--no-train", action="store_true", default=DEFAULTS["no_train"], help="do not call trainer.train()")
    parser.add_argument("--resume", type=str, default=DEFAULTS["resume"], help="checkpoint directory")
    
    # 配置文件
    parser.add_argument("--config-file", type=str, default=DEFAULTS["config_file"], help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default=DEFAULTS["dataset_config_file"], help="path to dataset config")
    
    # 模型设置
    parser.add_argument("--trainer", type=str, default=DEFAULTS["trainer"], help="name of trainer")
    parser.add_argument("--backbone", type=str, default=DEFAULTS["backbone"], help="name of CNN backbone")
    parser.add_argument("--head", type=str, default=DEFAULTS["head"], help="name of head")
    parser.add_argument("--model-file", type=str, default=DEFAULTS["model_file"], help="name of model file")
    
    # 评估模式
    parser.add_argument("--eval-only", action="store_true", default=DEFAULTS["eval_only"], help="白盒攻击评估")
    parser.add_argument("--eval-black", action="store_true", default=DEFAULTS["eval_black"], help="黑盒攻击评估")
    parser.add_argument("--model-dir", type=str, default=DEFAULTS["model_dir"], help="model directory for eval")
    parser.add_argument("--load-epoch", type=int, default=DEFAULTS["load_epoch"], help="load model at this epoch")
    
    # 攻击方法选择
    parser.add_argument("--black-attack", type=str, default=DEFAULTS["black_attack"], help="黑盒攻击方法: RAP, SIA")
    parser.add_argument("--white-attack", type=str, default=DEFAULTS["white_attack"], help="白盒攻击方法: PGD, FGSM")
    
    # 额外配置
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="random seed")
    parser.add_argument("--note", type=str, default=DEFAULTS["note"], help="training note")
    
    args = parser.parse_args()
    
    # 处理 --no-adv-training 覆盖
    if args.no_adv_training:
        args.adv_training = False

    # 启动主程序
    print("DEBUG: args.eval_only =", args.eval_only)
    print("DEBUG: args.adv_training =", args.adv_training)
    main(args)
