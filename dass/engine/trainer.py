import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dass.data import DataManager
from dass.optim import build_optimizer, build_lr_scheduler
from dass.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dass.modeling import build_head, build_backbone
from dass.evaluation import build_evaluator
from attack.attackFeature import PGD
from attack.purification import super_resolution
import clip
from torch import randperm
import os


def get_model(model):
    """
    获取实际的模型对象。
    如果是DataParallel或DistributedDataParallel包装的模型，则返回model.module。
    """
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


class SimpleNet(nn.Module):
    """
    一个简单的神经网络结构，由CNN主干网络（Backbone）组成，
    并且可以选择性地包含一个头部（Head，如MLP）用于分类。
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        # 构建主干网络 (Backbone)
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features  # 获取主干网络输出特征维度

        self.head = None
        # 如果配置中指定了头部名称和隐藏层，则构建头部网络
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features  # 更新特征维度为头部的输出维度

        self.classifier = None
        # 如果类别数大于0，构建分类器（全连接层）
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        # 前向传播：通过主干网络
        f = self.backbone(x)
        # 如果有头部，通过头部网络
        if self.head is not None:
            f = self.head(f)

        # 如果没有分类器，直接返回特征
        if self.classifier is None:
            return f

        # 通过分类器得到预测结果
        y = self.classifier(f)

        # 如果需要返回特征，则返回 (预测结果, 特征)
        if return_feature:
            return y, f

        return y


class TrainerBase:
    """
    迭代训练器的基类。
    处理模型注册、保存/加载、日志记录等通用功能。
    """

    def __init__(self):
        self._models = OrderedDict()  # 存储模型字典
        self._optims = OrderedDict()  # 存储优化器字典
        self._scheds = OrderedDict()  # 存储学习率调度器字典
        self._writer = None           # TensorBoard writer

    def register_model(self, name="model", model=None, optim=None, sched=None):
        """
        注册模型、优化器和调度器到训练器中。
        """
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        """
        获取模型名称列表。
        """
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
            self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        """
        保存模型检查点。
        """
        names = self.get_model_names()

        # If model_name is not provided, generate a default one based on config prefix
        if not model_name:
            prefix = "model"
            if hasattr(self, "cfg") and hasattr(self.cfg, "MODEL") and hasattr(self.cfg.MODEL, "FILE_PREFIX"):
                prefix = self.cfg.MODEL.FILE_PREFIX
            
            if is_best:
                model_name = f"{prefix}-best.pth.tar"
            else:
                model_name = f"{prefix}.pth.tar-{epoch + 1}"

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        """
        如果存在检查点，则恢复模型。
        """
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None, model_file=None):
        """
        加载指定模型。
        """
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # 默认加载最佳模型
        if model_file is None:
            model_file = "model-best.pth.tar"
            if epoch is not None:
                model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        """
        设置模型的模式（train/eval）。
        """
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        """
        更新学习率。
        """
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        """
        检测Loss是否为NaN或Inf。
        """
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        """
        初始化TensorBoard writer。
        """
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        """
        关闭TensorBoard writer。
        """
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        """
        写入标量数据到TensorBoard。
        """
        if self._writer is None:
            # 如果writer未初始化，则不做任何操作
            # 注意writer仅在需要训练时使用
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch, path=None, adv_training=False):
        """
        通用的训练循环。
        """
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        print("adv_training: ", adv_training)
        # self.before_train() # 注意：before_train通常在调用此方法前手动调用或在子类中处理
        if adv_training:
            # 如果是对抗训练，进行预处理（例如生成对抗样本）
            self.before_adv_train(path=path)
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch() # 每个epoch前的钩子
            if adv_training:
                self.run_epoch_adv() # 运行对抗训练的epoch
            else:
                self.run_epoch() # 运行普通训练的epoch
            self.after_epoch() # 每个epoch后的钩子（如测试、保存模型）
        self.after_train() # 训练结束后的钩子

    # 以下是需要在子类中实现的钩子方法或具体逻辑
    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def run_epoch_adv(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def forward_backward_adv(self, batch_dict):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)


class ClipModel(torch.nn.Module):
    """
    CLIP模型的包装器，用于对抗攻击或特征提取。
    """
    def __init__(self, model, num_classes=1000):
        super(ClipModel, self).__init__()
        self.model = model

        # temp, self.preprocess_val = clip.load(self.name, 'cpu')
        self.visual_encoder = self.model

        output_dim = self.visual_encoder.output_dim
        # 添加一个全连接层用于攻击embedding (对抗性微调或Prompt Tuning场景)
        self.fc = torch.nn.Linear(output_dim, 2)

    def forward(self, image):
        # 通过视觉编码器提取特征
        x = self.visual_encoder(image)
        # 通过全连接层（通常用于辅助攻击目标的映射）
        x = self.fc(x)
        return x


class SimpleTrainer(TrainerBase):
    """
    实现通用功能的简单训练器类。
    """

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        # 设置设备 (CPU/GPU)
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 保存一些常用变量作为属性
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader() # 构建数据加载器
        self.build_model()       # 构建模型
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname) # 构建评估器
        self.best_result = -np.inf # 记录最佳结果

    def check_cfg(self, cfg):
        """
        检查配置变量是否正确设置（可选）。
        """
        pass

    def build_data_loader(self):
        """
        创建必要的数据相关属性。
        """
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        # 使用DataManager管理数据加载
        dm = DataManager(self.cfg, batch_size)

        self.train_loader_x = dm.train_loader_x # 标记训练数据加载器
        self.train_loader_u = dm.train_loader_u  # 可选，无标签数据加载器
        self.val_loader = dm.val_loader  # 可选，验证集加载器
        self.test_loader = dm.test_loader # 测试集加载器

        # 可选：为对抗训练构建特定的数据加载器
        self.adv = 'notransform_noshuffle'
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        # 不使用变换且不打乱的数据管理器
        dm = DataManager(self.cfg, batch_size, self.adv)
        self.train_loader_x_notransform_noshuffle = dm.train_loader_x

        self.adv = 'noshuffle'
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_EMBEDDING_SIZE
        # 仅不打乱的数据管理器，用于生成Embedding
        dm = DataManager(self.cfg, batch_size, self.adv)
        self.train_loader_x_noshuffle = dm.train_loader_x

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        """
        构建并注册模型。
        默认构建一个分类模型及其优化器和调度器。
        """
        cfg = self.cfg

        print("Building model")
        # 使用SimpleNet构建模型
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        # 如果有初始化权重配置，则加载
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        # 构建优化器
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        # 构建学习率调度器
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # 注册模型组件
        self.register_model("model", self.model, self.optim, self.sched)

        # 处理多GPU情况
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self,path=None, adv_training=False):
        """
        调用父类的train方法开始训练。
        """
        super().train(self.start_epoch, self.max_epoch, path=path, adv_training=adv_training)

    def before_train(self):
        """
        训练前的准备工作：恢复模型、初始化TensorBoard等。
        """
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # 初始化summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # 记录开始时间
        self.time_start = time.time()

    def before_adv_train(self, path, attack='PGD'):
        """
        对抗训练前的准备：生成或加载对抗样本特征。
        """
        # 定义保存对抗特征的pkl文件路径
        pkl_path = '{}/{}_{}.pkl'.format(path, self.cfg.DATASET.NAME, self.cfg.MODEL.BACKBONE.NAME.replace(
                                                                         "/",
                                                                         "_"))
        # 如果文件存在，直接加载
        if os.path.isfile(pkl_path):
            self.train_pkl = torch.load(pkl_path, weights_only=False).to('cpu')
            print('loaded train_pkl')
            return
        
        # 否则开始生成对抗样本特征
        train_eps = self.cfg.DATASET.TRAIN_EPS
        # CLIP模型的归一化参数
        normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        # 加载CLIP模型
        clip_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device='cpu')
        # 构建代理模型用于生成攻击
        surrogate = ClipModel(model=get_model(clip_model.visual), num_classes=2).eval().to(self.device)

        if attack == 'PGD':
            # 初始化PGD攻击器
            attacker = PGD(train_eps / 255., preprocess=normalize, num_iters=10)
        else:
            attacker = None

        embedding_dim = surrogate.fc.in_features
        # 初始化存储特征的张量
        self.train_pkl = torch.empty(size=[len(self.train_loader_x_notransform_noshuffle.dataset), embedding_dim])
        
        # 遍历数据集生成对抗样本并提取特征
        for batch_idx, batch in enumerate(self.train_loader_x_notransform_noshuffle):
            inputs = batch['img'].to(self.device)
            # 生成对抗样本
            images_adv = attacker.run(surrogate, inputs, scaler=1, feature_layer='fc')
            # 检查扰动范围
            assert torch.max(images_adv - inputs) < (train_eps / 255. + 1e-6)
            assert torch.min(images_adv - inputs) > (-train_eps / 255 - 1e-6)
            
            images_adv = normalize(images_adv)
            # 使用CLIP模型提取特征
            with torch.no_grad():
                embedding = clip_model.encode_image(images_adv)

            # 保存特征到tensor中
            self.train_pkl[batch_idx * self.train_loader_x_notransform_noshuffle.batch_size: (
                                                                                                         batch_idx + 1) * self.train_loader_x_notransform_noshuffle.batch_size] = embedding.cpu()

        # 保存生成的特征到文件
        torch.save(self.train_pkl, pkl_path)
        print('generated train_pkl')
        del surrogate
        torch.cuda.empty_cache()

    def before_adv_test(self, path, attack='PGD'):
        """
        对抗测试前的准备：生成测试集的对抗样本。
        """
        # 定义保存路径
        pkl_path = '{}/{}_{}_{}.pkl'.format(path, self.cfg.DATASET.NAME, self.cfg.MODEL.BACKBONE.NAME.replace("/", "_"),
                                                                              attack)
        mean_value, std_value = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        mean = torch.tensor(mean_value).view(-1, 1, 1).to(self.device)
        std = torch.tensor(std_value).view(-1, 1, 1).to(self.device)
        normalize = transforms.Normalize(mean_value, std_value)
        self.mean, self.std = mean, std
        
        # 如果存在，直接加载
        if os.path.isfile(pkl_path):
            self.test_pkl = torch.load(pkl_path, weights_only=False)
            return
        
        # 初始化存储测试对抗样本的张量
        self.test_pkl = torch.empty(size=[len(self.test_loader.dataset), 3, 224, 224])
        test_eps = self.cfg.DATASET.TEST_EPS

        if attack == 'PGD':
            # 加载模型和攻击器
            temp_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device='cpu')
            surrogate = ClipModel(model=get_model(temp_model.visual), num_classes=2).eval().to(self.device)
            attacker = PGD(test_eps / 255., preprocess=normalize, num_iters=40) # PGD攻击，40次迭代
            
            # 遍历测试集生成对抗样本
            for batch_idx, batch in enumerate(self.test_loader):
                inputs = batch['img'].to(self.device)
                inputs *= std
                inputs += mean
                # 运行攻击
                images_adv = attacker.run(surrogate, inputs, scaler=1, feature_layer='fc')
                # 检查扰动限制
                assert torch.max(images_adv - inputs) < (test_eps / 255. + 1e-6)
                assert torch.min(images_adv - inputs) > (-test_eps / 255 - 1e-6)
                images_adv = normalize(images_adv)
                # 保存对抗样本
                self.test_pkl[batch_idx * self.test_loader.batch_size: (batch_idx + 1) * self.test_loader.batch_size] = images_adv.cpu()
            
            # 保存到文件
            torch.save(self.test_pkl, pkl_path)
            del surrogate

        else:
            raise NameError
        torch.cuda.empty_cache()

    def before_black_test(self, path, attack='RAP'):
        """
        黑盒测试前的准备。
        """
        pkl_path = '{}/{}_{}.pkl'.format(path, self.cfg.DATASET.NAME, attack)
        print("black test pkl_path:", pkl_path)
        mean_value, std_value = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        mean = torch.tensor(mean_value).view(-1, 1, 1)
        std = torch.tensor(std_value).view(-1, 1, 1)
        normalize = transforms.Normalize(mean_value, std_value)
        self.mean, self.std = mean, std
        if os.path.isfile(pkl_path):
            self.test_pkl = torch.load(pkl_path, weights_only=False)
            self.test_pkl = (self.test_pkl - mean) / std
            return
        else:
            raise FileNotFoundError(
                f"Adversarial examples not found at {pkl_path}. "
                f"Please run 'python black.py' to generate {attack} adversarial examples first."
            )

    def purify(self, baseline='super-resolution'):
        """
        对对抗样本进行净化（Purification），例如使用超分辨率。
        """
        if baseline == 'super-resolution':
            inputs = self.test_pkl
            mean, std = self.mean.squeeze(0), self.std.squeeze(0)
            outputs = super_resolution(inputs, mean, std)
        else:
            raise NameError

        self.test_pkl = outputs.to(self.device)

    def after_train(self):
        """
        训练结束后的清理工作。
        """
        print("Finish training")
        self.close_writer()

    def after_epoch(self):
        """
        每个epoch结束后的操作：测试、保存模型等。
        """
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        # 如果需要测试，并且是基于验证集最佳结果保存模型
        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    is_best=True
                )

        # 如果满足检查点保存频率或这是最后一个epoch，则保存模型
        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test_adv(self, split=None):
        """
        对抗性测试流程。
        """
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # 默认使用测试集
            data_loader = self.test_loader

        array_to_pkl = self.test_pkl
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(self.device)

        print(f"Evaluate on the *{split}* set")
        # 遍历数据加载器进行测试
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            # 获取对应的对抗样本
            input_adv = array_to_pkl[batch_idx * data_loader.batch_size: (batch_idx + 1) * data_loader.batch_size]
            input_adv = input_adv.to(input.device)

            # 限制噪声幅度 (claim small noise)
            x_adv = input_adv*std + mean
            x = input*std + mean
            noise = x_adv-x
            noise = torch.clamp(noise, -16 / 255.0, 16/255.0) # 限制噪声在L_inf ball内
            x_adv = x+noise
            x_adv = torch.clamp(x_adv, 0, 1) # 限制图像在[0, 1]范围
            
            # 验证约束
            assert (torch.max(x_adv - x) < (16/255.0 + 1e-6))
            assert (torch.min(x_adv - x) > (-16 / 255.0 - 1e-6))
            
            # 重新归一化
            input_adv = (x_adv-mean)/std

            # 模型推理
            output = self.model_inference(input_adv)
            # 记录评估结果
            self.evaluator.process(output, label.to(input.device))

        # 计算指标
        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def test_adaptive_attack(self, split=None):
        """
        Adaptive Attack Test:
        Generate PGD attacks using the full model gradient (including ODE part).
        This tests if the defense holds up when the attacker knows the defense mechanism.
        """
        self.set_model_mode("eval") # 必须设为eval，但我们需要梯度回传到输入
        # 注意：虽然是eval模式，但我们仍然可以通过 set_requires_grad(True) 来求输入的梯度
        
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set with Adaptive PGD Attack (Gradient through ODE)")
        
        # Setup Normalization
        mean_val = [0.48145466, 0.4578275, 0.40821073]
        std_val = [0.26862954, 0.26130258, 0.27577711]
        mean = torch.tensor(mean_val).view(-1, 1, 1).to(self.device)
        std = torch.tensor(std_val).view(-1, 1, 1).to(self.device)
        
        # Parameters
        test_eps = self.cfg.DATASET.TEST_EPS
        eps_val = test_eps / 255.0
        n_iters = 40  # Strong attack
        alpha = 2.0 / 255.0

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            label = label.to(self.device)
            
            # --- Adaptive PGD Attack Start ---
            
            # Start from clean images (already normalized in loader)
            images = input.clone().detach()
            
            # Initialize perturbation in normalized space
            # 为了计算方便，我们在归一化空间进行梯度更新，但在截断时还原到像素空间
            delta = torch.zeros_like(images).uniform_(-0.01, 0.01) # Small random init
            delta.requires_grad = True
            
            for _ in range(n_iters):
                # Forward pass through FULL model (including ODE)
                adv_input = images + delta
                
                # 重要：清空模型梯度
                self.model.zero_grad()
                
                output = self.model(adv_input)
                loss = F.cross_entropy(output, label)
                
                # Calculate gradient of loss w.r.t delta
                grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
                
                # PGD Update: Maximize Loss
                delta.data = delta.data + alpha * grad.sign()
                
                # Projection / Clamping
                # 1. Denormalize to pixel space
                x_adv = (images + delta) * std + mean
                x_clean = images * std + mean
                
                # 2. Clamp perturbation magnitude (L_inf)
                diff = x_adv - x_clean
                diff = torch.clamp(diff, -eps_val, eps_val)
                x_adv = x_clean + diff
                
                # 3. Clamp to valid image range [0, 1]
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
                
                # 4. Normalize back
                delta.data = ((x_adv - mean) / std) - images

            # Final adversarial images
            input_adv = images + delta.detach()
            
            # --- Adaptive PGD Attack End ---

            # Inference on adaptive adversarial examples
            output = self.model_inference(input_adv)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = f"{split}_adaptive/{k}"
            self.write_scalar(tag, v, self.epoch)

        print(f"Adaptive Attack Results: {list(results.values())[0]}")
        return list(results.values())[0]

    @torch.no_grad()
    def test(self, split=None):
        """
        通用测试流程（非对抗）。
        """
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    """
    使用有标签和无标签数据的训练器基类。
    通常用于领域自适应 (Domain Adaptation) 或 半监督学习 (Semi-supervised Learning)。
    
    Domain Adaptation: 有标签数据来自源域，无标签数据来自目标域。
    Semi-supervised Learning: 所有数据来自同一域。
    """

    def run_epoch(self):
        """
        运行一个epoch的训练循环（包含有标签和无标签数据）。
        """
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # 决定迭代次数：基于有标签数据、无标签数据或两者较小值
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            # 获取有标签数据 batch_x
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            # 获取无标签数据 batch_u
            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            
            # 执行前向传播和反向传播
            loss_summary = self.forward_backward(batch_x, batch_u)
            
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            # 打印日志
            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            # 记录到TensorBoard
            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """
    仅使用有标签数据的训练器基类。
    """

    def run_epoch(self):
        """
        运行标准训练的一个epoch。
        """
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        # mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(self.device)
        # std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(self.device)
        # normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        #
        # surrogate = ClipModel(model=self.clip_model.visual, num_classes=2).to(self.device)
        # attacker = PGD(16 / 255., preprocess=normalize, num_iters=10)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            
            # 前向和反向传播
            loss_summary = self.forward_backward(batch)
            
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            # 打印日志
            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            # 写入TensorBoard
            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def run_adv_training(self):
        """
        在线对抗训练（Online Adversarial Training）：在每个batch中动态生成对抗样本。
        """
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(self.device)

        train_eps = self.cfg.DATASET.TRAIN_EPS
        normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        # clip_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device='cpu')
        
        # 准备代理模型
        surrogate = ClipModel(model=self.model.image_encoder, num_classes=2).eval().to(self.device)
        attacker = PGD(train_eps / 255., preprocess=normalize, num_iters=10)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            
            # 准备输入数据
            inputs = batch['img'].to(self.device)
            inputs *= std
            inputs += mean # 还原归一化
            
            # 生成对抗样本
            self.model.image_encoder.eval()
            images_adv = attacker.run(surrogate, inputs, scaler=1, feature_layer='fc')
            self.model.image_encoder.train()
            
            # 检查扰动限制
            assert torch.max(images_adv - inputs) < (train_eps / 255. + 1e-6)
            assert torch.min(images_adv - inputs) > (-train_eps / 255 - 1e-6)
            
            # 重新归一化并替换batch中的图像
            images_adv = normalize(images_adv)
            batch['img'] = images_adv
            
            # 使用对抗样本进行训练
            loss_summary = self.forward_backward(batch)
            
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            # 日志记录...
            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def run_epoch_adv(self):
        """
        运行预计算对抗特征的训练epoch。
        这里使用`before_adv_train`中预先生成的对抗样本特征(`self.train_pkl`)。
        """
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # self.num_batches = len(self.train_loader_x)
        self.num_batches = len(self.train_loader_x_noshuffle) # 使用不打乱的加载器长度

        seed = torch.random.seed()
        torch.random.manual_seed(seed)

        # 重新打乱数据索引，以保持DataLoader和预计算特征的对应关系
        length = randperm(len(self.train_loader_x_noshuffle.dataset.data_source)).tolist()
        self.train_loader_x_noshuffle.dataset.data_source = [self.train_loader_x_noshuffle.dataset.data_source[i] for i
                                                             in length]
        self.train_pkl = self.train_pkl[torch.LongTensor(length)]

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x_noshuffle):

            data_time.update(time.time() - end)

            # 获取当前batch对应的预计算对抗特征
            images_adv = self.train_pkl[self.batch_idx * self.train_loader_x_noshuffle.batch_size: (
                                                                                                           self.batch_idx + 1) * self.train_loader_x_noshuffle.batch_size]
            batch_dict = {'batch': batch, 'images_adv': images_adv.to(self.device)}

            # 进行训练步骤（通常是Prompt Tuning或其他利用特征的训练）
            loss_summary = self.forward_backward_adv(batch_dict)
            
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            # 日志记录...
            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
