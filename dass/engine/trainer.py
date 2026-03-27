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
from attack.purification import super_resolution
from utils.adv_utils import (
    ImageNormalizer, ClipModel, get_model, create_pgd_attacker
)
import clip
from torch import randperm
import os


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
        self.before_train()
        # if adv_training:
        #     # 如果是对抗训练，进行预处理（例如生成对抗样本）
        #     self.before_adv_train(path=path)
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
        """创建必要的数据相关属性。"""
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        dm = DataManager(self.cfg, batch_size)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader

        # 不带归一化的训练 DataLoader（用于 PGD-bank 生成）
        # 使用单独的 BATCH_PGD_SIZE，可以更大以充分利用显存
        batch_size_pgd = getattr(self.cfg.DATALOADER.TRAIN_X, 'BATCH_PGD_SIZE', batch_size)
        dm_notransform = DataManager(self.cfg, batch_size_pgd, adv='notransform_noshuffle')
        self.train_loader_x_notransform_noshuffle = dm_notransform.train_loader_x
        self.val_loader_notransform = dm_notransform.val_loader  # 验证集也使用 notransform

        # 带归一化但不打乱的训练 DataLoader（用于生成 Embedding）
        batch_size_emb = self.cfg.DATALOADER.TRAIN_X.BATCH_EMBEDDING_SIZE
        dm_noshuffle = DataManager(self.cfg, batch_size_emb, adv='noshuffle')
        self.train_loader_x_noshuffle = dm_noshuffle.train_loader_x

        # 不带归一化的测试 DataLoader（用于对抗攻击）
        test_batch_size = self.cfg.DATALOADER.TEST.BATCH_SIZE
        dm_test_notransform = DataManager(self.cfg, test_batch_size, adv='notransform_noshuffle')
        self.test_loader_notransform = dm_test_notransform.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

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

        # 初始化 summary writer
        # 支持将多个实验写入统一根目录，便于跨实验对比。
        tb_root = getattr(self.cfg.TRAIN, "TENSORBOARD_DIR", "")
        if tb_root:
            tb_base_dir = tb_root
        elif f"{osp.sep}adv{osp.sep}" in osp.normpath(self.output_dir):
            # output_dir 形如: .../<cfg>/adv/<exp_name> -> 聚合到 .../<cfg>
            tb_base_dir = osp.dirname(osp.dirname(self.output_dir))
        else:
            tb_base_dir = self.output_dir

        run_name = osp.basename(osp.normpath(self.output_dir))
        writer_dir = osp.join(tb_base_dir, "tensorboard", run_name)
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        if self._writer is not None:
            self._writer.add_text("runtime/output_dir", self.output_dir, 0)
            self._writer.add_text("runtime/tensorboard_writer_dir", writer_dir, 0)

            args_cfg_path = osp.join(self.output_dir, "args_config.txt")
            if osp.isfile(args_cfg_path):
                with open(args_cfg_path, "r", encoding="utf-8", errors="ignore") as f:
                    args_cfg_text = f.read()
                self._writer.add_text("config/args_config", f"```\n{args_cfg_text}\n```", 0)

            # 记录完整配置，便于在 TensorBoard 里追踪实验参数
            self._writer.add_text("config/full", f"```\n{self.cfg.dump()}\n```", 0)
            self._writer.add_text("hparams/optimizer_name", str(self.cfg.OPTIM.NAME), 0)
            self._writer.add_text("hparams/lr_scheduler", str(self.cfg.OPTIM.LR_SCHEDULER), 0)
            self._writer.add_text("hparams/backbone", str(self.cfg.MODEL.BACKBONE.NAME), 0)
            self._writer.add_text("hparams/dataset", str(self.cfg.DATASET.NAME), 0)

            # 记录关键超参数（标量形式，便于对比）
            self.write_scalar("hparams/optim_lr", float(self.cfg.OPTIM.LR), 0)
            self.write_scalar("hparams/optim_max_epoch", float(self.cfg.OPTIM.MAX_EPOCH), 0)
            self.write_scalar("hparams/train_batch_size", float(self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE), 0)
            # self.write_scalar("hparams/mix_clean_ratio", float(getattr(self.cfg.TRAIN, "MIX_CLEAN_RATIO", 0.0)), 0)
            self.write_scalar("hparams/adv_n_ctx", float(getattr(self.cfg.TRAINER.ADV, "N_CTX", 0)), 0)
            self.write_scalar("hparams/epoch_test_batches", float(getattr(self.cfg.TEST, "EPOCH_TEST_BATCHES", -1)), 0)
            self.write_scalar("hparams/partial_test_batches", float(getattr(self.cfg.TEST, "PARTIAL_TEST_BATCHES", 10)), 0)
            self.write_scalar("hparams/checkpoint_freq", float(getattr(self.cfg.TRAIN, "CHECKPOINT_FREQ", 0)), 0)

            # 记录训练备注
            if hasattr(self.cfg, 'NOTE') and self.cfg.NOTE:
                self._writer.add_text("hparams/note", str(self.cfg.NOTE), 0)
                print(f"[TensorBoard] Recorded training note: {self.cfg.NOTE}")

        # 记录开始时间
        self.time_start = time.time()

    def _expand_dataset_and_pkl(self):
        """
        如果 train_pkl 包含多个 restart [N, R, dim]，将其展平为 [N*R, dim]，
        并同步扩展 DataLoader 的 data_source 和 clean_pkl。
        """
        if hasattr(self, 'train_pkl') and self.train_pkl.dim() == 3:
            N, num_restarts, dim = self.train_pkl.shape
            self.train_pkl = self.train_pkl.view(-1, dim)
            print(f"[_expand] Flattened train_pkl to {self.train_pkl.shape}")
            
            loaders = [
                getattr(self, 'train_loader_x', None),
                getattr(self, 'train_loader_x_noshuffle', None),
                getattr(self, 'train_loader_x_notransform_noshuffle', None)
            ]
            for loader in loaders:
                if loader is not None:
                    new_data_source = []
                    for item in loader.dataset.data_source:
                        for _ in range(num_restarts):
                            new_data_source.append(item)
                    loader.dataset.data_source = new_data_source
            
            if loaders[1] is not None:
                print(f"[_expand] Expanded dataset to {len(loaders[1].dataset.data_source)}")

            if hasattr(self, 'clean_pkl') and self.clean_pkl is not None:
                if self.clean_pkl.shape[0] == N:
                    self.clean_pkl = self.clean_pkl.repeat_interleave(num_restarts, dim=0)
                    print(f"[_expand] Expanded clean_pkl to {self.clean_pkl.shape}")

    def before_adv_train(self, path, attack='PGD'):
        """
        对抗训练前的准备：生成或加载对抗样本特征。
        
        支持两种攻击模式：
          - 'PGD':          特征扰动攻击（代理模型 + KL 散度）
          - 'PGD_whitebox':  白盒分类攻击（直接最大化 CLIP 零样本交叉熵）
        """
        pkl_path = '{}/{}_{}_v2.pkl'.format(
            path, self.cfg.DATASET.NAME, 
            self.cfg.MODEL.BACKBONE.NAME.replace("/", "_")
        )
        
        if os.path.isfile(pkl_path):
            self.train_pkl = torch.load(pkl_path, weights_only=False).to('cpu')
            print(f'[before_adv_train] Loaded train_pkl from {pkl_path}')
            self._expand_dataset_and_pkl()
            return
        
        normalizer = ImageNormalizer(device=self.device)
        train_eps = self.cfg.DATASET.TRAIN_EPS
        num_iters = getattr(self.cfg.DATASET, 'Train_PGD_NUM_ITERS', 60)
        num_restarts = getattr(self.cfg.DATASET, 'Train_PGD_NUM_RESTARTS', 10)
        
        model = get_model(self.model)
        image_encoder = model.image_encoder
        image_encoder.to(self.device)
        image_encoder.eval()
        dtype = model.dtype
        
        embedding_dim = image_encoder.output_dim
        data_loader = self.train_loader_x_notransform_noshuffle
        self.train_pkl = torch.empty(size=[len(data_loader.dataset), num_restarts, embedding_dim])

        if attack == 'PGD':
            print("[before_adv_train] Mode: feature-distortion PGD (surrogate)")
            clip_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device='cpu')
            if self.cfg.TRAINER.ADV.PREC == "fp32" or self.cfg.TRAINER.ADV.PREC == "amp":
                raise NotImplementedError("fp32/amp not supported for surrogate model")
            surrogate = ClipModel(model=get_model(clip_model.visual), num_classes=2).eval().to(self.device)
            attacker = create_pgd_attacker(train_eps, normalizer, self.cfg, num_iters=num_iters, num_restarts=num_restarts)

            print(f"[before_adv_train] eps={train_eps}/255, iters={num_iters}, "
                  f"restarts={num_restarts}, batches={len(data_loader)}")
            for batch_idx, batch in enumerate(data_loader):
                inputs_pixel = batch['img'].to(self.device)
                images_adv_pixel_all = attacker.run(surrogate, inputs_pixel, scaler=1, feature_layer='fc', return_all=True)

                images_adv_normalized_flat = normalizer.normalize(images_adv_pixel_all)
                with torch.no_grad():
                    embedding_flat = image_encoder(images_adv_normalized_flat.type(dtype))

                bs = inputs_pixel.shape[0]
                embedding = embedding_flat.view(bs, num_restarts, -1)
                start_idx = batch_idx * data_loader.batch_size
                end_idx = start_idx + embedding.shape[0]
                self.train_pkl[start_idx:end_idx] = embedding.cpu().float()

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")

            del surrogate

        elif attack == 'PGD_whitebox':
            print("[before_adv_train] Mode: white-box classification PGD")
            temp_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device=self.device)
            temp_model.float().eval()

            classnames = [self.lab2cname[i] for i in range(len(self.lab2cname))]
            prompts_text = [f"a photo of a {c.replace('_', ' ')}." for c in classnames]
            tokens = clip.tokenize(prompts_text).to(self.device)
            with torch.no_grad():
                text_features = temp_model.encode_text(tokens).float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = temp_model.logit_scale.exp().detach()
            attack_encoder = temp_model.visual

            eps_val = train_eps / 255.0
            alpha = eps_val / num_iters * 2.5

            print(f"[before_adv_train] eps={eps_val:.6f} ({train_eps}/255), iters={num_iters}, "
                  f"restarts={num_restarts}, batches={len(data_loader)}")

            for batch_idx, batch in enumerate(data_loader):
                images = batch['img'].to(self.device).float()
                labels = batch['label'].to(self.device)
                bs = images.shape[0]

                restart_embeddings = []
                for _r in range(num_restarts):
                    delta = torch.zeros_like(images).uniform_(-eps_val, eps_val)
                    delta = torch.clamp(images + delta, 0, 1) - images

                    for _ in range(num_iters):
                        delta.requires_grad_(True)
                        adv_norm = normalizer.normalize(images + delta)
                        feats = attack_encoder(adv_norm).float()
                        feats_n = feats / feats.norm(dim=-1, keepdim=True)
                        logits = logit_scale * feats_n @ text_features.T
                        loss = F.cross_entropy(logits, labels)
                        loss.backward()
                        grad = delta.grad.detach().sign()
                        delta = (delta.detach() + alpha * grad).clamp(-eps_val, eps_val)
                        delta = torch.clamp(images + delta, 0, 1) - images

                    adv_final = (images + delta.detach()).clamp(0, 1)
                    adv_normalized = normalizer.normalize(adv_final)
                    with torch.no_grad():
                        emb = image_encoder(adv_normalized.type(dtype))
                    restart_embeddings.append(emb.cpu().float())

                embedding = torch.stack(restart_embeddings, dim=1)
                start_idx = batch_idx * data_loader.batch_size
                end_idx = start_idx + bs
                self.train_pkl[start_idx:end_idx] = embedding

                if (batch_idx + 1) % 5 == 0:
                    print(f"  [{batch_idx + 1}/{len(data_loader)}] done")

            del temp_model

        else:
            raise ValueError(f"Unknown attack type: {attack}")

        torch.save(self.train_pkl, pkl_path)
        print(f'[before_adv_train] Saved to {pkl_path}')
        torch.cuda.empty_cache()
        self._expand_dataset_and_pkl()

    def before_clean_train(self, path):
        """
        生成干净图像的嵌入向量（用于混合训练）
        
        流程：
        1. 使用不带归一化的 DataLoader 获取 [0,1] 像素空间图像
        2. 归一化后提取特征（无攻击）
        """
        pkl_path = '{}/{}_{}_clean.pkl'.format(
            path, self.cfg.DATASET.NAME, 
            self.cfg.MODEL.BACKBONE.NAME.replace("/", "_")
        )
        
        if os.path.isfile(pkl_path):
            self.clean_pkl = torch.load(pkl_path, weights_only=False).to('cpu')
            print(f'[before_clean_train] Loaded clean_pkl from {pkl_path}')
            
            data_loader = getattr(self, 'train_loader_x_notransform_noshuffle', None)
            if data_loader is not None and self.clean_pkl.shape[0] != len(data_loader.dataset):
                ratio = len(data_loader.dataset) // self.clean_pkl.shape[0]
                self.clean_pkl = self.clean_pkl.repeat_interleave(ratio, dim=0)
                print(f"[before_clean_train] Expanded clean_pkl {ratio}x to {self.clean_pkl.shape}")
            return
        
        print("[before_clean_train] Generating clean embeddings...")
        
        normalizer = ImageNormalizer(device=self.device)
        
        model = get_model(self.model)
        image_encoder = model.image_encoder
        image_encoder.to(self.device) # 确保 image_encoder 在 GPU 上
        dtype = model.dtype
        
        embedding_dim = image_encoder.output_dim
        print(f"[before_clean_train] Embedding dimension: {embedding_dim}")
        
        data_loader = self.train_loader_x_notransform_noshuffle
        self.clean_pkl = torch.empty(size=[len(data_loader.dataset), embedding_dim])
        
        image_encoder.eval()
        
        print(f"[before_clean_train] Processing {len(data_loader)} batches...")
        for batch_idx, batch in enumerate(data_loader):
            inputs_pixel = batch['img'].to(self.device)
            
            # 直接归一化（无攻击）
            images_normalized = normalizer.normalize(inputs_pixel)
            
            with torch.no_grad():
                embedding = image_encoder(images_normalized.type(dtype))

            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + embedding.shape[0]
            self.clean_pkl[start_idx:end_idx] = embedding.cpu().float()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")

        torch.save(self.clean_pkl, pkl_path)
        print(f'[before_clean_train] Generated and saved clean_pkl to {pkl_path}')

    def before_adv_val(self, path, attack='PGD'):
        """
        生成验证集的对抗嵌入（用于每个 epoch 的验证）
        
        支持 'PGD'（特征扰动）和 'PGD_whitebox'（白盒分类攻击）
        """
        pkl_path = '{}/{}_{}_val_v2.pkl'.format(
            path, self.cfg.DATASET.NAME, 
            self.cfg.MODEL.BACKBONE.NAME.replace("/", "_")
        )
        
        if os.path.isfile(pkl_path):
            self.val_pkl = torch.load(pkl_path, weights_only=False).to('cpu')
            print(f'[before_adv_val] Loaded val_pkl from {pkl_path}')
            return
        
        if self.val_loader_notransform is None:
            print("[before_adv_val] No validation set available, skipping...")
            self.val_pkl = None
            return
        
        normalizer = ImageNormalizer(device=self.device)
        val_eps = self.cfg.DATASET.TRAIN_EPS
        num_iters = getattr(self.cfg.DATASET, 'Train_PGD_NUM_ITERS', 60)
        
        model = get_model(self.model)
        image_encoder = model.image_encoder
        image_encoder.to(self.device)
        image_encoder.eval()
        dtype = model.dtype
        
        embedding_dim = image_encoder.output_dim
        data_loader = self.val_loader_notransform
        self.val_pkl = torch.empty(size=[len(data_loader.dataset), embedding_dim])

        if attack == 'PGD':
            print("[before_adv_val] Mode: feature-distortion PGD (surrogate)")
            clip_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device='cpu')
            if self.cfg.TRAINER.ADV.PREC == "fp32" or self.cfg.TRAINER.ADV.PREC == "amp":
                raise NotImplementedError("fp32/amp not supported for surrogate model")
            surrogate = ClipModel(model=get_model(clip_model.visual), num_classes=2).eval().to(self.device)
            attacker = create_pgd_attacker(val_eps, normalizer, self.cfg, num_iters=num_iters)

            print(f"[before_adv_val] eps={val_eps}/255, iters={num_iters}, batches={len(data_loader)}")
            for batch_idx, batch in enumerate(data_loader):
                inputs_pixel = batch['img'].to(self.device)
                images_adv_pixel = attacker.run(surrogate, inputs_pixel, scaler=1, feature_layer='fc')

                images_adv_normalized = normalizer.normalize(images_adv_pixel)
                with torch.no_grad():
                    embedding = image_encoder(images_adv_normalized.type(dtype))

                start_idx = batch_idx * data_loader.batch_size
                end_idx = start_idx + embedding.shape[0]
                self.val_pkl[start_idx:end_idx] = embedding.cpu().float()

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")

            del surrogate

        elif attack == 'PGD_whitebox':
            print("[before_adv_val] Mode: white-box classification PGD")
            temp_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device=self.device)
            temp_model.float().eval()

            classnames = [self.lab2cname[i] for i in range(len(self.lab2cname))]
            prompts_text = [f"a photo of a {c.replace('_', ' ')}." for c in classnames]
            tokens = clip.tokenize(prompts_text).to(self.device)
            with torch.no_grad():
                text_features = temp_model.encode_text(tokens).float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = temp_model.logit_scale.exp().detach()
            attack_encoder = temp_model.visual

            eps_val = val_eps / 255.0
            alpha = eps_val / num_iters * 2.5

            print(f"[before_adv_val] eps={eps_val:.6f} ({val_eps}/255), iters={num_iters}, "
                  f"batches={len(data_loader)}")

            for batch_idx, batch in enumerate(data_loader):
                images = batch['img'].to(self.device).float()
                labels = batch['label'].to(self.device)

                delta = torch.zeros_like(images).uniform_(-eps_val, eps_val)
                delta = torch.clamp(images + delta, 0, 1) - images

                for _ in range(num_iters):
                    delta.requires_grad_(True)
                    adv_norm = normalizer.normalize(images + delta)
                    feats = attack_encoder(adv_norm).float()
                    feats_n = feats / feats.norm(dim=-1, keepdim=True)
                    logits = logit_scale * feats_n @ text_features.T
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    grad = delta.grad.detach().sign()
                    delta = (delta.detach() + alpha * grad).clamp(-eps_val, eps_val)
                    delta = torch.clamp(images + delta, 0, 1) - images

                adv_final = (images + delta.detach()).clamp(0, 1)
                adv_normalized = normalizer.normalize(adv_final)
                with torch.no_grad():
                    embedding = image_encoder(adv_normalized.type(dtype))

                start_idx = batch_idx * data_loader.batch_size
                end_idx = start_idx + embedding.shape[0]
                self.val_pkl[start_idx:end_idx] = embedding.cpu().float()

                if (batch_idx + 1) % 5 == 0:
                    print(f"  [{batch_idx + 1}/{len(data_loader)}] done")

            del temp_model

        else:
            raise ValueError(f"Unknown attack type: {attack}")

        torch.save(self.val_pkl, pkl_path)
        print(f'[before_adv_val] Saved to {pkl_path}')
        torch.cuda.empty_cache()

    def before_adv_test(self, path, attack='PGD'):
        """
        对抗测试前的准备：生成测试集的对抗嵌入。
        
        支持两种攻击模式（通过 attack 参数选择）：
          - 'PGD':          特征扰动攻击（旧方式，用随机代理模型扰乱特征分布）
          - 'PGD_whitebox':  白盒分类攻击（标准评测，直接最大化零样本分类交叉熵）
        
        流程：
        1. 使用不带归一化的 DataLoader，直接获取 [0,1] 像素空间图像
        2. PGD 攻击生成对抗图像
        3. 归一化后提取 embedding 保存
        """
        # 根据攻击模式选择不同的 pkl 文件名
        suffix = '_whitebox' if attack == 'PGD_whitebox' else ''
        pkl_path = '{}/{}_{}_test_v2{}.pkl'.format(
            path, self.cfg.DATASET.NAME,
            self.cfg.MODEL.BACKBONE.NAME.replace("/", "_"),
            suffix
        )

        if os.path.isfile(pkl_path):
            self.test_pkl = torch.load(pkl_path, weights_only=False).to('cpu')
            print(f'[before_adv_test] Loaded test_pkl ({attack}) from {pkl_path}')
            return

        data_loader = self.test_loader_notransform
        normalizer = ImageNormalizer(device=self.device)
        test_eps = self.cfg.DATASET.TEST_EPS
        num_iters = getattr(self.cfg.DATASET, 'Test_PGD_NUM_ITERS', 100)

        model = get_model(self.model)
        image_encoder = model.image_encoder
        image_encoder.to(self.device)
        image_encoder.eval()
        dtype = model.dtype

        embedding_dim = image_encoder.output_dim
        self.test_pkl = torch.empty(size=[len(data_loader.dataset), embedding_dim])

        if attack == 'PGD':
            print("[before_adv_test] Mode: feature-distortion PGD (surrogate)")
            temp_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device='cpu')
            if self.cfg.TRAINER.ADV.PREC == "fp32" or self.cfg.TRAINER.ADV.PREC == "amp":
                temp_model.float()

            surrogate = ClipModel(model=get_model(temp_model.visual), num_classes=2).eval().to(self.device)
            attacker = create_pgd_attacker(test_eps, normalizer, self.cfg, num_iters=num_iters)

            print(f"[before_adv_test] eps={test_eps}/255, iters={num_iters}, batches={len(data_loader)}")
            for batch_idx, batch in enumerate(data_loader):
                inputs_pixel = batch['img'].to(self.device)
                images_adv_pixel = attacker.run(surrogate, inputs_pixel, scaler=1, feature_layer='fc')

                images_adv_normalized = normalizer.normalize(images_adv_pixel)
                with torch.no_grad():
                    embedding = image_encoder(images_adv_normalized.type(dtype))

                start_idx = batch_idx * data_loader.batch_size
                end_idx = start_idx + embedding.shape[0]
                self.test_pkl[start_idx:end_idx] = embedding.cpu().float()

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")

            torch.save(self.test_pkl, pkl_path)
            print(f'[before_adv_test] Saved to {pkl_path}')
            del surrogate

        elif attack == 'PGD_whitebox':
            print("[before_adv_test] Mode: white-box classification PGD")
            temp_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device=self.device)
            temp_model.float().eval()

            classnames = [self.lab2cname[i] for i in range(len(self.lab2cname))]
            prompts_text = [f"a photo of a {c.replace('_', ' ')}." for c in classnames]
            tokens = clip.tokenize(prompts_text).to(self.device)
            with torch.no_grad():
                text_features = temp_model.encode_text(tokens).float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = temp_model.logit_scale.exp().detach()
            attack_encoder = temp_model.visual

            eps_val = test_eps / 255.0
            alpha = eps_val / num_iters * 2.5

            print(f"[before_adv_test] eps={eps_val:.6f} ({test_eps}/255), "
                  f"iters={num_iters}, alpha={alpha:.6f}, batches={len(data_loader)}")

            for batch_idx, batch in enumerate(data_loader):
                images = batch['img'].to(self.device).float()
                labels = batch['label'].to(self.device)

                delta = torch.zeros_like(images).uniform_(-eps_val, eps_val)
                delta = torch.clamp(images + delta, 0, 1) - images

                for _ in range(num_iters):
                    delta.requires_grad_(True)
                    adv_norm = normalizer.normalize(images + delta)
                    feats = attack_encoder(adv_norm).float()
                    feats_n = feats / feats.norm(dim=-1, keepdim=True)
                    logits = logit_scale * feats_n @ text_features.T
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    grad = delta.grad.detach().sign()
                    delta = (delta.detach() + alpha * grad).clamp(-eps_val, eps_val)
                    delta = torch.clamp(images + delta, 0, 1) - images

                adv_final = (images + delta.detach()).clamp(0, 1)
                adv_normalized = normalizer.normalize(adv_final)
                with torch.no_grad():
                    embedding = image_encoder(adv_normalized.type(dtype))

                start_idx = batch_idx * data_loader.batch_size
                end_idx = start_idx + embedding.shape[0]
                self.test_pkl[start_idx:end_idx] = embedding.cpu().float()

                if (batch_idx + 1) % 5 == 0:
                    print(f"  [{batch_idx + 1}/{len(data_loader)}] done")

            torch.save(self.test_pkl, pkl_path)
            print(f'[before_adv_test] Saved to {pkl_path}')
            del temp_model

        else:
            raise ValueError(f"Unknown attack type: {attack}")

        torch.cuda.empty_cache()

    def before_black_test(self, path, attack='RAP'):
        """
        黑盒测试前的准备。
        
        使用统一的 ImageNormalizer 处理归一化操作。
        """
        pkl_path = '{}/{}_{}.pkl'.format(path, self.cfg.DATASET.NAME, attack)
        print("black test pkl_path:", pkl_path)
        
        # 初始化归一化器
        self.normalizer = ImageNormalizer(device='cpu')
        
        if os.path.isfile(pkl_path):
            self.test_pkl = torch.load(pkl_path, weights_only=False)
            # 归一化加载的图像（假设保存的是像素空间图像）
            self.test_pkl = self.normalizer.normalize(self.test_pkl)
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
        
        使用统一的 ImageNormalizer 处理归一化操作。
        """
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        # 确保 normalizer 已初始化
        if not hasattr(self, 'normalizer'):
            self.normalizer = ImageNormalizer(device=self.device)
        else:
            self.normalizer.to(self.device)

        test_eps = self.cfg.DATASET.TEST_EPS / 255.0
        array_to_pkl = self.test_pkl

        print(f"Evaluate on the *{split}* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            
            # 获取对应的对抗样本
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + input.shape[0]
            input_adv = array_to_pkl[start_idx:end_idx].to(input.device)

            # 使用 normalizer 统一处理扰动限制
            input_adv = self.normalizer.clamp_perturbation(input_adv, input, test_eps)

            # 模型推理
            output = self.model_inference(input_adv)
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
        使用完整模型梯度（包括 ODE 部分）生成 PGD 攻击。
        测试防御是否能在攻击者了解防御机制时仍然有效。
        
        使用统一的 ImageNormalizer 处理归一化操作。
        """
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set with Adaptive PGD Attack")
        
        # 使用统一的归一化器
        if not hasattr(self, 'normalizer'):
            self.normalizer = ImageNormalizer(device=self.device)
        else:
            self.normalizer.to(self.device)
        
        # 攻击参数
        test_eps = self.cfg.DATASET.TEST_EPS
        eps_val = test_eps / 255.0
        n_iters = 40
        alpha = 2.0 / 255.0

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            label = label.to(self.device)
            
            # 从干净图像开始
            images = input.clone().detach()
            
            # 初始化扰动
            delta = torch.zeros_like(images).uniform_(-0.01, 0.01)
            delta.requires_grad = True
            
            for _ in range(n_iters):
                adv_input = images + delta
                self.model.zero_grad()
                
                output = self.model(adv_input)
                loss = F.cross_entropy(output, label)
                
                grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
                delta.data = delta.data + alpha * grad.sign()
                
                # 使用 normalizer 进行扰动限制
                adv_normalized = images + delta
                adv_clamped = self.normalizer.clamp_perturbation(adv_normalized, images, eps_val)
                delta.data = adv_clamped - images

            # 最终对抗样本
            input_adv = images + delta.detach()

            # 推理
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

        # epoch 级聚合指标
        for name, meter in losses.meters.items():
            self.write_scalar("epoch_train/" + name, meter.avg, self.epoch + 1)
        self.write_scalar("epoch_train/lr", self.get_current_lr(), self.epoch + 1)
        self.write_scalar("epoch_train/batch_time_avg", batch_time.avg, self.epoch + 1)
        self.write_scalar("epoch_train/data_time_avg", data_time.avg, self.epoch + 1)

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

        # epoch 级聚合指标
        for name, meter in losses.meters.items():
            self.write_scalar("epoch_train/" + name, meter.avg, self.epoch + 1)
        self.write_scalar("epoch_train/lr", self.get_current_lr(), self.epoch + 1)
        self.write_scalar("epoch_train/batch_time_avg", batch_time.avg, self.epoch + 1)
        self.write_scalar("epoch_train/data_time_avg", data_time.avg, self.epoch + 1)

    def run_epoch_adv(self):
        """
        运行预计算对抗特征的训练epoch。
        这里使用`before_adv_train`中预先生成的对抗样本特征(`self.train_pkl`)。

        支持两种混合方式（通过 cfg.TRAINER.ADV.MIX_CLEAN_RATIO 控制）：
        1. 数据混合: 按概率选择干净或对抗嵌入作为输入
        2. 损失混合: 使用对抗嵌入训练，同时在loss中混合干净损失
        """
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x_noshuffle)

        # 检查是否使用数据混合
        mix_clean_ratio = getattr(self.cfg.TRAINER.ADV, 'MIX_CLEAN_RATIO', 0.0)
        use_data_mixing = mix_clean_ratio > 0 and hasattr(self, 'clean_pkl') and self.clean_pkl is not None
        # 损失混合：只要有 clean_pkl 就使用
        use_clean_guidance = hasattr(self, 'clean_pkl') and self.clean_pkl is not None

        seed = torch.random.seed()
        torch.random.manual_seed(seed)

        # 重新打乱数据索引，以保持DataLoader和预计算特征的对应关系
        length = randperm(len(self.train_loader_x_noshuffle.dataset.data_source)).tolist()
        self.train_loader_x_noshuffle.dataset.data_source = [self.train_loader_x_noshuffle.dataset.data_source[i] for i
                                                             in length]
        self.train_pkl = self.train_pkl[torch.LongTensor(length)]
        
        # 同步打乱干净嵌入
        if use_clean_guidance:
            self.clean_pkl = self.clean_pkl[torch.LongTensor(length)]

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x_noshuffle):

            data_time.update(time.time() - end)

            # 获取当前batch对应的预计算特征
            start_idx = self.batch_idx * self.train_loader_x_noshuffle.batch_size
            end_idx = (self.batch_idx + 1) * self.train_loader_x_noshuffle.batch_size
            
            images_adv = self.train_pkl[start_idx:end_idx] # 已经是 [bs, dim]
            
            # 由于数据集已经展开，所以 batch 中的 label 也是展开后的
            # 不需要像之前那样在内部通过 repeat_interleave 展开

            # 数据混合: 按mix_clean_ratio概率随机选择干净或对抗嵌入
            if use_data_mixing:
                images_clean = self.clean_pkl[start_idx:end_idx]
                batch_size = images_adv.shape[0]
                
                # 为每个样本随机决定使用干净还是对抗嵌入
                mix_mask = torch.rand(batch_size) < mix_clean_ratio
                mix_mask = mix_mask.unsqueeze(1).expand_as(images_adv)
                
                # 混合：clean * mask + adv * (1 - mask)
                images_mixed = torch.where(mix_mask, images_clean, images_adv)
                batch_dict = {
                    'batch': batch, 
                    'images_adv': images_mixed.to(self.device),
                    'images_clean': images_clean.to(self.device)  # 保留clean用于损失混合
                }
            else:
                # 无数据混合，使用纯对抗嵌入
                images_clean = self.clean_pkl[start_idx:end_idx] if use_clean_guidance else None
                batch_dict = {
                    'batch': batch,
                    'images_adv': images_adv.to(self.device),
                    'images_clean': images_clean.to(self.device) if images_clean is not None else None
                }

            # 进行训练步骤
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

        # epoch 级聚合指标
        for name, meter in losses.meters.items():
            self.write_scalar("epoch_train/" + name, meter.avg, self.epoch + 1)
        self.write_scalar("epoch_train/lr", self.get_current_lr(), self.epoch + 1)
        self.write_scalar("epoch_train/batch_time_avg", batch_time.avg, self.epoch + 1)
        self.write_scalar("epoch_train/data_time_avg", data_time.avg, self.epoch + 1)
        self.write_scalar("epoch_train/mix_clean_ratio", float(mix_clean_ratio), self.epoch + 1)
        self.write_scalar("epoch_train/use_data_mixing", 1.0 if use_data_mixing else 0.0, self.epoch + 1)

        # 保存 epoch 级训练准确率，供 after_epoch 使用（避免重新评估）
        if "acc" in losses.meters:
            self._epoch_train_acc = losses.meters["acc"].avg

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
