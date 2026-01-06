"""
零样本 CLIP 分类器 (Zero-Shot CLIP Classifier)

本模块实现了基于 CLIP 的零样本分类基线，主要用于：
1. 作为 ODE-Prompt 的性能对比基准
2. 验证 CLIP 预训练模型的零样本迁移能力
3. 提供模板集成 (Prompt Ensembling) 的实现

主要类：
    - ZeroshotCLIP: 基本零样本分类器，使用单一提示模板
    - ZeroshotCLIP2: 增强版，使用多模板集成提高鲁棒性

零样本分类原理：
    1. 使用预定义模板（如 "a photo of a {}"）生成各类别的文本描述
    2. 通过 CLIP 文本编码器将描述编码为文本特征
    3. 将输入图像编码为图像特征
    4. 计算图像特征与各类别文本特征的余弦相似度
    5. 选择相似度最高的类别作为预测结果
"""

import torch
import torch.nn as nn

from dass.engine import TRAINER_REGISTRY, TrainerX
from dass.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .advpt import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

# 各数据集的自定义提示模板
# 这些模板经过精心设计，以匹配各数据集的图像分布
CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",           # 宠物数据集
    "OxfordFlowers": "a photo of a {}, a type of flower.",     # 花卉数据集
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",    # 飞机数据集
    "DescribableTextures": "{} texture.",                       # 纹理数据集
    "EuroSAT": "a centered satellite photo of {}.",            # 卫星图像数据集
    "StanfordCars": "a photo of a {}.",                        # 汽车数据集
    "Food101": "a photo of {}, a type of food.",               # 食物数据集
    "SUN397": "a photo of a {}.",                              # 场景数据集
    "Caltech101": "a photo of a {}.",                          # 通用物体数据集
    "UCF101": "a photo of a person doing {}.",                 # 动作识别数据集
    "ImageNet": "a photo of a {}.",                            # ImageNet 数据集
    "ImageNetSketch": "a photo of a {}.",                      # ImageNet 素描版
    "ImageNetV2": "a photo of a {}.",                          # ImageNet V2
    "ImageNetA": "a photo of a {}.",                           # ImageNet 对抗版
    "ImageNetR": "a photo of a {}.",                           # ImageNet 渲染版
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    """
    基本零样本 CLIP 分类器
    
    使用单一数据集特定模板进行分类。
    
    属性：
        text_features: 预计算的类别文本特征，形状 (n_cls, feature_dim)
        clip_model: CLIP 模型实例
    
    示例：
        >>> trainer = ZeroshotCLIP(cfg)
        >>> trainer.build_model()
        >>> logits = trainer.model_inference(images)
    """
    
    def build_model(self):
        """
        构建零样本 CLIP 模型
        
        流程：
            1. 加载预训练 CLIP 模型
            2. 使用数据集特定模板生成类别描述
            3. 预计算并缓存类别文本特征
        
        注意：
            - 文本特征在构建时预计算，推理时直接使用
            - 这是零样本学习的关键：无需训练即可分类
        """
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)
        clip_model = clip_model.to(torch.float32)

        # 生成类别描述文本
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        
        # 文本标记化
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        # 预计算文本特征（无梯度）
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            # L2 归一化，用于余弦相似度计算
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        """
        零样本推理
        
        参数：
            image: 输入图像张量，形状 (batch_size, 3, H, W)
        
        返回：
            logits: 分类分数，形状 (batch_size, n_cls)
        
        计算：
            logits = τ * (image_features @ text_features.T)
            其中 τ 是 CLIP 的温度参数 (logit_scale)
        """
        # 编码图像
        image_features = self.clip_model.encode_image(image)
        # L2 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 获取温度参数
        logit_scale = self.clip_model.logit_scale.exp()
        
        # 计算相似度分数
        logits = logit_scale * image_features @ self.text_features.t()
        
        return logits


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """
    模板集成零样本 CLIP 分类器
    
    使用多个提示模板的平均特征进行分类，提高鲁棒性。
    
    模板集成原理：
        1. 对每个类别，使用多个不同的提示模板
        2. 分别编码这些模板并取平均
        3. 使用平均特征进行分类
    
    优势：
        - 减少对单一模板的敏感性
        - 提高分类的鲁棒性和准确率
        - 类似于模型集成的效果
    """

    # 使用精选的 ImageNet 模板子集
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        """
        构建模板集成零样本 CLIP 模型
        
        流程：
            1. 加载预训练 CLIP 模型
            2. 冻结所有参数（零样本，无需训练）
            3. 对每个模板分别编码类别文本
            4. 计算所有模板的平均特征
            5. 归一化平均特征用于分类
        """
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        # 冻结所有参数（零样本学习不需要训练）
        for params in clip_model.parameters():
            params.requires_grad_(False)

        # 添加数据集特定模板到模板列表
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        # 计算模板集成特征
        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            # 生成当前模板的类别描述
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            
            # 编码文本并归一化
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 累加特征
            mean_text_features = mean_text_features + text_features
        
        # 计算平均并归一化
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
