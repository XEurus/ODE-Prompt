"""
ImageNet 提示模板集合

来源: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb

本模块提供了用于 CLIP 零样本分类的多样化提示模板。
这些模板经过精心设计，能够覆盖各种图像风格和拍摄条件。

模板设计原则：
    1. 多样性：包含不同的拍摄角度、光照条件、艺术风格
    2. 覆盖性：涵盖真实照片、绘画、素描、雕塑等多种媒介
    3. 描述性：包含大小、清晰度、质量等属性描述

使用方式：
    - IMAGENET_TEMPLATES: 完整的 80 个模板集合，用于最大化集成效果
    - IMAGENET_TEMPLATES_SELECT: 精选的 7 个模板，平衡性能和计算效率

模板集成的优势：
    1. 减少对单一模板措辞的敏感性
    2. 提高跨域泛化能力
    3. 增强分类的鲁棒性
"""

# 完整的 ImageNet 提示模板集合（80 个模板）
IMAGENET_TEMPLATES = [
    # 照片质量相关
    "a bad photo of a {}.",              # 质量差的照片
    "a photo of many {}.",               # 多个对象
    "a low resolution photo of the {}.", # 低分辨率
    "a photo of the hard to see {}.",    # 难以辨认
    "a bright photo of a {}.",           # 明亮的照片
    "a photo of a clean {}.",            # 干净的对象
    "a photo of a dirty {}.",            # 脏的对象
    "a dark photo of the {}.",           # 暗色照片
    "a cropped photo of the {}.",        # 裁剪的照片
    "a cropped photo of a {}.",          # 裁剪的照片变体
    "a close-up photo of a {}.",         # 特写照片
    "a close-up photo of the {}.",       # 特写照片变体
    "a blurry photo of the {}.",         # 模糊的照片
    "a blurry photo of a {}.",           # 模糊的照片变体
    "a jpeg corrupted photo of a {}.",   # JPEG 压缩失真
    "a jpeg corrupted photo of the {}.", # JPEG 压缩失真变体
    "a good photo of the {}.",           # 高质量照片
    "a good photo of a {}.",             # 高质量照片变体
    "a photo of the {}.",                # 标准照片
    "a photo of a {}.",                  # 标准照片变体
    "a photo of one {}.",                # 单个对象
    "a photo of my {}.",                 # 个人物品
    "a photo of a large {}.",            # 大型对象
    "a photo of the large {}.",          # 大型对象变体
    "a photo of a small {}.",            # 小型对象
    "a photo of the small {}.",          # 小型对象变体
    "a photo of a nice {}.",             # 好看的对象
    "a photo of the nice {}.",           # 好看的对象变体
    "a photo of a weird {}.",            # 奇怪的对象
    "a photo of the weird {}.",          # 奇怪的对象变体
    "a photo of the cool {}.",           # 酷的对象
    "a photo of a cool {}.",             # 酷的对象变体
    "a photo of the clean {}.",          # 干净的对象变体
    "a photo of the dirty {}.",          # 脏的对象变体
    
    # 艺术形式相关
    "a sculpture of a {}.",              # 雕塑
    "a sculpture of the {}.",            # 雕塑变体
    "a rendering of a {}.",              # 渲染图
    "a rendering of the {}.",            # 渲染图变体
    "a rendition of the {}.",            # 表现形式
    "a rendition of a {}.",              # 表现形式变体
    "graffiti of a {}.",                 # 涂鸦
    "graffiti of the {}.",               # 涂鸦变体
    "a tattoo of a {}.",                 # 纹身
    "a tattoo of the {}.",               # 纹身变体
    "a drawing of a {}.",                # 绘画
    "a drawing of the {}.",              # 绘画变体
    "a painting of the {}.",             # 油画
    "a painting of a {}.",               # 油画变体
    "a doodle of a {}.",                 # 涂鸦草图
    "a doodle of the {}.",               # 涂鸦草图变体
    "a sketch of a {}.",                 # 素描
    "a sketch of the {}.",               # 素描变体
    
    # 材质/风格相关
    "the embroidered {}.",               # 刺绣
    "a embroidered {}.",                 # 刺绣变体
    "the plastic {}.",                   # 塑料材质
    "a plastic {}.",                     # 塑料材质变体
    "the origami {}.",                   # 折纸
    "a origami {}.",                     # 折纸变体
    "the plushie {}.",                   # 毛绒玩具
    "a plushie {}.",                     # 毛绒玩具变体
    "the toy {}.",                       # 玩具
    "a toy {}.",                         # 玩具变体
    "a cartoon {}.",                     # 卡通
    "the cartoon {}.",                   # 卡通变体
    "a pixelated photo of the {}.",      # 像素化照片
    "a pixelated photo of a {}.",        # 像素化照片变体
    "a black and white photo of the {}.",# 黑白照片
    "a black and white photo of a {}.",  # 黑白照片变体
    
    # 媒体/场景相关
    "a {} in a video game.",             # 电子游戏中
    "the {} in a video game.",           # 电子游戏中变体
    "art of a {}.",                      # 艺术作品
    "art of the {}.",                    # 艺术作品变体
    
    # 网络风格
    "itap of the {}.",                   # "I took a picture of" 网络用语
    "itap of a {}.",                     # "I took a picture of" 变体
    "itap of my {}.",                    # "I took a picture of" 个人变体
]

# 精选的高效模板子集（7 个模板）
# 这些模板在保持多样性的同时减少计算开销
IMAGENET_TEMPLATES_SELECT = [
    "itap of a {}.",                     # 网络风格
    "a bad photo of the {}.",            # 低质量
    "a origami {}.",                     # 折纸艺术
    "a photo of the large {}.",          # 大型对象
    "a {} in a video game.",             # 电子游戏
    "art of the {}.",                    # 艺术作品
    "a photo of the small {}.",          # 小型对象
]
