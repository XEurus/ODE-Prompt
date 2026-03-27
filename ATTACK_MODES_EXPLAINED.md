# 攻击方式说明：旧版特征扰动 PGD 与新版白盒分类 PGD

本文档说明本仓库中两种 `PGD` 测试方式的区别，包括：

- 数学目标有什么不同
- 代码实现具体差在哪里
- 为什么旧方式会得到偏高的“对抗准确率”
- 什么时候应该使用哪一种方式

---

## 1. 结论先说

现在仓库里有两种白盒测试入口：

- `PGD`
  - 旧方式
  - 本质上是 **feature-distortion attack / surrogate attack**
  - 不是直接攻击最终分类结果
  - 更像“让图像特征偏移”
  - 通常攻击更弱

- `PGD_whitebox`
  - 新方式
  - 本质上是 **true white-box classification PGD**
  - 直接攻击 CLIP 的零样本分类 logits
  - 与常见论文中的白盒 PGD 评测更一致
  - 攻击更强，结果更可信

因此，如果目标是和论文里的 “PGD100, eps=1/255” 做可比评测，应优先使用：

```bash
ATTACK_MODE=PGD_whitebox
```

---

## 2. 旧攻击 `PGD`：它到底在做什么

### 2.1 代码入口

旧攻击在以下位置实现：

- `dass/engine/trainer.py` 中 `before_adv_test(..., attack='PGD')`
- `utils/adv_utils.py` 中 `ClipModel`
- `attack/attackFeature.py` 中 `PGD.run`

### 2.2 核心思路

旧攻击并不是直接拿“真实类别标签”去攻击模型分类结果，而是：

1. 取一个 CLIP 的视觉编码器
2. 外面套一个 **随机初始化的 2 类线性层**
3. 用这个随机代理模型的输出分布作为攻击目标
4. 最大化“干净图像输出分布”和“对抗图像输出分布”之间的 KL 散度

也就是说，旧攻击关注的是：

> “让特征变化足够大”

而不是：

> “让最终分类错掉”

### 2.3 对应代码

代理模型是这里定义的：

```143:177:utils/adv_utils.py
class ClipModel(nn.Module):
    """
    CLIP 视觉编码器包装器
    """
    
    def __init__(self, model, num_classes=2):
        super(ClipModel, self).__init__()
        self.visual_encoder = model
        output_dim = self.visual_encoder.output_dim
        self.fc = nn.Linear(output_dim, num_classes)
    
    def forward(self, image):
        x = self.visual_encoder(image)
        x = self.fc(x)
        return x
```

可以看到关键问题：`fc` 是一个新建的随机线性层，并不代表真实分类头。

实际攻击 loss 在这里：

```20:88:attack/attackFeature.py
def run(self, net, image, target=None, scaler=1, feature_layer='fc', return_all=False, *args):
    hook = SingleModelHook(net, feature_layer, use_inp=True)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    ...
    with torch.no_grad():
        if target is None:
            net(self.preprocess(image))
            clean_embeddings = hook.get_hooked_value().detach()
    ...
    for i in range(self.num_iters):
        image_adv = next(iteration)
        net(image_adv)

        loss = criterion(
            hook.get_hooked_value().log_softmax(dim=-1),
            clean_embeddings.softmax(dim=-1)
        )
        loss = loss * scaler
        loss.backward()
```

这里优化的是：

\[
\max_{\delta} \mathrm{KL}(p_{\text{adv}} \,\|\, p_{\text{clean}})
\]

其中：

- \( p_{\text{clean}} \) 是干净图像经过“随机 2 类头”后的 softmax
- \( p_{\text{adv}} \) 是对抗图像经过同一个“随机 2 类头”后的 softmax

### 2.4 数学上它为什么弱

因为它并没有直接优化真实任务目标。

CLIP 的真实分类逻辑是：

\[
\text{logits}(x) = s \cdot \hat{f}(x)\hat{T}^{\top}
\]

其中：

- \( \hat{f}(x) \) 是归一化后的图像特征
- \( \hat{T} \) 是所有类别文本提示的归一化特征
- \( s \) 是 `logit_scale`

但旧攻击优化的却是：

\[
\mathrm{KL}(\text{softmax}(W f(x+\delta)), \text{softmax}(W f(x)))
\]

其中 \( W \) 只是一个随机矩阵。

这意味着：

- 它没有使用真实类别标签 \( y \)
- 它没有使用真实文本特征 \( T \)
- 它没有直接推动 “正确类别 logit 下降、错误类别 logit 上升”

所以即便这个 loss 很大，也不一定真正破坏最终分类。

---

## 3. 新攻击 `PGD_whitebox`：现在做的是什么

### 3.1 代码入口

新攻击主要在：

- `dass/engine/trainer.py` 中 `before_adv_test(..., attack='PGD_whitebox')`

单独验证 CLIP 基线的脚本在：

- `test_whitebox_clip.py`

### 3.2 核心思路

新攻击直接按 CLIP 的真实零样本分类流程来做：

1. 为每个类别构造文本 prompt，例如 `a photo of a cat.`
2. 用 CLIP 文本编码器得到所有类别的文本特征
3. 用 CLIP 视觉编码器得到图像特征
4. 计算图像-文本相似度 logits
5. 直接最大化交叉熵损失，使模型分类错误

这是标准的白盒分类攻击。

### 3.3 对应代码

新版白盒分支在这里：

```882:935:dass/engine/trainer.py
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
```

独立验证版也用了相同思路：

```61:80:test_whitebox_clip.py
def whitebox_pgd_attack(images, labels):
    images = images.clone().detach()
    delta = torch.zeros_like(images).uniform_(-eps, eps).to(device)
    delta = torch.clamp(images + delta, 0, 1) - images

    for _ in range(num_iters):
        delta.requires_grad_(True)
        adv_images = images + delta
        adv_norm = normalizer.normalize(adv_images)
        image_features = clip_model.encode_image(adv_norm).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.T
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        grad = delta.grad.detach().sign()
        delta = (delta.detach() + alpha * grad)
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(images + delta, 0, 1) - images

    return (images + delta.detach()).clamp(0, 1)
```

### 3.4 数学形式

设：

- 输入图像为 \( x \)
- 扰动为 \( \delta \)
- 真实标签为 \( y \)
- 图像特征为 \( f(x) \)
- 文本特征矩阵为 \( T \)
- 温度缩放为 \( s \)

CLIP 分类 logits 为：

\[
z(x) = s \cdot \frac{f(x)}{\|f(x)\|} \cdot \left(\frac{T}{\|T\|}\right)^\top
\]

白盒 PGD 优化的是：

\[
\max_{\|\delta\|_{\infty} \le \epsilon} \mathcal{L}_{CE}(z(x+\delta), y)
\]

也就是直接让正确类别的交叉熵变大。

PGD 迭代是：

\[
\delta_{t+1} =
\Pi_{\|\delta\|_{\infty}\le \epsilon}
\left(
\delta_t + \alpha \cdot \mathrm{sign}\left(\nabla_{\delta}\mathcal{L}_{CE}(z(x+\delta_t), y)\right)
\right)
\]

再加上像素范围约束：

\[
x+\delta \in [0,1]
\]

---

## 4. 两种攻击的核心区别

### 4.1 优化目标不同

旧攻击：

\[
\max \mathrm{KL}(\text{代理模型输出分布变化})
\]

新攻击：

\[
\max \mathcal{L}_{CE}(\text{真实分类错误})
\]

一句话：

- 旧攻击：让“特征变”
- 新攻击：让“分类错”

### 4.2 是否使用真实标签

旧攻击：

- 通常不需要真实标签
- 只比较 clean / adv 的特征分布差异

新攻击：

- 明确使用真实标签 `labels`
- 直接对白盒分类边界施压

### 4.3 是否使用真实分类头

旧攻击：

- 不使用真实文本分类器
- 用 `ClipModel(..., num_classes=2)` 随机生成的 `fc`

新攻击：

- 直接使用 CLIP 的文本特征构成分类器
- 与 zero-shot 推理时完全一致

### 4.4 可解释性

旧攻击的结果只能说明：

> 特征空间受到了某种扰动

但不能强有力地说明：

> 最终分类在标准白盒威胁模型下是否真的鲁棒

新攻击则可以直接回答：

> 在标准白盒 PGD 下，模型最终分类是否还能保持正确

---

## 5. 为什么之前测出来的准确率很高

根本原因是：

旧 `PGD` 并不是直接攻击最终分类目标，所以它生成的“对抗样本”可能只是在代理空间中变了，但没有足够强地跨过真实分类边界。

因此你会看到：

- 干净准确率正常
- “对抗准确率”却异常偏高

这不是因为模型真的有那么强，而是因为攻击目标偏弱。

而在 `test_whitebox_clip.py` 中，用标准白盒攻击后，原版 CLIP 在 `PGD100, eps=1/255` 下的准确率掉到接近 0%，这就和论文现象一致了。

---

## 6. 代码级差异总结表

| 维度 | 旧 `PGD` | 新 `PGD_whitebox` |
|---|---|---|
| 攻击对象 | 代理模型 `ClipModel(visual + random fc)` | 原始 CLIP 分类流程 |
| 标签使用 | 不依赖真实标签 | 使用真实标签 |
| 文本特征 | 不使用 | 使用所有类的文本特征 |
| 优化目标 | KL 散度 | 交叉熵 |
| 数学目标 | 扰乱特征分布 | 直接让分类错 |
| 攻击强度 | 较弱 | 较强 |
| 与论文可比性 | 较弱 | 强 |
| 适合用途 | 特征敏感性分析 | 标准鲁棒性评测 |

---

## 7. 现在应该怎么用

评测论文风格白盒 PGD 时，建议在脚本中设置：

```bash
ATTACK_MODE=PGD_whitebox
TEST_EPS=1
PGD_ITERS=100
```

对应脚本位置：

```23:26:scripts/eval_pgd100_eps1.sh
# ==================== 可调参数 ====================
ATTACK_MODE=PGD_whitebox   # PGD | PGD_whitebox
TEST_EPS=1                 # 扰动强度 (x/255)
PGD_ITERS=100              # PGD 迭代次数
```

如果只是想保留旧流程做对比，也可以用：

```bash
ATTACK_MODE=PGD
```

这样两种攻击方式就可以在同一套评测流程里切换。

---

## 8. 推荐理解方式

可以把两种攻击理解成：

- `PGD`
  - “我让图像在某个代理特征空间里偏移得很厉害”

- `PGD_whitebox`
  - “我直接推动图像跨过真实分类边界”

前者更像“间接扰动”，后者才是“直接打分类器”。

---

## 9. 一句话总结

旧 `PGD` 评测的是“特征是否稳定”，新 `PGD_whitebox` 评测的是“分类是否真的鲁棒”。

如果目标是和标准鲁棒性论文对齐，应该使用 `PGD_whitebox`。
