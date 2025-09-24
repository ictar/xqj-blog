---
title: "玩转 TerraMind：从理解到微调"
description: "TerraMind 是首个针对地球观测 (EO) 领域提出的大规模、任意到任意（any-to-any）生成式多模态基础模型，它通过结合令牌级别和像素级别的双尺度表示进行预训练，以学习高层上下文信息和精细的空间细节。该模型旨在促进多模态数据整合、提供强大的生成能力并支持零样本和少样本应用，同时在地球观测基准测试中超越现有模型，并通过引入“模态思维”（TiM）进一步提升性能。"
summary: "TerraMind 是首个针对地球观测 (EO) 领域提出的大规模、任意到任意（any-to-any）生成式多模态基础模型，它通过结合令牌级别和像素级别的双尺度表示进行预训练，以学习高层上下文信息和精细的空间细节。该模型旨在促进多模态数据整合、提供强大的生成能力并支持零样本和少样本应用，同时在地球观测基准测试中超越现有模型，并通过引入“模态思维”（TiM）进一步提升性能。"
date: 2025-09-10
toc: true
draft: false
tags: ["GeoFM", "terramind", "遥感", "AI", "大模型", "EO"]
---

{{< toc >}}


过去几年，地球观测（Earth Observation, EO）正在进入一个“模型为王”的时代。卫星影像的数据量越来越大，模态越来越多（光学、雷达、气候指数、地理文本描述……），但如何真正把这些信息用起来，却始终是个挑战。

这时，**TerraMind** 出现了。它是**第一个端到端生成式、多模态地球观测基础模型**，由 IBM Research、ETH Zurich、Forschungszentrum Jülich 和 ESA Φ-Lab 联合开发。它不仅能整合不同模态的数据，还具备生成能力和出色的泛化能力。换句话说，它试图扮演“大一统模型”的角色，为各种下游任务（如土地覆盖分类、灾害监测、气候研究等）提供一个强大的“起点”。


# 为什么要有 TerraMind？

TerraMind 的提出主要是为了**解决 EO 领域中多模态数据集成、生成以及现有模型泛化能力不足等问题**。更具体地说，它针对以下几个痛点：

* **多模态数据利用不足**
  传统模型往往只针对某些特定模态（比如只用光学影像），或只服务于某个任务（比如分割）。TerraMind 则通过整合雷达、光学、土地覆盖图、NDVI、DEM、坐标元数据甚至自然语言描述，打通了“模态壁垒”。

* **缺乏生成式多模态能力**
  这是 TerraMind 的杀手锏。它是首个“大规模、任意到任意（any-to-any）”的生成式多模态 EO 模型。它不仅能完成传统分析任务，还能生成新的数据，支持零样本、少样本学习，降低标注成本。

* **泛化与数据效率挑战**
  过去的 EO 模型很容易“过拟合”到某个区域或某个任务，难以迁移。TerraMind 通过在海量无标注数据上的自监督预训练，加上小规模的有标注微调，实现了更强的空间、时间泛化。

* **现有地理空间基础模型的局限**
  很多 GFM（Geo-Foundation Model）其实是把计算机视觉的套路硬套过来，没考虑遥感数据的特殊性。TerraMind 在架构上专门做了优化，例如 **双尺度预训练**（token-level + pixel-level 表示），既能抓全局语义，又保留局部空间细节。

* **加速气候与可持续性应用**
  它不仅是个模型，还配套了分布式数据处理和采样框架，能直接连接卫星数据源，让开发者在气候和可持续性场景中更快落地应用。

简而言之，TerraMind 的目标是：**做一个通用、强大、能整合多模态的基础模型，为 EO 研究和应用提供更高效的“起点”。**

# TerraMind 从何而来？
![](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base/resolve/main/assets%2Fterramind_architecture.png)

## 输入：TerraMesh 数据集

TerraMind 在定制的全球大规模地理空间数据集 **TerraMesh** 上进行预训练，包含 **900 万个样本和九种地理空间模态**：
  * 光学卫星图像：Copernicus Sentinel-2 L1C 和 L2A (RGB)。
  * 雷达卫星图像：Copernicus Sentinel-1 GRD 和 RTC。
  * 任务特定模态：土地利用/土地覆盖 (LULC) 和归一化植被指数 (NDVI) 图。
  * 元数据：数字高程模型 (DEM) 和地理坐标（离散化为字符串）。
  * 自然语言：通过 LLaVA-Next 从 Sentinel-2 RGB 图像合成的图像描述 (captions)。
  
## 双尺度预训练（Dual-Scale Pretraining）

TerraMind 在像素级别和 token 级别上结合数据，是其核心创新。
  * **Token 级别**：编码高层上下文信息，学习跨模态关系，并实现扩展性。
  * **像素级别**：利用细粒度表示，捕捉关键的空间细微之处。

这种双尺度早期融合方法优于其他融合方法，能够实现人工数据生成、零样本和少样本应用。

## 两阶段预训练
1. **单模态分词器（Tokenizer）预训练**：为每种模态开发特定的分词器，将数据编码为离散 token 序列，或从 token 序列解码回原始形式。图像类模态（S-1, S-2, LULC, NDVI, DEM）使用**基于自编码器和有限标量量化 (FSQ) 的架构**。序列类模态（描述、地理位置）使用基于 WordPiece 的文本分词器。
2. **TerraMind 编码器-解码器预训练**：使用对称 Transformer 架构处理多模态 token 序列，并接受像素级别的跨模态输入。预训练目标是跨模态补丁分类问题，通过交叉熵损失来重建被遮蔽的目标 token。


# TerraMind 能做什么？

它的强项可以归结为三点：

1. **整合（Integration）**：把雷达、光学、DEM、LULC 等多模态放到一个统一表示里。
2. **生成（Generation）**：任意模态之间的生成任务（例如从雷达合成光学影像，或从 DEM 预测 NDVI）。
   * 从 Sentinel-2 L2A 光学数据开始，TerraMind 能高质量地生成雷达数据、土地利用图和数字高程图。
   * 即使从低信息量的地理位置信息开始，模型也能生成与上下文相关的光学图像（例如，从中东的地理位置生成沙漠图像），尽管结构上可能与真实值不同。
3. **泛化（Generalization）**：下游任务只需少量标注，就能快速适配，比如土地覆盖分类、灾害监测等。
   * **零样本学习**
        * **水体测绘**：TerraMindv1-B 在零样本设置下 IoU 达到 45.4%，若使用 DynamicWorld LULC 数据进行 ablation 实验，可提升至 69.8%，接近微调 SOTA 性能（84.4%）。
        * 地理定位：能够准确预测特定数据实例的地理位置，例如预测“裸地”类别的概率分布，并识别出撒哈拉、中东等区域。
   * **少样本学习**：在 EuroSAT 和 METER-ML 数据集上的 1-shot 5-way 分类任务中，TerraMind 的平均准确度至少比其他基准高出 10pp，表明其潜在空间结构良好。

此外，TerraMind 还引入了 **“多模态思维” (Thinking in Modalities, TiM)** 的创新概念，类似于大型语言模型中的“思维链 (chain-of-thought)”。通过在微调和推理过程中注入生成的人工数据，模型输出性能得以提升。例如，在水体测绘任务中，通过生成额外的 LULC 数据，TiM 微调比标准微调的 mIoU 提升了 2pp。

# 编码与重建（Tokenizer）

首先要理解 TerraMind 的 **Tokenizer**。它的工作原理类似“压缩—解压”：

* 输入影像 → 转换成 token 表示
* token 捕捉空间细节和上下文信息
* 再通过解码器重建影像 → 检验模型是否“理解”了输入

这部分主要是 **验证模型的信息保留能力**。

下面是一段实际的 Python 脚本（基于 Sentinel-2 L2A 数据），参考[IBM/terramind/notebooks/terramind_tokenizer_reconstruction.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_tokenizer_reconstruction.ipynb)：

```python
# 构建 tokenizer 模型
model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_s2l2a', pretrained=True)

# 加载示例影像并构造成张量
# Load an example Sentinel-2 image
examples = [
    '../examples/S2L2A/38D_378R_2_3.tif',
    '../examples/S2L2A/282D_485L_3_3.tif',
    '../examples/S2L2A/433D_629L_3_1.tif',
]
data = rxr.open_rasterio(examples[1])
# Convert to [B, C, 224, 224]
data = torch.Tensor(data.values, device='cpu').unsqueeze(0)

# 归一化（标准化）
mean = torch.Tensor(v1_pretraining_mean['untok_sen2l2a@224'])
std = torch.Tensor(v1_pretraining_std['untok_sen2l2a@224'])
input = (data - mean[None, :, None, None]) / std[None, :, None, None]

# 运行模型（编码→解码）
with torch.no_grad():
    reconstruction = model(input, timesteps=10)

# 反标准化（恢复到原始数量级）
reconstruction = reconstruction.cpu()
reconstruction = (reconstruction * std[None, :, None, None]) + mean[None, :, None, None]
# 解码出来的 reconstruction 要乘以 std 加上 mean 把数值带回“真实”量纲（与原始数据同一取值范围），否则看起来会像“均值为 0 的小数”。
```

在可视化时，可以对比输入影像与重建影像的 RGB 表达：

```python
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Input RGB
rgb_input = data[0, [3,2,1]].permute(1,2,0).cpu().numpy()
ax[0].imshow((rgb_input/2000).clip(0,1))
ax[0].set_title("Input"); ax[0].axis("off")

# Reconstructed RGB
rgb_recon = reconstruction[0, [3,2,1]].permute(1,2,0).cpu().numpy()
ax[1].imshow((rgb_recon/2000).clip(0,1))
ax[1].set_title("Reconstruction"); ax[1].axis("off")

plt.show()
```

效果：左边是原始 Sentinel-2 输入，右边是 Tokenizer 解码后的重建结果。

👉 这证明 **TerraMind 能用 token 表示影像，并且很好地保留信息**。


# 跨模态生成（Generation）
令人兴奋的是 TerraMind 的 **生成能力**。
它不仅能“重建”输入，还能实现 **跨模态生成 (Any-to-Any)**：

* 光学影像 → 生成雷达影像
* DEM → 生成光学或 NDVI
* 影像 → 生成 LULC 地图
* 甚至还能做 图像 ↔ 文本

这种 **任意模态到任意模态的生成能力**，是 TerraMind 相比传统模型的最大突破。
它意味着：即便某些数据缺失，也能通过生成补齐；即便标注有限，也能通过生成增强训练。

## 单 patch 示例

* 输入：单个 patch，可以是S2-L2A的数据 (224×224)
* 输出：多个目标模态，如 S1GRD、DEM、LULC 等。
* 特点：
    * 每个输出模态对应自己的 tokenizer（增加了内存消耗）。
    * 使用 diffusion steps（如 10 步）生成结果 → 保证输出的多样性与质量。
    * 可以直接对输入做标准化（standardize=True）。
* 步骤：
  1. 构建模型
    * 从 `FULL_MODEL_REGISTRY` 中实例化 TerraMind 模型。
    * 指定 **输入模态**（例如 `S2L2A`）和 **输出模态**（例如 `S1GRD, DEM, LULC`）。
    * 选择是否加载预训练权重（`pretrained=True`）。
    * 配置标准化（`standardize=True`，会自动应用预训练时的均值/方差）。
  2. 准备数据
    * 加载raster数据
    * 使用 `rioxarray.open_rasterio` 读取为数组。
    * 转换成模型要求的输入形状 [Batch, Channels, Height, Width]，例如 [1, C, 224, 224]。
  3. 加载输入数据：将输入数据绘制成 RGB 图像（帮助理解输入内容）。
  4. 执行扩散生成
    * 把输入送入 GPU/CPU（`input.to(device)`）。
    * 在 `torch.no_grad()` 下运行 扩散生成（`model(input, timesteps=10)`）
    * `timesteps` 控制扩散过程的迭代次数（步数越多，生成结果越精细但耗时更长）。
  5. 得到多模态输出
    * 模型会返回一个字典（`{模态名: 生成结果}`）
    * 每个输出模态有对应的 **生成张量**。
  6. 可视化结果。
    * 遍历输出模态，调用 `plot_modality()` 绘制每个结果。
    * 与输入图像并排显示，方便比较。

下面python片段实现了用 S-2 输入去生成其他模态（S-1、DEM、LULC……），参考 [IBM/terramind/notebooks/terramind_generation.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_generation.ipynb)：

```py
# 载入示例 S-2 L2A 并构成模型输入
data = rxr.open_rasterio(examples[example_id])          # rioxarray DataArray
data = torch.Tensor(data.values, device='cpu').unsqueeze(0)  # -> [B, C, H, W]

# 构建模型
model = FULL_MODEL_REGISTRY.build(
    'terramind_v1_base_generate',
    modalities=['S2L2A'],
    output_modalities=['S1GRD', 'DEM', 'LULC'],
    pretrained=True,
    standardize=True,
)
# 移到设备
_ = model.to(device)

# 运行生成（diffusion steps）
with torch.no_grad():
  generated = model(input, verbose=True, timesteps=10)
```

本质上，这是 跨模态翻译：**给你一个 Sentinel-2 的 patch，我可以“想象”出 SAR、DEM、NDVI 等其他模态。**

## 分块推理（Tiled Inference）

**为什么要用 tiled_inference？**

大瓦片（full tile）通常超出 GPU 内存，故把大图切成许多小 tile（patch）分别在 GPU 上推理，然后合并回整图。tiled_inference 是一个工具函数（TerraTorch/用户库提供），负责切 patch、按 batch 调用模型、并把 patch 输出合并成整幅输出。

总而言之：
* 输入：整块 tile（比如 Singapore, Santiago，可能是 1000×1000 或更大）。
* 问题：GPU 内存不够放下整张图。
* 步骤：
  1. 载入并（可选）裁剪大瓦片
  2. 转换为 [B,C,H,W] 
  3. 构建生成模型（指定输出模态、timesteps）
  4. 用 tiled_inference 切 patch 并批量调用 model_forward 
  5. 合并回整图得到 [C_total,H,W]
  6. 切分各模态通道
  7. 对连续模态反标准化、对分类模态 argmax
  8. 保存 GeoTIFF / 可视化 / 指标评估。

下面是一段实际的 Python 脚本，参考 [IBM/terramind/notebooks/large_tile_generation.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/large_tile_generation.ipynb)

```py
# 输入准备（crop / batch dim）
data = rxr.open_rasterio('examples/S2L2A/Santiago.tif').values
data = data[:, 500:1500]           # 可选裁剪以加速
input = torch.tensor(data, dtype=torch.float, device=device).unsqueeze(0)

# 构建并准备生成模型：以 S2L2A 为条件输入，生成 S1GRD 和 LULC
output_modalities = ['S1GRD', 'LULC']
model = FULL_MODEL_REGISTRY.build(
    'terramind_v1_base_generate',
    modalities=['S2L2A'],
    output_modalities=output_modalities,
    pretrained=True,
    standardize=True,
    timesteps=10,
)
model = model.to(device)

# 从 dict → tensor
def model_forward(x):
    generated = model(x)                       # dict: {modality: tensor}
    out = torch.concat([generated[m] for m in output_modalities], dim=1)
    return out

# tiled_inference 的输出与去批次维
pred = tiled_inference(model_forward, input, crop=256, stride=192, batch_size=16, verbose=True)
pred = pred.squeeze(0)   # 从 [1, C, H, W] -> [C, H, W]

# 把拼接的通道切回各模态
num_channels = {'S2L2A':12, 'S1GRD':2, 'S1RTC':2, 'DEM':1, 'LULC':10, 'NDVI':1}
num_channels = {m: num_channels[m] for m in output_modalities}
start_idx = np.cumsum([0] + list(num_channels.values()))
generated = {m: pred[i:i+c].cpu() for m, i, c in zip(output_modalities, start_idx, num_channels.values())}

# LULC 的后处理（从概率 -> 离散类别）
if 'LULC' in generated.keys():
    generated['LULC'] = generated['LULC'].argmax(dim=0)

```

故而，让 TerraMind 不只是小 patch demo，而是能够被应用到真实地球大范围场景。

# 微调任务（Fine-tuning with TerraTorch）

TerraMind 的另一大亮点，就是能很方便地在下游任务（如语义分割）上微调。我们以[IBM/terramind/notebooks/terramind_v1_base_sen1floods11.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_v1_base_sen1floods11.ipynb)为例，讲讲微调的主要流程：

1. **数据准备（DataModule）**
   使用 `GenericMultiModalDataModule` 来定义模态、路径、split 文件、标准化参数。例如：

   * 模态：`["S2L1C", "S1GRD"]`
   * 标签：`*_LabelHand.tif`
   * 训练/验证/测试划分：`flood_train_data.txt` 等

   这样做的好处是，不管你用的是光学+雷达，还是加 DEM、LULC，都能快速配置。

2. **加载预训练 Backbone**

   ```python
   model = BACKBONE_REGISTRY.build(
       "terramind_v1_base",
       modalities=["S2L1C", "S1GRD"],
       pretrained=True,
   )
   ```

3. **定义下游任务（SemanticSegmentationTask）**

   * Backbone：TerraMind
   * Neck：特征提取和 reshape
   * Decoder：UNetDecoder
   * Loss：Dice 或 CE

   同时支持冻结/解冻 backbone，调节学习率（1e-5 \~ 1e-4）。

4. **训练与测试**
   通过 PyTorch Lightning 的 `Trainer.fit()` 启动训练，保存最佳 checkpoint。
   然后在 `Trainer.test()` 上评估，最后就能进行预测和可视化。

整个微调脚本跑下来，就可以从“预训练基础模型”得到一个“适配任务的模型”，哪怕只训练了几个 epoch，就能看到效果。


# 总结

* **TerraMind 是什么？** 一个面向地球观测的多模态基础模型。
* **它能干什么？** 多模态整合、生成任务、下游泛化。
* **怎么玩？**

  * 先体验生成能力：输入影像 → tokens → 重建影像。
  * 再做下游微调：配置 DataModule，加载预训练模型，定义任务，训练/测试。

它不仅仅是一个模型，更像是一个 **面向 EO 的通用 AI 平台**。未来不论是气候研究、灾害响应，还是土地覆盖监测，都能借助它快速落地。


# 了解更多
- [Github: IBM/terramind](https://github.com/IBM/terramind/tree/main)
- [Hugging Face: ibm-esa-geospatial/TerraMind-1.0-base](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
- [Paper (arxiv): TerraMind: Large-Scale Generative Multimodality for Earth Observation](https://doi.org/10.48550/arXiv.2504.11171)