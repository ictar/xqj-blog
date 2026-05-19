---
title: "深度学习名词速查表"
date: 2026-05-19
summary: "深度学习核心概念、神经网络架构、优化算法及损失函数的快速查询手册。支持实时搜索与过滤。"
type: "cheatsheets"
---

## 基础层与架构 | Basic Layers & Architectures

### 感知机 | Perceptron
- **定义**: 神经网络的最基本结构单元，由输入、权重、偏置、激活函数和输出组成。它能解决线性可分问题，但无法解决异或（XOR）问题。
- **大白话**: 神经网络世界的“单个决策开关”。它根据输入信息的加权重要性，来决定是“拉闸”还是“合闸”。
- **怎么实现**: 将输入向量和权重向量做内积（点乘），加上一个偏置量（阈值调整器），最后塞进一个二分类激活函数（如阶跃函数或 Sigmoid）。
- **公式**: $$y = \sigma\left(\sum_{i=1}^n w_i x_i + b\right)$$
- **代码**:
  ```python
  import torch
  import torch.nn as nn

  # 单个神经元（感知机）
  perceptron = nn.Linear(in_features=10, out_features=1)
  x = torch.randn(1, 10)
  output = torch.sigmoid(perceptron(x))
  ```

### 多层感知机 | MLP
- **定义**: 由输入层、一个或多个隐藏层以及输出层组成的前馈神经网络。层与层之间是全连接的（Fully Connected）。
- **大白话**: 多个决策开关层叠在一起的“多层投票系统”。通过多层传递，它能识破单一开关看不懂的复杂逻辑（如异或逻辑）。
- **怎么实现**: 信号从输入层出发，经过多次「矩阵乘法 + 非线性变化（激活函数）」的嵌套叠加，最终输出到最后一层。
- **代码**:
  ```python
  import torch.nn as nn

  class MLP(nn.Module):
      def __init__(self):
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(784, 128),
              nn.ReLU(),
              nn.Linear(128, 10)
          )
      def forward(self, x):
          return self.net(x)
  ```

### 卷积神经网络 | CNN
- **定义**: 专门用于处理网格结构数据（如图像）的神经网络。其核心操作是局部连接和权重共享。
- **大白话**: 用一个“移动放大镜”在照片上逐行逐列扫描的识别系统，专门用来捕捉局部图案（比如边缘、眼睛、车轮）。
- **怎么实现**: 拿着一个权重矩阵（卷积核）在输入图像上一步步滑动，每次把镜框内的像素与核进行乘累加，得到缩减但特征突出的新图。
- **代码**:
  ```python
  import torch.nn as nn

  # 简单的卷积块
  conv_block = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
  )
  ```

### 卷积层 | Convolutional Layer
- **定义**: 卷积神经网络（CNN）的核心层。通过滑动卷积核对输入特征图进行局部运算以提取局部空间特征，具有局部连接和权重共享的特点。
- **大白话**: “特征捕手”。用一组滤镜（卷积核）去提取图案，比如一号滤镜找横线，二号滤镜找竖线。
- **怎么实现**: 输入矩阵与多个可学习的小矩阵（如 3x3 窗口）逐元素相乘并求和，滑过整个输入以生成多通道的特征响应图。
- **代码**:
  ```python
  import torch.nn as nn

  # 二维卷积层示例
  conv = nn.Conv2d(
      in_channels=3, 
      out_channels=64, 
      kernel_size=3, 
      stride=1, 
      padding=1
  )
  ```

### 池化层 | Pooling Layer
- **定义**: 用于在保留主要特征的同时降低特征图空间分辨率（下采样）的层，以减少计算量并增强模型对特征微小位移和形变的鲁棒性。
- **大白话**: “信息压缩器”或“脱水过滤器”。只保留最关键的那部分（比如局部最显眼的花纹），把次要的细节丢掉。
- **怎么实现**: 在特征图上以固定步长滑动窗口（如 2x2），只取窗口内的最大值（Max Pooling）或取均值（Average Pooling）。
- **代码**:
  ```python
  import torch.nn as nn

  # 最大池化层 (空间高宽减半)
  max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
  # 全局平均池化 (将高宽 HxW 压缩为 1x1 向量)
  gap = nn.AdaptiveAvgPool2d((1, 1))
  ```

### 全连接层 | Fully Connected Layer
- **定义**: 每个输入节点都与每个输出节点相连接的层，又称线性层（Linear Layer），常用于神经网络的末端进行特征高度汇总和最终分类/回归预测。
- **大白话**: “终审判决官”。它把前面提取的所有局部特征都汇总过来，根据各自权重全盘考虑，给出最终的分数或结论。
- **怎么实现**: 执行矩阵乘法 $y = x W^T + b$。将多维特征展平为一维向量，与权重矩阵直接点乘，生成目标类别的得分。
- **公式**: $$y = x W^T + b$$
- **代码**:
  ```python
  import torch.nn as nn

  # 将 512 维的高阶特征映射到 10 分类的 logits
  fc = nn.Linear(in_features=512, out_features=10)
  ```

### 残差连接 | Skip Connection
- **定义**: 在深度神经网络中，将某一层的输入直接跨越一层或多层，加到后面某层的输出上的连接方式。它是 ResNet（残差网络）的核心设计。
- **大白话**: “抄近道”或“直达通道”。在深山老林般的网络里修了一条高速公路，让后面的层直接听到前方的声音，也让梯度在回去时可以直接坐高铁。
- **怎么实现**: 将某层的输入 $x$ 跨过若干中间计算层，直接与最后的输出 $F(x)$ 进行相加（即 Element-wise Sum）：$y = F(x) + x$。
- **公式**: $$\mathcal{H}(x) = \mathcal{F}(x) + x$$
- **代码**:
  ```python
  import torch.nn as nn

  class ResidualBlock(nn.Module):
      def __init__(self, channels):
          super().__init__()
          self.conv = nn.Sequential(
              nn.Conv2d(channels, channels, 3, padding=1),
              nn.BatchNorm2d(channels),
              nn.ReLU(),
              nn.Conv2d(channels, channels, 3, padding=1),
              nn.BatchNorm2d(channels)
          )
          self.relu = nn.ReLU()
      def forward(self, x):
          # 残差连接：F(x) + x
          return self.relu(self.conv(x) + x)
  ```

### 骨干网络 | Backbone
- **定义**: 负责提取输入特征的基础神经网络部分，如目标检测、图像分割模型中的特征提取器。常见的 Backbone 包括 ResNet、VGG、EfficientNet 等。
- **特点**: Backbone 通常在海量数据集（如 ImageNet）上进行预训练，其提取的高阶特征会被传递给下游任务的特征融合层（Neck）与输出头（Head）。
- **大白话**: “地基与主骨架”。就像看图写作时的“眼睛”，负责从图片中提取出基础视觉信息，是支撑整个下游任务的核心基础。
- **怎么实现**: 通常直接复用经过 ImageNet 图像分类训练成熟的经典网络结构（如 ResNet），砍掉最尾端的分类头，只导出深层的多尺度特征响应图。
- **代码**:
  ```python
  import torchvision.models as models
  import torch.nn as nn

  # 使用预训练的 ResNet50 作为特征提取的 Backbone
  resnet = models.resnet50(pretrained=True)
  # 移除最后的分类全连接层，保留特征提取部分
  backbone = nn.Sequential(*list(resnet.children())[:-2])
  # 输入 shape: (batch_size, 3, 224, 224)
  # 输出特征 shape: (batch_size, 2048, 7, 7)
  ```

### 颈部与头部 | Neck & Head
- **定义**: 现代复杂深度学习模型（如目标检测 YOLO、图像分割 U-Net 等）中用于处理特征和产生输出的专门化结构。
  - **Neck (颈部)**: 位于 Backbone 和 Head 之间，负责收集、融合并增强来自 Backbone 的不同尺度特征（例如 FPN 特征金字塔、PANet）。
  - **Head (头部)**: 负责接收融合特征并输出最终的特定预测任务结果（例如分类类别、物体边界框、语义分割图）。
- **大白话**: 
  - **Neck (脖子)**: 特征的“搅拌机”。把高层的宏观抽象特征和底层的细粒度局部细节特征掺在一起搅拌均匀。
  - **Head (脑袋)**: 特征的“执行官”。接收搅拌好的特征并做出具体决策（比如框出物体位置或标出像素的类别）。
- **怎么实现**: Neck 通过特征金字塔（FPN）做双向跨尺度上采样和级联相加；Head 则通常是一组小卷积或全连接层，将特征通道映射为具体的任务输出图。
- **代码**:
  ```python
  import torch.nn as nn

  class SegModelWithNeckHead(nn.Module):
      def __init__(self):
          super().__init__()
          # 1. Backbone: 特征提取
          self.backbone = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU())
          # 2. Neck: 特征多尺度融合/增强
          self.neck = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
          # 3. Head: 映射到分割掩膜预测
          self.head = nn.Conv2d(64, 1, 1) # 输出 1 个通道的 logits
      def forward(self, x):
          feat = self.backbone(x)
          fused = self.neck(feat)
          return self.head(fused)
  ```

### 注意力机制 | Attention Mechanism
- **定义**: 允许模型在处理序列或图像 data 时，动态地给不同位置的特征分配不同权重的机制。自注意力（Self-Attention）是 Transformer 架构的基石。
- **大白话**: “重点关注仪”。就像看书时视线会自动落在加粗的关键字上一样，它能主动从一堆字或像素里找出谁跟谁最有关联，并针对性地加权关注。
- **怎么实现**: 计算输入特征的 Query（提问）、Key（索引）和 Value（内容）。通过 Q 与 K 的矩阵乘法计算相似度权重（注意力分数），再用该权重对 V 进行加权求和。
- **公式**: $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **代码**:
  ```python
  import torch
  import torch.nn as nn

  # PyTorch 内置的多头自注意力层
  self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
  x = torch.randn(4, 10, 512)
  attn_out, _ = self_attn(x, x, x)
  ```

### 归一化层 | Normalization Layers
- **定义**: 调整网络中间层特征输入的分布均值与方差以稳定训练。Batch Normalization (BN) 沿 Batch 维度计算归一化，而 Layer Normalization (LN) 沿 Feature 维度计算归一化（常用于 Transformer）。
- **大白话**: “强制班级标准化”。防止因为某些输入数据值太大或太小，导致网络情绪失控（梯度爆炸或消失）。强制将大家的表现拉回到标准的均值和方差内。
- **怎么实现**: 计算指定范围内的均值 $\mu$ 和方差 $\sigma^2$，将输入数据转化为均值为 0、方差为 1 的分布，并应用可学习的缩放参数 $\gamma$ 和平移参数 $\beta$。
- **代码**:
  ```python
  import torch.nn as nn

  # Batch Normalization (常用于卷积层)
  batch_norm = nn.BatchNorm2d(num_features=64)
  # Layer Normalization (常用于 Transformer 等序列模型)
  layer_norm = nn.LayerNorm(normalized_shape=512)
  ```

### U-Net | U-Net Architecture
- **定义**: 一种经典的对称式全卷积图像分割网络。由收缩路径（Encoder，特征提取）、扩张路径（Decoder，上采样）以及连接两者的跳跃连接（Skip Connection）组成，形状呈 "U" 字型。
- **大白话**: “镜像折叠照相馆”。前半部分（Encoder）负责给图像做级联压缩以看懂大局，后半部分（Decoder）负责把它还原放大；中间用一根根直连的“跨界传送桥”（Skip Connection），把前半部分没被压缩污染的清晰图像细节，直接复制拼给后半部分，以实现极其精准的像素级抠图。
- **怎么实现**: Encoder 用卷积+池化使图片变小、通道数翻倍；Decoder 用上采样使图片变大、通道数减半，并在每一次放大时，用 `torch.cat` 在通道维度拼上 Encoder 对应层的高清特征图，以此保留精准的空间坐标信息。
- **代码**:
  ```python
  import torch
  import torch.nn as nn

  class UNetMini(nn.Module):
      def __init__(self):
          super().__init__()
          # 下采样部分 (Encoder)
          self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU())
          self.pool = nn.MaxPool2d(2, 2)
          # 最底层 (Bottleneck)
          self.bottle = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
          # 上采样部分 (Decoder)
          self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
          self.dec1 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU())
          self.final = nn.Conv2d(64, 1, 1) # 输出单通道分割图

      def forward(self, x):
          # 1. Encoder 提取并保存细节特征
          x1 = self.enc1(x)
          # 2. 下采样到 Bottleneck
          x_pool = self.pool(x1)
          b = self.bottle(x_pool)
          # 3. Decoder 上采样并进行拼接 (Skip Connection)
          up_b = self.up(b)
          cat_x = torch.cat([up_b, x1], dim=1) # 拼接通道数
          return self.final(self.dec1(cat_x))
  ```

### 嵌入向量 | Embedding
- **定义**: 将高维离散特征（如单词 ID、类别标号、连续编号等）映射到低维连续稠密向量空间的技术。
- **大白话**: “语义翻译计算器”或“数字化肖像画”。它把人说的话或类别代号转成一串表示特征含义的数值坐标。在这个坐标空间里，相似的事物（如“猫”和“狗”）会被自动聚到距离很近的地方。
- **怎么实现**: 使用一个参数矩阵（Lookup Table），每一行对应一个离散标记的稠密向量。在反向传播中这些向量的数值会被自动微调，使上下文语义越接近的词，其向量积（余弦相似度）越大。
- **代码**:
  ```python
  import torch
  import torch.nn as nn

  # 词表大小为 1000，每个单词映射为 128 维的嵌入向量
  embedding_layer = nn.Embedding(num_embeddings=1000, embedding_dim=128)
  # 假定输入 2 个句子，每个句子 4 个单词对应的 token ID
  input_tokens = torch.tensor([[12, 45, 9, 201], [3, 88, 999, 12]])
  # 输出 shape: (2, 4, 128)
  embedded = embedding_layer(input_tokens)
  ```

### 隐空间 | Latent Space
- **定义**: 编码器在对原始高维数据进行特征降维和信息提取后，所生成的低维流形特征表示空间。它是隐藏在复杂数据背后的本质表示空间。
- **大白话**: “灵魂画手脑海中的灵感空间”。在这个空间里，省去了图像上万个像素的冗余叙述，只剩下了控制本质特征的几个“核心旋钮”（例如：发色、性别、高度、是否戴眼镜）。拨动这些旋钮，就能随心所欲拼装出全新的图片。
- **怎么实现**: 用多层卷积或全连接层将原始的高维数据压缩成一个低维特征向量（如 512 维的 Bottleneck 隐向量），后续的解码器（Decoder）将只根据这个低维特征向量来重构复原原始数据。
- **代码**:
  ```python
  import torch
  import torch.nn as nn

  class Autoencoder(nn.Module):
      def __init__(self):
          super().__init__()
          # 编码器：将 784 维的图像压缩至 32 维的 Latent Space
          self.encoder = nn.Sequential(
              nn.Linear(784, 128),
              nn.ReLU(),
              nn.Linear(128, 32) # 32 维的隐藏层
          )
          # 解码器：从 Latent Space 还原原始图像
          self.decoder = nn.Sequential(
              nn.Linear(32, 128),
              nn.ReLU(),
              nn.Linear(128, 784),
              nn.Sigmoid()
          )
      def forward(self, x):
          latent_vec = self.encoder(x.view(-1, 784)) # 获得隐变量
          reconstructed = self.decoder(latent_vec)
          return reconstructed, latent_vec
  ```

## 激活函数 | Activation Functions

### ReLU | Rectified Linear Unit
- **定义**: 线性整流单元，是深度学习中最常用的激活函数。在正半轴上是线性的，负半轴上输出为0。
- **大白话**: “单向阀门”。如果输入值是负数，一律拒之门外（输出 0）；如果是正数，则原封不动放行。
- **怎么实现**: 执行简单的求最大值操作：`max(0, x)`。在代码中，负数输入直接被截断清零。
- **公式**: $$f(x) = \max(0, x)$$
- **代码**:
  ```python
  import torch
  import torch.nn as nn

  # 激活函数调用
  relu = nn.ReLU()
  x = torch.tensor([-2.0, 3.0])
  print(relu(x)) # Tensor: [0.0, 3.0]
  ```

### Sigmoid | Logistic Function
- **定义**: 将输入值压缩到 $(0, 1)$ 区间的激活函数，常用于二分类模型的输出层。
- **大白话**: “概率转换器”。把从负无穷到正无穷的杂乱得分，全部折算成 0% 到 100% 之间的概率概率值。
- **怎么实现**: 用数学公式 $1 / (1 + e^{-x})$ 处理每个输入。当输入很大时输出逼近 1，输入很小时输出逼近 0。
- **公式**: $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- **代码**:
  ```python
  import torch
  import torch.nn as nn

  sigmoid = nn.Sigmoid()
  x = torch.tensor([0.0, 2.0])
  print(sigmoid(x)) # Tensor: [0.5, 0.8808]
  ```

### Softmax | Normalized Exponential
- **定义**: 将一个向量转化为概率分布的函数，所有元素在 $(0, 1)$ 之间且和为 1。常用于多分类任务的输出层。
- **大白话**: “比例切蛋糕”。把一组乱糟糟的得分（比如 3、1、0.1），按指数权重切成一份份加起来刚好等于 100% 的蛋糕片，代表每个选项的获胜概率。
- **怎么实现**: 对向量中的每个元素求 $e$ 的次幂，然后除以所有元素求幂后的总和，从而实现归一化。
- **公式**: $$S_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}$$
- **代码**:
  ```python
  import torch
  import torch.nn as nn

  softmax = nn.Softmax(dim=1)
  logits = torch.tensor([[1.0, 2.0, 3.0]])
  probs = softmax(logits)
  print(probs) # 各类别预测概率
  ```

## 优化算法 | Optimizers

### SGD | Stochastic Gradient Descent
- **定义**: 随机梯度下降。在每个训练步骤中只使用一个样本或一个小批量（Mini-batch）样本来计算梯度并更新参数。
- **大白话**: “摸黑下山法”。不花时间观察整座山的全貌，而是走一步算一步，沿着脚下最陡的一条小道往下挪，虽然会有些磕磕绊绊，但胜在速度快。
- **怎么实现**: 在每个迭代步，计算当前 mini-batch 样本下的损失函数对参数的梯度，并用参数减去学习率乘以梯度来更新参数。
- **公式**: $$	heta_{t+1} = 	heta_t - \eta \cdot 
abla_{	heta} L(	heta_t)$$
- **代码**:
  ```python
  import torch.optim as optim

  # 在 PyTorch 中使用带动量的 SGD
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  ```

### Adam | Adaptive Moment Estimation
- **定义**: 自适应矩估计。结合了 Momentum（动量，考虑一阶矩）和 RMSprop（自适应学习率，考虑二阶矩）优点的优化器。
- **大白话**: “带导航和避震的越野车”。它不仅记得之前的方向（有惯性），还会根据路况自动调整踩油门的力度（平缓的维度多走，走得陡峭颠簸的维度小步探路）。
- **怎么实现**: 分别计算梯度的指数移动平均（一阶矩，即动量）和梯度的平方指数移动平均（二阶矩，即自适应步长），然后动态缩放每个参数的更新步长。
- **代码**:
  ```python
  import torch.optim as optim

  # Adam 优化器
  optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
  ```

## 损失函数 | Loss Functions

### MSE | Mean Squared Error
- **定义**: 均方误差损失，又称 $L_2$ 损失。计算预测值与真实值差值的平方均值，主要用于回归任务。
- **大白话**: “偏差平分罚分”。如果射击离靶心越远，惩罚得分就呈平方级暴涨（比如偏离 2 米罚 4 分，偏离 3 米罚 9 分），所以它极力避免出现巨大偏差。
- **怎么实现**: 用每个样本的（预测值 - 真实值）求平方，然后对所有样本求平均值。
- **公式**: $$L_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
- **代码**:
  ```python
  import torch.nn as nn

  loss_fn = nn.MSELoss()
  y_pred = torch.tensor([1.5, 2.0])
  y_true = torch.tensor([1.0, 2.0])
  loss = loss_fn(y_pred, y_true)
  ```

### 交叉熵损失 | Cross Entropy Loss
- **定义**: 衡量两个概率分布之间差异的指标，在分类任务中用来作为损失函数。在 PyTorch 中，`CrossEntropyLoss` 已经内置了 Softmax 操作。
- **大白话**: “预测自信度考量”。如果模型对正确答案非常笃定（预测概率接近 100%），罚分为 0；如果押错了宝或者犹豫不决，罚分就会急剧上升。
- **怎么实现**: 计算正确类别的预测概率 $\hat{y}$ 的负自然对数 $-\log(\hat{y})$。若预测对的概率低，则损失值极大。
- **公式**: $$L_{CE} = -\sum_{i=1}^C y_i \log(\hat{y}_i)$$
- **代码**:
  ```python
  import torch.nn as nn

  loss_fn = nn.CrossEntropyLoss()
  # 假定 batch_size=2, 类别数=3
  logits = torch.randn(2, 3) # 模型输出的 logits
  targets = torch.tensor([1, 2]) # 真实的类标签
  loss = loss_fn(logits, targets)
  ```
