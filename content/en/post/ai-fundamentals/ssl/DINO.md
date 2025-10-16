---
title: "DINO: Self-Distillation with No Labels"
date: 2025-10-08
summary: "A student network learns from a teacher network using self-distillation, producing emergent semantic attention maps."
series: ["Self-Supervised Learning"]
tags: ["SSL", "Vision", "Representation Learning", "Joint Embedding", "Distillation Methods"]
---

<div class="model-card">

## 🏷️ Model Name
**DINO – Self-Distillation with No Labels**

## 🧠 Core Idea
> A student network learns from a teacher network using self-distillation, producing emergent semantic attention maps.

![DINO architecture](https://github.com/facebookresearch/dino/blob/main/.github/dino.gif?raw=true)

## 🖼️ Architecture

DINO PyTorch pseudocode w/o multi-crop:
```py
# gs, gt: student and teacher networks
# C: center (K)
# tps, tpt: student and teacher temperatures
# l, m: network and center momentum rates
gt.params = gs.params
for x in loader: # load a minibatch x with n samples
    x1, x2 = augment(x), augment(x) # random views
    s1, s2 = gs(x1), gs(x2) # student output n-by-K
    t1, t2 = gt(x1), gt(x2) # teacher output n-by-K
    loss = H(t1, s2)/2 + H(t2, s1)/2
    loss.backward() # back-propagate
    # student, teacher and center updates
    update(gs) # SGD
    gt.params = l*gt.params + (1-l)*gs.params
    C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)

def H(t, s):
    t = t.detach() # stop gradient
    s = softmax(s / tps, dim=1)
    t = softmax((t - C) / tpt, dim=1) # center + sharpen
    return - (t * log(s)).sum(dim=1).mean()
```

DINO employs two networks with the same underlying architecture ($g$) but with different parameter sets: <text style="color:green">a student network</text> ($g_{\theta_s}$) and <text style="color:green">a teacher network</text> ($g_{\theta_t}$). Each network $g$ consists of a backbone ($f$) (e.g., a Vision Transformer (ViT) or ResNet) and a projection head ($h$) such that $g = h \circ f$.
  * **Projection Head**: The projection head typically comprises <text style="color:purple">a 3-layer Multi-Layer Perceptron (MLP)</text>, followed by $\ell_2$ normalization and a final weight-normalized fully connected layer outputting $K$ dimensions.

Unlike some related methods (e.g., [BYOL](BYOL.md)), DINO operates effectively without needing an explicit predictor head in the student branch, resulting in the **exact same architecture for both the student and teacher**.

### 1️⃣ Data Augmentation and Input Views
DINO utilizes a <text style="color:purple">multi-crop training strategy</text> to generate multiple views of a single input image $x$. From one input image, a set of augmented views $V$ is generated.
* **Global Views**: The set $V$ includes at least two high-resolution global views ($x_{g1}, x_{g2}$), which typically cover a large area (e.g., greater than 50%) of the original image at a resolution like $224 \times 224$ pixels.
* **Local Views**: $V$ also includes several low-resolution local views (e.g., $96 \times 96$ pixels) that cover smaller areas (e.g., less than 50%).

View Distribution:
* The **student network** ($g_{\theta_s}$) processes **all** views (global and local).
* The **teacher network** ($g_{\theta_t}$) processes **only the global** views ($x_{g1}, x_{g2}$). This encourages "local-to-global" correspondence learning.

### 2️⃣ Output Processing and Collapse Avoidance
The networks output $K$-dimensional features, which are then converted into **probability distributions** using a <text style="color:purple">softmax function with a temperature parameter</text>.
* Student Output ($P_s$): The student output is normalized using a softmax function with a temperature $\tau_s$ (e.g., $\tau_s = 0.1$).
* Teacher Output ($P_t$): The teacher output undergoes two critical operations designed to prevent representation collapse (where the model outputs the same vector regardless of input):
    * <text style="color:purple">Centering</text>: A bias term $c$ is subtracted from the teacher's output. This center $c$ is continuously updated via an [Exponential Moving Average (EMA)](https://en.wikipedia.org/wiki/Exponential_smoothing) calculated over the batch outputs. Centering helps prevent one dimension from dominating the output.
    * <text style="color:purple">Sharpening</text>: A low temperature $\tau_t$ (e.g., linearly warmed up from $0.04$ to $0.07$ during training) is used in the teacher's softmax normalization. Sharpening encourages the opposite effect of centering, and the balance of the two avoids collapse.

### 3️⃣ Loss and Optimization
The objective is to **minimize the cross-entropy loss between the teacher's sharp distribution and the student's distribution, across different views**.
* Loss Formulation: The training minimizes the <text style="color:purple">cross-entropy loss</text> $H(P_t(x), P_s(x'))$ between the teacher distribution $P_t(x)$ of a global view $x$, and the student distribution $P_s(x')$ of any other view $x'$ (global or local) in the set $V$.
* Gradient Flow Control: A <text style="color:purple">stop-gradient (sg) operator</text> is applied to the teacher output, ensuring that the loss gradient only flows back to the student parameters $\theta_s$.
* Student Parameter Update ($\theta_s$): The student parameters $\theta_s$ are updated via <text style="color:purple">standard stochastic gradient descent (SGD), such as the AdamW optimizer</text>.
* Teacher Parameter Update ($\theta_t$): The teacher parameters $\theta_t$ are updated using an <text style="color:purple">Exponential Moving Average (EMA)</text> of the student parameters, following the rule $$\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$$ This update mimics a momentum encoder and ensures the teacher evolves smoothly, guiding the student training.
  * The momentum rate $\lambda$ often follows a cosine schedule, increasing from a base rate (e.g., 0.996) towards 1 during training.

## 🎯 Downstream Tasks
- Classification and Representation Quality Evaluation
- Transfer Learning (Fine-tuning)
- Similarity, Retrieval, and Dense Prediction Tasks

## 💡 Strengths

- **Simplicity and Architecture**: DINO presents a simple self-supervised approach that streamlines training by using a standard cross-entropy loss to predict the teacher network's output. It effectively works without needing a predictor head in the student branch, leading to the exact same architecture for both student and teacher networks.
- **Effective Collapse Avoidance**: DINO avoids model collapse using just centering and sharpening of the momentum teacher outputs. The centering operation only depends on first-order batch statistics.
- **Superior ViT Performance**: DINO demonstrates strong synergy with Vision Transformers (ViT). It achieves competitive performance, reaching 80.1% top-1 linear classification accuracy on ImageNet with a ViT-Base architecture.
- **High-Quality Features for k-NN**: DINO features are particularly effective when used with simple retrieval mechanisms. ViT features trained with DINO are excellent k-NN classifiers, achieving results nearly on par with or exceeding linear classifiers (e.g., 78.3% top-1 accuracy on ImageNet with a small ViT). The k-NN performance gain with DINO is substantially larger for ViT architectures compared to ResNet-50.
- **Emergent Segmentation Properties**: Self-supervised ViT features, when trained with DINO, explicitly contain information about the semantic segmentation of an image, including object boundaries. This scene layout information is directly accessible in the self-attention modules of the last block.
- **BN-Free Capability**: When applied to ViT architectures, the entire DINO system can be designed to be entirely Batch Normalization (BN)-free, including the projection heads. This is advantageous because training with BN, especially synchronized BN, can slow down training considerably.
- **Training Efficiency and Scalability**: DINO is computationally efficient, allowing high performance to be reached with a significant reduction of computational requirements compared to similar convnet-based SSL systems. It can also train models to high performance using small batch sizes (e.g., batch size 128 running on a single GPU).
- **Multi-Crop Effectiveness**: The multi-crop training strategy dramatically improves the accuracy/running-time tradeoff for DINO. For example, using additional small crops improved performance by +2% while reducing training time by half compared to the base two-crop setting.
- **Beneficial Teacher Dynamics**: The momentum teacher in DINO consistently outperforms the student during training. This momentum-based teacher performs a form of model ensembling (similar to Polyak-Ruppert averaging) that guides the student network by providing higher-quality target features.

## ⚠️ Limitations
- <text style="background-color:yellow">Dependence on Momentum Teacher</text>: The DINO framework requires a stabilizing mechanism and does not work in the complete absence of a momentum encoder. A simple attempt to stabilize the system by copying the student weights for the teacher fails to converge.
- <text style="background-color:yellow">Sensitivity to Hyperparameters (Collapse Risk)</text>: The collapse avoidance mechanism relies on the balance between centering and sharpening. If either centering or sharpening is missing, the training suffers convergence issues (KL divergence converges to zero, indicating collapse).
- <text style="background-color:yellow">Increased Computational Cost for Optimal ViT Results</text>: While ViT training is fast, achieving the highest performance often involves using smaller patch sizes (e.g., ViT-S/8 or ViT-B/8), which leads to a significant reduction in throughput (running time) and larger memory usage compared to larger patch sizes (e.g., ViT-S/16).
- <text style="background-color:yellow">Reduced Performance with Minimal Batches</text>: While DINO supports small batches, the performance of runs utilizing very small batches (e.g., batch size 128) is slightly below that of the standard training setup (bs=1024).

## 📚 Reference
- *Caron et al., 2021*  _[Emerging Properties in Self-Supervised Vision Transformers]_  🔗 [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- [Github:](https://github.com/facebookresearch/dino)
</div>