---
title: "BYOL"
date: 2025-10-08
summary: "BYOL: Bootstrap Your Own Latent"
series: ["Self-Supervised Learning"]
tags: ["SSL", "Vision", "Representation Learning", "Joint Embedding", "Distillation Methods"]
---

<div class="model-card">

## 🏷️ Model Name
**BYOL – Bootstrap Your Own Latent**

## 🧠 Core Idea
> Two interacting neural networks—the <text style="color: green">online network</text> and the <text style="color: green">target network </text> —— learn representations by predicting each other's outputs, **without using negative samples**.
>
> Objective: train the online network to <text style="color: red">predict the target network's projection</text> of the same image under a different augmented view.

![BYOL architecture](https://raw.githubusercontent.com/deepmancer/byol-pytorch/main/images/Byol.jpg)

## 🖼️ Architecture

BYOL employs two parallel neural networks: the online network (parameters $\theta$) and the target network (parameters $\xi$).
- <text style="color: green">Online Network</text> $(\theta)$: This network is comprised of three sequential stages:
    * An **encoder** ($f_{\theta}$): a convolutional neural network (such as <text style="color: purple">ResNet</text>) used to extract image feature representation $y_{\theta}$.
    * A **projector** ($g_{\theta}$): a <text style="color: purple">multi-layer perceptron (MLP)</text> projecting the encoder output $y_{\theta}$ into a smaller latent space to obtain the projection $z_{\theta}$.
    * A **predictor** ($q_{\theta}$): another <text style="color: purple">MLP</text>, with the same structure as the projector, is used to predict the projection of the target network.
- <text style="color: green">Target Network</text> $(\xi)$: This network has the same encoder ($f_{\xi}$) and projector ($g_{\xi}$) architecture as the online network, but it **does not have a predictor**. The target network weights $\xi$ are initialized to match the online network weights $\theta$.

And the pseudocode of BYOL's main learning algorithm:
```py
```

### 1️⃣ Data Augmentation and View Generation
1. An input image ($x$) is sampled from the dataset.
2. Two distributions of image augmentations, $T$ and $T'$, are used to create two different augmented views of the image: $v = t(x)$ and $v' = t'(x)$, where $t \sim T$ and $t' \sim T'$.
    - The augmentations typically include random cropping/resizing, random horizontal flip, color distortion, Gaussian blur, and solarization.

### 2️⃣ Forward Pass through Online and Target Branches

The two augmented views, $v$ and $v'$, are processed by the two distinct networks:
- Online Branch (View $v$):
    * The encoder $f_{\theta}$ processes $v$ to yield the representation $y_{\theta}$.
    * The projector $g_{\theta}$ maps $y_{\theta}$ to the projection $z_{\theta}$.
    * The predictor $q_{\theta}$ transforms $z_{\theta}$ to generate the prediction $q_{\theta}(z_{\theta})$.
- Target Branch (View $v'$):
    * The encoder $f_{\xi}$ processes $v'$ to yield the representation $y'_{\xi}$.
    * The projector $g_{\xi}$ maps $y'_{\xi}$ to the target projection $z'_{\xi}$.

### 3️⃣ Loss Calculation

The objective is to train the online network to predict the target network's output:
1. **Normalization**: Both the prediction $q_{\theta}(z_{\theta})$ and the target projection $z'_{\xi}$ are $\ell_2$-normalized.
2. **Stop Gradient**: A stop-gradient operator (sg) is applied to the target projection $z'_{\xi}$, ensuring that the gradient does not propagate back into the target network parameters $\xi$.
3. **Loss Definition**: The loss ($L_{\theta, \xi}$) is defined as the mean squared error (MSE) between the normalized prediction and the normalized target projection: $$L_{\theta, \xi} = \left\| q_{\theta}(z_{\theta}) - z'_{\xi} \right\|_2^2$$.
4. **Symmetrization**: The total BYOL loss ($L_{\text{BYOL}}$) is symmetrized by repeating the process with the inputs swapped (i.e., feeding $v'$ to the online network and $v$ to the target network to compute $\tilde{L}_{\theta, \xi}$): $$L{\text{BYOL}} = L_{\theta, \xi} + \tilde{L}_{\theta, \xi}$$.

### 4️⃣ Parameter Update

The two networks are updated differently in an asymmetric manner:
1. Online Network Update ($\theta$): The online parameters $\theta$ are updated by <text style="color:purple">a stochastic optimization step</text> (e.g., using the LARS optimizer) to minimize $L_{\text{BYOL}}$.
2. Target Network Update ($\xi$): The target parameters $\xi$ are updated using an <text style="color:purple">Exponential Moving Average (EMA)</text> of the online parameters $\theta$. This update is slow and steady, calculated as: $$\xi \leftarrow \tau\xi + (1-\tau)\theta$$ where $\tau \in$ is the momentum coefficient (often starting from $\tau_{\text{base}}=0.996$ and increasing to one). This EMA mechanism is crucial for stabilizing the bootstrap targets.

### 🎉 Final Output

After the self-supervised training is complete, everything except the encoder $f_{\theta}$ is discarded, and $f_{\theta}$ provides the learned image representation for downstream tasks.

## 🎯 Downstream Tasks
- ImageNet Evaluation Protocols
- Transfer Learning Tasks (General Vision)
  - Classification (Transfer to other classification tasks).
  - Semantic Segmentation (e.g., VOC2012 task).
  - Object Detection (Using architectures like Faster R-CNN).
  - Depth Estimation (e.g., NYU v2 dataset).

## 💡 Strengths
- Performance and Efficiency
    * State-of-the-Art Performance Without Negatives: BYOL achieves state-of-the-art results without relying on negative pairs.
    * High Accuracy.
    * Outperforms on Transfer Benchmarks
    * Parameter Efficiency
    * Simple and Powerful Objective
- Robustness and Training Flexibility
    * Insensitivity to Batch Size: BYOL is more resilient to changes in the batch size compared to contrastive counterparts. Its performance remains stable over a wide range of batch sizes (from 256 to 4096), only dropping for very small values due to reliance on batch normalization layers.
    * No Need for Large Batches: Since it does not rely on negative pairs, there is no need for large batches of negative samples.
    * Robustness to Augmentations: BYOL is incredibly robust to the choice of image augmentations compared to contrastive methods like SimCLR. When augmentations are drastically reduced to only random crops, BYOL shows much smaller performance degradation than SimCLR.
    * Effective Use of Moving Average Target: The use of a slow-moving exponential average (EMA) of the online network parameters for the target network provides stable targets, which is crucial for stabilizing the bootstrap step.

## ⚠️ Limitations

- Theoretical and Architectural Constraints
    * <text style="background-color:yellow">Susceptibility to Collapse</text>: The core objective of BYOL—predicting the target network's output—theoretically admits collapsed solutions (where the network outputs the same vector for all inputs). While BYOL employs specific mechanisms (predictor, EMA target) to avoid this, distillation methods generally are noted as being more prone to collapsing.
    * <text style="background-color:yellow">Dependence on Asymmetry</text>: The design requires an asymmetric architecture, specifically the addition of a predictor ($q_{\theta}$) to the online network, which is critical to preventing collapse in the unsupervised scenario. If this predictor is removed, the representation collapses.
- Generalization Constraints
    * <text style="background-color:yellow">Dependence on Vision-Specific Augmentations</text>: BYOL remains dependent on existing sets of augmentations that are specific to vision applications.
    * <text style="background-color:yellow">Generalization Difficulty to Other Modalities</text>: Generalizing BYOL to other modalities (e.g., audio, video, text) would require significant effort and expertise to design similarly suitable augmentations for each modality.

## 📚 Reference
- *Grill et al., 2020*  _[Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning]_  🔗 [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
- [Github: BYOL](https://github.com/google-deepmind/deepmind-research/tree/master/byol)
- [Github: deepmancer/byol-pytorch](https://github.com/deepmancer/byol-pytorch)

</div>