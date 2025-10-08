---
title: "Self-Supervised Learning"
date: 2025-10-07
weight: 4
summary: "Explore self-supervised learning methods such as contrastive learning, clustering, distillation, and mask modeling, and their applications to EO."
---

> **Self-Supervised Learning (SSL)** is a family of techniques for converting an unsupervised learning problem into a supervised one by creating surrogate labels from the unlabeled dataset. 

---

## Core Paradigms of Self-Supervised Learning

Three primary paradigms dominate the SSL landscape: Joint Embedding, Masked Image Modeling, and a hybrid approach that combines elements of both.

**Joint Embedding vs. MIM:**
| Joint Embedding	| Masked Image Modeling (MIM)	|
|--|--|
|**Pros:**|**Pros:**|
|✓ Produces highly semantic features, great for classification.|✓ Conceptually simple, with no need for positive/negative pairs.|
|✓ Architecture agnostic.|✓ Masking reduces pre-training time.|
|✓ Achieves competitive results in linear probing evaluations.|✓ Achieves competitive results with fine-tuning.|
||✓ Stronger fit for low-level tasks (e.g., denoising, super-resolution).|
|**Cons:**|**Cons:**|
|✗ May require very large batch sizes (e.g., SimCLR).|✗ Requires a Vision Transformer (ViT) backbone.|
|✗ Requires careful tuning of data augmentations.|✗ Weaker performance on abstract, high-level tasks like classification.|
|✗ Requires special mechanisms to handle negative samples or avoid collapse.||
|✗ Not well-suited for low-level tasks.||


### 1. Joint Embedding Architectures

> **Central idea**: Enforce invariance to data augmentations. <text style="color: violet;">A [Siamese network](https://en.wikipedia.org/wiki/Siamese_neural_network) architecture</text> with shared parameters processes two different augmented "views" of the same image and trains the model to produce similar or identical embeddings for both.

**Challenge**: <text style="background-color: yellow;">model collapse</text>, where the network learns a trivial solution by mapping all inputs to the same constant vector. 


| Method Category	| Core Idea	| Key Characteristics | Examples |
|--|--|--|--|
|**Contrastive**|Pull positive pairs (views of the same image) close in the embedding space while pushing negative pairs (views of different images) apart.|Requires negative samples, which can necessitate large batch sizes. Good for multimodal data.|[SimCLR](SimCLR.md), [MoCo](MoCo.md)|
|**Clustering**|Learn embeddings by grouping similar samples into clusters without using explicit negative pairs.|Jointly learns feature representations and cluster assignments.|[SwAV](SwAV.md), Deep Cluster|
|**Distillation**|A "student" network is trained to match the output distribution of a "teacher" network on different augmented views.|Avoids collapse via an asymmetric architecture (student vs. teacher). The teacher is often updated via an Exponential Moving Average (EMA) of the student's weights. Does not require negative samples.|[BYOL](BYOL.md), [DINO](DINO.md)|
|**Regularization**|Avoids collapse by imposing regularization terms on the embeddings, such as decorrelating feature dimensions.|Maximizes the information content of the embeddings by penalizing redundancy. No negative samples required. |Barlow Twins, VICReg|

### 2. Masked Image Modeling (MIM)

> **Central idea**: Reconstruction. The input image is split into patches, a significant portion of which (often ~75%) are masked. The model is then trained to predict the content of the masked patches based on the visible ones.

**Prediction Targets**: The model can be trained to predict various targets for the masked regions:
* Pixel Reconstruction: Reconstructing the raw pixel values (e.g., [MAE](MAE.md), SimMIM).
* Feature Regression: Predicting abstract feature representations (e.g., MaskFeat).
* Token Prediction: Predicting discrete visual tokens (e.g., BEiT).

### 3. Hybrid Architectures
 
> **Central idea**: Combine the principles of masking and joint embedding

**Examples:**
- **Image-based Joint Embedding Predictive Architecture (I-JEPA)**