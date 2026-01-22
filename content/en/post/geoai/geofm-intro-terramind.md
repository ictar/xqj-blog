---
title: "Mastering TerraMind: From Understanding to Fine-tuning"
description: "TerraMind is the first large-scale, any-to-any generative multimodal foundation model proposed for the Earth Observation (EO) field. It is pre-trained by combining token-level and pixel-level dual-scale representations to learn high-level contextual information and fine-grained spatial details. The model aims to facilitate multimodal data integration, provide powerful generative capabilities, and support zero-shot and few-shot applications, while outperforming existing models on Earth Observation benchmarks and further improving performance by introducing 'Thinking in Modalities' (TiM)."
summary: "TerraMind is the first large-scale, any-to-any generative multimodal foundation model proposed for the Earth Observation (EO) field. It is pre-trained by combining token-level and pixel-level dual-scale representations to learn high-level contextual information and fine-grained spatial details. The model aims to facilitate multimodal data integration, provide powerful generative capabilities, and support zero-shot and few-shot applications, while outperforming existing models on Earth Observation benchmarks and further improving performance by introducing 'Thinking in Modalities' (TiM)."
date: 2025-09-10
toc: true
draft: false
tags: ["GeoFM", "terramind", "Remote Sensing", "AI", "Large Model", "EO"]
---

{{< toc >}}


Over the past few years, Earth Observation (EO) has been entering a "model-centric" era. The volume of satellite imagery is increasing, and the number of modalities (optical, radar, climate indices, geographical text descriptions...) is growing, but how to truly utilize this information remains a challenge.

Enter **TerraMind**. It is the **first end-to-end generative, multimodal Earth Observation foundation model**, jointly developed by IBM Research, ETH Zurich, Forschungszentrum Jülich, and ESA Φ-Lab. It not only integrates data from different modalities but also possesses generative capabilities and excellent generalization. In other words, it attempts to play the role of a "unified model," providing a powerful "starting point" for various downstream tasks (such as land cover classification, disaster monitoring, climate research, etc.).


# Why TerraMind?

The proposal of TerraMind is primarily to **solve problems in the EO field such as multimodal data integration, generation, and the lack of generalization capability in existing models**. More specifically, it targets the following pain points:

*   **Underutilization of Multimodal Data**
    Traditional models often focus only on specific modalities (like using only optical imagery) or serve only a specific task (like segmentation). TerraMind breaks down "modality barriers" by integrating radar, optical, land cover maps, NDVI, DEM, coordinate metadata, and even natural language descriptions.

*   **Lack of Generative Multimodal Capabilities**
    This is TerraMind's killer feature. It is the first "large-scale, any-to-any" generative multimodal EO model. It can not only perform traditional analysis tasks but also generate new data, supporting zero-shot and few-shot learning, thereby reducing labeling costs.

*   **Generalization and Data Efficiency Challenges**
    Past EO models easily "overfit" to a specific region or task, making migration difficult. TerraMind achieves stronger spatial and temporal generalization through self-supervised pre-training on massive unlabeled data, combined with small-scale supervised fine-tuning.

*   **Limitations of Existing Geospatial Foundation Models**
    Many GFMs (Geo-Foundation Models) essentially force-fit Computer Vision paradigms without considering the unique characteristics of remote sensing data. TerraMind has optimized its architecture, for example, with **Dual-Scale Pre-training** (token-level + pixel-level representations), which captures both global semantics and retains local spatial details.

*   **Accelerating Climate and Sustainability Applications**
    It is not just a model; it also comes with a distributed data processing and sampling framework that can directly connect to satellite data sources, enabling developers to deploy applications faster in climate and sustainability scenarios.

In short, TerraMind's goal is: **To be a general-purpose, powerful foundation model capable of integrating multimodal data, providing a more efficient "starting point" for EO research and applications.**

# Where Does TerraMind Come From?
![](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base/resolve/main/assets%2Fterramind_architecture.png)

## Input: TerraMesh Dataset

TerraMind is pre-trained on a custom-built, large-scale global geospatial dataset called **TerraMesh**, containing **9 million samples and nine geospatial modalities**:
*   Optical Satellite Images: Copernicus Sentinel-2 L1C and L2A (RGB).
*   Radar Satellite Images: Copernicus Sentinel-1 GRD and RTC.
*   Task-Specific Modalities: Land Use/Land Cover (LULC) and Normalized Difference Vegetation Index (NDVI) maps.
*   Metadata: Digital Elevation Model (DEM) and geographical coordinates (discretized as strings).
*   Natural Language: Image captions synthesized from Sentinel-2 RGB images via LLaVA-Next.
  
## Dual-Scale Pretraining

TerraMind combines data at both pixel level and token level, which is its core innovation.
*   **Token Level**: Encodes high-level contextual information, learns cross-modal relationships, and enables scalability.
*   **Pixel Level**: Utilizes fine-grained representations to capture key spatial nuances.

This dual-scale early fusion method outperforms other fusion methods, enabling artificial data generation, zero-shot, and few-shot applications.

## Two-Stage Pretraining
1.  **Single-Modality Tokenizer Pretraining**: A specific tokenizer is developed for each modality to encode data into discrete token sequences or decode from token sequences back to the original form. Image-based modalities (S-1, S-2, LULC, NDVI, DEM) use an **architecture based on autoencoders and Finite Scalar Quantization (FSQ)**. Sequence-based modalities (descriptions, geolocation) use WordPiece-based text tokenizers.
2.  **TerraMind Encoder-Decoder Pretraining**: Uses a symmetric Transformer architecture to process multimodal token sequences and accepts pixel-level cross-modal inputs. The pre-training objective is a cross-modal patch classification problem, reconstructing masked target tokens through cross-entropy loss.


# What Can TerraMind Do?

Its strengths can be summarized in three points:

1.  **Integration**: Puts multiple modalities like radar, optical, DEM, LULC into a unified representation.
2.  **Generation**: Generative tasks between any modalities (e.g., synthesizing optical imagery from radar, or predicting NDVI from DEM).
    *   Starting from Sentinel-2 L2A optical data, TerraMind can generate high-quality radar data, land use maps, and digital elevation maps.
    *   Even starting from low-information geolocation info, the model can generate contextually relevant optical images (e.g., generating desert images from a Middle East location), although the structure might differ from the ground truth.
3.  **Generalization**: Downstream tasks require only a small amount of labeling to adapt quickly, such as land cover classification, disaster monitoring, etc.
    *   **Zero-Shot Learning**
        *   **Water Mapping**: TerraMindv1-B achieves 45.4% IoU in a zero-shot setting. Using DynamicWorld LULC data for ablation experiments, it improves to 69.8%, approaching fine-tuned SOTA performance (84.4%).
        *   Geolocation: Can accurately predict the geolocation of specific data instances, e.g., predicting the probability distribution for the "Bareland" category and identifying regions like the Sahara or the Middle East.
    *   **Few-Shot Learning**: In 1-shot 5-way classification tasks on EuroSAT and METER-ML datasets, TerraMind's average accuracy is at least 10pp higher than other benchmarks, indicating its good latent spatial structure.

Furthermore, TerraMind introduces the innovative concept of **"Thinking in Modalities" (TiM)**, similar to "chain-of-thought" in large language models. By injecting generated artificial data during fine-tuning and inference, model output performance is improved. For example, in water mapping tasks, generating additional LULC data via TiM fine-tuning improved mIoU by 2pp compared to standard fine-tuning.

![](https://research.ibm.com/_next/image?url=https%3A%2F%2Fresearch-website-prod-cms-uploads.s3.us.cloud-object-storage.appdomain.cloud%2FScreenshot_2025_10_20_at_3_49_35_PM_9f03a58e80.png&w=1920&q=85)

# Encoding and Reconstruction (Tokenizer)

First, one must understand TerraMind's **Tokenizer**. Its working principle is similar to "compression—decompression":

*   Input Image → Convert to token representation
*   Tokens capture spatial details and contextual information
*   Reconstruct image via decoder → Verify if the model "understood" the input

This part mainly **verifies the model's information retention capability**.

Below is an actual Python script (based on Sentinel-2 L2A data), referencing [IBM/terramind/notebooks/terramind_tokenizer_reconstruction.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_tokenizer_reconstruction.ipynb):

```python
# Build tokenizer model
model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_s2l2a', pretrained=True)

# Load example image and construct tensor
# Load an example Sentinel-2 image
examples = [
    '../examples/S2L2A/38D_378R_2_3.tif',
    '../examples/S2L2A/282D_485L_3_3.tif',
    '../examples/S2L2A/433D_629L_3_1.tif',
]
data = rxr.open_rasterio(examples[1])
# Convert to [B, C, 224, 224]
data = torch.Tensor(data.values, device='cpu').unsqueeze(0)

# Normalization (Standardization)
mean = torch.Tensor(v1_pretraining_mean['untok_sen2l2a@224'])
std = torch.Tensor(v1_pretraining_std['untok_sen2l2a@224'])
input = (data - mean[None, :, None, None]) / std[None, :, None, None]

# Run model (Encode -> Decode)
with torch.no_grad():
    reconstruction = model(input, timesteps=10)

# De-normalization (Restore to original scale)
reconstruction = reconstruction.cpu()
reconstruction = (reconstruction * std[None, :, None, None]) + mean[None, :, None, None]
# The decoded reconstruction needs to be multiplied by std and added to mean to bring the values back to the "real" scale (same range as original data), otherwise it will look like "small numbers with mean 0".
```

When visualizing, you can compare the RGB representation of the input image and the reconstructed image:

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

Result: The left is the original Sentinel-2 input, and the right is the reconstruction result after Tokenizer decoding.

👉 This proves that **TerraMind can represent images using tokens and retain information very well**.


# Cross-Modal Generation (Any-to-Any)
What's exciting is TerraMind's **generative capability**.
It can not only "reconstruct" input but also achieve **Cross-Modal Generation (Any-to-Any)**:

*   Optical Image → Generate Radar Image
*   DEM → Generate Optical or NDVI
*   Image → Generate LULC Map
*   It can even do Image ↔ Text

This **any-modality-to-any-modality generation capability** is TerraMind's biggest breakthrough compared to traditional models.
It means: even if some data is missing, it can be filled in through generation; even if labeling is limited, training can be augmented through generation.

## Single Patch Example

*   Input: Single patch, can be S2-L2A data (224×224)
*   Output: Multiple target modalities, such as S1GRD, DEM, LULC, etc.
*   Features:
    *   Each output modality corresponds to its own tokenizer (increases memory consumption).
    *   Uses diffusion steps (e.g., 10 steps) to generate results → ensures diversity and quality of output.
    *   Can standardize input directly (`standardize=True`).
*   Steps:
    1.  Build Model
        *   Instantiate TerraMind model from `FULL_MODEL_REGISTRY`.
        *   Specify **Input Modalities** (e.g., `S2L2A`) and **Output Modalities** (e.g., `S1GRD, DEM, LULC`).
        *   Select whether to load pre-trained weights (`pretrained=True`).
        *   Configure standardization (`standardize=True` applies pre-training mean/variance automatically).
    2.  Prepare Data
        *   Load raster data
        *   Use `rioxarray.open_rasterio` to read as array.
        *   Convert to input shape required by model [Batch, Channels, Height, Width], e.g., [1, C, 224, 224].
    3.  Load Input Data: Plot input data as RGB image (helps understand input content).
    4.  Execute Diffusion Generation
        *   Send input to GPU/CPU (`input.to(device)`).
        *   Run diffusion generation under `torch.no_grad()` (`model(input, timesteps=10)`)
        *   `timesteps` controls the number of iterations in the diffusion process (more steps mean finer generation but take longer).
    5.  Get Multimodal Output
        *   Model returns a dictionary (`{modality_name: generation_result}`)
        *   Each output modality has a corresponding **generation tensor**.
    6.  Visualize Results.
        *   Iterate through output modalities, call `plot_modality()` to plot each result.
        *   Display side-by-side with input image for comparison.

The following Python snippet implements generating other modalities (S-1, DEM, LULC...) using S-2 input, referencing [IBM/terramind/notebooks/terramind_generation.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_generation.ipynb):

```py
# Load example S-2 L2A and construct model input
data = rxr.open_rasterio(examples[example_id])          # rioxarray DataArray
data = torch.Tensor(data.values, device='cpu').unsqueeze(0)  # -> [B, C, H, W]

# Build model
model = FULL_MODEL_REGISTRY.build(
    'terramind_v1_base_generate',
    modalities=['S2L2A'],
    output_modalities=['S1GRD', 'DEM', 'LULC'],
    pretrained=True,
    standardize=True,
)
# Move to device
_ = model.to(device)

# Run generation (diffusion steps)
with torch.no_grad():
  generated = model(input, verbose=True, timesteps=10)
```

Essentially, this is cross-modal translation: **Give you a Sentinel-2 patch, I can "imagine" other modalities like SAR, DEM, NDVI, etc.**

## Tiled Inference

**Why use tiled_inference?**

Large tiles (full tile) usually exceed GPU memory, so big images are cut into many small tiles (patches) to perform inference separately on the GPU, and then merged back into the full image. `tiled_inference` is a utility function (provided by TerraTorch/user library) responsible for cutting patches, calling the model in batches, and merging patch outputs into a full output.

In summary:
*   Input: Full tile (e.g., Singapore, Santiago, could be 1000×1000 or larger).
*   Problem: GPU memory isn't enough to hold the whole image.
*   Steps:
    1.  Load and (optionally) crop large tile
    2.  Convert to [B,C,H,W]
    3.  Build generation model (specify output modalities, timesteps)
    4.  Use `tiled_inference` to cut patches and call `model_forward` in batches
    5.  Merge back to full image to get [C_total,H,W]
    6.  Split channels for each modality
    7.  De-standardize continuous modalities, argmax classification modalities
    8.  Save GeoTIFF / Visualize / Evaluate metrics.

Below is an actual Python script, referencing [IBM/terramind/notebooks/large_tile_generation.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/large_tile_generation.ipynb)

```py
# Input preparation (crop / batch dim)
data = rxr.open_rasterio('examples/S2L2A/Santiago.tif').values
data = data[:, 500:1500]           # Optional crop for speed
input = torch.tensor(data, dtype=torch.float, device=device).unsqueeze(0)

# Build and prepare generation model: condition on S2L2A, generate S1GRD and LULC
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

# From dict -> tensor
def model_forward(x):
    generated = model(x)                       # dict: {modality: tensor}
    out = torch.concat([generated[m] for m in output_modalities], dim=1)
    return out

# Output of tiled_inference and remove batch dim
pred = tiled_inference(model_forward, input, crop=256, stride=192, batch_size=16, verbose=True)
pred = pred.squeeze(0)   # From [1, C, H, W] -> [C, H, W]

# Split concatenated channels back to modalities
num_channels = {'S2L2A':12, 'S1GRD':2, 'S1RTC':2, 'DEM':1, 'LULC':10, 'NDVI':1}
num_channels = {m: num_channels[m] for m in output_modalities}
start_idx = np.cumsum([0] + list(num_channels.values()))
generated = {m: pred[i:i+c].cpu() for m, i, c in zip(output_modalities, start_idx, num_channels.values())}

# Post-processing for LULC (from probability -> discrete category)
if 'LULC' in generated.keys():
    generated['LULC'] = generated['LULC'].argmax(dim=0)

```

Thus, enabling TerraMind to be not just a small patch demo, but applicable to real-world large-scale Earth scenarios.

# Fine-tuning Tasks (Fine-tuning with TerraTorch)

Another highlight of TerraMind is its convenience for fine-tuning on downstream tasks (like semantic segmentation). We use [IBM/terramind/notebooks/terramind_v1_base_sen1floods11.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_v1_base_sen1floods11.ipynb) as an example to explain the main flow of fine-tuning:

1.  **Data Preparation (DataModule)**
    Use `GenericMultiModalDataModule` to define modalities, paths, split files, standardization parameters. For example:

    *   Modalities: `["S2L1C", "S1GRD"]`
    *   Labels: `*_LabelHand.tif`
    *   Train/Val/Test Splits: `flood_train_data.txt`, etc.

    The benefit of this is that whether you use Optical+Radar, or add DEM, LULC, you can configure it quickly.

2.  **Load Pre-trained Backbone**

    ```python
    model = BACKBONE_REGISTRY.build(
        "terramind_v1_base",
        modalities=["S2L1C", "S1GRD"],
        pretrained=True,
    )
    ```

3.  **Define Downstream Task (SemanticSegmentationTask)**

    *   Backbone: TerraMind
    *   Neck: Feature extraction and reshape
    *   Decoder: UNetDecoder
    *   Loss: Dice or CE

    Also supports freezing/unfreezing backbone, adjusting learning rate (1e-5 \~ 1e-4).

4.  **Training and Testing**
    Start training via PyTorch Lightning's `Trainer.fit()`, save the best checkpoint.
    Then evaluate on `Trainer.test()`, and finally perform prediction and visualization.

Running through the entire fine-tuning script allows you to get a "task-adapted model" from a "pre-trained foundation model," seeing results even after training for just a few epochs.

## Using TiM in Fine-tuning

To use TiM in fine-tuning, simply use a backbone ending with `_tim` and specify the modalities to use via `backbone_tim_modalities`. For example:
```yaml
backbone: terramind_v1_base_tim      # Instead of terramind_v1_base
backbone_tim_modalities: [LULC]      # or S1GRD / NDVI / DEM / …
```

# Summary

*   **What is TerraMind?** A multimodal foundation model for Earth Observation.
*   **What can it do?** Multimodal integration, generative tasks, downstream generalization.
*   **How to play?**
    *   Experience generative capability first: Input Image → Tokens → Reconstruct Image.
    *   Then do downstream fine-tuning: Configure DataModule, load pre-trained model, define task, train/test.

It is not just a model, but more like a **General-Purpose AI Platform for EO**. In the future, whether it's climate research, disaster response, or land cover monitoring, it can help achieve rapid deployment.


# Learn More
- [Github: IBM/terramind](https://github.com/IBM/terramind/tree/main)
- [Hugging Face: ibm-esa-geospatial/TerraMind-1.0-base](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
- [Paper (arxiv): TerraMind: Large-Scale Generative Multimodality for Earth Observation](https://doi.org/10.48550/arXiv.2504.11171)
- [Introducing Thinking-in-Modalities with TerraMind](https://research.ibm.com/blog/thinking-in-modalities-terramind)
