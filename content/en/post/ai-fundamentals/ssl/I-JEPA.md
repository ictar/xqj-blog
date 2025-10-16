---
title: "I-JEPA: Image-based Joint Embedding Predictive Architecture"
date: 2025-10-09
summary: "A non-generative, self-supervised framework predicting high-level feature representations of masked regions from visible context, enabling scalable and efficient visual pretraining."
series: ["Self-Supervised Learning"]
tags: ["SSL", "Vision", "Representation Learning", "Joint Embedding", "Masked Image Modeling"]
---

<div class="model-card">

### 🏷️ Model Name
**I-JEPA - Image-based Joint Embedding Predictive Architecture** 


## 🧠 Core Idea

> **“Predict what you can’t see — not in pixels, but in meaning.”**

![I-JEPA architecture](https://private-user-images.githubusercontent.com/7530871/245483236-dbad94ab-ac35-433b-8b4c-ca227886d311.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjAwMDE5OTYsIm5iZiI6MTc2MDAwMTY5NiwicGF0aCI6Ii83NTMwODcxLzI0NTQ4MzIzNi1kYmFkOTRhYi1hYzM1LTQzM2ItOGI0Yy1jYTIyNzg4NmQzMTEucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MTAwOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTEwMDlUMDkyMTM2WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9Mzk2MDliZTI3Y2U5MDQ2ZWFhMjkwMjU0OGRjY2IwMjQwNWZjMDBiOTVjNzMzOWVhOWIxOTg2NmVhYWM5OWE4ZCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ._w3JT7VjXtnNM_Wl0cweg9GrHov0xo22E0X3oL6pTOY)

---

## 🖼️ Architecture

```markdown
                    +-------------------------+
                    |     Input Image         |
                    +-------------------------+
                                |
                                v
                +------------------------------------+
                |     Random Masking of Regions      |
                +------------------------------------+
                   | Visible Patches | Masked Patches |
                   |-----------------|----------------|
                     |                 |
                     v                 v
          +------------------+     +------------------+
          |  Context Encoder |     |   Target Encoder |
          |  (f_context)     |     |   (f_target)     |
          +------------------+     +------------------+
                     |                 |
                     v                 v
          +------------------+     +------------------+
          |  Predictor Head  | --> |  Target Features |
          +------------------+     +------------------+
                     |
                     v
          +----------------------------+
          | Loss: MSE in feature space |
          +----------------------------+
```

## 💡 Strengths

## ⚠️ Limitations

## 📚 Reference
* **Paper:** [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)  
* **Code:** [GitHub – facebookresearch/ijepa](https://github.com/facebookresearch/ijepa)  

</div>