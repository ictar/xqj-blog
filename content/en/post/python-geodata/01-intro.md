---
title: "Lesson 1: Introduction to Remote Sensing Data and Python Setup"
description: "Learn about raster and vector data and set up your Python environment"
summary: "Learn about raster and vector data and set up your Python environment"
date: 2025-07-02
draft: false
tags: ["raster", "vector", "remote sensing", "python"]
---

## 🛰️ What is Remote Sensing Data?

Remote sensing data typically includes two types: raster data (e.g., satellite images) and vector data (e.g., boundaries, roads). This article introduces these concepts and helps set up your Python environment.

## 🛠️ Python Environment Setup

Recommended conda setup:

```bash
conda create -n geopython python=3.10
conda activate geopython
conda install geopandas rasterio rioxarray matplotlib jupyterlab
```

## 🔗 Coming Next
Next, we'll dive into reading raster files and exploring metadata and band information.