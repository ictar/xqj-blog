---
title: "第 1 课：遥感数据与 Python 环境入门"
description: "了解遥感数据的基本类型，并搭建处理环境"
summary: "了解遥感数据的基本类型，并搭建处理环境"
date: 2025-07-11
draft: false
tags: ["raster", "vector", "遥感", "python"]
---

## 🛰️ 什么是遥感数据？

遥感数据分为两类：栅格（如遥感影像）与矢量（如边界、道路等）。本节将介绍它们的基本概念，并配置 Python 开发环境。

## 🛠️ 环境准备

推荐使用 Conda 环境：

```bash
conda create -n geopython python=3.10
conda activate geopython
conda install geopandas rasterio rioxarray matplotlib jupyterlab
```


## 🔗 下一节预告
下一节我们将开始读取栅格数据文件，探索元数据与波段信息。