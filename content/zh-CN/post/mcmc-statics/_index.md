---
title: "蒙特卡洛-马尔可夫链统计方法"
date: 2026-01-24
type: "docs"
layout: "section"
weight: 2
---

![MCMC Statistics Cover](/img/contents/post/mcmc-statics/mcmc_statics_cover.png)

欢迎来到 **蒙特卡洛-马尔可夫链统计方法** 系列，在这里你将学习从概率论基础到 MCMC 采样的原理与实践。本系列基于米兰理工大学的 [PhD课程：MONTECARLO-MARKOV CHAINS STATISTICAL METHODS](https://www11.ceda.polimi.it/manifestidott/manifestidott/controller/MainPublic.do?evn_dettaglioinsegnamento=evento&aa=2024&k_cf=82&k_corso_la=1378&ac_ins=0&lang=EN&c_insegn=095929&jaf_currentWFID=main)，借助 ChatGPT，结合 Python/Jupyter Notebook 代码与图示帮助理解。

## 系列文章

1. [什么是概率？](./probability/)
2. [随机变量与采样方法基础](./random-variables/)
   - 概率密度函数与期望
   - 简单分布的采样方法
   - 常用基础采样算法介绍
3. [蒙特卡洛方法](./monte-carlo/)
   - 重要性采样（Importance Sampling）
   - 方差缩减技术
4. [理解马尔可夫链](./markov-chains/)
   - 什么是马尔可夫过程
   - 平稳分布与收敛性
   - 构建简单的状态转移过程
5. [MCMC 初识](./intro-mcmc/)
   - 为什么我们需要 MCMC？
   - 从马尔可夫链到抽样
   - 理论与直觉
6. [Metropolis 算法详解：从原理到 Python 实现](./metropolis/)
   - 核心困境：无法计算的归一化常数
   - 随机游走 Metropolis 算法详解
   - 高维分布中的采样表现与调参
7. [Metropolis-Hastings 算法：打破对称性的束缚](./metropolis-hastings/)
   - 为什么我们需要“不对称”的提议？
   - 哈斯廷斯修正项 (Hastings Correction) 的推导与直觉
   - 实战案例：利用 Log-Normal 解决边界问题
8. [Gibbs 采样详解：分而治之的降维智慧](./gibbs-sampling/)
   - 高维困境与“曼哈顿漫步”直觉
   - 及其数学原理：布鲁克引理 (Brook's Lemma)
   - 离散与连续分布的 Python 代码实现
9. [确定性优化算法详解：梯度下降的数学本质与代码实战](./deterministic-optimization/)
   - 凸优化与非凸优化的几何直觉
   - 牛顿法 (Newton's Method) 与二阶近似
   - 坐标下降 (Coordinate Descent) 与 Gibbs 采样的联系
   - 最速下降法 (Steepest Descent) 的优缺点
10. [随机优化算法详解：模拟退火与 Pincus 定理](./stochastic-optimization/)
    - 从能量最小化到概率最大化：模拟退火的物理本质
    - 高温探索与低温锁定：Metropolis 采样的另一种视角
    - Pincus 定理：退火算法收敛到全局最优的数学证明
11. [收敛性诊断](./convergence/)
12. [Python 实战：MCMC 建模](./python/)

## 快速访问
