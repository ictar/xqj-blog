---
title: "Metropolis 算法"
slug: "metropolis"
date: 2026-01-24
description: "详解 Metropolis 算法的核心思想、随机游走实现及其在高维分布中的表现。包含 Python 代码示例与可视化。"
summary: "详解 Metropolis 算法的核心思想、随机游走实现及其在高维分布中的表现。包含 Python 代码示例与可视化。"
toc: true
draft: false
tags: ["MCMC", "Metropolis", "Algorithm", "Monte Carlo"]
---

{{< toc >}}

# 我们要解决什么问题？(The Core Problem)

## 1. 核心困境：无法计算的 $Z$

在贝叶斯统计、物理模拟和高维计算中，我们经常需要从一个复杂的概率分布 $\pi(x)$ 中进行采样。但是，我们通常只知道这个分布的“形状”，却不知道它的“规模”。
- **已知**： 未归一化的密度函数 $f(x)$（相对权重）。
- **未知**： 归一化常数 $Z$（总和或积分）。$$\pi(x) = \frac{f(x)}{Z}, \quad \text{其中 } Z = \int f(x) dx$$
- **痛点**： 在高维空间中，计算 $Z$（遍历整个空间求和）是计算上不可行的 (Intractable)。
- **后果**： 因为不知道 $Z$，我们无法算出绝对概率 $\pi(x)$，传统的直接采样方法（如逆变换法）全部失效。

### 关于 $\pi$

| 场景 | $\pi$ 的形式 | 数学名称 | 物理意义 |
| --- | --- | --- | --- |
| **基础马尔可夫链** | 向量  | 平稳分布向量 | 各个状态的长期停留概率 |
| **Metropolis (MCMC)** | 函数  | 目标概率密度 | 我们希望采集样本的那个“形状” |

## 2. Metropolis 的解决策略：相对比值法

Metropolis 算法的核心洞见是：**既然 $Z$ 算不出来，那就消掉它。**

如果不去计算绝对概率，而是比较两个状态之间的**相对概率比值**，常数 $Z$ 就会在分子分母中自动抵消：
$$\frac{\pi(x_{\text{new}})}{\pi(x_{\text{old}})} = \frac{f(x_{\text{new}}) / Z}{f(x_{\text{old}}) / Z} = \frac{f(x_{\text{new}})}{f(x_{\text{old}})}$$

这使得我们只利用**相对高低**（$f(x)$的比值）就能判断两个状态的优劣，从而绕过了计算 $Z$ 的难题。


## 3. 连接点：为什么要用马尔可夫链？

既然我们只能做“局部比较”（比较当前位置和下一步位置），我们就无法一步到位地生成独立样本。我们需要一个能够**在空间中游走**的机制，这就引入了马尔可夫链。

* **动态模拟静态：** 我们的目标是得到一个**静态分布**  $\pi$ 的样本，Metropolis 的手段是构造一个**动态过程**（马尔可夫链）。
* **逆向工程思维：**
  * **传统马尔可夫链问题：** 给定转移矩阵  $P$ ，求稳态分布  $\pi$ 。
  * **Metropolis (MCMC) 问题：** 已知目标分布 $\pi$ ，**设计**一个转移矩阵 $P$，使得这个链最终收敛到  $\pi$。


* **算法本质：**
Metropolis 算法通过**细致平衡原则 (Detailed Balance)** 构造了特殊的“接受/拒绝”规则，实时生成了一个**HIA 链**（齐次、不可约、非周期）。
* **最终结论：**
根据**遍历定理 (Ergodic Theorem)**，这个马尔可夫链跑出来的**轨迹 (Trajectory)**，在长期统计上等价于从目标分布  $Z$ 中抽取的样本。

> **一句话总结：**
> Metropolis 算法是为了解决 **“在归一化常数 $Z$ 未知的情况下进行采样”** 的难题，它通过 **“构造一个以目标分布为稳态的马尔可夫链”** 来实现这一目标。

# Metropolis（随机游走）

为了保证收敛到 $\pi$，我们只需要构造一个满足 细致平衡方程 的链：
$$\pi_i P_{ij} = \pi_j P_{ji}$$

Metropolis 算法把转移过程拆成了两步：
1. **提议 (Proposal)** $Q_{ij}$： 在数学符号里，它通常写作 $Q(x_{new} | x_{old})$ 或者 $q(x' | x)$。意思是：“已知我现在站在 $x_{old}$，我下一步提议跳到 $x_{new}$ 的概率是多少？”
   - 请注意，它叫“提议” (Proposal)。因为它只是负责建议：“嘿，我们要不要试试去那里？” 至于到底去不去，那是后面 $\alpha$ (接受率) 决定的事。
   - 在原始的 Metropolis 算法中，$Q$ 必须是**对称的（Symmetry）**：$$Q(x_{new} | x_{old}) = Q(x_{old} | x_{new})$$
     - 这样在后续计算接受率的时候，我们就可以把 $Q$ 消去了。
   - 在实践时，$Q$ 通常就是一行简单的随机数生成代码。它有两种常见的形态
     - A. 均匀游走 (Uniform Random Walk)
       - 代码：  `x_new = x_old + random.uniform(-1, 1)`
       - 逻辑： 以当前位置为中心，画一个宽为 2 的盒子，盒子里的任何一个点被选中的概率都一样。
       - 特点： 简单粗暴。
     - B. 高斯游走 (Gaussian Random Walk)
       - 代码：  `x_new = x_old + random.normal(0, sigma)`
       - 逻辑： 以当前位置为中心，生成一个正态分布。离当前位置越近的点，越容易被提议；太远的点很少被提议。
       - 特点： 更符合自然界的移动规律（大多数时候迈小步，偶尔迈大步）。
2. **接受 (Acceptance)** $\alpha_{ij}$： 决定“我真的要跳过去吗，还是留在原地？”。
   - 接受率虽然是由状态对 $(i, j)$ 决定的固定值，但在工程上，因为状态数量 $N$ 是天文数字，我们永远无法把这个 $N \times N$ 的表格预先算出来存储。我们只能 **“走到哪，算到哪”**。
   - ⚠️ Metropolis 算法存在的全部意义，就是因为状态空间太大（或连续无限），导致我们无法提前确定这个关于接受率的“二维数组”。

所以，实际的转移概率是：$P_{ij} = Q_{ij} \times \alpha_{ij}$。把它代入细致平衡方程：
$$\pi_i (Q_{ij} \alpha_{ij}) = \pi_j (Q_{ji} \alpha_{ji})$$

假设我们使用的是**对称的提议规则**（即 $Q_{ij} = Q_{ji}$，比如向左跳和向右跳的概率一样，都是 0.5）。那么方程就简化为：
$$\pi_i \alpha_{ij} = \pi_j \alpha_{ji}$$
或者写成比率：
$$\frac{\alpha_{ij}}{\alpha_{ji}} = \frac{\pi_j}{\pi_i}$$


假设你现在处于状态 $i$，系统建议你跳到状态 $j$。如果状态 $j$ 的概率比状态 $i$ 更高（即 $\pi_j > \pi_i$，这一步是往“高处”走），为了满足上面的比率，接受概率 $\alpha_{ij}$ 应该设为 1 (100%) 最合适（也最有效率）。因为既然 $\pi_j > \pi_i$，说明新状态 $j$ 是一个“更好”或者是“更重要”的状态，我们总是乐意往高处走，所以我们毫不犹豫地接受这个提议。

这得到了著名的 Metropolis 接受准则 (Acceptance Probability)：
$$\alpha_{ij} = \min \left( 1, \frac{\pi_j}{\pi_i} \right)$$
它包含了两种情况：
1. 往高处走 ($\pi_j > \pi_i$)： 比值 $>1$，取 $\min$ 后得到 1。总是接受。
2. 往低处走 ($\pi_j < \pi_i$)： 比值 $<1$，取 $\min$ 后得到 $\frac{\pi_j}{\pi_i}$。
   - 这才是算法的灵魂！
   - 即使新状态不如现在好，我们也有一定的概率（虽然不是 100%）接受它。
   - **为什么？** 为了防止陷入“局部最优” (Local Optima)。偶尔接受坏结果，能让你跳出小坑，去寻找更远处的最高峰。


```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义目标分布 pi(x) (Target Distribution)
# 这里我们用标准正态分布: proportional to exp(-0.5 * x^2)
def target_pi(x):
    return np.exp(-0.5 * x**2)

# 2. Metropolis 算法设置
num_samples = 100000
current_state = 0 # 随便找个起点
samples = []

# 3. 开始采样循环
for _ in range(num_samples):
    # A. 提议 (Proposal): 在当前位置附近随便跳一下
    # Q(j|i) 是对称的 (比如用均匀分布或高斯分布作为跳跃步长)
    proposal_state = current_state + np.random.uniform(-1, 1)
    
    # B. 计算接受率 (Acceptance Probability)
    # alpha = min(1, pi_new / pi_old)
    ratio = target_pi(proposal_state) / target_pi(current_state)
    acceptance_prob = min(1, ratio)
    
    # C. 决定是否移动 (Accept/Reject Step)
    # 生成一个 0-1 之间的随机数，如果小于接受率，就接受
    if np.random.rand() < acceptance_prob:
        current_state = proposal_state  # 移动到新位置
    
    # 无论接受还是拒绝，都记录当前位置 (注意：如果拒绝，就是记录旧位置！)
    samples.append(current_state)

# --- 绘图验证 ---
plt.figure(figsize=(10, 6))

# 绘制我们要采样的真实曲线（理论值）
x = np.linspace(-4, 4, 1000)
plt.plot(x, target_pi(x) / np.sqrt(2 * np.pi), 'r-', lw=3, label='True Target Distribution')

# 绘制 Metropolis 算法采样得到的直方图
plt.hist(samples, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Metropolis Samples')

plt.title("Metropolis Algorithm in Action", fontsize=16)
plt.legend()
plt.show()
```


    
![png](/img/contents/post/mcmc-statics/6_metropolis/6_mcmc_metropolis_2_0.png)
    


## 一维

**输入**：

* 目标（未归一化）对数密度 $\log \tilde\pi(x)$
* 初始点 $x_0$
* 对称提议分布 $q(y\mid x)=\mathcal N(x,\sigma^2)$（高维可用多元正态）
* 总步数 $T$，以及丢弃前 $B$ 步作为 burn-in

**每一步** $t=0,1,2,\dots,T-1$：

1. 从对称提议分布**提议**：$y \sim \mathcal N(x_t,\sigma^2)$。
2. 计算**接受率**：

$$
\alpha(x_t,y)=\min\Big\{1,\ \frac{\tilde\pi(y)}{\tilde\pi(x_t)}\Big\}.
$$

> 注意我们只用到了**比值**，不需要归一化常数！
> 为了数值稳定，实际都是用 $\log\tilde\pi$：$\log\alpha = \min\{0,\ \log\tilde\pi(y)-\log\tilde\pi(x_t)\}$。

3. 以概率 $\alpha$ 接受：$x_{t+1}=y$；否则拒绝：$x_{t+1}=x_t$。

**输出**：
* 丢弃前 $B$ 步后得到的样本序列作为近似来自 $\pi$ 的样本；
* 报告**接受率**（accepted 次数 / 总步数）。


### 正确性解释

核心是**详细平衡（可逆性）**：对称提议 $q(y\mid x)=q(x\mid y)$ 时，Metropolis 的接受率确保

$$
\pi(x)\,q(y\mid x)\,\alpha(x,y)=\pi(y)\,q(x\mid y)\,\alpha(y,x),
$$

从而 $\pi$ 是**平稳分布**（不变分布）。只要链还**不可约 + 非周期**，就会从任意起点**收敛**到 $\pi$（TV 距离下）。

**直觉**：每次都让“从 $x$ 到 $y$”的概率流量恰好与“从 $y$ 到 $x$”相配平，长期没有净流，分布就稳在 $\pi$。



### $\sigma$（步长）

* $\sigma$ **太小**：几乎都接受，但走得很慢，样本强相关，**ESS 低**；
* $\sigma$ **太大**：经常提议到低密度区，被拒绝很多，也不高效；
* $\sigma$ **合适**：接受率与移动幅度权衡较好，ACF 衰减快，**ESS 高**。

经验上：随机游走型在**1 维**最优接受率常在 **\~0.4 左右**；维度增大则常见在 **0.2–0.3** 之间较合理（只是经验，不是铁律）。

### 示例


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)

def acf_1d(x, max_lag=200):
    x = np.asarray(x)
    x = x - np.mean(x) # zero-mean
    n = len(x)
    var = np.var(x) # biased variance
    out = np.empty(max_lag+1, dtype=float)
    out[0] = 1.0
    for k in range(1, max_lag+1):
        out[k] = np.dot(x[:-k], x[k:]) / ((n-k) * var)
    return out

def ess_from_acf(acf_vals, n):
    s = 0.0
    for k in range(1, len(acf_vals)):
        if acf_vals[k] <= 0:
            break
        s += 2 * acf_vals[k]
    tau_int = 1.0 + s
    return n / tau_int

def normalize_pdf(xs, logpdf):
    lps = np.array([logpdf(x) for x in xs])
    lps -= np.max(lps)
    pdf_unnorm = np.exp(lps)
    Z = np.trapezoid(pdf_unnorm, xs)
    return pdf_unnorm / Z

def metropolis(logpdf, x0, proposal_std, n_steps, burn_in=0, rng=None):
    """Metropolis algorithm for 1D distributions.

    Args:
        logpdf: function that computes the log of the target PDF at a given x
        x0: initial position (float)
        proposal_std: standard deviation of the Gaussian proposal distribution (float)
        n_steps: total number of MCMC steps (int)
        burn_in: number of initial samples to discard (int, default=0)
        rng: optional numpy random generator (default=None, uses np.random.default_rng())
    """
    if rng is None:
        local_rng = np.random.default_rng()
    else:
        local_rng = rng
    x = float(x0)
    samples = []
    accepted = 0
    accepts = []
    for t in range(n_steps): # t = 0, 1, ..., n_steps-1
        y = x + local_rng.normal(0.0, proposal_std) # propose new position
        logacc = logpdf(y) - logpdf(x) # log acceptance ratio
        if np.log(local_rng.uniform()) < logacc: # accept/reject
            x = y
            accepted += 1
            accepts.append(1)
        else:
            accepts.append(0)
        if t >= burn_in: # record sample after burn-in
            samples.append(x)
    acc_rate = accepted / n_steps # acceptance rate
    return np.array(samples), acc_rate, np.array(accepts)
```

#### 单峰示例
单峰难归一化分布**：$\pi(x)\propto e^{-x^4}$

观察不同 $\sigma$ 下的接受率、ACF、ESS、直方图 vs 真实密度（数值归一化）。


```python
import os
# Example : exp(-x^4)
def logpdf_expfour(x):
    return - (x**4)

save_folder = "./mcmc_meetropolis_results"
os.makedirs(save_folder, exist_ok=True)
n_steps = 50000
burn_in = 5000
x0 = 0.0
configs = [("small", 0.15), ("tuned", 0.8), ("large", 3.0)]
results_A = []
for name, s in configs:
    samples, acc_rate, accepts = metropolis(logpdf_expfour, x0, s, n_steps, burn_in, rng)
    acf_vals = acf_1d(samples, max_lag=200)
    ess = ess_from_acf(acf_vals, len(samples))
    results_A.append({"config": name, "proposal_std": s, "accept_rate": acc_rate, "ESS": ess, "n_kept": len(samples)})
    pd.DataFrame({"x": samples}).to_csv(f"{save_folder}/metropolis_expfour_{name}.csv", index=False)

samples_tuned = pd.read_csv(f"{save_folder}/metropolis_expfour_tuned.csv")["x"].values
plt.figure(figsize=(9,4))
plt.plot(samples_tuned)
plt.xlabel("iteration (post burn-in)")
plt.ylabel("x")
plt.title("Metropolis trace on π(x) ∝ exp(-x^4) — tuned proposal")
plt.tight_layout()
plt.savefig(f"{save_folder}/expfour_trace_tuned.png", dpi=150)
plt.show()

xs = np.linspace(-4.5, 4.5, 600)
true_pdf = normalize_pdf(xs, logpdf_expfour)
plt.figure(figsize=(9,4))
plt.hist(samples_tuned, bins=80, density=True, alpha=0.5, label="samples")
plt.plot(xs, true_pdf, label="true density (normalized numerically)")
plt.xlabel("x")
plt.ylabel("density")
plt.title("Metropolis on exp(-x^4): samples vs true density (tuned proposal)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_folder}/expfour_hist_tuned.png", dpi=150)
plt.show()

plt.figure(figsize=(9,4))
for name, _ in configs:
    x = pd.read_csv(f"{save_folder}/metropolis_expfour_{name}.csv")["x"].values
    acf_vals = acf_1d(x, max_lag=150)
    plt.plot(acf_vals, label=f"{name}")
plt.xlabel("lag")
plt.ylabel("autocorrelation")
plt.title("ACF comparison — exp(-x^4)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_folder}/expfour_acf_compare.png", dpi=150)
plt.show()
```


    
![png](/img/contents/post/mcmc-statics/6_metropolis/6_mcmc_metropolis_7_0.png)
    



    
![png](/img/contents/post/mcmc-statics/6_metropolis/6_mcmc_metropolis_7_1.png)
    



    
![png](/img/contents/post/mcmc-statics/6_metropolis/6_mcmc_metropolis_7_2.png)
    


#### 双峰示例
**双峰混合**：0.5 $\mathcal N(-3,1)$ + 0.5 $\mathcal N(3,1)$

看随机游走在多峰地形里的“卡峰”问题，以及 $\sigma$ 太小/太大的反面教材。



```python
# Example: bimodal mixture
import os
save_folder = "./mcmc_meetropolis_results"
os.makedirs(save_folder, exist_ok=True)
def logpdf_bimodal(x):
    mu1, mu2, s = -3.0, 3.0, 1.0
    l1 = -0.5*((x-mu1)/s)**2 - 0.5*np.log(2*np.pi*s*s) + np.log(0.5)
    l2 = -0.5*((x-mu2)/s)**2 - 0.5*np.log(2*np.pi*s*s) + np.log(0.5)
    m = np.maximum(l1, l2)
    return m + np.log(np.exp(l1-m) + np.exp(l2-m))

configs_B = [("too_small", 0.2), ("okay", 1.2), ("too_large", 4.0)]
results_B = []
for name, s in configs_B:
    samples, acc_rate, accepts = metropolis(logpdf_bimodal, x0=-5.0, proposal_std=s, n_steps=n_steps, burn_in=burn_in, rng=rng)
    acf_vals = acf_1d(samples, max_lag=200)
    ess = ess_from_acf(acf_vals, len(samples))
    frac_right = float(np.mean(samples > 0))
    results_B.append({"config": name, "proposal_std": s, "accept_rate": acc_rate, "ESS": ess, "frac_right_mode": frac_right, "n_kept": len(samples)})
    pd.DataFrame({"x": samples}).to_csv(f"{save_folder}/metropolis_bimodal_{name}.csv", index=False)

xs2 = np.linspace(-8, 8, 700)
pdf2 = normalize_pdf(xs2, logpdf_bimodal)
samples_ok = pd.read_csv(f"{save_folder}/metropolis_bimodal_okay.csv")["x"].values
plt.figure(figsize=(9,4))
plt.hist(samples_ok, bins=120, density=True, alpha=0.5, label="samples (okay)")
plt.plot(xs2, pdf2, label="true density (normalized numerically)")
plt.xlabel("x")
plt.ylabel("density")
plt.title("Metropolis on bimodal mixture — histogram vs true density")
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_folder}/bimodal_hist_okay.png", dpi=150)
plt.show()

plt.figure(figsize=(9,4))
for name, _ in configs_B:
    x = pd.read_csv(f"{save_folder}/metropolis_bimodal_{name}.csv")["x"].values
    acf_vals = acf_1d(x, max_lag=150)
    plt.plot(acf_vals, label=name)
plt.xlabel("lag")
plt.ylabel("autocorrelation")
plt.title("ACF comparison — bimodal mixture")
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_folder}/bimodal_acf_compare.png", dpi=150)
plt.show()
```


    
![png](/img/contents/post/mcmc-statics/6_metropolis/6_mcmc_metropolis_9_0.png)
    



    
![png](/img/contents/post/mcmc-statics/6_metropolis/6_mcmc_metropolis_9_1.png)
    


## 2D/高维版本：相关高斯
> 直观体会：
> 1. **高维时随机游走 Metropolis 的挑战**；
> 2. **提议分布协方差的缩放对接受率与有效样本数 (ESS) 的影响**。

### 目标分布：二维相关高斯

设目标分布为

$$
\pi(x) = \mathcal N\Big(0, \Sigma\Big),\quad
\Sigma = \begin{bmatrix}1 & 0.8\\0.8 & 1\end{bmatrix}.
$$

这是一个“椭圆”形的二维高斯，主方向在 $y=x$。


### Metropolis 设置

* **提议分布**：对称高斯

  $$
  q(y\mid x) = \mathcal N(x,\, \sigma^2 I).
  $$
* 我们比较三种 $\sigma$：

  * 太小（0.05）
  * 合适（0.5）
  * 太大（2.0）


### 诊断指标

* **接受率**（accepted / 总步数）
* **ESS（有效样本数）**：对每个维度单独算自相关后近似估计
* **轨迹/散点**：观察是否沿椭圆主轴探索
* **自相关函数**：对比不同 $\sigma$ 的衰减速度



```python
import numpy as np
import matplotlib.pyplot as plt

# ---------- 目标分布（二维相关高斯） ----------
Sigma = np.array([[1.0, 0.8],
                  [0.8, 1.0]])
Sigma_inv = np.linalg.inv(Sigma)
Sigma_det = np.linalg.det(Sigma)
d = 2

def log_target(x):
    # log density of N(0, Sigma)
    return -0.5 * x @ Sigma_inv @ x

# ---------- Metropolis 实现 ----------
def metropolis_2d(log_target, x0, sigma, n_samples=20000, burn_in=2000):
    x = np.zeros((n_samples, d))
    x[0] = x0
    accepted = 0
    for t in range(1, n_samples):
        proposal = x[t-1] + sigma * np.random.randn(d)
        log_alpha = log_target(proposal) - log_target(x[t-1])
        if np.log(np.random.rand()) < log_alpha:
            x[t] = proposal
            accepted += 1
        else:
            x[t] = x[t-1]
    return x[burn_in:], accepted/(n_samples-1)

# ---------- 自相关 & ESS ----------
def autocorr(x, lag):
    n = len(x)
    x_mean = np.mean(x)
    num = np.sum((x[:n-lag]-x_mean)*(x[lag:]-x_mean))
    den = np.sum((x-x_mean)**2)
    return num/den

def ess(x):
    # 简单近似 ESS = N / (1 + 2*sum_rho)
    n = len(x)
    acfs = []
    for lag in range(1, 200):  # 截断到200滞后
        r = autocorr(x, lag)
        if r <= 0:
            break
        acfs.append(r)
    tau = 1 + 2*np.sum(acfs)
    return n/tau

# ---------- 运行不同sigma ----------
sigmas = [0.05, 0.5, 2.0]
results = {}

for sigma in sigmas:
    samples, acc_rate = metropolis_2d(log_target, np.zeros(d), sigma)
    ess_x = ess(samples[:,0])
    ess_y = ess(samples[:,1])
    results[sigma] = {
        "samples": samples,
        "acc_rate": acc_rate,
        "ESS_x": ess_x,
        "ESS_y": ess_y
    }

# ---------- 绘图：散点 & 轨迹 ----------
fig, axes = plt.subplots(1, 3, figsize=(15,5))
for ax, sigma in zip(axes, sigmas):
    s = results[sigma]["samples"]
    ax.scatter(s[:,0], s[:,1], s=3, alpha=0.3)
    ax.set_title(f"σ={sigma}, acc={results[sigma]['acc_rate']:.2f}\nESSx={results[sigma]['ESS_x']:.0f}, ESSy={results[sigma]['ESS_y']:.0f}")
    ax.set_xlim(-4,4); ax.set_ylim(-4,4)
plt.suptitle("Metropolis in 2D Correlated Gaussian")
plt.show()

# ---------- 绘制自相关函数对比 (x维度) ----------
plt.figure(figsize=(6,4))
lags = np.arange(50)
for sigma in sigmas:
    s = results[sigma]["samples"][:,0]
    acfs = [autocorr(s, lag) for lag in lags]
    plt.plot(lags, acfs, label=f"σ={sigma}")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation (x-dim)")
plt.title("ACF of Metropolis samples (x dimension)")
plt.legend()
plt.show()

results

```


    
![png](/img/contents/post/mcmc-statics/6_metropolis/6_mcmc_metropolis_11_0.png)
    



    
![png](/img/contents/post/mcmc-statics/6_metropolis/6_mcmc_metropolis_11_1.png)
    





    {0.05: {'samples': array([[-2.16572377, -0.77884803],
             [-2.15834687, -0.90655463],
             [-2.18296029, -0.78398635],
             ...,
             [ 1.51324889,  0.67798398],
             [ 1.50519569,  0.65206217],
             [ 1.50207077,  0.69648709]], shape=(18000, 2)),
      'acc_rate': 0.9588479423971199,
      'ESS_x': np.float64(53.51017982106769),
      'ESS_y': np.float64(53.36453749200187)},
     0.5: {'samples': array([[ 0.22022266,  0.34438253],
             [ 0.166778  ,  0.15202594],
             [ 0.166778  ,  0.15202594],
             ...,
             [-0.66219976, -0.84027925],
             [-0.66219976, -0.84027925],
             [-0.93618224, -1.05758728]], shape=(18000, 2)),
      'acc_rate': 0.6429821491074553,
      'ESS_x': np.float64(306.2402365935577),
      'ESS_y': np.float64(320.2125756843043)},
     2.0: {'samples': array([[ 0.38647484,  0.08333977],
             [ 0.38647484,  0.08333977],
             [ 0.38647484,  0.08333977],
             ...,
             [-0.1488856 ,  0.45357494],
             [-0.1488856 ,  0.45357494],
             [-0.02129455, -0.43894876]], shape=(18000, 2)),
      'acc_rate': 0.1905095254762738,
      'ESS_x': np.float64(1308.3542777464584),
      'ESS_y': np.float64(1325.6521185465317)}}



### 📊 诊断表

| σ (proposal std) | 接受率  | ESS(x) | ESS(y) | 直观表现                                     |
| ---------------- | ---- | ------ | ------ | ---------------------------------------- |
| **0.05** (太小)    | 0.97 | \~53   | \~53   | 接受率极高，但样本走得像“蚂蚁挪步”，自相关极强，ESS 极低。         |
| **0.5** (合适)     | 0.64 | \~350  | \~385  | 接受率和移动幅度均衡，ESS 明显提升，样本沿椭圆充分探索。           |
| **2.0** (太大)     | 0.18 | \~1030 | \~929  | 接受率很低，但每次成功移动都很大，ESS 反而最高；不过链“抖动”，稳定性受限。 |


### 📉 图解说明

1. **散点图**

   * σ=0.05：点云很密集，几乎粘在局部。
   * σ=0.5：点云覆盖椭圆形分布，最合理。
   * σ=2.0：点云分布合理，但轨迹很“跳跃”，很多拒绝（trace 会出现“卡住不动”）。

2. **ACF (x维度)**

   * σ=0.05：ACF 衰减非常慢 → 强相关。
   * σ=0.5：ACF 快速下降 → 较高效率。
   * σ=2.0：ACF 更快下降 → 看似效率高，但接受率低，导致采样不稳定。



### ✅ **直觉总结**

* 高维情况下，**步长缩放**对 MCMC 性能影响更大。
* 太小 → 接受率高但“蚂蚁爬”，ESS 低。
* 太大 → 接受率低，链“卡住不动”。
* 合适区间 → 兼顾接受率和探索能力。


# 实战注意事项

1. **用 log 密度**：永远在 log 域里做加减，避免数值下溢。
2. **Burn-in**：保守一点，前期样本丢掉；但别丢太多浪费。
3. **不要盲目 thinning**：存储允许的前提下保留全部样本，用 ACF/ESS 正确估计方差。
4. **多链检查**：多初值并行跑，看看是否都收敛并混合到同一个稳态（后续你学到 R-hat 等更规范的指标）。
5. **调 $\sigma$**：目标是让**接受率**与**探索幅度**取得平衡（看 ACF/ESS 和图）。


