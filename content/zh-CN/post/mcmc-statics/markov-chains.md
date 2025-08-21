---
title: "理解马尔可夫链"
slug: "markov-chains"
description: "了解马尔可夫过程，以及马尔可夫链的平稳分布与收敛性"
summary: "了解马尔可夫过程，以及马尔可夫链的平稳分布与收敛性"
date: 2025-08-21
toc: true
draft: false
tags: ["马尔可夫链", "数学", "python"]
---

{{< toc >}}


# 马尔可夫链的基本概念

## 随机过程（Stochastic Process）

### 什么是随机过程

* 随机过程就是一组按“时间/索引”排列的随机变量 $\{X_t: t\in \mathcal{T}\}$。
* $\mathcal{T}$ 是索引集合：可以是离散的（$t=0,1,2,\dots$）也可以是连续的（$t\in \mathbb{R}_{\ge 0}$）。
* 每个 $X_t$ 取值于一个**状态空间** $\mathcal{S}$（可离散/连续）。

### 离散时间 vs 连续时间

* **离散时间（DT）**：$t=0,1,2,\dots$。本节主要讲**离散时间马尔可夫链（DTMC）**。
* **连续时间（CT）**：$t\in\mathbb{R}_{\ge 0}$，对应**连续时间马尔可夫链（CTMC）**，用生成元而非转移矩阵描述。


## 马尔可夫性质（Markov Property）

> **无记忆性**：下一步只取决于当前，不取决于更久远的历史。

离散时间、**齐次**马尔可夫链（时间不变）定义为：

$$
\mathbb{P}(X_{t+1}=j \mid X_t=i, X_{t-1},\dots,X_0)=\mathbb{P}(X_{t+1}=j\mid X_t=i)=p_{ij},
$$

其中 $p_{ij}$ 与 $t$ 无关（齐次）。如果允许随时间变化，就是**非齐次**马尔可夫链。

**推论（Chapman–Kolmogorov）**：多步转移概率满足

$$
P^{(n+m)} = P^{(n)}P^{(m)},
$$

特别地，$n$ 步转移矩阵 $P^{(n)}=P^n$。

## 转移概率矩阵（Transition Matrix）

### 定义与性质

* 对于有限状态空间 $\mathcal{S}=\{1,\dots, S\}$，定义**转移矩阵** $P=[p_{ij}]_{S\times S}$，其中

  $$
  p_{ij}=\mathbb{P}(X_{t+1}=j\mid X_t=i).
  $$
* **行随机（row-stochastic）**：每一行是一组概率

  $$
  p_{ij}\ge 0,\quad \sum_{j=1}^S p_{ij}=1\quad(\forall i).
  $$
* 记分布向量为行向量 $\pi_t=[\mathbb{P}(X_t=1),\dots,\mathbb{P}(X_t=S)]$，则

  $$
  \pi_{t+1}=\pi_t P,\quad \pi_t=\pi_0 P^t.
  $$

## 示例


```python
import numpy as np
import matplotlib.pyplot as plt

# ============ 基础工具 ============

def is_row_stochastic(P, tol=1e-12):
    """检查转移矩阵是否“行随机”（每行和约为1，元素>=0）"""
    P = np.asarray(P, dtype=float)
    nonneg = np.all(P >= -tol) # 检查非负性
    rowsum_one = np.allclose(P.sum(axis=1), 1.0, atol=1e-10) # 检查每行和是否约为1
    return bool(nonneg and rowsum_one), P.sum(axis=1)

def n_step_transition(P, n):
    """n 步转移矩阵：P^n"""
    return np.linalg.matrix_power(np.asarray(P, dtype=float), n)

def simulate_markov_chain(P, init_state, n_steps, rng=None):
    """
    从单个初始状态模拟一条马尔可夫链路径。
    P: 行随机矩阵；init_state: int (0..S-1)；返回数组 shape=(n_steps+1,)
    """
    if rng is None:
        rng = np.random.default_rng()
    P = np.asarray(P, dtype=float)
    S = P.shape[0] # 状态数
    path = np.empty(n_steps+1, dtype=int) # 初始化路径
    path[0] = int(init_state) # 确保初始状态是整数
    for t in range(n_steps): # 逐步生成路径
        i = path[t]
        path[t+1] = rng.choice(S, p=P[i]) # 从当前状态 i 选择下一个状态
    return path

def simulate_many(P, pi0, n_steps, n_runs=10000, rng=None):
    """
    模拟多条路径，估计各时刻的经验分布（与理论 pi0 P^t 对比）。
    返回：
      emp_dist: shape=(n_steps+1, S) 经验分布
      th_dist : shape=(n_steps+1, S) 理论分布
    """
    if rng is None:
        rng = np.random.default_rng()
    P = np.asarray(P, dtype=float)
    S = P.shape[0]
    # 理论分布随时间演化
    th = np.zeros((n_steps+1, S))
    th[0] = pi0
    for t in range(n_steps):
        th[t+1] = th[t] @ P
    # 经验分布
    counts = np.zeros((n_steps+1, S), dtype=int)
    init_states = rng.choice(S, size=n_runs, p=pi0)
    for r in range(n_runs):
        s0 = init_states[r]
        path = simulate_markov_chain(P, s0, n_steps, rng=rng)
        for t in range(n_steps+1):
            counts[t, path[t]] += 1
    emp = counts / n_runs
    return emp, th
```

### 示例 1：两状态（天气示例：晴=S，雨=R）

#### 模型设定（两状态天气链）

状态：0=晴 (Sunny)，1=雨 (Rainy)
转移矩阵：

$$
P=\begin{bmatrix}
\text{晴→晴} & \text{晴→雨}\\
\text{雨→晴} & \text{雨→雨}
\end{bmatrix}=\begin{bmatrix}
0.8 & 0.2\\
0.4 & 0.6
\end{bmatrix}
$$

含义：晴→晴 0.8、晴→雨 0.2；雨→晴 0.4、雨→雨 0.6。

* **行随机**：每行和为 1（合法概率矩阵）。
* **不可约**：每个状态都能到达另一个状态（两行都含非零对向转移）。
* **非周期**：对角元 $p_{00},p_{11}>0$（有自环），周期为 1。
  ⇒ **链是遍历的（ergodic）**：存在唯一平稳分布，且从任意初值都会收敛到它。


```python
# ============ 示例 1：两状态（晴/雨） ============

# 状态编码：0=晴(S), 1=雨(R)
P2 = np.array([[0.8, 0.2],
               [0.4, 0.6]], dtype=float)

ok, rowsums = is_row_stochastic(P2)
print("P2 行随机性检查：", ok, " 行和=", rowsums)

# 单一路径模拟与可视化
rng = np.random.default_rng(2025)
path = simulate_markov_chain(P2, init_state=0, n_steps=50, rng=rng)

plt.figure(figsize=(9,3))
plt.plot(range(len(path)), path, marker='o')
plt.hlines(0, -1, len(path)-1, colors='green', linestyles='dashed', label="Sunny")
plt.hlines(1, -1, len(path)-1, colors='yellow', linestyles='dashed', label="Rainy")
plt.text(10, -0.4, "样本路径（状态随时间跳变的折线）", fontsize=12, color='red')
plt.title("Sample path of 2-state Markov chain (0=Sunny, 1=Rainy)")
plt.xlabel("time t")
plt.ylabel("state")
plt.legend()
plt.show()

# 多路径统计 vs 理论分布
pi0 = np.array([0.5, 0.5])  # 初始分布
emp, th = simulate_many(P2, pi0, n_steps=20, n_runs=5000, rng=rng)

# 画“处于 Sunny 的概率”随时间变化：理论 vs 经验
plt.figure(figsize=(9,4))
plt.plot(th[:,0], label="theory P(X_t=Sunny)")
plt.plot(emp[:,0], label="empirical P(X_t=Sunny)")
plt.text(3, 0.6, "理论分布与模拟经验分布几乎重合（随样本数增大吻合更好）。", fontsize=12, color='red')
plt.title("Distribution evolution in 2-state chain")
plt.xlabel("time t")
plt.ylabel("probability of Sunny")
plt.legend()
plt.show()

# n 步转移矩阵示例
P2_5 = n_step_transition(P2, 5)
print("P2^5 =\n", np.round(P2_5, 4))
```

    P2 行随机性检查： True  行和= [1. 1.]



    
![png](/img/contents/post/mcmc-statics/4_markov-chains/4_mcmc_markovchains_6_1.png)
    



    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_6_2.png)
    


    P2^5 =
     [[0.6701 0.3299]
     [0.6598 0.3402]]


### 示例 2：三状态（含吸收态 C）

$$
P_3=
\begin{bmatrix}
0.6 & 0.4 & 0.0\\
0.2 & 0.5 & 0.3\\
0.0 & 0.0 & 1.0
\end{bmatrix},
$$


```python
# ============ 示例 2：三状态（含吸收态 C） ============

# 状态编码：0=A, 1=B, 2=C(吸收)
P3 = np.array([[0.6, 0.4, 0.0],
               [0.2, 0.5, 0.3],
               [0.0, 0.0, 1.0]], dtype=float)

ok3, rowsums3 = is_row_stochastic(P3)
print("P3 行随机性检查：", ok3, " 行和=", rowsums3)

path3 = simulate_markov_chain(P3, init_state=0, n_steps=40, rng=rng)

plt.figure(figsize=(9,3))
plt.plot(range(len(path3)), path3, marker='o')
plt.text(10, 1.5, "样本路径（状态随时间跳变的折线）\n状态 C 一旦进入就不离开（吸收态）。", fontsize=12, color='red')
plt.title("Sample path of 3-state chain (2 is absorbing)")
plt.xlabel("time t")
plt.ylabel("state (0=A, 1=B, 2=C)")
plt.show()

# 多路径统计：观测吸收到 C 的概率随时间变化
pi0_3 = np.array([1.0, 0.0, 0.0])  # 从 A 起步
emp3, th3 = simulate_many(P3, pi0_3, n_steps=20, n_runs=5000, rng=rng)

plt.figure(figsize=(9,4))
plt.plot(th3[:,2], label="theory P(X_t=C)")
plt.plot(emp3[:,2], label="empirical P(X_t=C)")
plt.text(5, 0.2, "理论分布与模拟经验分布几乎重合（随样本数增大吻合更好）。\n吸收态 C 的概率随时间单调上升", fontsize=12, color='red')
plt.title("Absorption probability into state C over time")
plt.xlabel("time t")
plt.ylabel("probability in C")
plt.legend()
plt.show()

# n 步转移矩阵示例
P3_5 = n_step_transition(P3, 5)
print("P3^5 =\n", np.round(P3_5, 4))
```

    P3 行随机性检查： True  行和= [1. 1. 1.]



    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_8_1.png)
    



    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_8_2.png)
    


    P3^5 =
     [[0.242  0.2856 0.4724]
     [0.1428 0.1706 0.6866]
     [0.     0.     1.    ]]


## 小结
* **随机过程**：按时间索引的一族随机变量。
* **离散/连续时间**：本节以离散时间链为主。
* **马尔可夫性质**：下一步只依赖当前；齐次链的转移概率与时间无关。
* **转移矩阵**：行随机矩阵；多步转移用幂 $P^n$；分布随时间演化 $\pi_t=\pi_0 P^t$。
* **例子**：两状态（天气）、三状态（含吸收态）展示了基本计算与模拟。

# 马尔可夫链的长期行为和收敛性

## 平稳分布 (Stationary Distribution)

* **定义**：一个概率向量 $\pi$，如果

  $$
  \pi P = \pi, \quad \sum_i \pi_i = 1, \; \pi_i \geq 0
  $$

  那么 $\pi$ 称为该马尔可夫链的 **平稳分布**。

* **意义**：如果链在某个时刻的分布是 $\pi$，那么在之后任意时刻仍然保持 $\pi$。它描述了 **长期状态分布**。

## 状态分类

* **可达性 (Reachability)**：状态 $i \to j$ 如果存在某个 $n$，使得 $(P^n)_{ij} > 0$。
* **常返/暂留**：

  * **常返 (Recurrent)**：从 $i$ 出发，最终必然返回 $i$。
  * **暂留 (Transient)**：有非零概率永远不返回。
* **不可约 (Irreducible)**：所有状态两两可达 → 链是一个整体。
* **周期性 (Periodicity)**：

  * 状态 $i$ 的周期：$\gcd\{ n : (P^n)_{ii} > 0 \}$。
  * 若周期 = 1，则为 **非周期 (aperiodic)**。


## 遍历定理 (Ergodic Theorem)

* **定理**：
  对一个有限马尔可夫链，如果它是 **不可约** 且 **非周期**，则存在唯一平稳分布 $\pi$，并且：

  $$
  \lim_{n \to \infty} P(X_n = j \mid X_0 = i) = \pi_j \quad \forall i,j
  $$

  同时，时间平均收敛到概率平均：

  $$
  \frac{1}{N}\sum_{t=1}^N \mathbf{1}_{\{X_t=j\}} \to \pi_j
  $$

## 混合时间 (Mixing Time)

> 收敛速度的度量

* **定义**：链从初始分布 $\mu$ 到接近平稳分布所需的时间。
* 常用距离：**全变差距离 (total variation distance)**

  $$
  d(t) = \max_\mu \| \mu P^t - \pi \|_{TV}
  $$
* 混合时间：最小 $t$，使得 $d(t) \leq \epsilon$。


## 示例

### 示例 1：两状态马尔可夫链

> 最简单的演示，清晰看到收敛到平稳分布。

转移矩阵：

$$
P = \begin{bmatrix}
0.9 & 0.1 \\
0.5 & 0.5
\end{bmatrix}
$$

#### (a) 平稳分布

解方程：

$$
\pi P = \pi
$$

即：

$$
\pi_0 = 0.9\pi_0 + 0.5\pi_1 \quad\Rightarrow\quad 0.1\pi_0 = 0.5\pi_1
$$

结合 $\pi_0 + \pi_1 = 1$，解得：

$$
\pi = (0.833..., \; 0.166...)
$$


#### (b) 性质分析

* **不可约**。因为两个状态互相可达（两行都含非零对向转移）。
* **非周期**：因为 $P_{00}>0, P_{11}>0$（有自环），可保持在原状态 → 周期 = 1

综上，根据**遍历定理**，该有限马尔可夫链是**遍历的（ergodic）**，即存在唯一平稳分布，且从任意初值都会收敛到它。

#### (c) 收敛过程



```python
import numpy as np
import matplotlib.pyplot as plt

P = np.array([[0.9, 0.1],
              [0.5, 0.5]])

# 初始分布
mu = np.array([1.0, 0.0])  

distributions = [mu]
for _ in range(20):
    mu = mu @ P
    distributions.append(mu)

distributions = np.array(distributions)

plt.plot(distributions[:,0], label="Pr[state=0]")
plt.plot(distributions[:,1], label="Pr[state=1]")
plt.axhline(0.833, color="gray", linestyle="--", label="π0")
plt.axhline(0.167, color="gray", linestyle="--", label="π1")
plt.text(2.5, 0.7, "从状态 0 出发，概率分布逐步收敛到 (0.833, 0.167)", fontsize=12, color='red')
plt.xlabel("Step")
plt.ylabel("Probability")
plt.legend()
plt.title("Convergence to Stationary Distribution")
plt.show()
```


    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_13_0.png)
    


### 示例 2：三状态马尔可夫链

> 展示了更复杂链中平稳分布的存在性和唯一性。

转移矩阵：

$$
P = \begin{bmatrix}
0.5 & 0.5 & 0.0 \\
0.2 & 0.5 & 0.3 \\
0.0 & 0.3 & 0.7
\end{bmatrix}
$$

* **不可约**：所有状态可互相到达。
* **非周期**：存在自循环概率 $P_{ii} > 0$。
* **平稳分布**：解 $\pi P = \pi$，得到唯一 $\pi$。
* **长期行为**：所有初始分布都会收敛到 $\pi$。


```python
import numpy as np
import matplotlib.pyplot as plt

# 三状态马尔可夫链转移矩阵
P = np.array([[0.5, 0.5, 0.0],
              [0.2, 0.5, 0.3],
              [0.0, 0.3, 0.7]])

# 初始分布（全部在状态0）
mu = np.array([1.0, 0.0, 0.0])

# 计算平稳分布：解 pi P = pi
eigvals, eigvecs = np.linalg.eig(P.T)
stat_dist = eigvecs[:, np.isclose(eigvals, 1)]
stat_dist = stat_dist[:,0]
stat_dist = stat_dist / stat_dist.sum()  # 归一化
print("Stationary distribution:", stat_dist.real)

# 迭代分布演化
distributions = [mu]
for _ in range(30):
    mu = mu @ P
    distributions.append(mu)

distributions = np.array(distributions)

# 绘图
plt.figure(figsize=(8,5))
for i in range(3):
    plt.plot(distributions[:, i], label=f"Pr[state={i}]")
    plt.axhline(stat_dist[i].real, linestyle="--", color="gray")

plt.text(5, 0.8, "从初始状态全在 0 出发，\n随着步数增加，概率分布逐渐收敛到这组平稳分布；\n横虚线表示平稳分布值。", fontsize=12, color='red')
plt.xlabel("Step")
plt.ylabel("Probability")
plt.title("Convergence of 3-State Markov Chain")
plt.legend()
plt.show()

```

    Stationary distribution: [0.16666667 0.41666667 0.41666667]



    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_15_1.png)
    


### 示例 3：环形随机游走 (Random Walk on a Cycle)

> 展示了 **周期性** 对收敛性的影响。

* 状态：$\{0,1,2,\dots,n-1\}$。

* 转移规则：从当前位置 $i$，以概率 0.5 移动到 $(i−1)\bmod n$，以概率 0.5 移动到 $(i+1)\bmod n$。即

  $$
  P(i \to i+1 \bmod n) = 0.5, \quad P(i \to i-1 \bmod n) = 0.5
  $$

* **不可约**：从任意状态可到任意状态。

* **周期性**：如果 $n$ 是偶数，则周期 = 2；如果 $n$ 是奇数，则非周期。

* **平稳分布**：均匀分布 $\pi_i = 1/n$。

* **收敛性**：

  * 若 $n$ 奇数 → 链是不可约且非周期 → 收敛到均匀分布。
  * 若 $n$ 偶数 → 链有周期性（周期=2） → 链会在“奇数/偶数类”之间来回跳，无法收敛到均匀分布。


```python
import numpy as np
import matplotlib.pyplot as plt

def ring_rw_transition_matrix(n):
    """生成 n 状态的环形随机游走转移矩阵"""
    P = np.zeros((n, n))
    for i in range(n):
        P[i, (i-1)%n] = 0.5
        P[i, (i+1)%n] = 0.5
    return P

def simulate_chain(P, steps=30, start_state=0):
    """从单点分布开始，计算分布演化"""
    n = P.shape[0]
    mu = np.zeros(n)
    mu[start_state] = 1.0
    distributions = [mu]
    for _ in range(steps):
        mu = mu @ P
        distributions.append(mu)
    return np.array(distributions)

# 参数
steps = 30
P5 = ring_rw_transition_matrix(5)
P6 = ring_rw_transition_matrix(6)

# 模拟
dist5 = simulate_chain(P5, steps)
dist6 = simulate_chain(P6, steps)

# 平稳分布（对于奇数 n，均匀分布；偶数情况不存在唯一收敛）
pi5 = np.ones(5) / 5
pi6 = np.ones(6) / 6

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(8,8), sharex=True)

# n=5
for i in range(5):
    axes[0].plot(dist5[:, i], label=f"state {i}")
axes[0].hlines(pi5, 0, steps, colors="gray", linestyles="--", linewidth=1)
axes[0].set_title("Ring Random Walk (n=5, odd → converges to uniform)")
axes[0].set_ylabel("Probability")
axes[0].legend()

# n=6
for i in range(6):
    axes[1].plot(dist6[:, i], label=f"state {i}")
axes[1].hlines(pi6, 0, steps, colors="gray", linestyles="--", linewidth=1)
axes[1].set_title("Ring Random Walk (n=6, even → oscillates)")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Probability")
axes[1].legend()

plt.tight_layout()
plt.show()

```


    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_17_0.png)
    


### 示例 4: 混合时间 (Mixing Time) 的数值度量（比如 total variation distance 收敛速度）


```python
# Simulate Markov chains and compute total variation distance (TV) to the stationary distribution.
# We will:
# 1. Define several transition matrices (fast/slow 3-state, cycle random walks n=5 and n=6).
# 2. Compute TV distance over time starting from state 0.
# 3. Compute mixing times tau(epsilon) for epsilons = [0.1, 0.01, 0.001].
# 4. Plot TV vs time for comparisons and show a table of mixing times.
#
# Note: Plots use matplotlib (no seaborn) and each figure is a single plot as requested.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def stationary_from_P(P):
    # solve pi = pi P  with sum(pi)=1  -> transpose eigenvector of P^T for eigenvalue 1
    w, v = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(w - 1.0))
    pi = np.real(v[:, idx])
    pi = pi / pi.sum()
    pi = np.maximum(pi, 0)
    pi = pi / pi.sum()
    return pi

def tv_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

def tv_curve(P, p0, t_max):
    n = P.shape[0]
    pis = stationary_from_P(P)
    p = p0.copy()
    tvs = []
    for t in range(t_max + 1):
        tvs.append(tv_distance(p, pis))
        p = p @ P
    return np.array(tvs), pis

def mixing_time_from_tvs(tvs, eps):
    # minimal t such that tvs[t] <= eps
    below = np.where(tvs <= eps)[0]
    return int(below[0]) if below.size > 0 else np.nan

# Define chains
# 3-state fast chain
P_fast = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6]
])

# 3-state slow chain (more "sticky" on state 0)
P_slow = np.array([
    [0.9, 0.08, 0.02],
    [0.2, 0.7, 0.1],
    [0.15, 0.15, 0.7]
])

# Cycle random walk
def cycle_P(n):
    P = np.zeros((n, n))
    for i in range(n):
        P[i, (i+1) % n] = 0.5
        P[i, (i-1) % n] = 0.5
    return P

P_cycle5 = cycle_P(5)
P_cycle6 = cycle_P(6)

# initial distribution: start at state 0
def e0(n): 
    v = np.zeros(n); v[0]=1.0; return v

t_max = 200

# compute tv curves
tvs_fast, pi_fast = tv_curve(P_fast, e0(3), t_max)
tvs_slow, pi_slow = tv_curve(P_slow, e0(3), t_max)

tvs_c5, pi_c5 = tv_curve(P_cycle5, e0(5), t_max)
tvs_c6, pi_c6 = tv_curve(P_cycle6, e0(6), t_max)

# compute mixing times for selected epsilons
epsilons = [1e-1, 1e-2, 1e-3]
rows = []
for name, tvs in [
    ("3-state fast", tvs_fast),
    ("3-state slow", tvs_slow),
    ("cycle n=5", tvs_c5),
    ("cycle n=6", tvs_c6)
]:
    entry = {"chain": name}
    for eps in epsilons:
        entry[f"tau({eps})"] = mixing_time_from_tvs(tvs, eps)
    rows.append(entry)

df_mix = pd.DataFrame(rows)

# Plot 1: 3-state fast vs slow
plt.figure(figsize=(8,4))
plt.plot(tvs_fast, label='3-state fast')
plt.plot(tvs_slow, label='3-state slow')
plt.yscale('log')  # show both fast and slow clearly on log scale
plt.xlabel('time t')
plt.ylabel('TV distance (log scale)')
plt.title('Total Variation distance: 3-state fast vs slow (start at state 0)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: cycle n=5 vs n=6
plt.figure(figsize=(8,4))
plt.plot(tvs_c5, label='cycle n=5')
plt.plot(tvs_c6, label='cycle n=6')
plt.yscale('log')
plt.xlabel('time t')
plt.ylabel('TV distance (log scale)')
plt.title('Total Variation distance: cycle random walk n=5 vs n=6 (start at state 0)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Also print stationary distributions for reference
pi_table = pd.DataFrame({
    "chain": ["3-state fast", "3-state slow", "cycle n=5", "cycle n=6"],
    "stationary": [pi_fast, pi_slow, pi_c5, pi_c6]
})
pi_table['stationary_str'] = pi_table['stationary'].apply(lambda x: np.array2string(x, precision=4, separator=', '))
pi_table = pi_table[['chain','stationary_str']]
display("Stationary distributions", pi_table)

```


    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_19_0.png)
    



    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_19_1.png)
    



    'Stationary distributions'



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chain</th>
      <th>stationary_str</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3-state fast</td>
      <td>[0.2857, 0.4286, 0.2857]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3-state slow</td>
      <td>[0.6466, 0.2328, 0.1207]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cycle n=5</td>
      <td>[0.2, 0.2, 0.2, 0.2, 0.2]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cycle n=6</td>
      <td>[0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]</td>
    </tr>
  </tbody>
</table>
</div>

