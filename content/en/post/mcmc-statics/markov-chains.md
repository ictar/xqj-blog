---
title: "Understanding Markov Chains"
slug: "markov-chains"
description: "Learn about Markov processes, stationary distributions, and convergence of Markov chains."
summary: "Learn about Markov processes, stationary distributions, and convergence of Markov chains."
date: 2025-08-21
toc: true
draft: false
tags: ["Markov Chain", "Mathematics", "python"]
---

{{< toc >}}


# Basic Concepts of Markov Chains

## Stochastic Process

### What is a Stewardship Process

* A stochastic process is a collection of random variables $\{X_t: t\in \mathcal{T}\}$ indexed by "time/index".
* $\mathcal{T}$ is the index set: it can be discrete ($t=0,1,2,\dots$) or continuous ($t\in \mathbb{R}_{\ge 0}$).
* Each $X_t$ takes values in a **State Space** $\mathcal{S}$ (discrete or continuous).

### Discrete Time vs Continuous Time

* **Discrete Time (DT)**: $t=0,1,2,\dots$. This section focuses on **Discrete Time Markov Chains (DTMC)**.
* **Continuous Time (CT)**: $t\in\mathbb{R}_{\ge 0}$, corresponding to **Continuous Time Markov Chains (CTMC)**, described by generators rather than transition matrices.


## Markov Property

> **Memorylessness**: The next step depends only on the current state, not on the distant history.

A discrete-time, **homogeneous** Markov chain (time-invariant) is defined as:

$$
\mathbb{P}(X_{t+1}=j \mid X_t=i, X_{t-1},\dots,X_0)=\mathbb{P}(X_{t+1}=j\mid X_t=i)=p_{ij},
$$

where $p_{ij}$ is independent of $t$ (homogeneous). If it is allowed to change with time, it is a **non-homogeneous** Markov chain.

**Corollary (Chapman–Kolmogorov)**: Multi-step transition probabilities satisfy

$$
P^{(n+m)} = P^{(n)}P^{(m)},
$$

In particular, the $n$-step transition matrix $P^{(n)}=P^n$.

## Transition Probability Matrix

### Definition and Properties

* For a finite state space $\mathcal{S}=\{1,\dots, S\}$, define the **Transition Matrix** $P=[p_{ij}]_{S\times S}$, where

  $$
  p_{ij}=\mathbb{P}(X_{t+1}=j\mid X_t=i).
  $$
* **Row-stochastic**: Each row is a probability distribution

  $$
  p_{ij}\ge 0,\quad \sum_{j=1}^S p_{ij}=1\quad(\forall i).
  $$
* Let the distribution vector be a row vector $\pi_t=[\mathbb{P}(X_t=1),\dots,\mathbb{P}(X_t=S)]$, then

  $$
  \pi_{t+1}=\pi_t P,\quad \pi_t=\pi_0 P^t.
  $$

## Examples


```python
import numpy as np
import matplotlib.pyplot as plt

# ============ Basic Tools ============

def is_row_stochastic(P, tol=1e-12):
    """Check if transition matrix is 'row-stochastic' (row sums approx 1, elements >= 0)"""
    P = np.asarray(P, dtype=float)
    nonneg = np.all(P >= -tol) # Check non-negativity
    rowsum_one = np.allclose(P.sum(axis=1), 1.0, atol=1e-10) # Check row sums approx 1
    return bool(nonneg and rowsum_one), P.sum(axis=1)

def n_step_transition(P, n):
    """n-step transition matrix: P^n"""
    return np.linalg.matrix_power(np.asarray(P, dtype=float), n)

def simulate_markov_chain(P, init_state, n_steps, rng=None):
    """
    Simulate a single Markov chain path from an initial state.
    P: row-stochastic matrix; init_state: int (0..S-1); Returns array shape=(n_steps+1,)
    """
    if rng is None:
        rng = np.random.default_rng()
    P = np.asarray(P, dtype=float)
    S = P.shape[0] # Number of states
    path = np.empty(n_steps+1, dtype=int) # Initialize path
    path[0] = int(init_state) # Ensure initial state is int
    for t in range(n_steps): # Generate path step by step
        i = path[t]
        path[t+1] = rng.choice(S, p=P[i]) # Choose next state from current state i
    return path

def simulate_many(P, pi0, n_steps, n_runs=10000, rng=None):
    """
    Simulate many paths, estimate empirical distribution at each time step (compare with theoretical pi0 P^t).
    Returns:
      emp_dist: shape=(n_steps+1, S) Empirical distribution
      th_dist : shape=(n_steps+1, S) Theoretical distribution
    """
    if rng is None:
        rng = np.random.default_rng()
    P = np.asarray(P, dtype=float)
    S = P.shape[0]
    # Theoretical distribution evolution
    th = np.zeros((n_steps+1, S))
    th[0] = pi0
    for t in range(n_steps):
        th[t+1] = th[t] @ P
    # Empirical distribution
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

### Example 1: Two States (Weather Example: Sunny=S, Rainy=R)

#### Model Setup (2-State Weather Chain)

States: 0=Sunny, 1=Rainy
Transition Matrix:

$$
P=\begin{bmatrix}
\text{Sunny→Sunny} & \text{Sunny→Rainy}\\
\text{Rainy→Sunny} & \text{Rainy→Rainy}
\end{bmatrix}=\begin{bmatrix}
0.8 & 0.2\\
0.4 & 0.6
\end{bmatrix}
$$

Meaning: Sunny→Sunny 0.8, Sunny→Rainy 0.2; Rainy→Sunny 0.4, Rainy→Rainy 0.6.

* **Row-stochastic**: Row sums are 1 (valid probability matrix).
* **Irreducible**: Each state can reach the other (both rows contain non-zero cross transitions).
* **Aperiodic**: Diagonal elements $p_{00},p_{11}>0$ (self-loops), period is 1.
  ⇒ **Chain is ergodic**: Unique stationary distribution exists, and converges to it from any initial value.


```python
# ============ Example 1: Two States (Sunny/Rainy) ============

# State encoding: 0=Sunny(S), 1=Rainy(R)
P2 = np.array([[0.8, 0.2],
               [0.4, 0.6]], dtype=float)

ok, rowsums = is_row_stochastic(P2)
print("P2 Row stochasticity check:", ok, " Row sums =", rowsums)

# Single path simulation and visualization
rng = np.random.default_rng(2025)
path = simulate_markov_chain(P2, init_state=0, n_steps=50, rng=rng)

plt.figure(figsize=(9,3))
plt.plot(range(len(path)), path, marker='o')
plt.hlines(0, -1, len(path)-1, colors='green', linestyles='dashed', label="Sunny")
plt.hlines(1, -1, len(path)-1, colors='yellow', linestyles='dashed', label="Rainy")
plt.text(10, -0.4, "Sample Path (State jumping over time)", fontsize=12, color='red')
plt.title("Sample path of 2-state Markov chain (0=Sunny, 1=Rainy)")
plt.xlabel("time t")
plt.ylabel("state")
plt.legend()
plt.show()

# Multi-path statistics vs Theoretical distribution
pi0 = np.array([0.5, 0.5])  # Initial distribution
emp, th = simulate_many(P2, pi0, n_steps=20, n_runs=5000, rng=rng)

# Plot "Probability of Sunny" over time: Theory vs Empirical
plt.figure(figsize=(9,4))
plt.plot(th[:,0], label="theory P(X_t=Sunny)")
plt.plot(emp[:,0], label="empirical P(X_t=Sunny)")
plt.text(3, 0.6, "Theory and empirical match closely (Better with more samples).", fontsize=12, color='red')
plt.title("Distribution evolution in 2-state chain")
plt.xlabel("time t")
plt.ylabel("probability of Sunny")
plt.legend()
plt.show()

# n-step transition matrix example
P2_5 = n_step_transition(P2, 5)
print("P2^5 =\n", np.round(P2_5, 4))
```

    P2 Row stochasticity check: True  Row sums = [1. 1.]



    
![png](/img/contents/post/mcmc-statics/4_markov-chains/4_mcmc_markovchains_6_1.png)
    



    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_6_2.png)
    


    P2^5 =
     [[0.6701 0.3299]
     [0.6598 0.3402]]


### Example 2: Three States (With Absorbing State C)

$$
P_3=
\begin{bmatrix}
0.6 & 0.4 & 0.0\\
0.2 & 0.5 & 0.3\\
0.0 & 0.0 & 1.0
\end{bmatrix},
$$


```python
# ============ Example 2: Three States (With Absorbing State C) ============

# State encoding: 0=A, 1=B, 2=C(Absorbing)
P3 = np.array([[0.6, 0.4, 0.0],
               [0.2, 0.5, 0.3],
               [0.0, 0.0, 1.0]], dtype=float)

ok3, rowsums3 = is_row_stochastic(P3)
print("P3 Row stochasticity check:", ok3, " Row sums =", rowsums3)

path3 = simulate_markov_chain(P3, init_state=0, n_steps=40, rng=rng)

plt.figure(figsize=(9,3))
plt.plot(range(len(path3)), path3, marker='o')
plt.text(10, 1.5, "Sample Path\nState C is never left once entered (Absorbing).", fontsize=12, color='red')
plt.title("Sample path of 3-state chain (2 is absorbing)")
plt.xlabel("time t")
plt.ylabel("state (0=A, 1=B, 2=C)")
plt.show()

# Multi-path stats: Observe prob of absorbing into C over time
pi0_3 = np.array([1.0, 0.0, 0.0])  # Start from A
emp3, th3 = simulate_many(P3, pi0_3, n_steps=20, n_runs=5000, rng=rng)

plt.figure(figsize=(9,4))
plt.plot(th3[:,2], label="theory P(X_t=C)")
plt.plot(emp3[:,2], label="empirical P(X_t=C)")
plt.text(5, 0.2, "Theory and empirical match closely.\nProb of Absorbing State C increases monotonically.", fontsize=12, color='red')
plt.title("Absorption probability into state C over time")
plt.xlabel("time t")
plt.ylabel("probability in C")
plt.legend()
plt.show()

# n-step transition matrix example
P3_5 = n_step_transition(P3, 5)
print("P3^5 =\n", np.round(P3_5, 4))
```

    P3 Row stochasticity check: True  Row sums = [1. 1. 1.]



    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_8_1.png)
    



    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_8_2.png)
    


    P3^5 =
     [[0.242  0.2856 0.4724]
     [0.1428 0.1706 0.6866]
     [0.     0.     1.    ]]


## Summary
* **Stochastic Process**: A family of random variables indexed by time.
* **Discrete/Continuous Time**: This section focuses on Discrete Time Chains.
* **Markov Property**: Next step depends only on current; homogeneous chain transition prob is time-independent.
* **Transition Matrix**: Row-stochastic matrix; Multi-step transition uses power $P^n$; Distribution evolves as $\pi_t=\pi_0 P^t$.
* **Examples**: Two-state (Weather), Three-state (Absorbing) demonstrate basic calculation and simulation.

# Long-term Behavior and Convergence of Markov Chains

## Stationary Distribution

* **Definition**: A probability vector $\pi$, if

  $$
  \pi P = \pi, \quad \sum_i \pi_i = 1, \; \pi_i \geq 0
  $$

  Then $\pi$ is called the **Stationary Distribution** of the Markov chain.

* **Significance**: If the chain's distribution is $\pi$ at some moment, it remains $\pi$ at any subsequent moment. It describes the **Long-term State Distribution**.

## State Classification

* **Reachability**: State $i \to j$ if there exists some $n$ such that $(P^n)_{ij} > 0$.
* **Recurrent/Transient**:

  * **Recurrent**: Starting from $i$, strictly returns to $i$ eventually.
  * **Transient**: There is non-zero probability of never returning.
* **Irreducible**: All states are reachable from each other → Chain is a whole.
* **Periodicity**:

  * Period of state $i$: $\gcd\{ n : (P^n)_{ii} > 0 \}$.
  * If Period = 1, it is **Aperiodic**.


## Ergodic Theorem

* **Theorem**:
  For a finite Markov chain, if it is **Irreducible** and **Aperiodic**, then there exists a unique stationary distribution $\pi$, and:

  $$
  \lim_{n \to \infty} P(X_n = j \mid X_0 = i) = \pi_j \quad \forall i,j
  $$

  Also, time average converges to probability average:

  $$
  \frac{1}{N}\sum_{t=1}^N \mathbf{1}_{\{X_t=j\}} \to \pi_j
  $$

## Mixing Time

> Measure of convergence speed

* **Definition**: Time required for the chain to get close to stationary distribution from initial distribution $\mu$.
* Common Distance: **Total Variation Distance**

  $$
  d(t) = \max_\mu \| \mu P^t - \pi \|_{TV}
  $$
* Mixing Time: Smallest $t$ such that $d(t) \leq \epsilon$.


## Examples

### Example 1: Two-State Markov Chain

> Simplest demo, clearly see convergence to stationary distribution.

Transition Matrix:

$$
P = \begin{bmatrix}
0.9 & 0.1 \\
0.5 & 0.5
\end{bmatrix}
$$

#### (a) Stationary Distribution

Solve equation:

$$
\pi P = \pi
$$

i.e.:

$$
\pi_0 = 0.9\pi_0 + 0.5\pi_1 \quad\Rightarrow\quad 0.1\pi_0 = 0.5\pi_1
$$

Combined with $\pi_0 + \pi_1 = 1$, we get:

$$
\pi = (0.833..., \; 0.166...)
$$


#### (b) Property Analysis

* **Irreducible**. Because two states are reachable from each other.
* **Aperiodic**: Because $P_{00}>0, P_{11}>0$ (self-loops), can stay at original state → Period = 1

In summary, by **Ergodic Theorem**, this finite Markov chain is **Ergodic**, i.e., unique stationary distribution exists and converges to it from any initial value.

#### (c) Convergence Process



```python
import numpy as np
import matplotlib.pyplot as plt

P = np.array([[0.9, 0.1],
              [0.5, 0.5]])

# Initial distribution
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
plt.text(2.5, 0.7, "Start from state 0, prob distribution converges to (0.833, 0.167)", fontsize=12, color='red')
plt.xlabel("Step")
plt.ylabel("Probability")
plt.legend()
plt.title("Convergence to Stationary Distribution")
plt.show()
```


    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_13_0.png)
    


### Example 2: Three-State Markov Chain

> Demonstrates existence and uniqueness of stationary distribution in more complex chains.

Transition Matrix:

$$
P = \begin{bmatrix}
0.5 & 0.5 & 0.0 \\
0.2 & 0.5 & 0.3 \\
0.0 & 0.3 & 0.7
\end{bmatrix}
$$

* **Irreducible**: All states mutually reachable.
* **Aperiodic**: Self-loop prob $P_{ii} > 0$.
* **Stationary Distribution**: Solve $\pi P = \pi$, get unique $\pi$.
* **Long-term Behavior**: All initial distributions converge to $\pi$.


```python
import numpy as np
import matplotlib.pyplot as plt

# 3-State Markov Chain Transition Matrix
P = np.array([[0.5, 0.5, 0.0],
              [0.2, 0.5, 0.3],
              [0.0, 0.3, 0.7]])

# Initial distribution (All in state 0)
mu = np.array([1.0, 0.0, 0.0])

# Calculate Stationary Distribution: Solve pi P = pi
eigvals, eigvecs = np.linalg.eig(P.T)
stat_dist = eigvecs[:, np.isclose(eigvals, 1)]
stat_dist = stat_dist[:,0]
stat_dist = stat_dist / stat_dist.sum()  # Normalize
print("Stationary distribution:", stat_dist.real)

# Iterative distribution evolution
distributions = [mu]
for _ in range(30):
    mu = mu @ P
    distributions.append(mu)

distributions = np.array(distributions)

# Plot
plt.figure(figsize=(8,5))
for i in range(3):
    plt.plot(distributions[:, i], label=f"Pr[state={i}]")
    plt.axhline(stat_dist[i].real, linestyle="--", color="gray")

plt.text(5, 0.8, "Start from all in 0,\nprob distribution converges to this stationary distribution;\nHorizontal dashed lines indicate stationary values.", fontsize=12, color='red')
plt.xlabel("Step")
plt.ylabel("Probability")
plt.title("Convergence of 3-State Markov Chain")
plt.legend()
plt.show()

```

    Stationary distribution: [0.16666667 0.41666667 0.41666667]



    
![png](/img/contents/post/mcmc-statics/4_markov-chains//4_mcmc_markovchains_15_1.png)
    


### Example 3: Random Walk on a Cycle

> Demonstrates effect of **Periodicity** on convergence.

* States: $\{0,1,2,\dots,n-1\}$.

* Transition Rule: From current $i$, move to $(i−1)\bmod n$ with prob 0.5, to $(i+1)\bmod n$ with prob 0.5. i.e.,

  $$
  P(i \to i+1 \bmod n) = 0.5, \quad P(i \to i-1 \bmod n) = 0.5
  $$

* **Irreducible**: Can reach any state from any state.

* **Periodicity**: If $n$ is even, Period=2; If $n$ is odd, Aperiodic.

* **Stationary Distribution**: Uniform distribution $\pi_i = 1/n$.

* **Convergence**:

  * If $n$ odd → Chain is irreducible and aperiodic → Converges to uniform.
  * If $n$ even → Chain has periodicity (Period=2) → Chain oscillates between "odd/even classes", fails to converge to uniform.


```python
import numpy as np
import matplotlib.pyplot as plt

def ring_rw_transition_matrix(n):
    """Generate n-state ring random walk transition matrix"""
    P = np.zeros((n, n))
    for i in range(n):
        P[i, (i-1)%n] = 0.5
        P[i, (i+1)%n] = 0.5
    return P

def simulate_chain(P, steps=30, start_state=0):
    """Compute distribution evolution from single point"""
    n = P.shape[0]
    mu = np.zeros(n)
    mu[start_state] = 1.0
    distributions = [mu]
    for _ in range(steps):
        mu = mu @ P
        distributions.append(mu)
    return np.array(distributions)

# Parameters
steps = 30
P5 = ring_rw_transition_matrix(5)
P6 = ring_rw_transition_matrix(6)

# Simulation
dist5 = simulate_chain(P5, steps)
dist6 = simulate_chain(P6, steps)

# Stationary distribution (Uniform for odd n; Even n doesn't have unique convergence)
pi5 = np.ones(5) / 5
pi6 = np.ones(6) / 6

# Plot
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
    


### Example 4: Numerical Measure of Mixing Time (e.g. total variation distance convergence speed)


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
