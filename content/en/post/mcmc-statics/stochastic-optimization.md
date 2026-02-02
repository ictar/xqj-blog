---
title: "Stochastic Optimization Explained: Simulated Annealing & Pincus Theorem"
slug: "stochastic-optimization"
date: 2026-02-02
summary: "When optimization problems are trapped in the maze of local optima, deterministic algorithms are often helpless. This article takes you into the world of stochastic optimization, exploring how to transform the problem of finding minimum energy into finding maximum probability. We will delve into the physical intuition and mathematical principles of the Simulated Annealing algorithm, demonstrate its elegant mechanism of 'high-temperature exploration, low-temperature locking' through dynamic visualization, and derive the Pincus Theorem in detail, mathematically proving why the annealing algorithm can find the global optimal solution."
tags: ["Stochastic Optimization", "Simulated Annealing", "Optimization Algorithms", "Machine Learning", "Pincus Theorem", "Python Implementation"]
keywords: ["Stochastic Optimization", "Simulated Annealing", "Global Optimum", "Non-convex Optimization", "Pincus Theorem"]
series: ["MCMC"]
toc: true
draft: false
---

# Stochastic Optimization

> In an environment full of uncertainty (noise) or extreme complexity (non-convexity), how do we use "randomness" to find the best solution?

## From Deterministic to Stochastic Optimization

### Problem Definition: Starting from the Deterministic World

It all starts with a classic optimization puzzle. Minimize a system's state, described mathematically as:
$$\min_{x \in Q} E(x) = m$$
Where:
- $x$ is the parameter we are looking for (e.g., model weights, molecular configuration).
- $E(x)$ is our energy function (called Loss Function in machine learning). The goal is to make it as small as possible.
- $m$ is the theoretical global minimum.

In the world of Deterministic Methods, we usually move step by step along the gradient direction, like a blind person walking down a hill. This is effective in simple terrains (convex functions), but in complex real-world problems, we easily get trapped in "pits" of local optima.

### Welcome to Stochastic World

To escape the trap of local optima, we use a key transformation: **We convert the problem of "finding minimum energy" into "finding maximum probability".**

This transformation is based on the **[Boltzmann Distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution)** in physics. We define a new probability density function (PDF):
$$f(x) = A e^{-E(x)}$$

Where:
- $f(x)$: Probability density function, must be positive.
- $A = \frac{1}{\int e^{-E(x)} \, dx}$ is the **Normalization Constant**.
  - In statistical physics, the reciprocal of $A$ has a famous name: Partition Function, usually denoted by $Z$. $$Z = \frac{1}{A} = \int e^{-E(x)} \, dx$$

There is an ingenious correspondence here:
- The smaller $E(x)$ (lower energy, which is what we want).
- The larger $e^{-E(x)}$.
- This means the **minimum point** of $E(x)$ corresponds exactly to the **peak (maximum)** of the probability distribution $f(x)$.

Thus, the original problem corresponds to:
$$\min E(x) \iff \max f(x)$$

#### Why make this transformation?

Because in the stochastic world, we are no longer obsessed with "going down every step", but view the solution space as a probability field. We allow the algorithm to accept "bad results" with a certain probability, and it is this mechanism that gives us the opportunity to jump out of local traps.

#### Introducing Temperature Parameter $\lambda$
To control the search process, we introduce a crucial parameter $\lambda$. So the probability density function becomes:
$$f(x, \lambda) = A_\lambda e^{-\lambda E(x)}$$
Where $\lambda$ is positive and inversely proportional to temperature $T$: $$\lambda = \frac{1}{T} \ge 0$$

Thus, $A$ becomes $A = \frac{1}{\int e^{-\lambda E(x)} \, dx}$

This $\lambda$ (or $T$) acts like a regulator, determining the "resolution" or "contrast" of the terrain. We can change the shape of the probability distribution $f(x, \lambda)$ by adjusting it.

#### Two Extreme States: Exploration and Exploitation
By analyzing the limit cases of $\lambda$, we can perfectly reveal the mechanism of stochastic optimization:

**State A: High Temperature Mode**

When $\lambda \to 0$ (meaning $T \to \infty$):
- Mathematically: $-\lambda E(x) \to 0$, leading to $e^{-\lambda E(x)} \to 1$.
- Result: $f(x, \lambda) \to \text{Constant}$.
- Physical Picture: The probability distribution tends to be uniform throughout the space. Regardless of $E(x)$, the probability of sampling any point is almost equal.
- Significance: This is the **Exploration** phase. The algorithm moves violently in space like gas molecules, easily crossing any high mountains and deep valleys, ensuring we don't miss the region where the global optimum is located.

**State B: Low Temperature Mode**

When $\lambda \to \infty$ (meaning $T \to 0$):
- Mathematically: Differences are amplified infinitely. As long as $E(x)$ is slightly larger, $e^{-\lambda E(x)}$ decays extremely fast.
- Result: The probability distribution becomes a sharp peak (similar to a Dirac $\delta$ function), having value only at the lowest energy point $m$.
- Physical Picture: The system "freezes".
- Significance: This is the **Exploitation** phase. The algorithm locks onto the lowest point of the current region and stops running around, thus obtaining a high-precision solution.


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- 1. Define Energy Function ---
# We design an asymmetric double well:
# A deep pit (global optimum), a shallow pit (local optimum)
def energy_function(x):
    # (x^2 - 1)^2 is standard W-shaped double well
    # + 0.3*x tilts it, making the left pit deeper than the right
    return (x**2 - 1)**2 + 0.3 * x

# --- 2. Prepare Data ---
x = np.linspace(-2.5, 2.5, 500)
E = energy_function(x)

# Find true global min for plotting
min_idx = np.argmin(E)
global_min_x = x[min_idx]
global_min_y = E[min_idx]

# --- 3. Setup Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

# Top: Energy Landscape E(x) - Static
ax1.plot(x, E, 'k-', linewidth=3, label='Energy $E(x)$')
ax1.scatter(global_min_x, global_min_y, color='gold', s=150, zorder=5, edgecolors='k', label='Global Min')
ax1.set_title("The Problem: Energy Landscape $E(x)$", fontsize=14)
ax1.set_ylabel("Energy")
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(-2.5, 2.5)

# Bottom: Probability Distribution f(x) - Changes with T
line, = ax2.plot([], [], 'r-', linewidth=3, alpha=0.8)
fill_poly = ax2.fill_between(x, np.zeros_like(x), np.zeros_like(x), color='red', alpha=0.3)
ax2.set_xlim(-2.5, 2.5)
ax2.set_ylabel("Probability Density $f(x)$")
ax2.set_xlabel("x")
ax2.grid(True, alpha=0.3)

# Dynamic Text: Lambda and Temperature
text_info = ax2.text(0.05, 0.9, '', transform=ax2.transAxes, fontsize=12, 
                     bbox=dict(facecolor='white', alpha=0.8))

# --- 4. Animation Logic ---
# Lambda from 0.1 (High T) to 15.0 (Low T)
# Log scale to show high T phase slowly, low T phase quickly
lambdas = np.logspace(np.log10(0.1), np.log10(15.0), 100)

def init():
    line.set_data([], [])
    return line,

def update(frame_lambda):
    global fill_poly
    
    # === Core Physics ===
    # 1. Boltzmann Factor (Unnormalized Prob)
    # e^(-lambda * E)
    unnormalized_prob = np.exp(-frame_lambda * E)
    
    # 2. Normalization Constant A (Partition Function Z)
    # Numerical integration using trapezoidal rule
    integral_Z = np.trapezoid(unnormalized_prob, x)
    A = 1.0 / integral_Z
    
    # 3. Final PDF f(x)
    # f(x) = A * e^(-lambda * E)
    pdf = A * unnormalized_prob
    # =================
    
    # Update curve
    line.set_data(x, pdf)
    
    # Update fill (remove old, add new)
    fill_poly.remove()
    fill_poly = ax2.fill_between(x, 0, pdf, color='red', alpha=0.3)
    
    # Dynamically adjust Y limit (peaks get higher)
    ax2.set_ylim(0, np.max(pdf) * 1.2)
    
    # Update title and text
    T = 1 / frame_lambda
    ax2.set_title(f"The Stochastic Solution: Probability Distribution", fontsize=14)
    text_info.set_text(f"$\lambda$ = {frame_lambda:.2f} (Inverse Temp)\n$T$ = {T:.2f} (Temperature)")
    
    return line, fill_poly, text_info

# --- 5. Generate and Save ---
print("Generating animation...")
ani = FuncAnimation(fig, update, frames=lambdas, init_func=init, blit=False)

# Save as GIF
ani.save('simulated_annealing.gif', writer=PillowWriter(fps=15))
print("✅ Saved as 'simulated_annealing.gif'")
```

![](/img/contents/post/mcmc-statics/10_stochastic_optimization/simulated_annealing.gif)

This is a classic **"Double Well"** energy function:
- It has a global optimum (deep pit).
- And a local optimum (shallow pit).

At high temperatures, the probability distribution covers both pits; as temperature drops ($\lambda$ increases), the probability distribution gradually "abandons" the local optimum and squeezes entirely into the "peak" of the global optimum.

## From MCMC to Optimization: The Art of Annealing (The Bridge: Simulated Annealing)

**Simulated Annealing (SA)** is a general probabilistic optimization algorithm.
- Name Origin: From the "annealing" process in metallurgy.
  - Physical Annealing: Heat metal to high temp (atoms move freely), then cool slowly. Atoms have time to find the lowest energy crystal structure, making metal hard and defect-free.
  - Algorithmic Annealing: Throw parameter $x$ into high temp (random walk), then slowly lower $T$. $x$ has time to escape local optima and settle in global optimum.
- Core Feature: It is an algorithm that **"allows regret"**. During search, it accepts not only "good" solutions but also "bad" solutions with some probability (to jump out of pits).

Steepest Descent mentioned before is "snobbish", only going downhill. If the terrain is like an egg carton (countless small pits), gradient descent dies in the first pit it falls into.

SA's advantages:
- Global Search Capability: Because it accepts "bad solutions" at high T, it can climb over hills to explore unknown territories.
- Gradient-Free: No derivatives needed ($f(x)$ can even be discontinuous).
- Versatile: Works on any ugly function as long as you can calculate the value.
  - Applicable even if $f(x)$ is non-convex.

### Algorithm Flow

1. **Transform**: We don't solve $\min E(x)$ directly, but construct a probability distribution: $$f(x) \propto e^{-E(x)/T}$$
   - Here $\lambda = 1/T$.
   - Intuition: Turn "terrain height" into "probability density". Deeper pit ($E$ small) -> higher probability. Higher mountain ($E$ large) -> lower probability.
2. **High Temperature Exploration** (High T Sampling):
   - Action: Set a high initial temperature $T_{max}$.
   - Phenomenon:
     - When $T$ is large, $E(x)/T \approx 0$, so $e^0 \approx 1$.
     - The whole probability distribution is Flat, close to uniform.
     - Your "particle" (sample) will run around the map (since probability of going anywhere is similar), easily crossing mountains and escaping local traps.
3. **Cooling / Annealing**: Slowly lower temperature $T$ (increase $\lambda$).
   - Action: Lower $T$ according to a schedule.
     - Theoretical: $T(t) \sim \frac{c}{\log(1+t)}$ (Guarantees global opt, but strictly slow).
     - Practical: $T_{new} = T_{old} \times 0.99$.
   - Phenomenon: As T drops, distribution "deforms". Flat areas get lower, deep pits get deeper (peaks get sharper).
4. **Low Temperature Exploitation**: Loop till Low T.
   - Action: Continue "Sample -> Cool -> Sample" until $T$ is very low.
     - ⚠️ But not 0, or division is undefined.
   - Phenomenon: Probability distribution becomes a needle (Dirac Delta). Particle is essentially "locked" in the deepest pit.
5. **Sample & Average**:
   - Action: Collect samples $x_1, \dots, x_n$ at low temp stage, calculate their mean.
   - Conclusion: This mean $\bar{x}$ is our estimated global minimum $x_{min}$.

> Note:
> - At first $T$ is huge, $\Delta E / T$ small, $P$ near 1. Algorithm accepts almost all bad moves (Crazy run).
> - Later $T$ is tiny, $\Delta E / T$ huge, $P$ near 0. Algorithm rejects almost all bad moves (Becomes gradient descent).

#### Examples

##### 1D Example
Using a function with multiple traps to show "Temperature Control" + "Metropolis Sampling": $E(x) = x^2 - \cos(\pi x)$.


```python
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Objective Function ---
def E(x):
    return x**2 - np.cos(np.pi * x)

# --- 2. Sampling Core: Metropolis Criterion ---
# This is a "sampler", generates sample based on current T
def sample_one_step(x_curr, T):
    # a. Propose: Random step left or right
    x_next = x_curr + np.random.uniform(-0.5, 0.5)
    
    # b. Delta E
    dE = E(x_next) - E(x_curr)
    
    # c. Accept/Reject
    # Core Logic: If new energy is lower, go; if higher, go with prob (depends on T)
    if dE < 0 or np.random.rand() < np.exp(-dE / T):
        return x_next # Accept
    else:
        return x_curr # Reject

# --- Main Flow ---
def run_stochastic_optimization():
    x = -2.5       # Start (Deliberately far local opt)
    T = 10.0       # T_max
    T_min = 0.01   # T_min
    alpha = 0.99   # Cooling rate
    
    path = []      
    temps = []     
    
    print(f"{'Step':<6} | {'Temp':<8} | {'Current x':<10} | {'Action'}")
    print("-" * 45)
    
    step = 0
    # Flow 4: Till very low T
    while T > T_min:
        
        # Flow 2 & 3: Sample & Cool
        x = sample_one_step(x, T)
        
        path.append(x)
        temps.append(T)
        
        # Print intermediate
        if step % 200 == 0:
            status = "Explore 🎲" if T > 1.0 else "Exploit 🎯"
            print(f"{step:<6} | {T:<8.4f} | {x:<10.4f} | {status}")
            
        T = T * alpha 
        step += 1
        
    # Flow 5: Avg Samples
    final_samples = path[-100:]
    estimated_min = np.mean(final_samples)
    
    print("-" * 45)
    print(f"✅ Final Estimate: x = {estimated_min:.4f}")
    print(f"✅ True Min: x = 0.0000")
    
    return path, temps

# --- Run & Visualize ---
path, temps = run_stochastic_optimization()

plt.figure(figsize=(10, 6))
plt.plot(path, alpha=0.6, label='Particle Path')
plt.xlabel('Iterations (Time)')
plt.ylabel('Position x')
plt.title('Algorithm Flow: From Exploration (High T) to Exploitation (Low T)')
plt.axhline(0, color='r', linestyle='--', label='Global Min (x=0)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

    Step   | Temp     | Current x  | Action
    ---------------------------------------------
    0      | 10.0000  | -2.5000    | Explore 🎲
    200    | 1.3398   | 2.2058     | Explore 🎲
    400    | 0.1795   | 0.2473     | Exploit 🎯
    600    | 0.0241   | -0.0782    | Exploit 🎯
    ---------------------------------------------
    ✅ Final Estimate: x = -0.0115
    ✅ True Min: x = 0.0000

    
![png](/img/contents/post/mcmc-statics/10_stochastic_optimization/10_mcmc_stochastic_optimization_5_1.png)
    

Graph Analysis:
- First half (Left): Curve oscillates violently. High temp exploration, particle doesn't care about pits, jumping everywhere.
- Second half (Right): Curve becomes a straight line. Low temp locking, particle trapped near $x=0$.
- Result: Averaging the "static" points at the end gives precise minimum.

##### N-Dimensional Example

Using the famous **Rastrigin Function**. It's notorious: macroscopically a bowl (global opt exists), microscopically full of pits (egg carton).

Formula ($A=10$):
$$f(\mathbf{x}) = 10n + \sum_{i=1}^n (x_i^2 - 10 \cos(2\pi x_i))$$
Global min at origin $\mathbf{x} = [0, \dots, 0]$, value 0.


```python
import numpy as np
import matplotlib.pyplot as plt

# --- 1. N-D Objective (Rastrigin) ---
def rastrigin(x):
    # x is vector
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# --- 2. N-D Proposal ---
def get_neighbor(x_curr, step_size=0.5):
    # Key: Random walk in N-D space
    # size=len(x_curr) ensures dimension match
    perturbation = np.random.uniform(-step_size, step_size, size=len(x_curr))
    return x_curr + perturbation

# --- 3. Simulated Annealing Main ---
def simulated_annealing_nd(n_dim=2, n_iter=2000):
    # Init: Random start in range [-5.12, 5.12]
    current_x = np.random.uniform(-5.12, 5.12, size=n_dim)
    current_E = rastrigin(current_x)
    
    # Track Best So Far
    best_x = current_x.copy()
    best_E = current_E
    
    # Temps
    T = 100.0
    T_min = 1e-4
    alpha = 0.99 
    
    path = [current_x] 
    energy_history = [current_E]

    print(f"Start {n_dim}-D Optimization...")
    print(f"Start: {np.round(current_x, 2)}, Energy: {current_E:.2f}")

    iter_count = 0
    while T > T_min and iter_count < n_iter:
        # 1. Propose
        new_x = get_neighbor(current_x)
        new_x = np.clip(new_x, -5.12, 5.12) # Clip to domain
        
        new_E = rastrigin(new_x)
        
        # 2. Delta E
        dE = new_E - current_E
        
        # 3. Metropolis Criterion
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            current_x = new_x
            current_E = new_E
            
            if current_E < best_E:
                best_x = current_x.copy()
                best_E = current_E
        
        path.append(current_x)
        energy_history.append(current_E)
        
        T *= alpha
        iter_count += 1
        
    print(f"End. Final Loc: {np.round(best_x, 4)}")
    print(f"Final Energy: {best_E:.6f} (Theoretical 0.0)")
    
    return np.array(path), energy_history, best_x

# --- Run ---
DIMENSION = 2
path, energies, final_sol = simulated_annealing_nd(n_dim=DIMENSION, n_iter=3000)

# --- 4. Visualize (2D only) ---
if DIMENSION == 2:
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Terrain & Path
    plt.subplot(1, 2, 1)
    x_grid = np.linspace(-5.12, 5.12, 100)
    y_grid = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = rastrigin(np.array([X[i,j], Y[i,j]]))
            
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Energy')
    
    # Path: White start, Red end
    plt.plot(path[:, 0], path[:, 1], 'w-', linewidth=0.5, alpha=0.6)
    plt.scatter(path[0, 0], path[0, 1], c='white', s=50, label='Start')
    plt.scatter(final_sol[0], final_sol[1], c='red', marker='*', s=200, label='End')
    plt.legend()
    plt.title(f"2D Rastrigin Optimization Path\n(Escaping many local minima)")
    
    # Subplot 2: Energy Drop
    plt.subplot(1, 2, 2)
    plt.plot(energies)
    plt.yscale('log') 
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Log Scale)')
    plt.title('Energy Minimization Process')
    plt.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.show()
```

    Start 2-D Optimization...
    Start: [-0.04  4.85], Energy: 28.08
    End. Final Loc: [1.0266 2.0164]
    Final Energy: 5.312337 (Theoretical 0.0)

    
![png](/img/contents/post/mcmc-statics/10_stochastic_optimization/10_mcmc_stochastic_optimization_8_1.png)
    

1. Key code: `perturbation = np.random.uniform(-step_size, step_size, size=len(x_curr))`. This is the core of N-D extension. We generate an N-D random vector at once.
2. Terrain (Left): Many deep blue circles, each is a local trap.
   - **Gradient Descent**: Likely falls into the nearest blue circle and dies.
   - **Simulated Annealing**: White path scurries around (especially early on). Jumps in, jumps out, until T drops, sucked into the middle deepest pit (Red Star).

### Proof of Correctness: Pincus Theorem
> Pincus Theorem provides the **Theoretical Guarantee** for SA convergence.

#### Pincus Theorem (Mark Pincus, 1968)

Pincus Theorem is a bridge proving: When we lower T to extremely low ($\lambda \to \infty$), the "weighted average" (expectation) of a function converges to its "global minimum point".

It turns an **"Optimization Problem" into an "Integration Problem"**.

Assume objective $f(x)$ on domain $D$, global min $x^*$. Pincus Theorem states:
$$x^* = \lim_{\lambda \to \infty} \frac{\int_D x \cdot e^{-\lambda f(x)} \, dx}{\int_D e^{-\lambda f(x)} \, dx}$$
Or in expectation form:
$$x^* = \lim_{\lambda \to \infty} \mathbb{E}_{\lambda}[x]$$

#### Proof

Goal: Limit of fraction:
$$\langle x \rangle_\lambda = \frac{\int x \cdot e^{-\lambda E(x)} \, dx}{\int e^{-\lambda E(x)} \, dx}$$
Prove finding $x^*$.

**Step 1: Extract "Greatest Common Divisor"**

Assume unique global min $x^*$. For any $x \neq x^*$, $E(x) > E(x^*)$.

Factor out $e^{-\lambda E(x^*)}$ from numerator and denominator.
- Denom: $\int e^{-\lambda E(x)} \, dx = e^{-\lambda E(x^*)} \int e^{-\lambda (E(x) - E(x^*))} \, dx$
- Num: $\int x e^{-\lambda E(x)} \, dx = e^{-\lambda E(x^*)} \int x \cdot e^{-\lambda (E(x) - E(x^*))} \, dx$

Substitute back, $e^{-\lambda E(x^*)}$ cancels out!
$$\langle x \rangle_\lambda = \frac{\int x \cdot e^{-\lambda (E(x) - E(x^*))} \, dx}{\int e^{-\lambda (E(x) - E(x^*))} \, dx}$$

Now exponent is $-\lambda (E(x) - E(x^*))$. Note $\Delta E = E(x) - E(x^*) \ge 0$.

**Step 2: Split Integration Domain (Neighborhood vs. Far Away)**

Split domain into:
1. Small neighborhood $U_\epsilon$ around min: $|x - x^*| < \epsilon$.
2. Rest $R$: Far away.

As $\lambda \to \infty$:
- Region $R$: $\Delta E \ge \delta > 0$. Term $e^{-\lambda \Delta E} \le e^{-\lambda \delta}$. Decays exponentially to 0. Contributions from far regions are negligible.
- Neighborhood $U_\epsilon$: $\Delta E \approx 0$, so $e^{-\lambda \Delta E} \approx 1$. Integral dominated by this tiny region.

**Step 3: Taylor Expansion**

Near $x^*$:
$$E(x) \approx E(x^*) + E'(x^*)(x-x^*) + \frac{1}{2}E''(x^*)(x-x^*)^2$$

Since $x^*$ is min, $E'(x^*) = 0$, $E''(x^*) = k > 0$.
$$E(x) - E(x^*) \approx \frac{1}{2} k (x-x^*)^2$$

Substitute into integral (only neighborhood):
$$\text{Denom} \approx \int_{x^*-\epsilon}^{x^*+\epsilon} e^{-\lambda \frac{1}{2} k (x-x^*)^2} \, dx$$

This is a **Gaussian Integral**. Recall $\int e^{-ax^2} dx = \sqrt{\frac{\pi}{a}}$, here $a = \frac{\lambda k}{2}$.
$$\text{Denom} \approx \sqrt{\frac{2\pi}{\lambda k}}$$

Similarly for Numerator: $x \approx x^*$ is constant in small neighborhood.
$$
\text{Num} \approx x^* \cdot \sqrt{\frac{2\pi}{\lambda k}}
$$

**Step 4: Cancellation**

$$\lim_{\lambda \to \infty} \langle x \rangle_\lambda \approx \frac{x^* \cdot \sqrt{\frac{2\pi}{\lambda k}}}{\sqrt{\frac{2\pi}{\lambda k}}}$$
Roots, $\pi$, derivative $k$, and $\lambda$ all cancel out! Remaining:
$$= x^*$$
