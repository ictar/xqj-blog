---
title: "[Course Notes] Probability Theory and Mathematical Statistics | Preliminaries"
date: 2025-07-05
description: "Based on Teacher Song Hao's 'Probability Theory and Mathematical Statistics' course on Bilibili: Preliminaries"
tags: ["Probability Theory", "Mathematical Statistics", "Addition Principle", "Multiplication Principle", "Permutations and Combinations", "Course Notes", "Mathematics"]
---

## Addition Principle and Multiplication Principle

According to [Wikipedia](https://en.wikipedia.org/wiki/Rule_of_product):
- **Addition Principle (Rule of Sum)**: If there are $a$ ways to do something and $b$ ways to do another thing, and these two things cannot be done simultaneously (mutually exclusive), then there are $a+b$ ways to choose one of the actions.
- **Multiplication Principle (Rule of Product)**: If there are $a$ ways to do something and $b$ ways to do another thing, then there are $a\times b$ ways to do both.


### Addition Principle or Multiplication Principle?

It depends on **whether it is done in steps**. If it involves steps, use multiplication; otherwise, use addition.

## Permutations and Combinations

### Permutations

#### Permutation without Repetition
Taking $m$ ($1\le m \le n$) different elements from $n$ different elements and arranging them in a sequence, the number of permutations is denoted as $P^m_n$ (or $A^m_n$ or $_nP_m$).
1. If $m \lt n$, then $P^m_n = A^m_n = n(n-1)...(n-m+1)$
2. If $m = n$, then $P^n_n = A^n_n = n(n-1)...1 = n!$

> The second case is called a full permutation.

#### Permutation with Repetition

Taking $m$ elements from $n$ different elements **with replacement** and arranging them in a sequence, there are $n^m$ ways of permutation.

### Combinations
Taking $m$ different elements from $n$ different elements, regardless of order, to form a group, the number of combinations is $C^m_n = \binom{n}{m} = \frac{P^m_n}{m!} = \frac{n(n-1)...(n-m+1)}{m!}$.

> Dividing by $m!$ is to remove duplicates caused by ordering.

Note:
- $0! = 1$
- $C^m_n = C^{n-m}_n$
- $C^m_n = C^{m-1}_{n-1} + C^m_{n-1}$ (Pascal's Identity)

### Permutation or Combination?

It depends on **whether order matters**. If order matters, it is a permutation; otherwise, it is a combination.


## References
- []()
