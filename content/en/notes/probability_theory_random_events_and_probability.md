---
title: "[Course Notes] Probability Theory and Mathematical Statistics | Random Events and Probability"
date: 2025-07-05
description: "Based on Teacher Song Hao's 'Probability Theory and Mathematical Statistics' course on Bilibili: Random Events and Probability"
tags: ["Probability Theory", "Mathematical Statistics", "Random Events", "Probability", "Random Experiments", "Course Notes", "Mathematics"]
---

## Random Experiments and Random Events

### Definitions
In probability theory, an *experiment* possessing the following three characteristics is called a **random experiment**:
1. Reproducibility under identical conditions
2. Multiplicity of outcomes
3. Uncertainty

The **sample space** ($\Omega$) is the *set* of all possible outcomes of a random experiment.
- Each possible outcome in a random experiment is called a **sample point** ($\omega$), i.e., $\omega \in \Omega$.

Any subset of the sample space is called a **random event** ($A$), i.e., $A \sub \Omega$.
  - "Event $A$ occurs" = A sample point belonging to $A$ appears.
  - If a subset contains only one element, this subset is called an **elementary event**, i.e., $|A|=1$.
  - An event containing no sample points is called an **impossible event** $\emptyset$, i.e., $|\emptyset|=0$.
  - **Certain event** = $\Omega$
  - $\emptyset \sub A \sub \Omega$

> The sample space can be finite/infinite/discrete/continuous.

![alt text](/img/contents/样本空间和样本点示意图.png)

**Example**: Tossing two coins simultaneously. Event A is "one head and one tail", Event B is "at least one head".
- Random Experiment $E$: Toss two coins simultaneously and observe the occurrence of heads and tails.
- Sample Space $\Omega$ = {(Head, Head), (Head, Tail), (Tail, Head), (Tail, Tail)}
  - ⚠️ The sample space here is a finite discrete set.
- Random Event $A$ = {(Head, Tail), (Tail, Head)}
- Random Event $B$ = {(Head, Head), (Head, Tail), (Tail, Head)}

### Relationships Between Events
1. Inclusion: $A \sub B$. Indicates that the occurrence of $A$ inevitably leads to the occurrence of $B$.
   - $A = B \hArr A \sub B \text{ and } B \sub A$
2. Union (Sum): $A \cup B$. Indicates that at least one of $A, B$ occurs.
   - Sometimes also denoted as $A + B$
   - $A \sub (A \cup B) \sub \Omega$
   - $A + A = A$
   - $A + \Omega = \Omega$
   - $A_1 \cup A_2 \cup ... \cup A_n$
   - Countably infinite: $A_1 + A_2 + ...$
3. Intersection (Product): $A \cap B $. Indicates that $A, B$ occur simultaneously.
   - Also denoted as $AB$
   - $AB \sub A$
   - $AA = A$
   - $A \cap \emptyset = \emptyset$
   - $A \cap \Omega = A$
4. Difference of Events: $A - B$. Indicates that $A$ occurs, but $B$ does not occur.
   - $A - B = A - AB = A\bar{B}$ 
5. Mutually Exclusive (Disjoint): $AB = \emptyset$. Indicates that $A, B$ cannot occur simultaneously.
   - $A_1, A_2, ..., A_n$ are mutually exclusive if $A_iA_j = \emptyset$ for all $i \neq j$.
6. Complementary Events: $A \cup B = \Omega$ and $A \cap B = \emptyset$.
   - The complement of $A$ can be denoted by $\bar{A}$. That is, $\bar{A} = \Omega - A$.
     - $A\bar{A} = \emptyset$; $\bar{\bar{A}} = A$
   - Mutually Exclusive v.s. Complementary Events
     - $A, B$ are complementary $\Rightarrow$ $A, B$ are mutually exclusive. The converse is not true (because the condition $A \cup B = \Omega$ might not hold).
     - Complementary events apply between **two** events, while mutually exclusive applies between **multiple** events.
     - $A, B$ are mutually exclusive $\nRightarrow \bar{A}$ and $\bar{B}$ are compatible or incompatible.
     - $A, B$ are complementary $\Rightarrow$ $\bar{A}$ and $\bar{B}$ are complementary.
7. Complete Set of Events: $A_1, A_2, ..., A_n$ must satisfy $A_i \cap A_j = \emptyset$ and $\sum A_i = \Omega$.

### Operations on Events
1. Commutative Law: $A\cup B = B \cup A$, $A\cap B = B \cap A$
2. Associative Law: $(A\cup B) \cup C = A \cup ( B \cup C)$, $(A\cap B) \cap C = A \cap (B \cap C)$
3. Distributive Law: $(A\cup B) \cap C = (A \cap C) \cup ( B \cap C)$, $(A\cap B) \cup C = (A \cup C) \cap (B \cup C)$
4. Idempotent Law (Double Complement): $\bar{\bar{A}} = A$
5. De Morgan's Laws: $\overline{A \cup B} = \bar{A} \cap \bar{B}$, $\overline{A \cap B} = \bar{A} \cup \bar{B}$

> You can understand the above operations by drawing diagrams.


## Frequency and Probability

In statistics, the **frequency** $f_i$ of an event $i$ is the ratio of the number of times event $i$ is observed in an experiment to the total number of experiments. Frequency exhibits stability. (Source: [Wikipedia: Frequency (statistics)](https://en.wikipedia.org/wiki/Frequency_(statistics)))

The axiomatic definition of **probability** is: Assume the sample space of a random event $E$ is $\Omega$. Then for every event $A$ in $\Omega$, there exists a real-valued function $P(A)$, satisfying:
1. Non-negativity: $P(A) \ge 0$
2. Normalization: $P(\Omega) = 1$
3. Countable Additivity: For a countable set of **pairwise mutually exclusive** events $\{A_i\}_{i\in N}$, we have: $\sum _{i=1}^{\infty }P(A_{i})=P\left(\bigcup _{i=1}^{\infty }A_{i}\right)$

Any function $P$ satisfying the above conditions can serve as a **probability function** for the sample space $\Omega$, and the function value $P(A)$ is called the probability of event $A$ in $\Omega$. (Source: [Wikipedia: Probability](https://en.wikipedia.org/wiki/Probability))


### Properties of Probability
1. The probability of an impossible event is 0, i.e., $P(\emptyset) = 0$. The converse is not true, i.e., $P(A) = 0 \nRightarrow A = \emptyset$.
   - This implies that events with probability 0 can still happen. Consider an infinite continuous sample space.
2. Addition Rule: For any events $A, B$, $P(A+B) = P(A) + P(B) - P(AB)$
   - Proof: $P(A+B) = P(A+(B-AB)) = P(A) + P(B-AB) = P(A) + P(B) - P(AB)$
   - $P(A+B+C) = P(A) + P(B) + P(C) - P(AB) - P(AC) - P(BC) + 2P(ABC)$
3. Finite Additivity: For countable **pairwise mutually exclusive** events $A_1, A_2, ..., A_n$, $P\left(\bigcup _{i=1}^{n }A_{i}\right) = \sum _{i=1}^{n }P(A_{i})$
   - $A, B$ are mutually exclusive (in this case $P(AB)=0$) $\Rightarrow P(A+B) = P(A) + P(B)$. The converse is not true.
4. $P(\bar{A}) = 1 - P(A)$; $P(A) + P(\bar{A}) = 1$
5. For any events $A, B$, $P(A-B) = P(A) - P(AB)$
     - $B \sub A \Rightarrow P(A-B) = P(A) - P(B)$, and $P(A) \ge P(B)$

## Classical Probability and Geometric Probability


## Conditional Probability and Multiplication Rule

## Law of Total Probability and Bayes' Theorem

## Independence of Events and Bernoulli Trials

## References
- [Wikipedia: Sample Space](https://en.wikipedia.org/wiki/Sample_space)
