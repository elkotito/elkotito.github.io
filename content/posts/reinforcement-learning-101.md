---
title: "Reinforcement Learning 101"
description: "Introduction to classical Reinforcement Learning techniques."
date: 2025-05-15
tags: ["Reinforcement Learning", "Bellman Equation", "Banach Fixed-Point Theorem", "Value Iteration", "Policy Iteration", "Q-Learning", "SARSA", "Vanilla Policy Gradient", "A2C", "REINFORCE"]
draft: true
math: true
---

# Introduction

What are discounted rewards. What is MDP. What is policy. What is the goal of learning the agent. Start from evaluating a state. Then how to find a policy.)

# Bellman Equation

$$
\begin{aligned}
v_\pi(s) &= \mathbb{E_{\pi}}[G_t \mid S_t = s] \\\\\[0.5em]
&= \mathbb{E_{\pi}}[R_t + \gamma G_{t+1} \mid S_t = s] \qquad \text{(recursive definition of } G_t \text{)} \\\\\[0.5em]
&= \mathbb{E_{\pi}}[R_t \mid S_t = s] + \gamma \mathbb{E_{\pi}}[G_{t+1} \mid S_t = s] \qquad \text{(linearity of expectation)} \\\\\[0.5em]
&= \mathbb{E_{\pi}}[R_t \mid S_t = s] + \gamma \sum_{s^\prime} p(s^\prime \mid s)\mathbb{E_{\pi}}[G_{t+1} \mid S_t = s,S_{t+1} = s^\prime] \qquad \text{(law of total expectation)} \\\\\[0.5em]
&= \mathbb{E_{\pi}}[R_t \mid S_t = s] + \gamma \sum_{s^\prime} p(s^\prime \mid s)\mathbb{E_{\pi}}[G_{t+1} \mid S_{t+1} = s^\prime] \qquad \text{(conditional independence)} \\\\\[0.5em]
&= \mathbb{E_{\pi}}[R_t \mid S_t = s] + \gamma \sum_{s^\prime} p(s^\prime \mid s) v_{\pi}(s^\prime) \qquad \text{(definition of } v_{\pi} \text{)} \\\\\[0.5em]
&= \mathbb{E_{\pi}}[R_t \mid S_t = s] + \gamma \sum_{r, a, s^\prime} p(r, a, s^\prime \mid s) v_{\pi}(s^\prime) \qquad \text{(marginalization)} \\\\\[0.5em]
&= \sum_{r, a, s^\prime} p(r, a, s^\prime \mid s) r + \gamma \sum_{r, a, s^\prime} p(r, a, s^\prime \mid s) v_{\pi}(s^\prime) \qquad \text{(definition of } \mathbb{E_\pi}  \text{)} \\\\\[0.5em]
&= \sum_{r, a, s^\prime} p(r, a, s^\prime \mid s) (r + \gamma v_{\pi}(s^\prime)) \\\\\[0.5em]
&= \sum_{a}\pi(a \mid s) \sum_{r, s^\prime} p(r, s^\prime \mid s, a) (r + \gamma v_{\pi}(s^\prime)) \qquad \text{(conditional probability)} \\\\\[0.5em]
\end{aligned}
$$

We defined our value function $v_\pi(s)$ as total discounted future rewards in state $s$. We can introduce an additional function $q_\pi(s, a)$ called action-value function. 
Action-value function represents total discounted future rewards if our agent in state $s$ takes an action $a$ and keeps following policy $\pi$ until the episode ends. 
Mathematically we can express it like follows.


$$
q_\pi(s, a) = \mathbb{E_\pi}[G_t \mid S_t = s, A_t = a] = \sum_{r, s^\prime} p(r, s^\prime \mid s, a)(r + \gamma v_\pi(s^\prime))
$$

We no longer need to sum over actions $a$ since we condition on action $a$.
Having defined action-value function we can use it to express value function.

$$
v_\pi(s) =  \sum_{a}\pi(a \mid s) q_\pi(s, a)
$$

Both will be very useful in the future as we can see. It is also posssible to express $q_\pi(s, a)$ in terms of $q_\pi(s^\prime, a^\prime)$.

Sometimes the equations above are called expected Bellman Equations. There are also optimal Bellman Equation expressed below.


# Banach Fixed-Point Theorem

# Value Iteration

There exists a proof that Bellman Optimality Equation is a contraction mapping from Banach Fixed-Point Theorem. Having said that we can apply Bellman Optimality Equation iteratively to compute optimal value function $v^* (s)$.

1. Iteratively apply Bellman Optimality Equation for a value function until it convergences to $v^* (s)$. How do we know when it happens? We need to observe both $v_{\text{old}} (s)$ and $v_{\text{new}}(s)$ until they stop changing by a prior value $\epsilon$.  
$$
v_\pi(s) = \sum_{a}\pi(a \mid s) \sum_{r, s^\prime} p(r, s^\prime \mid s, a) (r + \gamma v_{\pi}(s^\prime))
$$
 
2. Once we computed an optimal value function $v^* (s)$, we can extract an associated optimal action-value function $q^* (s, a)$ and optimal policy $\pi^* (a \mid s)$ in a single step, using the equations we already know.
$$
q^* (s, a) = \sum_{r, s^\prime} p(r, s^\prime \mid s, a)(r + \gamma v^* (s^\prime)) \\\\[0.5em]
\pi^* (s) = \argmax_{a}{q^*(s, a)}
$$
Keep in mind we never update our policy in the 1. step of the algorithm. Any intermediate value function $v(s)$ might not correspond to any reasonable policy $\pi$.


# Policy Iteration

There exists a proof that Bellman Expectation Equation is a contraction mapping from Banach Fixed-Point Theorem. Having said that we can apply Bellman Expectation Equation iteratively to compute value function $v(s)$ for our policy $\pi(a \mid s)$.



# Q-Learning
Model-free environment. Value function applied to Q instead of V

# SARSA

Model-free environment. Q-learning that accounts for exploration.


# Advantage Actor-Critic (A2C)


# Vanilla Policy Gradient


# REINFORCE