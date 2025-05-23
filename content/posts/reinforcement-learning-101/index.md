---
title: "Reinforcement Learning 101"
description: "Introduction to classical model-based Reinforcement Learning techniques."
date: 2025-05-19
tags: ["Reinforcement Learning", "Bellman Optimality Equation", "Bellman Expectation Equation", "Banach Fixed-Point Theorem", "Contraction Mapping", "Value Iteration", "Policy Iteration"]
draft: false
---

# Introduction
Imagine playing a computer game. In such a game you observe the screen and based on what you see on the screen you make a decision.
With each decision you either get some points or not. Making a decision changes what you see on the screen. 
The same process we could describe in a more abstract mathematical language. In such a formal language, you are the agent.
* You observe a state of the environment $s_t$ given a timestamp $t$.
* You execute an action $a_t$ by following your policy function $\pi$.
* You get a reward $r_t$. 
* The environment transitions to a new state $s_{t+1}$.    

The goal of reinforcement learning is to find an optimal policy function $\pi^*$ that maximizes the total future rewards $G_t$. In other words, you want to find actions that let you get as many points as possible in the game.

$$
G_t = R_{t} + R_{t+1} + R_{t+2} + \ldots = \sum_{k=0}^{\infty} R_{t+k}
$$

However, with such a definition of $G_t$, the sum will be infinite which imposes certain mathematical complications. 
Having said that, for mathematical convenience, we modify and replace $G_t$ with discounted cumulative rewards where $0 \leq \gamma \leq 1$ represents a discount factor.

$$
G_t = R_{t} + \gamma R_{t+1} + \gamma^2 R_{t+2} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k} = R_{t} + \gamma G_{t+1}
$$

Why is it finite? To illustrate that, we can assume a constant reward for each timestamp $R_t = R$, and we will end up with a finite geometric series.

$$
G_t = R + \gamma R + \gamma^2 R + \ldots = R \sum_{k=0}^{\infty} \gamma^k = \frac{R}{1-\gamma}
$$

What mathematical framework should we use to model such a decision-making process? A natural choice is probability theory because
usually our environment is not deterministic. It means that if we are in a state $s_t$, and we take an action $a_t$ we don't always get the same reward $r_t$.
Having said that, we can represent $S_t, R_t, A_t$ as random variables and the policy function $\pi$ will be represented as a probability distribution over actions.
Notice that $G_t$ will also be a random variable, hence now we would need to maximize the expected discounted cumulative reward.

$$
\mathbb{E}[G_t \mid S_t = s_t, A_{t-1} = a_{t-1}, S_{t-1} = s_{t-1}, \ldots, S_0 = s_0, A_0 = a_0]
$$


For mathematical simplicity, we often assume such a process to be a Markov Decision Process (MDP) i.e. the next state depends only on the current state and action, not the whole history of states and actions.

$$
p(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = p(s_{t+1} \mid s_t, a_t)
$$

It allows us to simplify the expected discounted cumulative reward and derive Bellman Equations in the next section.

$$
\mathbb{E}[G_t \mid S_t = s]
$$


# Bellman Equations

## Value function

We already know that the goal of reinforcement learning is to find an optimal policy $\pi^* (s)$ that maximizes the expected discounted cumulative reward for every state $s$. 
In reinforcement learning, we define our objective as a value function $v_\pi(s)$. 

$$
\begin{aligned}
v_\pi(s) &= \mathbb{E_\pi}[G_t \mid S_t = s] \qquad \text{(recursive definition of } G_t \text{)}  \\\\\[0.5em]
&= \mathbb{E_\pi}[R_t + \gamma G_{t+1} \mid S_t = s] \qquad \text{(linearity of expectation)} \\\\\[0.5em]
&= \mathbb{E_\pi}[R_t \mid S_t = s] + \gamma \mathbb{E_\pi}[G_{t+1} \mid S_t = s] \qquad \text{(law of total expectation)} \\\\\[0.5em]
&= \mathbb{E_\pi}[R_t \mid S_t = s] + \gamma \sum_{s^\prime} p(s^\prime \mid s)\mathbb{E_\pi}[G_{t+1} \mid S_t = s,S_{t+1} = s^\prime] \qquad \text{(conditional independence)} \\\\\[0.5em]
&= \mathbb{E_\pi}[R_t \mid S_t = s] + \gamma \sum_{s^\prime} p(s^\prime \mid s)\mathbb{E_\pi}[G_{t+1} \mid S_{t+1} = s^\prime] \qquad \text{(definition of } v_\pi \text{)} \\\\\[0.5em]
&= \mathbb{E_\pi}[R_t \mid S_t = s] + \gamma \sum_{s^\prime} p(s^\prime \mid s) v_\pi(s^\prime) \qquad \text{(marginalization)} \\\\\[0.5em]
&= \mathbb{E_\pi}[R_t \mid S_t = s] + \gamma \sum_{r, a, s^\prime} p(r, a, s^\prime \mid s) v_\pi(s^\prime) \qquad \text{(definition of } \mathbb{E_\pi} \text{)} \\\\\[0.5em]
&= \sum_{r, a, s^\prime} p(r, a, s^\prime \mid s) r + \gamma \sum_{r, a, s^\prime} p(r, a, s^\prime \mid s) v_\pi(s^\prime) \\\\\[0.5em]
&= \sum_{r, a, s^\prime} p(r, a, s^\prime \mid s) (r + \gamma v_\pi(s^\prime)) \qquad \text{(probability chain rule)}  \\\\\[0.5em]
&= \sum_{a}p (a \mid s) \sum_{r, s^\prime} p(r, s^\prime \mid s, a) (r + \gamma v_\pi(s^\prime)) \\\\\[0.5em]
&= \sum_{a}\pi (a \mid s) \sum_{r, s^\prime} p(r, s^\prime \mid s, a) (r + \gamma v_\pi(s^\prime)) \\\\\[0.5em]
\end{aligned}
$$

A bit of math, but the takeaway is that we can express $v_\pi(s)$ as a function of itself i.e. $v_\pi(s^\prime)$, making it a recursive definition. We will see why it is useful once we derive the Value Iteration and Policy Iteration algorithms.
Because $v(s)$ depends on the policy $\pi(a \mid s)$, we often write $v_\pi(s).$ 


## Action-value function
We defined the value function $v_\pi(s)$ as the expected discounted cumulative reward in state $s$. We can introduce an additional useful function $q_\pi(s, a)$ called the action-value function. 
The action-value function represents the expected discounted cumulative reward if an agent in state $s$ takes an action $a$ and keeps following policy $\pi$ until the episode ends.
We introduce it for mathematical convenience and ease of communication as it will appear in many equations.

$$
q_\pi(s, a) = \mathbb{E_\pi}[G_t \mid S_t = s, A_t = a] = \sum_{r, s^\prime} p(r, s^\prime \mid s, a)(r + \gamma v_\pi(s^\prime))
$$

We no longer need to sum over all actions $a$ since we condition our probability distribution on action $a$.
Having defined the action-value function, we can use it to express the value function.

$$
v_\pi(s) =  \sum_{a}\pi(a \mid s) q_\pi(s, a)
$$

## Bellman Expectation Equation

A derived equation in the value function section is also known as the Bellman Expectation Equation. It is called "Expectation" because we sum over all actions which is equivalent to taking the expected value over the action-value function as you can see below.

$$
v_\pi(s) = \sum_{a} \pi(a \mid s) \sum_{r, s^\prime} p(r, s^\prime \mid s, a) (r + \gamma v_\pi(s^\prime)) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}[ q _{\pi} (s, a)  ]
$$

## Bellman Optimality Equation

There is also the variant of the Bellman Expectation Equation known as the Bellman Optimality Equation where instead of taking expectation over all actions we act greedily and always take an action that maximizes the value function. We will see why it is useful later in this post.

$$
v(s) = \max_{a} \sum_{r, s^\prime} p(r, s^\prime \mid s, a) (r + \gamma v(s^\prime)) = \max_{a} q  (s, a) \\\\\[0.5em]
$$

Unlike the Bellman Expectation Equation, notice that our value function $v(s)$ doesn't depend on the policy $\pi$, hence we used $v(s)$ instead of $v_\pi(s)$ on purpose.
This observation is a foundation of the off-policy algorithms in reinforcement learning (e.g. Q-Learning) as you will learn later.

# Contraction Mapping

You might wonder why we introduced the Bellman Expectation Equation and the Bellman Optimality Equation in the first place. 
It will become obvious once we switch our focus a bit from reinforcement learning and understand contraction mapping first.
This part of math will be useful to derive algorithms to find the optimal policy.


## Definition
A contraction mapping is a function $T: X \rightarrow X$ that maps a metric space $X$ onto itself with the property that the distance $d$ between any two points $x$ and $y$ in the space is reduced after applying the function $T$.
Such a reduction in distance we can express with a constant $0 \leq \kappa < 1$ such that:

$$d(T(x), T(y)) \leq \kappa \cdot d(x, y)$$

In other words, applying the function $T$ brings points closer together by at least a factor of $\kappa$.

## Banach Fixed-Point Theorem

The Banach Fixed-Point Theorem states that if $T$ is a contraction mapping then:

1. $T$ has a unique fixed point $x^*$ such that $T(x^ *) = x^ *$.
2. For any initial point $x_0$ in the space, the sequence $x_0, T(x_0), T(T(x_0)), \ldots$ converges to that fixed point  $x^*$.

In other words, if we keep applying the function $T$ starting with some initial point $x_0$ it will converge to the point $x^*$. 


## Applications in RL

Now you probably wonder why is it useful in reinforcement learning? 
Imagine for a second that any point $x$, $y$, $\ldots$ in the metric space is a function, not a real number as you are probably used to think at school.
The takeaway from the Banach Fixed-Point Theorem is that if we found a function $T$ that is a contraction mapping, then we would have an iterative algorithm to find the true function $x$, beginning from any function $x_0$. 

Well, it seems like there is a proof that the Bellman Expectation Equation and the Bellman Optimality Equations are contraction mappings!
Notice that we couldn't even think about proving such a fact without the Bellman Equations, because the recursive formulation of these equations allows us to use it as an operator $T$ as it maps a metric space onto itself $T: X \rightarrow X$.

For the Bellman Expectation Equation, the operator $T^\pi$ is defined as:

$$T^\pi (v(s)) = \sum_{a} \pi(a \mid s) \sum_{r, s^\prime} p(r, s^\prime \mid s, a) (r + \gamma v_\pi(s^\prime))$$

For the Bellman Optimality Equation, the operator $T^*$ is defined as:

$$T^* (v(s)) = \max_{a} \sum_{r, s^\prime} p(r, s^\prime \mid s, a) (r + \gamma v(s^\prime))$$

Both of these operators directly lead to two common algorithms. The contraction property ensures that:
1. In Value Iteration, repeatedly applying $T^*$ will converge to the optimal value function $v^ *$
2. In Policy Iteration, repeatedly applying $T^\pi$ for a fixed policy $\pi$ will converge to the value function $v_\pi$ for that policy

This mathematical foundation is why we can be confident that these iterative algorithms will eventually find the solutions we're looking for.

# Algorithms

Below you can find two common algorithms that allow us to find the optimal policy $\pi^*(s)$.

## Value Iteration

As we've seen, the Bellman Optimality Equation is a contraction mapping. This means we can apply it iteratively to compute the optimal value function $v^* (s)$.

1. Iteratively apply the Bellman Optimality Equation for a value function until it converges to the optimal value function $v^*(s)$.
$$
v(s) = \max_{a} \sum_{r, s^\prime} p(r, s^\prime \mid s, a) (r + \gamma v(s^\prime))
$$
How do we know when convergence happens? We need to observe both $v_t(s)$ and $v_{t+1}(s)$ until they stop changing by a small predefined value $\epsilon$.

2. Once we computed the optimal value function $v^* (s)$, we can extract the corresponding optimal action-value function $q^* (s, a)$ and the optimal policy $\pi^* (s)$ in a single step, using the equations we already know.
$$
q^* (s, a) = \sum_{r, s^\prime} p(r, s^\prime \mid s, a)(r + \gamma v^* (s^\prime)) \\\\[0.5em]
\pi^* (s) = \argmax_{a}{q^*(s, a)}
$$
We can extract the optimal policy $\pi^ * (s)$ because the optimal policy is the one that maximizes the value function for each state $s$.

What is interesting about the Value Iteration algorithm is that we never update a policy during the algorithm. 
We only update  our value function $v(s)$. Any intermediate value function $v(s)$ that is not optimal might not correspond to any reasonable policy $\pi(s)$.


## Policy Iteration

As we've established, the Bellman Expectation Equation is a contraction mapping. This allows us to apply it recursively to compute the value function $v_\pi(s)$ for our policy $\pi(s)$. 

What we compute in Policy Iteration is not an optimal value function $v^* (s)$ as in the Value Iteration algorithm, but a value function $v_\pi(s)$ that corresponds to a policy $\pi(s)$. 
Now having computed $v_\pi (s)$ given policy $\pi$, how do we find the optimal policy? We will use the Policy Improvement Theorem.

### Policy Improvement Theorem

The theorem states that acting greedily with respect to $q_\pi(s, a)$ i.e. extracting our new policy $\pi^\prime(s) = \argmax_a q_\pi(s, a)$ improves or doesn't change our current policy $\pi(s)$.
It is true because of the following inequality as the $\max_a$ function will always be equal to or higher than a weighted average of $q_\pi(s, a)$.

$$
v_\pi(s) = \sum_a \pi(a \mid s) q_\pi(s, a) \leq \max_a q_\pi(s, a) = v_{\pi^\prime}(s)
$$

Improving a policy means that our value function is better for every state $s$. In other words, we can expect a higher cumulative discounted reward as $v_\pi(s) \leq v_{\pi^\prime}(s)$.


### Algorithm

1. Iteratively apply the Bellman Expectation Equation for a value function until it converges to the value function $v_\pi (s)$ for policy $\pi$.
$$
v_\pi(s) = \sum_{a} \pi(a \mid s) \sum_{r, s^\prime} p(r, s^\prime \mid s, a) (r + \gamma v_{\pi}(s^\prime))
$$

2. Update the policy $\pi(s)$ by acting greedily with respect to $q_\pi(s, a)$ to obtain a new policy $\pi^*$.
$$\pi^\prime(s) = \argmax_a q_\pi(s, a)$$

3. Repeat until the policy stops changing for each state $s$ i.e. $\|\pi_{t+1}(s) - \pi_{t}(s)\| \leq \epsilon $.

# Example

Let's play a game. We are given a grid world in which each cell represents a state $s$. 
We can take 4 actions (up, down, left, right). We win the game when we obtain a gift in the shortest time possible while avoiding falling down.
We can model the shortest possible time as assigning $-1$ rewards with each timestep. 

![Frozen Lake empty](frozen_lake_empty.png)

## Code

You can find my naive implementation of these algorithms on my GitHub:

https://github.com/elkotito/rl-tutorial/blob/main/model_based_algorithms.py

## Policy

You can see an optimal policy on the image below:

![Frozen Lake policy](frozen_lake_policy.png)


# Final thoughts

In this article we discussed algorithms to learn the agent's behavior expressed as policy $\pi$ in model-based environments. A model-based environment means that we have direct access to the environment's dynamics i.e. $p(r, s^\prime \mid s, a)$ probabilities. 
As you can tell, it is rarely the case in the real world. In the next article we will discuss model-free environments which is a situation when you can rely only on data collected by interacting with environment.