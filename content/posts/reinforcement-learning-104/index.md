---
title: "Reinforcement Learning 104: Deep Q-Networks"
description: "Deep Q-Networks: DQN, experience replay, target networks, reward clipping, overestimation bias, Double DQN, Dueling DQN, Noisy DQN, Distributional DQN, Bootstrapped DQN."
date: 2026-06-26
tags:
  [
    "Reinforcement Learning",
    "Deep Reinforcement Learning",
    "DQN",
    "Deep Q-Network",
    "Experience Replay",
    "Target Network",
    "Reward Clipping",
    "Overestimation Bias",
    "Double DQN",
    "Dueling DQN",
    "Noisy DQN",
    "Distributional DQN",
    "Bootstrapped DQN",
  ]
draft: true
---

## Introduction

In the previous article, we replaced the tabular action-value function $q(s, a)$ with a parameterized approximation: $q_\theta(s, a)$. We will use neural networks to approximate $q_\theta(s, a)$.

Deep Q-Network (DQN) is the version of approximate Q-learning that made this practical at a scale. When we combine Q-learning with neural networks, then we must deal with the instability created by correlated data, moving targets, changing behavior distributions, and large TD errors. This article presents solutions to deal with those problems.

Before we continue, we need to understand why we moved forward with approximate Q-learning, not approximate SARSA or Expected SARSA.

### Why not SARSA?

SARSA is an on-policy algorithm. It learns the value of the policy that is currently acting. Its target uses the actual next action $a^\prime$, which should be sampled from the current policy:

$$
r + \gamma q_\theta(s^\prime, a^\prime)
$$

If we collect and store transitions and train on them later, that next action may have been sampled from an older policy. The update is then no longer strictly on-policy. To stay faithful to SARSA, we need fresh data from the current policy, or at least data that is not too stale. This limits how freely we can reuse stored experience.

There is another issue. In the tabular case, SARSA can be understood through generalized policy iteration: evaluate the current policy, then improve it by acting greedily with respect to the learned values. With neural networks, this story becomes less clean. A gradient update on a batch is only a partial policy-evaluation step on the states and actions we sampled. It does not give us a reliable global estimate of $q^\pi$. At the same time, the policy is derived from the same changing $q_\theta$, so the policy may change before evaluation has caught up.

This makes approximate SARSA awkward. It needs fresh on-policy data to evaluate a moving policy, while policy improvement happens only indirectly through changes in the approximate action-value function.

Q-learning is off-policy. The behavior policy that collects data can be different from the target policy being learned. This means the agent can explore with an $\epsilon$-greedy policy, store transitions in a replay buffer, and later reuse them for training.

This fits neural-network training better than strict on-policy value learning. One environment interaction can be reused in many gradient updates, and random batches from a replay buffer are less correlated than consecutive game frames. This does not mean off-policy learning is always better. It means off-policy control is compatible with replay, and replay is very useful when training neural networks.

What about off-policy Expected SARSA? Its target is:

$$
r + \gamma \sum_{a^\prime} \pi(a^\prime \mid s^\prime) q_\theta(s^\prime, a^\prime)
$$

If the target policy is $\epsilon$-greedy, then Expected SARSA learns values for a policy that continues to explore in the future. Those values include the cost of future random exploratory actions. This can be useful if we want to keep exploring after learning the best policy e.g. in non-stationary environment. In our case we explore to discover useful transitions, but after training we usually want to act greedily and make the best decision according to the learned values and keep winning games!

Therefore, we do not necessarily want $q_\theta$ to encode the long-term cost of future exploratory actions. We want it to approximate the value of the greedy policy we would like to deploy. This is exactly what Q-learning does.

$$
r + \gamma \max_{a^\prime} q_\theta(s^\prime, a^\prime)
$$

So have we established that off-policy methods are always better than on-policy methods? Not quite. On-policy actor-critic methods (more on them later) can also work very well with neural networks. For example A3C avoided replay buffer by using many parallel actors. Instead of decorrelating data with a replay buffer, it decorrelated data with many environments running asynchronously. SARSA could also do that, the difference is how the policy is improved.

Now, let's go back to DQN and problems we outlined in the previous article.

## Problem 1: Correlated Data

Stochastic gradient descent works best when batches give a useful estimate of the training objective. Consecutive RL samples are not like that. During one episode, adjacent states are strongly correlated.

Imagine an agent watching the ball move across the screen. Frame $t$ and frame $t+1$ are almost the same. If we train directly on the latest transition at every step, the network sees a long stream from one small region of experience. The gradient points too strongly toward what just happened and too weakly toward the broader game.

This hurts both optimization and behavior. The agent can overfit to recent experience, forget older situations, or move the action-value function in a direction that only makes sense for the latest part of the trajectory.

There is a second effect. The data distribution depends on the policy. If a small parameter update makes "move left" look slightly better than "move right", the $\epsilon$-greedy behavior policy may start visiting more left-side states. After more left-side updates, the estimate may swing again. This feedback between value estimates, actions, and future data is one source of policy oscillation.

Policy oscillation means the learned greedy policy keeps changing because small value changes alter the data the agent sees next, and the new data then pushes the values somewhere else. The policy is not smoothly improving; it is chasing its own recent consequences.

### Experience Replay

DQN appends each transition to a replay buffer $\mathcal{D}$ and trains on random batches $B$ sampled from it. This helps in several ways.

$$
\{(s_i, a_i, r_i, s_i^\prime)\}_{i=1}^{B} \sim \mathcal{D}
$$

1. It breaks short-term correlations. A random batch may contain transitions from different episodes, different game situations, and different moments in training. The samples are not truly independent, but they are much less correlated than consecutive frames.

2. It improves data efficiency. One transition can be used in many gradient updates of our neural network. This matters because environment interaction is often the expensive part.

3. It slows sudden shifts in what the network trains on. This makes learning more stable: a small change in the current policy does not immediately replace the whole training batch with new, similar transitions.

Other way to think about replay batch is that it can be viewed as a sample-based model of the world. A real model would estimate:

$$
p(r, s^\prime \mid s, a)
$$

and let us ask what would happen for arbitrary state-action pairs. A replay buffer does not do that. It cannot generate new transitions for states and actions we never tried, but it is a non-parametric collection of one-step samples from the environment.

What are disadvantages of replay?

1. If a replay buffer is large, then it is memory intensive. For Atari, one `uint8` grayscale frame is $84 \times 84 = 7{,}056$ bytes. A transition contains $s_t$ and $s_{t+1}$, each a stack of 4 frames, so naive storage needs $8 \times 7{,}056 = 56{,}448$ bytes per transition i.e. ~56GB for $1M$ transitions. Practical buffers avoid most of this duplication by storing frames once and reconstructing overlapping states, which is why Atari replay buffers are often closer to ~6-7GB.

2. Uniform random sampling to form a batch is not optimal as it treats all transitions as equally useful. In reality, some transitions have larger TD errors or contain rare rewards. Prioritized replay improves this idea by sampling more informative transitions more often, but vanilla DQN uses uniform sampling.

3. Replay trains on transitions collected by older versions of the policy. Q-learning can use them because it is off-policy, but too much old experience can slow adaptation when the current policy starts reaching different states.

## Problem 2: Moving Targets

Experience replay helps with correlated data, but it does not solve the moving target problem. In approximate Q-learning, the target is:

$$
y(\theta) = r + \gamma \max_{a^\prime}q_\theta(s^\prime, a^\prime)
$$

In the previous article, we established that we use semi-gradients for targets $y(\theta)$, so during one gradient update the target is treated as fixed. However, this is not enough. Once we update the network parameters $\theta$, the target $y(\theta)$ also changes, because the same network is used both to predict $q_\theta(s,a)$ and to build the bootstrap target $y(\theta)$.

Why is this a problem? Is it because the target is wrong? No. Bootstrap targets are usually wrong. That is expected. For example, consider the scalar recursive equation:

$$
q = 1 + 0.9q
$$

The true solution is:

$$
q^* = 10
$$

Suppose we start from $q_0 = 0$. It does not matter that $q_0$ is wrong. If we keep applying the update:

$$
q_{k+1} = 1 + 0.9q_k
$$

we get:

$$
q_1 = 1,\quad q_2 = 1.9,\quad q_3 = 2.71,\dots
$$

Every intermediate value is wrong, but the sequence still moves toward the fixed point. That is the core idea behind the Bellman optimality equation, contraction mappings, and the Banach fixed-point theorem. The target does not need to be correct at every step. It only needs to be a useful next approximation.

The problem in DQN is different. We are not applying the exact recursive equation to the whole action-value function. We are learning from finite, noisy samples using gradient descent. If we just use a noisy sample to update $q_k$, then training might behave in a very unpredictable way.

Ideally, we would like to do something closer to this:

1. Keep an old action-value function $q_0$ fixed.
2. Use many noisy samples and many gradient updates to learn a better approximation $q_1$.
3. Once $q_1$ has been fitted reasonably well, replace $q_0$ with $q_1$.
4. Repeat the process.

In other words, we do not want the target to change every time the online network takes a single gradient step. We want to hold the previous approximation fixed for a while, train the new approximation against it, and only then move to the next step of the recursive process. It is then more similar to how recursive Bellman optimality equations work.

### Target Network

To apply this idea in practice, DQN introduces a second neural network called the target network, with parameters $\theta^-$. DQN separates the network being trained from the network used to build the bootstrap target.

There are two networks:

- The online network $q_\theta$, updated by gradient descent.
- The target network $q_{\theta^-}$, used to compute bootstrap targets.

The target becomes:

$$
y =
\begin{cases}
r, & \text{if } s^\prime \text{ is terminal} \\
r + \gamma \max_{a^\prime}q_{\theta^-}(s^\prime, a^\prime), & \text{otherwise}
\end{cases}
$$

The online network is still trained with batches $B$ sampled from a replay buffer.

$$
L(\theta) = \frac{1}{B}\sum_{i=1}^{B} \frac{1}{2}\left(y_i - q_\theta(s_i, a_i)\right)^2
$$

During these gradient updates, the target network parameters $\theta^-$ are held fixed. This means the online network is repeatedly trained against targets generated by the same fixed action-value approximation. Then periodically, the online network copies its parameters to the target network:

$$
\theta^- \leftarrow \theta
$$

Conceptually, the target network plays the role of the old approximation $q_0$. The online network tries to learn a better approximation $q_1$ using many noisy samples and gradient updates. After some time, $q_1$ becomes the new fixed reference point, and the process continues. This makes training less reactive to noisy samples and closer in spirit to approximate value iteration, although there is still no convergence guarantee like in the tabular setting.

## Problem 3: Unstable Gradients

The TD error for one transition is:

$$
\delta = y - q_\theta(s, a)
$$

For squared loss, the gradient has the form:

$$
\nabla_\theta L(\theta)
= -\delta \nabla_\theta q_\theta(s, a)
$$

So the size of the TD error directly scales the update. If rewards have very different magnitudes across games, then the same learning rate can be too small for one game and too large for another. Bootstrapping can amplify this. If one target produces a large Q-value, future targets may contain that large value:

$$
r + \gamma \max_{a^\prime}q_{\theta^-}(s^\prime, a^\prime)
$$

Then earlier states learn large values too. If those values are overestimates, the max operator can keep selecting them and propagating them backward.

### Reward Clipping

DQN clips rewards to:

$$
r \in [-1, 1]
$$

It makes true optimal Q-values bounded by roughly:

$$
\frac{1}{1 - \gamma}
$$

For $\gamma = 0.99$, that is $100$. This gives good gradients in a pragmatic sense: the TD errors are kept in a range where one learning rate can work across many games. Without clipping, the scale can be much larger and very different across games.

Reward clipping has also a cost. If one action gives reward $+1$ and another gives reward $+100$, clipping makes both immediate rewards equal to $+1$. The agent can no longer distinguish good rewards from great rewards at that time step. It learns from signs and frequencies more than from exact reward magnitudes. So reward clipping is more like an engineering trick to make learning process stable.

## Vanilla DQN

Now we can combine all these ideas and implement a basic DQN training algorithm.

1. Initialize the online network $q_\theta$.
2. Initialize the target network with the same parameters: $\theta^- \leftarrow \theta$.
3. Initialize a replay buffer $\mathcal{D}$.
4. Act with an $\epsilon$-greedy behavior policy based on $q_\theta$.
5. Store each transition $(S_t, A_t, R_t, S_{t+1})$ in $\mathcal{D}$.
6. Sample a batch from $\mathcal{D}$.
7. For each sampled transition, compute:

$$
y_i =
\begin{cases}
r_i, & \text{if } s_i^\prime \text{ is terminal} \\
r_i + \gamma \max_{a^\prime} q_{\theta^-}(s_i^\prime, a^\prime), & \text{otherwise}
\end{cases}
$$

8. Update $\theta$ by minimizing:

$$
L(\theta) =
\frac{1}{B}\sum_{i=1}^{B}
\frac{1}{2}\left(y_i - q_\theta(s_i, a_i)\right)^2
$$

9. Periodically update the target network with a hard copy of the online network.

$$
\theta^- \leftarrow \theta
$$

This is DQN: approximate Q-learning with replay buffer, a target network, and reward clipping.

### Implementation

There are a few practical issues that can show up when training DQN on Atari games.

Store frames as `uint8` in the replay buffer to save memory, then convert and normalize them on the GPU:

```python
states = states.to(self.device).float().div_(255.0)
```

If learning stalls early, check the gradient norms. In my runs, very small gradients with Adam made the model fail to learn. Using a larger Adam epsilon helped:

```python
self.optimizer = Adam(self.policy_network.parameters(), lr=config.learning_rate, eps=1e-4)
```

I also used `LeakyReLU` instead of `ReLU` in the convolutional neural network to avoid dying ReLU neurons.

```python
nn.LeakyReLU(negative_slope=0.01)
```

Atari environments return both `terminated` and `truncated`. Only true terminations should stop bootstrapping; time-limit truncations can still use the next-state Q-value.

```python
transition = Transition(state=state, action=action, reward=clipped_reward, next_state=next_state, done=terminated)
```

### Code

Feel free to take a look at [my implementation on GitHub](https://github.com/elkotito/dqn-implementations). I recommend implementing it yourself rather than having an agent do it. It is the best way to learn.

