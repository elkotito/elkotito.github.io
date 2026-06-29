---
title: "Reinforcement Learning 104: Deep Q-Networks (Part 1)"
description: "Deep Q-Networks: DQN, Experience Replay, Prioritized Experience Replay, Target Networks, Reward Clipping, Overestimation Bias, Double DQN, Dueling DQN."
date: 2026-06-26
tags:
  [
    "Reinforcement Learning",
    "Deep Reinforcement Learning",
    "DQN",
    "Deep Q-Network",
    "Experience Replay",
    "Prioritized Experience Replay",
    "Target Network",
    "Reward Clipping",
    "Overestimation Bias",
    "Double DQN",
    "Dueling DQN",
  ]
draft: true
---

## Introduction

In the previous article, we replaced the tabular action-value function $q(s, a)$ with a parameterized approximation: $q_\theta(s, a)$. We will use neural networks to approximate $q_\theta(s, a)$.

Deep Q-Network (DQN) is the version of approximate Q-learning that made it practical at a scale. When we combine Q-learning with neural networks, then we must deal with the instability created by correlated data, moving targets, changing behavior distributions, and large TD errors. This article presents solutions to deal with those problems.

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

Stochastic gradient descent works best when batches give a useful estimate of the training objective. In supervised learning, we sample examples independently from the same training distribution and use the minibatch gradient as an unbiased estimate of the expected gradient:

$$
\nabla_\theta J(\theta) = \mathbb{E}_X[\nabla_\theta L(X, \theta)] \approx
\frac{1}{B}\sum_{i=1}^{B}\nabla_\theta L(X_i, \theta)
$$

Consecutive RL samples are not like independently sampled training examples. During one episode, adjacent states are strongly correlated.

Imagine an agent watching the ball move across the screen. Frame $t$ and frame $t+1$ are almost the same. If we train directly on the latest transition at every step, the network sees a long stream from one small region of experience. The gradient points too strongly toward what just happened and too weakly toward the broader game.

There is also a distribution problem. In supervised learning, the dataset is usually fixed. In RL, the behavior policy creates the data. If $q_\theta$ changes, then an $\epsilon$-greedy policy may choose different actions and reach different states. If we train only on the newest transitions, the training distribution can immediately become dominated by whatever the current policy just visited, instead of staying stable and well mixed.

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

The problem in DQN is different. We are not applying the exact recursive equation to the whole action-value function. We are learning from finite, noisy samples using gradient descent. If we just use a noisy sample to update $q_k$, then training might behave in a very unpredictable way. Ideally, we would like to do something closer to this:

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

Conceptually, the target network plays the role of the old approximation $q_0$. The online network tries to learn a better approximation $q_1$ using many noisy samples and gradient updates. After some time, $q_1$ becomes the new fixed reference point, and the process continues. This makes training less reactive to noisy samples and closer in spirit to approximate value iteration, although there are still no convergence guarantees like in the tabular setting.

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

So the size of the TD error directly scales the update. Since action-values estimate discounted sums of rewards, larger reward scales lead to larger target values and larger TD errors. If rewards have very different magnitudes across games, then the same learning rate can be too small for one game and too large for another.

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

8. Update $\theta$ by minimizing loss function:

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

### Tips & Tricks

There are a few practical issues that can show up when training DQN on Atari games.

1. Store frames as `uint8` in the replay buffer to save memory, then convert and normalize them on the GPU:

```python
states = states.to(self.device).float().div_(255.0)
```

2. If learning stalls early, check the gradient norms. In my runs, very small gradients with Adam made the model fail to learn. Using a larger Adam epsilon helped:

```python
self.optimizer = Adam(self.policy_network.parameters(), lr=config.learning_rate, eps=1e-4)
```

3. I also used `LeakyReLU` instead of `ReLU` in the convolutional neural network to avoid dying ReLU neurons.

```python
nn.LeakyReLU(negative_slope=0.01)
```

4. Atari environments return both `terminated` and `truncated`. Only true terminations should stop bootstrapping; time-limit truncations can still use the next-state Q-value.

```python
transition = Transition(state=state, action=action, reward=clipped_reward, next_state=next_state, done=terminated)
```

### Atari Pong Demo

Pong is a two-player Atari game. Here, vanilla DQN uses a Convolutional Neural Network that sees the state as the last four frames stacked together. After about 10 hours of training on a MacBook, the learned policy controls the paddle on the right and starts exploiting certain strategies that feel like reward hacking.

{{< rawhtml >}}
<video autoplay muted loop playsinline preload="metadata" style="display: block; width: 360px; max-width: 100%; height: auto; margin: 0 auto;">

  <source src="dqn_output.webm" type="video/webm">
  <source src="dqn_output.mp4" type="video/mp4">
</video>
{{< /rawhtml >}}

### Implementation

Feel free to take a look at [my implementation on GitHub](https://github.com/elkotito/dqn-implementations). I recommend implementing it yourself rather than having an agent do it. It is the best way to learn.

## Prioritized Replay Buffer

Uniform replay treats every stored transition as equally useful. That is simple and it already helps a lot, but it is not how learning usually feels. Some transitions are boring because the network already predicts them well. Others are surprising: the reward was unexpected, the state was rare, or the bootstrap target disagrees strongly with the current estimate. The TD error gives us a simple way to measure that surprise:

$$
\delta_i = y_i - q_\theta(s_i, a_i)
$$

If $|\delta_i|$ is large, then transition $i$ currently creates a large learning signal. Prioritized replay uses this idea by sampling those transitions more often than transitions with small TD errors. A common priority is:

$$
p_i = |\delta_i| + \epsilon
$$

where $\epsilon > 0$ keeps every transition sampleable. Without this small constant, a transition with zero priority might never be seen again, even though it could become useful later after the network changes. Then the sampling probability is:

$$
P(i) = \frac{p_i^\alpha}{\sum_j p_j^\alpha}
$$

The parameter $\alpha$ controls how much prioritization we use. If $\alpha = 0$, then $p_i^\alpha = 1$ for every transition, so we end up with uniform replay. If $\alpha$ is larger, high-error transitions are sampled more often. In practice, a common choice is $\alpha = 0.6$. It is enough to prefer high-error transitions, but not so aggressive that the replay buffer is dominated only by the largest errors.

So we changed our sampling distribution. How does that affect the gradient? Before, in uniform replay, we sampled uniformly from the $N$ stored transitions:

$$
P_{\text{Uniform}}(i) = \frac{1}{N}
$$

With uniform replay, the expected minibatch gradient estimates the average gradient over the whole buffer:

$$
\nabla_\theta J(\theta)
\approx
\frac{1}{N}\sum_{i=1}^{N}\nabla_\theta L_i(\theta)
$$

Prioritized replay samples transition $i$ with probability $P(i)$ instead, so before correction the expectation becomes:

$$
\nabla_\theta J(\theta) \approx \sum_{i=1}^{N}P(i)\nabla_\theta L_i(\theta)
$$

That is the bias introduced by prioritized sampling. High-priority transitions are no longer just seen more often, but they also count more in expectation. To correct that bias, multiply each sampled gradient by the ratio between its uniform probability and its prioritized probability:

$$
\frac{1}{N \cdot P(i)}
$$

Then the weighted prioritized gradient becomes the uniform replay gradient again:

$$
\nabla_\theta J(\theta)
\approx
\sum_{i=1}^{N}
P(i)\frac{1}{N \cdot P(i)}\nabla_\theta L_i(\theta)
= \frac{1}{N}\sum_{i=1}^{N}\nabla_\theta L_i(\theta)
$$

This may look like it removes the benefit of prioritized replay, but it does not. The bias correction changes the weight of a transition after it has already been sampled. It does not make the sampling process uniform again. High-priority transitions are still selected more often, so the agent spends more updates looking at them.

Implementation-wise, we still need a way to sample high-priority transitions efficiently. More on that in the next section, but once we have these high-priority samples in the batch, we correct the bias by weighting their losses. For the full correction, the weight is:

$$
w_i = \frac{1}{N \cdot P(i)}
$$

In PyTorch, this just means computing a per-sample loss for each sampled transition $i_k$, multiplying it by its weight, and then averaging over the batch:

$$
L(\theta) =
\frac{1}{B}\sum_{k=1}^{B}
w_{i_k} \frac{1}{2}\left(y_{i_k} - q_\theta(s_{i_k}, a_{i_k})\right)^2
$$

The factor $\frac{1}{B}$ is only the minibatch average. The bias correction still uses $N$ because $P(i)$ is a sampling probability over the replay buffer.

The weights $w_i$ are coefficients on the per-sample losses, but because the final scalar loss is weighted, the gradients with respect to $\theta$ are weighted too.

In practice, Prioritized Replay Buffer often does not use the full correction immediately. It introduces a parameter $\beta$:

$$
w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta
$$

Here $N$ is the number of transitions in the replay buffer. The parameter $\beta$ controls how strongly we correct the sampling bias. If $\beta = 0$, there is no bias correction. If $\beta = 1$, the bias correction is full.

Sometimes we intentionally want stronger gradients from high-priority samples. Early in training, the policy, targets, and priorities are changing quickly, and high-error transitions often contain the useful signal the network is currently missing. If we fully correct the bias immediately, many of those transitions are downweighted after we sampled them. That can make prioritization less aggressive exactly when we want it to move learning quickly.

So Prioritized Replay Buffer usually starts with weaker correction, for example $\beta = 0.4$, and anneals $\beta$ upward toward $1.0$ during training. Later, as the value estimates become more stable, stronger correction reduces the extra influence of high-priority samples and makes the update less biased relative to uniform replay.

### Training Curves

Here is one Pong run comparing uniform replay with prioritized replay. The prioritized replay agent reaches positive returns earlier, although both runs end up close after enough environment steps.

![Pong mean episode return with prioritized replay vs uniform replay](per_return.webp)

Since high-priority transitions appear more often in batches, the sampled batch TD error can stay higher while the policy improves.

![Pong TD error with prioritized replay vs uniform replay](per_td_error.webp)

### Implementation

The replay buffer itself can still be a normal circular buffer: arrays for states, actions, rewards, next states, and done flags. The extra part is one priority per stored transition. The algorithm is straightforward:

1. Sample a priority-weighted batch from the replay buffer.
2. Train on that batch and compute the TD error for each sampled transition.
3. Convert each TD error into a new priority.
4. Store that priority next to the transition in the regular replay buffer.

New transitions do not have a TD error yet, so they usually get the current maximum priority. That makes sure they can be sampled at least once.

The naive sampling approach is to build a probability distribution over the current buffer from these priorities, then sample from that distribution. This works, but it requires scanning across the priority array. Sampling one transition is $O(N)$, so a batch can become expensive for large replay buffers.

A sum tree is a data structure that makes this weighted sampling efficient. It stores priority sums in a binary tree, so we can sample a transition in $O(\log N)$ instead of scanning the whole buffer. Adding a new transition or updating an existing priority is also $O(\log N)$.

The actual transitions do not live in the tree. The tree only maps priority mass to replay-buffer indices. The replay buffer can still overwrite old transitions in circular order. When slot $i$ is overwritten, we also replace the tree entry for slot $i$ with the new transition's priority.

For the implementation details, see my [prioritized replay buffer code](https://github.com/elkotito/dqn-implementations/blob/main/src/buffers/prioritized_replay_buffer.py).

## Double DQN
