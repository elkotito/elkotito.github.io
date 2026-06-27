---
title: "Reinforcement Learning 103: Approximate Methods"
description: "Approximate model-free RL: function approximation, regression targets, loss functions, semi-gradient TD, approximate SARSA, Expected SARSA, Q-learning, and the deadly triad."
date: 2026-06-23
tags:
  [
    "Reinforcement Learning",
    "Model-free Reinforcement Learning",
    "Function Approximation",
    "Approximate Q-learning",
    "Semi-gradient Methods",
    "SARSA",
    "Expected SARSA",
    "Q-learning",
    "Deadly Triad",
  ]
draft: false
---

## Introduction

In the previous article, we learned model-free control with tabular action-values:

$$
q(s, a)
$$

Strictly speaking, $q$ is already a function from state-action pairs to numbers. In the tabular setting, we represent that function with an explicit table. Every state-action pair has its own cell. If the agent observes a transition from $(s, a)$, it updates that one cell and leaves the others unchanged. That is conceptually clean and enough for small grid worlds, but it does not scale.

If the state is an image, then each possible image would need a separate table row. If the state is a continuous vector, like the position and angle of a robot arm, then the table has no natural finite set of keys. Even if we discretize the state space, nearby states are often related and should share information. If a spaceship is moving directly toward us, then slightly different pixel positions should not require learning the same lesson from scratch.

So instead of storing one value for every possible state-action pair, we approximate the action-value function with a parameterized function:

$$
q_\theta(s, a)
$$

Here $\theta$ is the parameter vector. It may be the weights of a linear model, a neural network, or any other differentiable function approximator. The state $s$ and action $a$ are still inputs to the function; in an implementation they have to be represented somehow, for example as feature vectors, tensors, or embeddings. We may write the model input as $x = \phi(s, a)$, where $\phi$ is the chosen representation. The goal is to choose $\theta$ so that $q_\theta(s, a)$ is close to the action-value function we care about.

This changes the problem in an important way. In tabular learning, the estimates are independent. Updating $q(s, a)$ does not directly modify $q(\tilde{s}, \tilde{a})$ for another pair $(\tilde{s}, \tilde{a})$.

With function approximation, the same parameters are shared across many states and actions. One update to $\theta$ can improve many predictions at once, which is exactly why approximation is useful, but the same update can also damage many predictions at once. Generalization and interference come from the same mechanism.

This article is about that replacement:

$$
q(s, a) \quad \rightarrow \quad q_\theta(s, a)
$$

We will first turn action-value prediction into a regression problem. Then we will see why reinforcement learning is not ordinary supervised learning, why TD methods use semi-gradients, and how approximate SARSA, Expected SARSA, and Q-learning are written with function approximation.

## Supervised Learning

The tabular update had the form:

$$
q(s, a) \leftarrow q(s, a) + \alpha(y - q(s, a))
$$

The target $y$ depended on the algorithm. For example, for SARSA:

$$
y = r + \gamma q(s^\prime, a^\prime)
$$

The update moves the current estimate toward the target. When $q$ is a table, this is a direct assignment to one table entry. With function approximation, the estimate is produced by a function:

$$
q_\theta(s, a)
$$

So the update can no longer say "change this table cell". Instead, it must say "change the parameters so that the function output at $(s, a)$ moves toward the target". This makes the problem look like supervised learning. Abstractly, the input is the state-action pair:

$$
x = (s, a)
$$

For an actual function approximator, this usually means feeding in the chosen representation $\phi(s, a)$. The model prediction is:

$$
\hat{y} = q_\theta(s, a)
$$

And the target is some number $y$ that we want the prediction to match. The analogy is useful, but incomplete. In ordinary supervised learning, we usually start with a fixed dataset:

$$
\{(x_i, y_i)\}_{i=1}^{N}
$$

The inputs are known, the labels are known, and the optimization problem is fixed before training begins. In reinforcement learning, none of this is so clean.

- The input $(s, a)$ appears only when the agent or another behavior policy visits it.
- The data distribution depends on the policy used to collect experience.
- The target $y$ is not given by a teacher. It is estimated from returns in Monte Carlo learning, or bootstrapped from the current action-value estimate in TD learning.

So approximate RL borrows tools from supervised learning, but the learning problem keeps moving while we train.

## Loss Function

For a sampled pair $(S_t, A_t)$ with target $y_t$, the simplest objective is squared error:

$$
L_t(\theta) = \frac{1}{2}\left(y_t - q_\theta(S_t, A_t)\right)^2
$$

Huber loss is also common in deep RL because occasional large TD errors can produce large gradients. The important question is where $y_t$ comes from. Monte Carlo targets use sampled returns. TD targets bootstrap from another action-value estimate, which makes the target available earlier, but also makes it depend on the current approximation.

After that deciding on the target, we still need optimize this loss when both the target and the data distribution come from interactions.

### Monte Carlo

The Monte Carlo target is the sampled return:

$$
y_t^{\text{MC}} = G_t
$$

The Monte Carlo target has a nice property: once the episode is finished, $G_t$ is just a number. It does not depend on $\theta$. From the point of view of gradient descent, it behaves like an ordinary supervised label. This means Monte Carlo does not have bootstrap bias: errors in the current estimate $q_\theta$ do not enter the target.

The downside is that we must wait until the return is known. For long episodes this delays learning. For continuing tasks, there may be no natural episode end. Monte Carlo targets can also have high variance because $G_t$ is a sum over the rest of the trajectory. Each $R_t$, $R_{t+1}$, $R_{t+2}$ etc. is a random variable and the more random variables we sum together, the higher the variance.

### Temporal Difference

The one-step SARSA target is:

$$
y_t^{\text{SARSA}}(\theta) = R_t + \gamma q_\theta(S_{t+1}, A_{t+1})
$$

Compared with Monte Carlo, this target usually has lower variance because it does not sum over the whole remaining trajectory. The cost is that it uses our current estimate $q_\theta$. If $q_\theta(S_{t+1}, A_{t+1})$ is wrong, then the target is wrong too.

You might wonder now: why would a model learn from a target that can be wrong? Recall contraction mappings and the Banach fixed-point theorem. The whole idea is that recursive Bellman-style equations can start from a wrong estimate, and repeatedly applying them can improve that estimate.

Having said that, the wrong target itself is not automatically the problem. The problem to keep in mind is the target can be systematically shifted every time we update $\theta$. It might lead to unstable learning. We will solve that problem in the next article by introducing a target network.

Another thing to keep in mind is that the TD target has bootstrap bias. In approximate TD, we do not know the true value $q^\pi$, so we rely on the estimate produced by our approximator, for example a neural network $q_\theta$. In statistics, bias is the difference between the expected value of an estimator and the true value.

In SARSA, once the next state $S_{t+1}$ has been observed, the target is a sum with two random variables: the reward $R_t$ and the sampled next action $A_{t+1}$. Expected SARSA has the same bootstrap bias as SARSA, but lower variance in the target because it removes the next-action random variable $A_{t+1}$ and replaces it with a policy expectation:

$$
y_t^{\text{Expected SARSA}}(\theta) =
R_t + \gamma \sum_{a^\prime}\pi(a^\prime \mid S_{t+1})q_\theta(S_{t+1}, a^\prime)
$$

Q-learning uses the Bellman optimality target:

$$
y_t^{\text{Q-learning}}(\theta) =
R_t + \gamma \max_{a^\prime}q_\theta(S_{t+1}, a^\prime)
$$

Similarly to Expected SARSA, Q-learning does not sample the next action for the target. However, Q-learning can introduce overestimation bias: when action-value estimates are noisy, the max tends to select actions whose estimates are too high. We will describe that in more detail in the next article.

To sum up, Monte Carlo uses completed returns, but it has to wait and can have high variance. TD uses imperfect bootstrap targets, but it can update after one transition and often has lower variance. A useful empirical reminder is the paper [TD or not TD: Analyzing the Role of Temporal Differencing in Deep Reinforcement Learning](https://arxiv.org/abs/1806.01175), which found that finite-horizon Monte Carlo targets can be competitive in deep RL settings. The practical choice is a bias-variance and engineering tradeoff, not a universal rule.

### Training Distribution

Now assume we can produce a target $y_t$ from Monte Carlo or TD. The next question is what objective this sampled loss represents.

In supervised learning, the loss is usually an average over the training dataset. In approximate RL, it is better to think of the loss as an expectation over the state-action pairs we train on. For a discrete state-action space, an ideal weighted squared error would look like:

$$
J(\theta) =
\frac{1}{2}\sum_{s,a}\mu(s, a)
\left(y(s, a) - q_\theta(s, a)\right)^2
$$

Here $\mu(s, a)$ is the distribution that says how much we care about each state-action pair. For continuous variables, the corresponding sums become integrals, but the idea is the same.

This weighting matters because with function approximation we usually cannot be perfectly accurate everywhere. The model has limited capacity, the state-action space is large, and some regions are visited much more often than others. The distribution $\mu$ tells the optimizer where errors are expensive.

In on-policy learning, samples come from the same policy we are evaluating or improving. A rough way to think about the update distribution is:

$$
\mu_\pi(s, a) = d_\pi(s)\pi(a \mid s)
$$

where $d_\pi(s)$ is the state visitation distribution under policy $\pi$. We multiply by $\pi(a \mid s)$ because $\mu_\pi$ is a distribution over state-action pairs. The first term says how often the policy reaches state $s$; the second says how often it chooses action $a$ once it is there. This means the loss emphasizes states the policy actually visits and actions it actually chooses.

In off-policy learning, the samples come from a behavior policy $b$:

$$
\mu_b(s, a) = d_b(s)b(a \mid s)
$$

The target may still describe a different policy, but the updates happen where the behavior policy provides data.

A common misconception is to read $\mu(s, a)$ as how important the state is in some human sense, but it is the training distribution. If the agent rarely visits a dangerous state, then ordinary SGD rarely updates that state. If we want to learn more accurately there, we need more samples, different exploration, replay, prioritization, or explicit weighting.

In practice, we cannot compute loss $J(\theta)$ directly:

- The state-action space may be huge or continuous.
- The target value for every pair $(s, a)$ is not known in advance.
- The visitation distribution $\mu$ is observed through rollouts rather than available as a full table.

The useful part is that we do not need to compute $J(\theta)$ exactly to improve $\theta$. Rollouts give us sampled state-action pairs from this distribution, so we can use stochastic gradient descent: estimate an update direction from sampled experience instead of evaluating the full expectation.

## Stochastic Gradient Descent

### Monte Carlo

First consider the easy case where the target $y_t$ is fixed with respect to $\theta$. This is true for a completed Monte Carlo return $G_t$.

The sample loss is:

$$
L_t(\theta) =
\frac{1}{2}\left(y_t - q_\theta(S_t, A_t)\right)^2
$$

The gradient is:

$$
\nabla_\theta L_t(\theta)
= -\left(y_t - q_\theta(S_t, A_t)\right)
\nabla_\theta q_\theta(S_t, A_t)
$$

Gradient descent subtracts this gradient:

$$
\theta \leftarrow \theta + \alpha
\left(y_t - q_\theta(S_t, A_t)\right)
\nabla_\theta q_\theta(S_t, A_t)
$$

This is the basic approximate action-value update. It says: change the parameters in the direction that would increase the prediction if the target is larger than the prediction, and decrease the prediction if the target is smaller. For a linear function approximator, this becomes especially simple. Suppose:

$$
q_\theta(s, a) = \theta^\top \phi(s, a)
$$

where $\phi(s, a)$ is a feature vector. Then:

$$
\nabla_\theta q_\theta(s, a) = \phi(s, a)
$$

and the update is:

$$
\theta \leftarrow \theta + \alpha
\left(y_t - q_\theta(S_t, A_t)\right)
\phi(S_t, A_t)
$$

This looks very close to a tabular update. In fact, a table can be viewed as a special case of linear approximation where every state-action pair has its own one-hot feature. The difference is that in a general feature representation, multiple state-action pairs share features. Updating $\theta$ for one pair also changes predictions for other pairs with overlapping features.

### Semi-gradient Methods

TD targets are different from Monte Carlo targets because they can depend on $\theta$.

For SARSA, the target is:

$$
y_t(\theta) = R_t + \gamma q_\theta(S_{t+1}, A_{t+1})
$$

The loss is:

$$
L_t(\theta) =
\frac{1}{2}\left(y_t(\theta) - q_\theta(S_t, A_t)\right)^2
$$

If we take the full gradient, the target also has a derivative:

$$
\nabla_\theta L_t(\theta) = \left(y_t(\theta) - q_\theta(S_t, A_t)\right) \left( \nabla_\theta y_t(\theta) - \nabla_\theta q_\theta(S_t, A_t) \right)
$$

For the SARSA target:

$$
\nabla_\theta y_t(\theta) = \gamma \nabla_\theta q_\theta(S_{t+1}, A_{t+1})
$$

So the full gradient would update parameters through both the current prediction and the next-state prediction inside the target.

TD methods usually do not do that. They use a semi-gradient. A semi-gradient treats the target as fixed for the purpose of the current update, even when the target was computed from the current action-value function. The semi-gradient update ignores $\nabla_\theta y_t(\theta)$:

$$
\theta \leftarrow \theta + \alpha
\left(y_t(\theta) - q_\theta(S_t, A_t)\right)
\nabla_\theta q_\theta(S_t, A_t)
$$

In PyTorch, this means stopping gradients through the target while keeping gradients through the current prediction. A common pattern is:

```python
with torch.no_grad():
    target = reward + gamma * neural_network(next_state, next_action)

prediction = neural_network(state, action)
loss = 0.5 * (target - prediction).pow(2).mean()
```

The intuition is practical. The bootstrap target is acting as a temporary label. We want to move the current action-value estimate toward the estimate of the next step. If we allowed the update to also change the next-step estimate, the model could reduce the loss by moving the target instead of improving the current prediction.

For Q-learning with function approximation, the same issue appears. The target:

$$
y_t(\theta) = R_t + \gamma \max_{a^\prime}q_\theta(S_{t+1}, a^\prime)
$$

also depends on $\theta$. Approximate Q-learning normally uses the semi-gradient update:

$$
\theta \leftarrow \theta + \alpha \left(R_t + \gamma \max_{a^\prime}q_\theta(S_{t+1}, a^\prime) - q_\theta(S_t, A_t) \right) \nabla_\theta q_\theta(S_t, A_t)
$$

The fact that a behavior policy collected the transition does not make the target fixed. The sampled transition $(S_t, A_t, R_t, S_{t+1})$ is fixed after it is observed, but the bootstrap value $\max_{a^\prime}q_\theta(S_{t+1}, a^\prime)$ still depends on the parameters. One way to make the target fixed with respect to the current update is to use a separate target network:

$$
y_t = R_t + \gamma \max_{a^\prime}q_{\theta^-}(S_{t+1}, a^\prime)
$$

where $\theta^-$ is held constant while updating $\theta$. This idea becomes central in Deep Q-Networks, and we will return to it in the next article.

## Algorithms

The approximate algorithms differ mainly in how they build the target. The training loop has the same shape:

1. Collect transitions $(S, A, R, S^\prime)$ and optionally $A'$ by acting in the environment.
2. Form a minibatch $\mathcal{B}$ from the collected transitions. This may be a short segment of recent experience or the most recent rollout.
3. Build one scalar target $y_i$ for each transition in the minibatch.
4. Take one gradient or semi-gradient step using the average minibatch loss.

Here $i$ indexes transitions in the minibatch. The sampled loss is:

$$
L_{\mathcal{B}}(\theta)
= \frac{1}{|\mathcal{B}|}
\sum_{i \in \mathcal{B}}
\frac{1}{2}\left(y_i - q_\theta(S_i, A_i)\right)^2
$$

For Monte Carlo, wait until the return is known and set:

$$
y_i = G_i
$$

For one-step TD methods, use $y_i = R_i$ if $S_i^\prime$ is terminal. Otherwise:

$$
y_i(\theta) =
\begin{cases}
R_i + \gamma q_\theta(S_i^\prime, A_i^\prime),
& \text{SARSA} \\
R_i + \gamma \sum_{a^\prime}\pi(a^\prime \mid S_i^\prime)q_\theta(S_i^\prime, a^\prime),
& \text{Expected SARSA} \\
R_i + \gamma \max_{a^\prime}q_\theta(S_i^\prime, a^\prime),
& \text{Q-learning}
\end{cases}
$$

For TD methods, treat $y_i(\theta)$ as fixed during the current update. In code, this means detaching the target before computing the minibatch loss.

## Why Approximate RL Is Hard

At this point, approximate RL may look like a small modification of supervised learning:

1. Collect experience.
2. Build targets.
3. Take gradient steps.

The difficulty is that each of those steps hides a problem.

### Generalization and Interference

Function approximation generalizes; that is the point. If two state-action pairs share features, then learning from one pair can improve predictions for the other. This is what lets us handle large state-action spaces without experiencing every possible state-action combination separately.

The same shared parameters also create interference. An update from one transition can change predictions for many other state-action pairs, including ones we did not intend to change. In tabular learning, a bad update is local to one cell; in approximate learning, it can spread through the function approximator. With neural networks this is especially visible, because a single gradient step can change the representation used by many inputs. Sometimes that improves generalization, and sometimes it damages behavior that was previously working.

### Correlated Data

The clean SGD picture assumes that mini-batches give a reasonably representative estimate of the gradient of the training objective. This is easiest when samples are approximately independent and identically distributed (IID), or at least well mixed. The samples do not have to be perfectly independent, but strong correlation can make optimization worse.

RL data is naturally correlated, so it violates this IID picture. During a rollout, consecutive states are close in time and often close in content. If the agent is driving down a road, many adjacent frames look almost the same. If we train directly on this stream, the model may overfit to the most recent part of experience and move away from older knowledge.

Correlated updates also make the gradient estimates less representative of the broader training distribution. Instead of getting a useful average direction, the optimizer may chase whatever small region the agent recently visited.

This is one reason experience replay is useful. A replay buffer lets the agent train on a more mixed batch of past transitions instead of only the latest transition, but then it works only for off-policy algorithms.

### Moving Targets

We already saw this in the semi-gradient section. During one gradient step, we detach the TD target and treat it like a temporary label.

But the label is fixed only for that step. After we update $\theta$, the action-value estimates change. The next TD target is then built from the new $q_\theta$, so the target changes too.

This is the moving target problem: the model is training against targets produced by its own changing predictions. The problem is not simply that the targets are wrong. Bootstrap targets are usually approximate. The problem is that a noisy or overestimated value can be used as a target, learned by earlier state-action pairs, and then reused in later targets. A deeper explanation will be in the next article.

### Moving Data Distribution

In ordinary supervised learning, the dataset is often fixed. In RL, the data distribution is produced by a behavior policy. If that behavior policy is tied to the current $q_\theta$, then changing $\theta$ also changes what data we collect.

For example, if the behavior policy chooses actions $\epsilon$-greedily according to $q_\theta$, then changing $\theta$ changes both how the agent exploits and how it explores. That changes which states it visits, which actions it tries, and which rewards it observes. This is clearly true for on-policy methods such as SARSA, where the policy being evaluated is also the policy collecting data.

For Q-learning, the target is greedy rather than defined by the behavior policy, so the update can use off-policy data, but it is the behavior policy that determines which transitions are observed. The moving distribution comes from data collection, not from the target itself. With a fixed offline dataset this movement disappears, although the target may still ask for values of actions that are rare or missing in the data.

### Sharp Value Boundaries

Function approximators are often smooth functions. Similar inputs tend to produce similar outputs. That is helpful when similarity in input space matches similarity in action-value, but in RL, tiny changes in state can sometimes cause huge changes in return.

Imagine a helicopter flying close to a tree. A small change in position may be the difference between passing safely and crashing. The reward changes abruptly because one trajectory continues and the other terminates with a large penalty. In other words, two inputs can look close in raw pixels or coordinates while requiring very different value predictions.

This does not always mean the true action-value function is mathematically non-differentiable, although it can be. The practical issue is the action-value function may have sharp boundaries or high curvature. A smooth approximator can smear the boundary and assign unsafe values to states near failure.

### The Deadly Triad

The feedback loop above is especially dangerous in what is usually called the deadly triad:

1. Function approximation
2. Bootstrapping
3. Off-policy learning

None of these is suspicious on its own. Function approximation is how we handle large state-action spaces. Bootstrapping is how we update before the full return is known. Off-policy learning is how we learn from data generated by a different policy.

The trouble starts when they are combined. With function approximation, an update for one state-action pair also changes predictions for other pairs. With bootstrapping, those changed predictions are used as targets for later updates. With off-policy learning, the data may be weak exactly where the target asks for values, so the model has to extrapolate.

For example, suppose an action-value is too high in a poorly covered next state. A bootstrapped target uses that value. The update then increases the estimate for a previous state-action pair. Because parameters are shared, that same update may also change other predictions. Later targets can use those changed predictions again. The error is no longer isolated in one table entry; it becomes part of the training signal.

This is where the tabular fixed-point intuition stops being enough. The update may still settle down, but it is no longer guaranteed to behave like a simple contraction. It can oscillate or diverge.

This is not just a neural-network problem. Divergence examples exist even with linear function approximation. Papers such as [Breaking the Deadly Triad with a Target Network](https://arxiv.org/abs/2101.08862) study how target networks can stabilize some of these dynamics.

The deadly triad doesn't mean that every approximate off-policy TD method fails. DQN uses all three ingredients and can work very well. The whole point is that stability is no longer automatically guaranteed.

## Final Thoughts

Tabular model-free control represents the action-value function with one number per state-action pair. Approximate model-free control replaces that explicit table representation with a parameterized function:

$$
q_\theta(s, a)
$$

This lets the agent generalize to large, continuous, or high-dimensional state-action spaces. It also makes each update less local. One gradient step can change many predictions at once.

Monte Carlo targets turn RL into the most supervised-looking regression problem, because the completed return is a fixed number. TD targets update sooner, but they bootstrap from current estimates, so the target itself can move. Semi-gradient methods handle this by treating the target as fixed during the current update.

Approximate SARSA, Expected SARSA, and Q-learning all share the same update shape.
They differ in the target used to compute $\delta_t$: sampled next action, expected next action, or greedy next action.

$$
\theta \leftarrow \theta + \alpha \delta_t
\nabla_\theta q_\theta(S_t, A_t)
$$

The main lesson is that function approximation changes the learning dynamics. Correlated samples, moving data distributions, sharp value boundaries, moving targets, and the deadly triad all appear because we no longer have independent table entries.

The next step is to make approximate Q-learning work with neural networks. That leads to Deep Q-Networks, where replay buffers and target networks are introduced as practical answers to the instability described here.
