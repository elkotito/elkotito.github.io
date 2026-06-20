---
title: "Reinforcement Learning 103a: Approximate Methods for Model-free RL"
description: "Approximate model-free Reinforcement Learning from first principles: function approximation, regression targets, loss functions, semi-gradient TD, approximate SARSA, Expected SARSA, Q-learning, and the deadly triad."
date: 2026-06-20
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
draft: true
---

## Introduction

In the previous article, we learned model-free control with tabular action-values:

$$
q(s, a)
$$

The table version is conceptually clean. Every state-action pair has its own cell. If the agent observes a transition from $(s, a)$, it updates that one cell and leaves the others unchanged. That is enough for small grid worlds, but it does not scale.

If the state is an image, then each possible image would need a separate table row. If the state is a continuous vector, like the position and angle of a robot arm, then the table has no natural finite set of keys. Even if we discretize the state space, nearby states are often related and should share information. If a spaceship is moving directly toward us, then slightly different pixel positions should not require learning the same lesson from scratch.

So instead of storing one value for every possible state-action pair, we approximate the action-value function with a parameterized function:

$$
q_\theta(s, a)
$$

Here $\theta$ is the parameter vector. It may be the weights of a linear model, a neural network, or any other differentiable function approximator. The goal is to choose $\theta$ so that $q_\theta(s, a)$ is close to the action-value function we care about.

This changes the problem in an important way. In tabular learning, the estimates are independent. Updating $q(s, a)$ does not directly modify $q(\tilde{s}, \tilde{a})$ for another pair $(\tilde{s}, \tilde{a})$.

With function approximation, the same parameters are shared across many states and actions. One update to $\theta$ can improve many predictions at once, which is exactly why approximation is useful, but the same update can also damage many predictions at once. Generalization and interference come from the same mechanism.

This article is about that replacement:

$$
q(s, a) \quad \rightarrow \quad q_\theta(s, a)
$$

We will first turn value prediction into a regression problem. Then we will see why reinforcement learning is not ordinary supervised learning, why TD methods use semi-gradients, and how approximate SARSA, Expected SARSA, and Q-learning are written with function approximation.

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

So the update can no longer say "change this table cell". Instead, it must say "change the parameters so that the function output at $(s, a)$ moves toward the target". This makes the problem look like supervised learning. The input is:

$$
x = (s, a)
$$

The model prediction is:

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

## Loss Functions

For a sampled pair $(S_t, A_t)$ with target $y_t$, the simplest objective is squared error:

$$
L_t(\theta) = \frac{1}{2}\left(y_t - q_\theta(S_t, A_t)\right)^2
$$

The factor $\frac{1}{2}$ is only there because it makes the derivative cleaner. It does not change the minimizer.

Huber loss is also common in deep RL because occasional large TD errors can produce large gradients. Here, MSE is enough because it keeps the update algebra simple.

The important question is where $y_t$ comes from. Monte Carlo targets use sampled returns. TD targets bootstrap from another value estimate, which makes the target available earlier but also makes it depend on the current approximator.

After that, we still need to ask how to optimize this loss when both the target and the data distribution come from interaction.

## Finding Targets

### Monte Carlo

The Monte Carlo target is the sampled return:

$$
y_t^{\text{MC}} = G_t
$$

The Monte Carlo target has a nice property: once the episode is finished, $G_t$ is just a number. It does not depend on $\theta$. From the point of view of gradient descent, it behaves like an ordinary supervised label.

The downside is that we must wait until the return is known. For long episodes this delays learning. For continuing tasks, there may be no natural episode end. Monte Carlo targets can also have high variance because the full return depends on every random event that happens after time $t$.

This is why TD methods are popular: they update sooner and usually have lower variance. But it is not true that TD is always better. TD introduces bias through bootstrapping. A useful empirical reminder is the paper [TD or not TD: Analyzing the Role of Temporal Differencing in Deep Reinforcement Learning](https://arxiv.org/abs/1806.01175), which found that finite-horizon Monte Carlo targets can be competitive in deep RL settings. The practical choice is a bias-variance and engineering tradeoff, not a universal rule.

### Temporal Difference

The one-step SARSA target is:

$$
y_t^{\text{SARSA}} = R_t + \gamma q_\theta(S_{t+1}, A_{t+1})
$$

The cost is that the target uses our current estimate $q_\theta$. If $q_\theta(S_{t+1}, A_{t+1})$ is wrong, then the target is wrong too. This is what people mean when they say TD targets are biased: the expected TD target equals the true Bellman target only when the current action-value estimate in the bootstrap term is already correct.

This bias is not necessarily bad. Bootstrapping often gives much more sample-efficient learning. It just means we should remember that TD is not a fixed-label supervised learning problem. The label is partly produced by the model we are training.

Expected SARSA reduces one source of randomness by replacing the sampled next action with a policy expectation:

$$
y_t^{\text{Expected SARSA}} =
R_t + \gamma \sum_{a^\prime}\pi(a^\prime \mid S_{t+1})q_\theta(S_{t+1}, a^\prime)
$$

Q-learning uses the Bellman optimality target:

$$
y_t^{\text{Q-learning}} =
R_t + \gamma \max_{a^\prime}q_\theta(S_{t+1}, a^\prime)
$$

## Optimizing the Loss

Now assume we can produce a target $y_t$ from Monte Carlo or TD. The next question is how to minimize the loss.

In supervised learning, the loss is usually an average over the training dataset. In approximate RL, it is better to think of the loss as an expectation over the state-action pairs we train on.

For a discrete state-action space, an ideal weighted squared error would look like:

$$
J(\theta) =
\frac{1}{2}\sum_s \sum_a \mu(s, a)
\left(y(s, a) - q_\theta(s, a)\right)^2
$$

Here $\mu(s, a)$ is the distribution that says how much we care about each state-action pair. For continuous states, the sums become integrals, but the idea is the same.

This weighting matters because with function approximation we usually cannot be perfectly accurate everywhere. The model has limited capacity, the state space is large, and some regions are visited much more often than others. The distribution $\mu$ tells the optimizer where errors are expensive.

In on-policy learning, samples come from the same policy we are evaluating or improving. A rough way to think about the update distribution is:

$$
\mu_\pi(s, a) = d_\pi(s)\pi(a \mid s)
$$

where $d_\pi(s)$ is the state visitation distribution under policy $\pi$. This means the loss emphasizes states the policy actually visits and actions it actually chooses.

In off-policy learning, the samples come from a behavior policy $b$:

$$
\mu_b(s, a) = d_b(s)b(a \mid s)
$$

The target may still describe a different policy, but the updates happen where the behavior policy provides data.

This corrects an easy misconception. The weight $\mu(s, a)$ is not automatically "how important the state is" in some human sense. It is the training distribution. If the agent rarely visits a dangerous state, then ordinary SGD rarely updates that state. If we want to learn more accurately there, we need more samples, different exploration, replay, prioritization, or explicit weighting.

In practice, we cannot compute $J(\theta)$ exactly.

- The state space may be huge or continuous.
- The true target $y(s, a)$ is unknown.
- The visitation distribution $\mu$ is only available through rollouts.

So we use stochastic gradient descent. Each sampled transition gives a noisy estimate of the gradient of the larger objective. Instead of summing over all states and actions, we update from the state-action pairs that appear in experience.

This is the "distribution of rollouts" view: the agent's experience stream defines which parts of the state-action space are trained more often.

## Gradient Descent with Fixed Targets

First consider the easy case where the target $y_t$ is fixed with respect to $\theta$. This is true for a completed Monte Carlo return $G_t$.

The sample loss is:

$$
L_t(\theta) =
\frac{1}{2}\left(y_t - q_\theta(S_t, A_t)\right)^2
$$

Define the prediction error:

$$
\delta_t = y_t - q_\theta(S_t, A_t)
$$

Then the gradient is:

$$
\nabla_\theta L_t(\theta)
= -\delta_t \nabla_\theta q_\theta(S_t, A_t)
$$

Gradient descent subtracts this gradient:

$$
\theta \leftarrow \theta + \alpha \delta_t \nabla_\theta q_\theta(S_t, A_t)
$$

This is the basic approximate value update. It says: change the parameters in the direction that would increase the prediction if the target is larger than the prediction, and decrease the prediction if the target is smaller.

For a linear function approximator, this becomes especially simple. Suppose:

$$
q_\theta(s, a) = \theta^\top x(s, a)
$$

where $x(s, a)$ is a feature vector. Then:

$$
\nabla_\theta q_\theta(s, a) = x(s, a)
$$

and the update is:

$$
\theta \leftarrow \theta + \alpha \delta_t x(S_t, A_t)
$$

This looks very close to a tabular update. In fact, a table can be viewed as a special case of linear approximation where every state-action pair has its own one-hot feature.

The difference is that in a general feature representation, multiple state-action pairs share features. Updating $\theta$ for one pair also changes predictions for other pairs with overlapping features.

## Semi-gradient Methods

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
\nabla_\theta L_t(\theta)
=
\left(y_t(\theta) - q_\theta(S_t, A_t)\right)
\left(
\nabla_\theta y_t(\theta)
-
\nabla_\theta q_\theta(S_t, A_t)
\right)
$$

For the SARSA target:

$$
\nabla_\theta y_t(\theta)
=
\gamma \nabla_\theta q_\theta(S_{t+1}, A_{t+1})
$$

So the full gradient would update parameters through both the current prediction and the next-state prediction inside the target.

TD methods usually do not do that. They use a semi-gradient. A semi-gradient treats the target as fixed for the purpose of the current update, even when the target was computed from the current value function.

So we still define:

$$
\delta_t = y_t - q_\theta(S_t, A_t)
$$

but the update ignores $\nabla_\theta y_t$:

$$
\theta \leftarrow \theta + \alpha \delta_t \nabla_\theta q_\theta(S_t, A_t)
$$

The intuition is practical. The bootstrap target is acting as a temporary label. We want to move the current estimate toward the estimate of the next step. If we also change the next-step estimate to reduce the same loss, the model can reduce the error by moving the target instead of improving the current prediction.

This does not mean semi-gradient methods are mathematically meaningless. There are important convergence results for on-policy TD with linear function approximation under suitable assumptions. But the guarantee is weaker than ordinary supervised learning, and once we combine nonlinear approximators, control, or off-policy learning, simple convergence stories disappear.

For Q-learning with function approximation, the same issue appears. The target:

$$
y_t(\theta) = R_t + \gamma \max_{a^\prime}q_\theta(S_{t+1}, a^\prime)
$$

also depends on $\theta$. Approximate Q-learning normally uses the semi-gradient update:

$$
\theta \leftarrow \theta + \alpha
\left(
R_t + \gamma \max_{a^\prime}q_\theta(S_{t+1}, a^\prime)
- q_\theta(S_t, A_t)
\right)
\nabla_\theta q_\theta(S_t, A_t)
$$

The fact that a behavior policy collected the transition does not make the target fixed. The sampled transition $(S_t, A_t, R_t, S_{t+1})$ is fixed after it is observed, but the bootstrap value $\max_{a^\prime}q_\theta(S_{t+1}, a^\prime)$ still depends on the parameters.

One way to make the target fixed with respect to the current update is to use a separate target network:

$$
y_t = R_t + \gamma \max_{a^\prime}q_{\theta^-}(S_{t+1}, a^\prime)
$$

where $\theta^-$ is held constant while updating $\theta$. This idea becomes central in Deep Q-Networks.

## Approximate SARSA

Approximate SARSA is the function-approximation version of the tabular SARSA update.

The policy $\pi$ is usually derived from the current action-value estimates. For example, it may be $\epsilon$-greedy with respect to $q_\theta$.

At time $t$, we have:

$$
S_t, A_t, R_t, S_{t+1}, A_{t+1}
$$

where $A_{t+1}$ is sampled from the same policy $\pi$ that the algorithm is evaluating.

For a non-terminal next state, the target is:

$$
y_t = R_t + \gamma q_\theta(S_{t+1}, A_{t+1})
$$

The TD error is:

$$
\delta_t =
R_t + \gamma q_\theta(S_{t+1}, A_{t+1})
- q_\theta(S_t, A_t)
$$

and the semi-gradient update is:

$$
\theta \leftarrow \theta + \alpha \delta_t
\nabla_\theta q_\theta(S_t, A_t)
$$

For a terminal next state:

$$
\delta_t = R_t - q_\theta(S_t, A_t)
$$

There are two sources of sampling noise in the non-terminal SARSA target.

First, the environment samples the reward and next state:

$$
R_t, S_{t+1}
$$

Second, the policy samples the next action:

$$
A_{t+1} \sim \pi(\cdot \mid S_{t+1})
$$

This second source of randomness is not always necessary. If the action space is small and discrete, we can average over next actions exactly. That gives Expected SARSA.

## Approximate Expected SARSA

Expected SARSA uses the same sampled transition $(S_t, A_t, R_t, S_{t+1})$, but it does not sample the next action for the target. It computes the expected next value under the target policy:

$$
\sum_{a^\prime}\pi(a^\prime \mid S_{t+1})q_\theta(S_{t+1}, a^\prime)
$$

For a non-terminal next state, the target is:

$$
y_t =
R_t + \gamma
\sum_{a^\prime}\pi(a^\prime \mid S_{t+1})q_\theta(S_{t+1}, a^\prime)
$$

The TD error is:

$$
\delta_t =
R_t + \gamma
\sum_{a^\prime}\pi(a^\prime \mid S_{t+1})q_\theta(S_{t+1}, a^\prime)
- q_\theta(S_t, A_t)
$$

and the semi-gradient update is:

$$
\theta \leftarrow \theta + \alpha \delta_t
\nabla_\theta q_\theta(S_t, A_t)
$$

Expected SARSA usually has lower variance than SARSA because it removes the randomness from the sampled next action. The reward and next state are still sampled from the environment, but the action part of the target is averaged exactly.

The cost is that we need to know the action probabilities and sum over actions. This is cheap for small discrete action spaces. It is harder for large or continuous action spaces.

Expected SARSA can be on-policy or off-policy.

It is on-policy when the transitions are collected by the same policy $\pi$ used inside the expectation. It is off-policy when a behavior policy $b$ collects the transition, but the expectation uses a different target policy $\pi$.

The current action $A_t$ in the sampled transition does not have to come from $\pi$ for the one-step target to make sense, because $q_\pi(s, a)$ is defined as: take action $a$ first, then follow $\pi$. But the behavior policy still matters because it determines which state-action pairs get updated. Without coverage, we cannot learn accurate values for pairs that never appear in the data.

## Approximate Q-learning

Approximate Q-learning uses the Bellman optimality target. It does not evaluate the exploratory behavior policy. It learns toward the greedy target:

$$
q^*(s, a)
$$

For a sampled transition:

$$
(S_t, A_t, R_t, S_{t+1})
$$

the non-terminal target is:

$$
y_t = R_t + \gamma \max_{a^\prime}q_\theta(S_{t+1}, a^\prime)
$$

The TD error is:

$$
\delta_t =
R_t + \gamma \max_{a^\prime}q_\theta(S_{t+1}, a^\prime)
- q_\theta(S_t, A_t)
$$

and the semi-gradient update is:

$$
\theta \leftarrow \theta + \alpha \delta_t
\nabla_\theta q_\theta(S_t, A_t)
$$

For a terminal next state:

$$
\delta_t = R_t - q_\theta(S_t, A_t)
$$

This is the direct approximate version of tabular Q-learning.

It is off-policy because the target uses a greedy action in the next state, regardless of which action the behavior policy actually takes next. The behavior policy can be $\epsilon$-greedy, fully exploratory, or even an older policy that produced logged data. The update still points toward the greedy Bellman optimality target.

However, off-policy does not mean "data does not matter." The behavior policy controls the training distribution. If the data never covers an important state-action pair, Q-learning cannot learn its value. With function approximation, lack of coverage is even more dangerous because the model may extrapolate values for unseen actions and then bootstrap from those extrapolations.

Approximate Q-learning is the conceptual bridge to DQN. DQN keeps the same target idea, but adds engineering tools such as replay buffers and target networks to make neural Q-learning stable enough in practice.

## Algorithm View

The approximate control algorithms share the same loop.

1. Initialize parameters $\theta$.
2. Choose an action $A_t$ using a behavior policy, often $\epsilon$-greedy with respect to $q_\theta$.
3. Take the action and observe $R_t$ and $S_{t+1}$.
4. Build a target $y_t$.
5. Compute the TD error:

   $$
   \delta_t = y_t - q_\theta(S_t, A_t)
   $$

6. Apply the semi-gradient update:

   $$
   \theta \leftarrow \theta + \alpha \delta_t
   \nabla_\theta q_\theta(S_t, A_t)
   $$

The algorithms differ only in step 4.

SARSA samples the next action:

$$
y_t = R_t + \gamma q_\theta(S_{t+1}, A_{t+1})
$$

Expected SARSA averages over the target policy:

$$
y_t =
R_t + \gamma \sum_{a^\prime}\pi(a^\prime \mid S_{t+1})
q_\theta(S_{t+1}, a^\prime)
$$

Q-learning uses the greedy target:

$$
y_t = R_t + \gamma \max_{a^\prime}q_\theta(S_{t+1}, a^\prime)
$$

For terminal next states, all three targets reduce to:

$$
y_t = R_t
$$

This similarity is useful. Approximate SARSA, Expected SARSA, and Q-learning are not three unrelated algorithms. They are three ways to choose the bootstrap term in the same approximate regression update.

## Why Approximate RL Is Hard

At this point, approximate RL may look like a small modification of supervised learning:

1. Collect a sample.
2. Build a target.
3. Take a gradient step.

The difficulty is that each of those steps hides a problem.

### Generalization and Interference

Function approximation generalizes. That is the point.

If two states share features, then learning from one state can improve predictions in the other. This is what lets us handle large state spaces. We do not need to experience every possible image or every possible robot position.

But shared parameters also cause interference. An update from one transition can change values for many other states and actions, including states we did not intend to change.

In tabular learning, a bad update is local. In approximate learning, a bad update can spread through the function approximator.

This is especially visible with neural networks. A single gradient step can change the representation used by many states. Sometimes that improves generalization. Sometimes it makes the agent forget behavior that was previously working.

### Correlated Data

Supervised learning usually works best when mini-batches are reasonably mixed. The samples do not have to be perfectly independent, but strong correlation can make optimization worse.

RL data is naturally correlated. During a rollout, consecutive states are close in time and often close in content. If the agent is driving down a road, many adjacent frames look almost the same. If we train directly on this stream, the model may overfit to the most recent part of experience and move away from older knowledge.

Correlated updates also make the gradient estimates less representative of the broader training distribution. Instead of getting a useful average direction, the optimizer may chase whatever small region the agent recently visited.

This is one reason experience replay is useful in DQN. A replay buffer lets the agent train on a more mixed batch of past transitions instead of only the latest transition.

### Moving Data Distribution

In ordinary supervised learning, the dataset is often fixed. In online RL, the dataset depends on the current policy.

When $q_\theta$ changes, the greedy action can change:

$$
\argmax_a q_\theta(s, a)
$$

If the behavior policy is $\epsilon$-greedy with respect to $q_\theta$, then changing $\theta$ changes how the agent explores and exploits. That changes which states it visits, which actions it tries, and which rewards it observes.

This is clearly true for on-policy methods such as SARSA, where the policy being evaluated is also the policy collecting data.

For Q-learning, there is a nuance. The target policy is greedy, so the algorithm is off-policy and can learn from data generated by another behavior policy. In that sense, Q-learning is less tied to the current behavior policy than SARSA. But if the behavior policy used online is still derived from $q_\theta$, then the data distribution still changes during training.

So the better statement is:

- SARSA's target and data collection are both tied to the current policy.
- Q-learning's target is greedy and off-policy, but its online data distribution can still move if the behavior policy changes.

Offline Q-learning has a fixed dataset, but then a different problem appears: the target may ask for values of actions that are poorly covered by the dataset.

### Sharp Value Boundaries

Function approximators usually impose some kind of smoothness. Similar inputs often produce similar outputs.

That is helpful when similarity in input space matches similarity in value. But in RL, tiny changes in state can sometimes cause huge changes in return.

Imagine a helicopter flying close to a tree. A small change in position may be the difference between passing safely and crashing. The reward changes abruptly because one trajectory continues and the other terminates with a large penalty.

This does not always mean the true value function is mathematically non-differentiable, although it can be. The practical issue is broader: the value function may have sharp boundaries or high curvature. A smooth approximator can smear the boundary and assign unsafe values to states near failure.

The same problem appears in driving. Two states can look almost identical, except one is still safe and the other is too close to another car. The action-values can change dramatically near collision boundaries.

Approximation works best when the features make behaviorally similar states close to each other. Raw distance in the input space is not always the right notion of similarity.

### Non-stationary Targets

Even if the environment is stationary, the learning target may not be.

For TD learning, the target contains the current value estimate:

$$
R_t + \gamma q_\theta(S_{t+1}, A_{t+1})
$$

or:

$$
R_t + \gamma \max_{a^\prime}q_\theta(S_{t+1}, a^\prime)
$$

When $\theta$ changes, these targets change too. The model is not training against a fixed label. It is chasing labels that are produced by its own current predictions.

Policy improvement adds another source of movement. If the policy changes, then the state visitation distribution changes and the meaning of $q_\pi$ changes. The value of a state-action pair depends on what policy is followed afterward, so changing the policy changes the values we are trying to approximate.

This is what non-stationarity means in this context. The environment dynamics may be fixed, but the supervised-learning problem induced by the RL algorithm keeps changing.

### The Deadly Triad

The most important stability warning is the deadly triad. It is the combination of three ingredients:

1. Function approximation.
2. Bootstrapping.
3. Off-policy learning.

Each ingredient is useful by itself.

Function approximation lets us generalize across large state spaces. Bootstrapping lets us update before the full return is known. Off-policy learning lets us learn about one policy from data generated by another policy.

The problem is that all three together can create a feedback loop.

With function approximation, an update in one state changes predictions in other states. With bootstrapping, those predictions are used as targets for more updates. With off-policy learning, the update distribution may not match the distribution of the target policy, so the function approximator may be forced to extrapolate in regions where the data is weak.

Errors can then reinforce themselves. A value is overestimated in a poorly covered next state. A bootstrapped target uses that overestimate. The update increases the value of a previous state-action pair. Because parameters are shared, this may further increase other predictions. The process can oscillate or diverge.

This is not just a neural-network problem. Divergence examples exist even with linear function approximation. Baird's counterexample is the classic case. Modern papers such as [Breaking the Deadly Triad with a Target Network](https://arxiv.org/abs/2101.08862) study how target networks can stabilize some of these dynamics.

The deadly triad does not mean every approximate off-policy TD method always fails. DQN is proof that these ingredients can work very well in practice. It means the simple tabular convergence intuition no longer applies. Once all three ingredients are present, stability becomes something we must design for.

## Final thoughts

Tabular model-free control learns one number per state-action pair. Approximate model-free control replaces that table with a function:

$$
q_\theta(s, a)
$$

This lets the agent generalize to large, continuous, or high-dimensional state spaces. It also makes each update less local. One gradient step can change many predictions at once.

Monte Carlo targets turn RL into the most supervised-looking regression problem, because the completed return is a fixed number. TD targets update sooner, but they bootstrap from current estimates, so the target itself can move. Semi-gradient methods handle this by treating the target as fixed during the current update.

Approximate SARSA, Expected SARSA, and Q-learning all share the same update shape:

$$
\theta \leftarrow \theta + \alpha \delta_t
\nabla_\theta q_\theta(S_t, A_t)
$$

They differ in the target used to compute $\delta_t$: sampled next action, expected next action, or greedy next action.

The main lesson is that function approximation is not just a scaling trick. It changes the learning dynamics. Correlated samples, moving data distributions, sharp value boundaries, non-stationary targets, and the deadly triad all appear because we are no longer updating independent table entries.

The next step is to make approximate Q-learning work with neural networks. That leads to Deep Q-Networks, where replay buffers and target networks are introduced as practical answers to the instability described here.
