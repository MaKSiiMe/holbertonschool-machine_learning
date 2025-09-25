# Gradient Descent — Optimization Landscape

![Optimization landscape illustration showing valleys and paths](https://images.unsplash.com/photo-1526401485004-2fda9f4cae69?q=80&w=1600&auto=format&fit=crop) 
<sub>Photo by Unsplash (free to use). Illustrative cover image for optimization landscapes.</sub>

## Optimization in Practice: Mechanics, Pros & Cons of the Most-Used Techniques
<a id="optimization-in-practice-mechanics-pros--cons-of-the-most-used-techniques"></a>

Modern deep learning isn’t just about building bigger networks—it’s about optimizing them well. Below is a pragmatic guide to the what, why, how, and trade-offs of seven core techniques you’ll meet in almost every serious project.

---

## Table of Contents
- [1) Feature Scaling (Standardization & Min-Max)](#1-feature-scaling-standardization--min-max)
- [2) Batch Normalization (BN)](#2-batch-normalization-bn)
- [3) Mini-Batch Gradient Descent](#3-mini-batch-gradient-descent)
- [4) Gradient Descent with Momentum](#4-gradient-descent-with-momentum)
- [5) RMSProp](#5-rmsprop)
- [6) Adam (Adaptive Moment Estimation)](#6-adam-adaptive-moment-estimation)
- [7) Learning Rate (LR) Decay](#7-learning-rate-lr-decay)
- [A Practical Playbook](#a-practical-playbook)
- [Minimal End-to-End Skeleton (Keras)](#minimal-end-to-end-skeleton-keras)
- [When to Use What (quick heuristics)](#when-to-use-what-quick-heuristics)

---

## 1) Feature Scaling (Standardization & Min-Max)
<a id="1-feature-scaling-standardization--min-max"></a>

What it is. Transform features to comparable ranges so gradients don’t get dominated by large-scale features.

How.
- Standardization (z-score): $x' = \frac{x - \mu}{\sigma + \varepsilon}$ (mean 0, std 1; use a small epsilon like 1e-8 for stability).
- Min-Max scaling: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ (range [0, 1] by default).

Why it matters. Speeds up convergence (especially for GD/SGD), stabilizes training, and helps regularization behave consistently.

Pros.
- Faster, more stable training.
- Essential for distance-based models; recommended for neural nets too.

Cons / pitfalls.
- Data leakage: always fit scalers on train only, then transform val/test.
- Outliers can distort Min-Max; standardization is typically more robust.

Quick example (NumPy):
```python
import numpy as np

X = ...  # shape (m, n)
mu = X.mean(axis=0)
sigma = X.std(axis=0, ddof=0)
X_std = (X - mu) / (sigma + 1e-8)  # standardization
```

---

## 2) Batch Normalization (BN)
<a id="2-batch-normalization-bn"></a>

What it is. Normalizes layer activations over a mini-batch, then learns scale/shift $(\gamma, \beta)$:

$$
\hat{z} = \frac{z - \mu_B}{\sqrt{\sigma^2_B + \varepsilon}},\quad y = \gamma \, \hat{z} + \beta
$$

Why it matters.
- Stabilizes and accelerates training.
- Allows higher learning rates.
- Acts as a mild regularizer through batch noise.

Pros.
- Faster convergence; less sensitivity to init and LR.
- Often improves final accuracy.

Cons / pitfalls.
- Small batches make BN statistics noisy; consider LayerNorm/GroupNorm if batch ≪ 32.
- BN behaves differently at inference (uses running averages).
- Interactions with Dropout require care (usually place BN before nonlinearity, and tune dropout rates).

Keras sketch:
```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(256, use_bias=False),      # bias often unnecessary before BN
    layers.BatchNormalization(epsilon=1e-7),
    layers.Activation('relu'),
    layers.Dense(10, activation='softmax')
])
```

---

## 3) Mini-Batch Gradient Descent
<a id="3-mini-batch-gradient-descent"></a>

What it is. Compute gradients on small batches (e.g., 32–256) instead of full dataset or single samples.

Why it matters.
- Efficiency: parallelizable, GPU-friendly.
- Generalization: a bit of gradient noise often helps escape sharp minima.
- Memory: no need to load full dataset into RAM.

Pros.
- Great speed/accuracy trade-off.
- Works hand-in-glove with BN and modern hardware.

Cons / pitfalls.
- Batch size is a hyperparameter; too large can hurt generalization.
- Always shuffle each epoch to avoid learning order artifacts.

---

## 4) Gradient Descent with Momentum
<a id="4-gradient-descent-with-momentum"></a>

Idea. Accumulate a velocity vector to smooth updates:

$$
v_t = \beta \, v_{t-1} + (1 - \beta)\, \nabla_w \, \mathcal{L}(w_t),\quad
w_{t+1} = w_t - \alpha \, v_t
$$

Pros.
- Dampens oscillations across ravines; accelerates along consistent directions.
- Often yields faster convergence than vanilla SGD.

Cons / pitfalls.
- Too high $\beta$ or $\alpha$ can overshoot minima.
- Works best with well-tuned LR schedules.

Rules of thumb. $\beta \in [0.8, 0.95]$ is common; pair with step/exponential LR decay.

---

## 5) RMSProp
<a id="5-rmsprop"></a>

Idea. Keep an exponentially decayed average of squared gradients to adapt step sizes per parameter:

$$
\begin{aligned}
 s_t &= \beta \, s_{t-1} + (1 - \beta)\, (\nabla_w \, \mathcal{L})^2, \\
 w_{t+1} &= w_t - \frac{\alpha}{\sqrt{s_t + \varepsilon}}\,\nabla_w \, \mathcal{L}
\end{aligned}
$$

Pros.
- Handles varying gradient scales; typically fast and stable.
- Less tuning than raw SGD.

Cons / pitfalls.
- Still needs $\alpha$ and $\beta$ tuning.
- Can plateau without LR scheduling.

Defaults to try. $\alpha \in [10^{-4}, 10^{-3}],\ \beta \approx 0.9,\ \varepsilon = 1\mathrm{e}{-8}$.

---

## 6) Adam (Adaptive Moment Estimation)
<a id="6-adam-adaptive-moment-estimation"></a>

Idea. Momentum + RMSProp with bias corrections:

$$
\begin{aligned}
 m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, \\
 v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\end{aligned}
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t},\quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t},\quad
w_{t+1} = w_t - \alpha\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t + \varepsilon}}
$$

Pros.
- Strong out-of-the-box performance; low tuning burden.
- Works well with sparse/ill-scaled gradients.

Cons / pitfalls.
- Sometimes slightly worse final generalization than SGD+Momentum.
- Consider AdamW (decoupled weight decay) for better regularization.

Defaults to try. $\alpha = 1\mathrm{e}{-3},\ \beta_1 = 0.9,\ \beta_2 = 0.999,\ \varepsilon = 1\mathrm{e}{-8}$.

---

## 7) Learning Rate (LR) Decay
<a id="7-learning-rate-lr-decay"></a>

Why. Large steps early to explore; small steps later to refine.

Common schedules.
- Step decay: drop LR by a factor every $k$ epochs.
- Exponential decay: $\alpha_t = \alpha_0 \, e^{-\lambda t}$.
- Inverse time decay: $\alpha_t = \frac{\alpha_0}{1 + \text{decay}\cdot t}$.

Pros.
- Improves stability and final accuracy.
- Works with any optimizer.

Cons / pitfalls.
- Schedule hyperparameters matter (drop factor, step size, decay rate).
- Too aggressive → underfitting; too mild → slow training.

Keras sketch (inverse time):
```python
import tensorflow as tf

alpha0 = 1e-2
schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    alpha0, decay_steps=1000, decay_rate=1.0, staircase=True
)
opt = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9)
```

---

## A Practical Playbook
<a id="a-practical-playbook"></a>

- Always scale features for neural nets (fit on train; transform val/test).
- Start with Adam for baselines; ensure the model works and metrics are sane.
- If you need the last few %, try SGD+Momentum with a solid LR schedule.
- Use BatchNorm in deep nets with reasonably sized batches (≥ 32). For tiny batches, try LayerNorm/GroupNorm.
- Tune LR first. Then batch size, momentum/betas, and weight decay.
- Log and plot: loss curves, LR vs. step, gradient norms. Many “mystery bugs” are optimizer misconfigurations.

---

## Minimal End-to-End Skeleton (Keras)
<a id="minimal-end-to-end-skeleton-keras"></a>
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 1) Data
X_train, y_train = ...  # your dataset
X_val,   y_val   = ...

# 2) Model
def build():
    inp = layers.Input(shape=(X_train.shape[1],))
    x = layers.Dense(256, use_bias=False)(inp)
    x = layers.BatchNormalization(epsilon=1e-7)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(10, activation='softmax')(x)
    return models.Model(inp, out)

model = build()

# 3) Optimizer + LR schedule (Adam baseline + step-wise exponential decay)
initial_lr = 1e-3
schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr, decay_steps=2000, decay_rate=0.96, staircase=True
)
opt = tf.keras.optimizers.Adam(learning_rate=schedule)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4) Train
model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_val, y_val))
```

---

## When to Use What (quick heuristics)
<a id="when-to-use-what-quick-heuristics"></a>

- Messy scales / many numeric features → scale features (standardization).
- Deep networks / unstable training → add BN (or LayerNorm for small batches).
- Need a fast, good baseline → Adam (+ mild LR decay).
- Chasing top accuracy in vision/NLP → SGD+Momentum with tuned schedules often wins late-game.
- Highly sparse gradients (NLP, recsys) → Adam/RMSProp shine.
- Training plateaus → adjust LR schedule before changing everything else.