# Linear Regression: Two Approaches Compared

This document explains the **main differences** and **algorithm life cycles** of the two linear regression implementations in this project, with their **math formulas**.

---

## 1. Overview and Main Differences

| Aspect | **1D closed-form** (`lr_1d.py` / `lr_1d.ipynb`) | **Multi-dimensional gradient descent** (`lin_reg.ipynb`) |
|--------|--------------------------------------------------|----------------------------------------------------------|
| **Model** | One input $x$: $\hat{y} = a x + b$ | Many inputs: $\hat{y} = \theta^T x$ (vector $\theta$, vector $x$) |
| **Parameters** | Slope $a$ and intercept $b$ (2 scalars) | Weight vector $\theta$ of size $n+1$ (bias + $n$ features) |
| **How parameters are found** | **Closed-form**: direct formulas from the normal equations (no iteration) | **Gradient descent**: iterative updates until cost converges |
| **Cost** | Not minimized explicitly; solution minimizes sum of squared residuals by construction | Mean squared error $J(\theta)$ minimized by taking steps in the direction $-\nabla J$ |
| **Data** | Single feature per sample (e.g. from `data_1d.csv`) | Multiple features per sample (e.g. diabetes: 10 features, 442 samples) |
| **Typical use** | Simple curves, teaching, one predictor | Many predictors, scalable to large $n$ when closed-form is expensive |

**In short:** the 1D notebook solves for $a$ and $b$ in one shot with formulas; the other fits $\theta$ by repeatedly updating it using the gradient of the cost.

---

## 2. 1D Linear Regression (Closed-Form)

### 2.1 Goal and model

Fit a line to pairs $(x_i, y_i)$:

$$
\hat{y} = a x + b
$$

- $a$ = slope, $b$ = intercept.  
- Predictions: $\hat{y}_i = a x_i + b$.

### 2.2 Math formulas

**Slope (minimizes sum of squared residuals):**

$$
a = \frac{\sum_{i}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i}(x_i - \bar{x})^2} = \frac{\text{Cov}(X,Y)}{\text{Var}(X)}
$$

**Intercept:**

$$
b = \bar{y} - a \bar{x}
$$

**Identities used in code:**

- Denominator: $\sum_i x_i^2 - n\bar{x}^2 = \sum_i (x_i - \bar{x})^2$
- Numerator of $a$: $\sum_i x_i y_i - n \bar{x}\bar{y} = \sum_i (x_i - \bar{x})(y_i - \bar{y})$

**Goodness of fit — R²:**

- Residual sum of squares: $SS_{\text{res}} = \sum_i (y_i - \hat{y}_i)^2$
- Total sum of squares: $SS_{\text{tot}} = \sum_i (y_i - \bar{y})^2$

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

$R^2$ is the fraction of variance in $y$ explained by the model (0 to 1, higher is better).

### 2.3 Algorithm life cycle

1. **Load data**  
   Read $(x, y)$ pairs (e.g. from `data_1d.csv`).  
   Build vectors $\mathbf{x}$ and $\mathbf{y}$ (length $n$).

2. **Compute sufficient statistics**  
   Means $\bar{x}$, $\bar{y}$; sums $\sum x_i$, $\sum y_i$; dot products $\mathbf{x} \cdot \mathbf{x}$, $\mathbf{x} \cdot \mathbf{y}$ (or equivalent).

3. **Compute parameters**  
   - Denominator: $\text{denom} = \sum_i x_i^2 - n\bar{x}^2$.  
   - $a = \bigl(\sum_i x_i y_i - n\bar{x}\bar{y}\bigr) / \text{denom}$.  
   - $b = \bar{y} - a\bar{x}$.

4. **Predict**  
   $\hat{y}_i = a x_i + b$ for all $i$ (and for any new $x$: $\hat{y} = a x + b$).

5. **Evaluate fit**  
   Compute $SS_{\text{res}}$, $SS_{\text{tot}}$, then $R^2$.

6. **Optional**  
   Plot data and the fitted line; report $R^2$.

No loops over “iterations”; the solution is direct.

---

## 3. Multi-Dimensional Linear Regression (Gradient Descent)

### 3.1 Goal and model

Fit a linear model with $n$ features (and a bias). Each sample is a vector $x^{(i)} \in \mathbb{R}^{n+1}$ with $x_0 = 1$ (bias term). The model is:

$$
\hat{y}^{(i)} = \theta^T x^{(i)} = h_\theta(x^{(i)})
$$

In matrix form, with design matrix $X$ of shape $(n+1) \times m$ and $\theta$ of shape $(n+1) \times 1$:

$$
h_\theta(X) = \theta^T X \quad \Rightarrow \quad \text{shape } (1 \times m)
$$

### 3.2 Math formulas

**Cost (mean squared error):**

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^m \bigl( h_\theta(x^{(i)}) - y^{(i)} \bigr)^2
$$

**Gradient of the cost (vector of partial derivatives):**

$$
\frac{\partial J}{\partial \theta} = \frac{1}{m} X \, (h_\theta(X) - y)^T
$$

Result is a column vector of shape $(n+1) \times 1$.

**Gradient descent update (one step):**

$$
\theta \leftarrow \theta - \alpha \frac{\partial J}{\partial \theta}
$$

$\alpha$ is the learning rate.

**Prediction for a new point:**

$$
\hat{y} = \theta^T x_{\text{new}}
$$

($x_{\text{new}}$ must include the bias component, e.g. first entry 1.)

### 3.3 Algorithm life cycle

1. **Load data**  
   Get design matrix $X$ (e.g. shape $n \times m$) and targets $y$ (e.g. shape $1 \times m$).  
   Example: `load_regression_data()` from `utils`.

2. **Add bias**  
   Prepend a row of ones to $X$ so that $X$ has shape $(n+1) \times m$ and the first component of $\theta$ is the intercept:  
   $X \gets [\mathbf{1}; X]$.

3. **Set hyperparameters**  
   Choose learning rate $\alpha$ and number of iterations (e.g. 500).

4. **Initialize $\theta$**  
   Random (or zero) vector of shape $(n+1) \times 1$.

5. **Gradient descent loop** (repeat for each iteration):  
   - Compute predictions: $h_\theta(X) = \theta^T X$.  
   - Compute cost: $J(\theta) = \frac{1}{2m} \sum (h_\theta(X) - y)^2$ (optional: log or plot).  
   - Compute gradient: $\frac{\partial J}{\partial \theta} = \frac{1}{m} X (h_\theta(X) - y)^T$.  
   - Update: $\theta \leftarrow \theta - \alpha \frac{\partial J}{\partial \theta}$.

6. **Stopping**  
   After the chosen number of iterations (or when $J(\theta)$ changes little).

7. **Predict**  
   For new input $x_{\text{new}}$ (with bias): $\hat{y} = \theta^T x_{\text{new}}$.

8. **Optional**  
   Plot $J(\theta)$ vs iteration to check convergence.

---

## 4. Summary Diagram (Life Cycles)

**1D closed-form:**

```
Load (x, y) → Compute means/sums/dots → a, b from formulas → Predict & R²
```

**Gradient descent:**

```
Load X, y → Add bias row → Init θ, set α, iterations
    → Loop: h = θ'X, J(θ), ∇J, θ = θ − α∇J → Predict
```

---

## 5. When to Use Which

- **1D closed-form:** One predictor, small data, or you want an exact solution in one pass. Simple and interpretable ($a$, $b$, $R^2$).
- **Gradient descent:** Many features, large datasets, or when you will extend the same idea to other models (e.g. logistic regression, neural nets) where a closed-form solution is not available.

Both methods minimize the same type of loss (sum of squared errors) for linear regression; they differ in **how** they find the minimizing parameters (direct formulas vs iterative steps).
