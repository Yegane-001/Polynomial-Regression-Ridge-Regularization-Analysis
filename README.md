# 📈 Polynomial Regression & Ridge Regularization Analysis

A vectorized, from-scratch implementation and empirical evaluation of **Polynomial Regression** and **L2 Regularization (Ridge)** in Python. This project investigates the foundational concepts of model capacity, the **Bias-Variance Tradeoff**, and hyperparameter tuning using closed-form linear algebra solutions.

---

## 🛠️ Key Features

- **From-Scratch Implementation:** Parameter estimation using the closed-form **Normal Equation** without relying on high-level machine learning frameworks (e.g., Scikit-Learn).
- **Model Capacity Analysis:** Systematic evaluation of fitting behavior across polynomial degrees ($1 \le \text{degree} \le 12$).
- **Overfitting Mitigation (Ridge Regularization):** Implementation of an $L_2$ penalty term to constrain weight magnitude, featuring hyperparameter ($\lambda$) optimization across logarithmic scales.
- **Comprehensive Visualizations:** High-resolution plots illustrating regression fit curves, training vs. validation error trends, and coefficient shrinkage.

---

## 📐 Mathematical Formulation

### 1. Standard Normal Equation (Unregularized)
For a design matrix $X$ and target vector $y$, the optimal parameter vector $w^*$ that minimizes the Mean Squared Error (MSE) is given by:

$$w^* = (X^T X)^{-1} X^T y$$

### 2. Ridge Regularized Normal Equation
To penalize large weights and reduce variance in high-degree polynomials, an $L_2$ regularization term is added to the objective function:

$$w^* = (X^T X + \lambda I')^{-1} X^T y$$

> **Note:** $I'$ is a modified identity matrix where $I'_{0,0} = 0$, ensuring that the bias term ($w_0$) remains unregularized.

---

## 📊 Empirical Findings & Analysis

### Dataset A (Regression Curve Analysis)
* **Degree 1 (Linear):** Exhibits severe **Underfitting** (High Bias). Fails to capture the non-linear oscillating trend of the data.
* **Degree 3 (Cubic):** Provides an optimal fit, balancing complexity and generalization.
* **Degree 9 (Unregularized):** Exhibits severe **Overfitting** (High Variance). Shows extreme oscillations between data points due to noise sensitivity.

### Dataset B (Error Dynamics & Regularization)
* **Underfitting Region ($1 \le \text{Degree} \le 4$):** Both training and validation errors remain elevated.
* **Overfitting Region ($\text{Degree} \ge 10$):** Training MSE continues to decay towards zero, while Validation MSE spikes exponentially.
* **Ridge Regularization ($\text{Degree} = 9$):** Sweeping $\lambda$ demonstrates that optimal generalization performance is achieved in the range of $10^{-4} \le \lambda \le 10^{-2}$, effectively stabilizing the validation error.
