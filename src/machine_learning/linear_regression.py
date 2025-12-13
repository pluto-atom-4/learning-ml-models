"""
Linear Regression Algorithm Explained Step-by-Step
---------------------------------------------------
Linear Regression is a fundamental machine learning algorithm that models the linear relationship
between input features (X) and a target variable (y). It assumes the relationship can be expressed as
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b, where w are weights (coefficients) and b is the bias (intercept).
This algorithm is essential for understanding predictive modeling and serves as the foundation for
more complex regression techniques.

Here is how the process works:

1. **Problem Formulation**: Find the best-fit line through data points.
   - Objective: Minimize the sum of squared residuals (errors)
   - Residual = actual value (y) - predicted value (ŷ)
   - We want to find weights that minimize: Σ(y - ŷ)² = Σ(y - Xw)²

2. **Normal Equation Derivation**: Closed-form solution to find optimal weights.
   - Start with error function: E(w) = (y - Xw)ᵀ(y - Xw)
   - Take derivative and set to zero: dE/dw = -2Xᵀ(y - Xw) = 0
   - Solve for w: XᵀXw = Xᵀy
   - Final solution: w = (XᵀX)⁻¹Xᵀy

3. **Bias Term Handling**: Augment data with bias column.
   - Add column of ones to X: X_b = [1, x₁, x₂, ..., xₙ]
   - This allows the algorithm to learn a non-zero intercept
   - First element of solution vector θ becomes the intercept

4. **Numerical Stability**: Use pseudo-inverse for robustness.
   - Instead of computing (XᵀX)⁻¹ directly, use pseudo-inverse: (XᵀX)⁺
   - Pseudo-inverse handles singular matrices and numerical errors
   - More stable than direct matrix inversion

5. **Coefficient Extraction**: Separate intercept and coefficients.
   - θ[0] = intercept (bias term)
   - θ[1:] = coefficients for each feature
   - Store separately for easier prediction and interpretation

6. **Prediction**: Apply learned model to new data.
   - For new sample x: ŷ = b + w₁x₁ + w₂x₂ + ... + wₙxₙ
   - Vectorized form: ŷ = X @ w + b
   - Return list of predictions for batch inputs

Example: Predicting house price from square footage
- X = [[1000], [1500], [2000], [2500]]  # square footage
- y = [150000, 225000, 300000, 375000]  # price
- After fitting: intercept ≈ 75000, coefficient ≈ 100 (price per sq ft)
- Prediction for 2200 sq ft: 75000 + 100*2200 = 295000

Time Complexity: O(n³) for matrix inversion where n = number of features
Space Complexity: O(n²) for storing the XᵀX matrix

Advantages:
- Simple, interpretable, and fast
- Works well with linear relationships
- Foundation for understanding advanced techniques

Limitations:
- Assumes linear relationship between features and target
- Sensitive to outliers
- Fails when XᵀX is singular (use regularization like Ridge or Lasso)

This algorithm is crucial for interviews and demonstrates understanding of
optimization, linear algebra, and fundamental machine learning concepts.
"""

from typing import List

import numpy as np


class LinearRegression:
    """
    Simple Linear Regression using the Normal Equation.
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        """
        Fit linear regression model using the normal equation.
        """
        X = np.array(X)
        y = np.array(y)

        # Add bias column of ones
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Normal equation
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:].tolist()

    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Predict using the trained model.
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.array(X)
        return (self.intercept_ + X @ np.array(self.coef_)).tolist()
