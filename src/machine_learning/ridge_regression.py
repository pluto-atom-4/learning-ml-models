"""
Ridge Regression Algorithm Explained Step-by-Step
--------------------------------------------------
Ridge Regression extends Linear Regression by adding L2 regularization to prevent overfitting.
The key difference is adding a penalty term to the error function that discourages large weights.
Ridge Regression solves: minimize ||y - Xw||² + α||w||², where α controls the strength of regularization.
This technique is essential for handling multicollinearity and improving model generalization on new data.

Here is how the process works:

1. **Problem Formulation**: Add regularization penalty to standard regression.
   - Original Linear Regression: min ||y - Xw||²
   - Ridge Regression: min ||y - Xw||² + α||w||²
   - α (alpha) is the regularization strength (hyperparameter)
   - Larger α forces weights closer to zero, preventing overfitting

2. **Bias Term Exemption**: Don't regularize the intercept.
   - Bias/intercept term should not be penalized
   - Create identity matrix I and set I[0,0] = 0
   - This ensures only feature weights are regularized, not intercept
   - Intercept still learns from data without L2 penalty

3. **Closed-Form Solution**: Modified normal equation for Ridge.
   - Standard normal equation: w = (XᵀX)⁻¹Xᵀy
   - Ridge solution: w = (XᵀX + αI)⁻¹Xᵀy
   - Adding αI to XᵀX makes matrix more well-conditioned
   - Reduces numerical instability and singular matrix issues

4. **Augmented Data Matrix**: Add bias column as in Linear Regression.
   - Augment X with column of ones: X_b = [1, x₁, x₂, ..., xₙ]
   - Allows learning non-zero intercept
   - First parameter in θ becomes the intercept term

5. **Numerical Stability**: Use pseudo-inverse with regularization.
   - Compute: θ = (XᵀX + αI)⁺ Xᵀy using pinv
   - Pseudo-inverse is more numerically stable than direct inversion
   - Handles rank-deficient matrices gracefully

6. **Coefficient Extraction and Prediction**.
   - θ[0] = intercept (not regularized)
   - θ[1:] = regularized feature coefficients
   - Prediction: ŷ = intercept + X @ coefficients

Example: House price prediction with Ridge Regression
- If features are correlated (multicollinearity exists)
- Ridge forces related weights to be smaller and similar
- Prevents extreme weight values that might overfit
- Trade-off: Slight bias increase for significant variance reduction

Impact of α (regularization strength):
- α = 0: Ridge becomes standard Linear Regression
- α small: Weak regularization, similar to Linear Regression
- α large: Strong regularization, weights shrink significantly
- α → ∞: Weights approach zero, model becomes constant

Time Complexity: O(n³) for matrix operations where n = number of features
Space Complexity: O(n²) for storing the (XᵀX + αI) matrix

Advantages:
- Handles multicollinearity (correlated features)
- Prevents overfitting on high-dimensional data
- More stable than Linear Regression when XᵀX is singular
- Works well in practice for many datasets

Limitations:
- Does not perform feature selection (keeps all features)
- Requires hyperparameter tuning (choosing α)
- Assumes linear relationship between features and target

Key Interview Point: Ridge vs Lasso
- Ridge: Shrinks weights continuously (keeps all features)
- Lasso: Can force weights exactly to zero (feature selection)
- Both use regularization but with different penalties (L2 vs L1)

This algorithm demonstrates regularization concepts crucial for building
robust machine learning models and preventing overfitting in practice.
"""

from typing import List

import numpy as np


class RidgeRegression:
    """
    Ridge Regression using the closed-form solution.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: List[List[float]], y: List[float]):
        X = np.array(X)
        y = np.array(y)

        # Add bias column
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        n_features = X_b.shape[1]

        # Identity matrix (no regularization on bias term)
        I = np.eye(n_features)
        I[0, 0] = 0

        theta = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y

        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:].tolist()

    def predict(self, X: List[List[float]]) -> List[float]:
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.array(X)
        return (self.intercept_ + X @ np.array(self.coef_)).tolist()
