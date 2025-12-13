"""
Lasso Regression Algorithm Explained Step-by-Step
--------------------------------------------------
Lasso Regression (Least Absolute Shrinkage and Selection Operator) extends Linear Regression using
L1 regularization to achieve both regularization and automatic feature selection. Unlike Ridge Regression
which shrinks weights, Lasso can force weights exactly to zero, effectively removing features.
This makes Lasso powerful for high-dimensional data where many features may be irrelevant.

Here is how the process works:

1. **Problem Formulation**: Add L1 regularization penalty.
   - Standard Linear Regression: min ||y - Xw||²
   - Lasso: min ||y - Xw||² + α Σ|wⱼ|
   - α controls regularization strength (hyperparameter)
   - L1 penalty: sum of absolute values of weights (unlike L2 in Ridge)
   - L1 penalty creates sparsity: many weights become exactly zero

2. **L1 vs L2 Regularization**:
   - L2 (Ridge): Penalty proportional to w²
   - L1 (Lasso): Penalty proportional to |w|
   - L1 creates "sharp corners" in constraint region
   - Sharp corners cause solutions to land on axes (weights = 0)
   - L2 creates smooth circles, doesn't force zeros

3. **Soft Thresholding**: Core operation in coordinate descent.
   - Threshold function: soft_threshold(z, λ) = sign(z) * max(|z| - λ, 0)
   - If |z| < λ: return 0 (weight forced to zero)
   - If |z| > λ: shrink by λ and keep sign
   - Implements the L1 penalty effect on each coordinate

4. **Coordinate Descent Optimization**: Iterative algorithm.
   - Update one weight at a time while keeping others fixed
   - For each feature j:
     * Compute residual with current weights: r = y - Xw + wⱼXⱼ
     * Compute correlation with residual: ρ = Xⱼ · r
     * Apply soft threshold to get new weight: wⱼ = soft_threshold(ρ/||Xⱼ||², α)
   - Repeat until convergence (weights and intercept stabilize)

5. **Parameter Initialization**: Start from a reasonable baseline.
   - Initialize all weights to zero: w = [0, 0, ..., 0]
   - Initialize intercept: intercept = mean(y)
   - Start from zero helps with sparsity (many features remain zero)

6. **Convergence Checking**: Iterative refinement with stopping criterion.
   - Track changes in weights and intercept across iterations
   - Stop when changes fall below tolerance threshold (tol)
   - Prevents excessive iterations while ensuring solution stability
   - Both weight and intercept changes must be small to stop

7. **Feature Selection Outcome**:
   - After convergence, weights may be exactly zero
   - Non-zero weights indicate selected features
   - Number of selected features depends on α
   - Larger α → more features dropped to zero

Example: Gene expression data with 20,000 features
- Most genes irrelevant to target disease
- Lasso with α=1.0 might select only 50-100 relevant genes
- Ridge would keep all 20,000 with small weights
- Lasso result is more interpretable for biologists

Hyperparameter α (regularization strength):
- α = 0: Lasso becomes standard Linear Regression (no sparsity)
- α small: Few features eliminated, similar to Ridge behavior
- α optimal: Best bias-variance tradeoff (cross-validation to find)
- α large: Most features eliminated, very sparse model

Time Complexity: O(n * m * max_iter) where n = samples, m = features
Space Complexity: O(n * m) for storing the data matrix

Advantages:
- Automatic feature selection (sets weights to exactly zero)
- More interpretable models (fewer features)
- Handles high-dimensional data well
- Can identify irrelevant features

Limitations:
- No closed-form solution (requires iterative optimization)
- Slower than Ridge Regression
- When many features are correlated, may arbitrarily pick one
- Requires careful hyperparameter tuning (α selection)

Key Interview Points:
1. Why Lasso over Ridge? When you need interpretability and feature selection
2. Why Ridge over Lasso? When all features are likely relevant
3. What about Elastic Net? Combines L1 and L2 for both benefits
4. How to choose α? Cross-validation with different α values

This algorithm demonstrates sparse modeling and feature selection,
essential concepts for dealing with high-dimensional real-world data.
"""

from typing import List

import numpy as np


class LassoRegression:
    """
    Lasso Regression using coordinate descent.
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def soft_threshold(self, rho, alpha):
        if rho > alpha:
            return rho - alpha
        elif rho < -alpha:
            return rho + alpha
        else:
            return 0.0

    def fit(self, X: List[List[float]], y: List[float]):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.coef_ = np.zeros(n_features)
        self.intercept_ = np.mean(y)

        for _ in range(self.max_iter):
            coef_old = self.coef_.copy()
            intercept_old = self.intercept_

            # Update intercept
            self.intercept_ = np.mean(y - X @ self.coef_)

            # Update each coefficient
            for j in range(n_features):
                residual = y - (self.intercept_ + X @ self.coef_) + self.coef_[j] * X[:, j]
                rho = np.dot(X[:, j], residual)
                norm_sq = np.dot(X[:, j], X[:, j])
                self.coef_[j] = self.soft_threshold(rho / norm_sq, self.alpha) if norm_sq > 0 else 0

            # Check convergence
            if np.linalg.norm(self.coef_ - coef_old) < self.tol and np.abs(self.intercept_ - intercept_old) < self.tol:
                break

    def predict(self, X: List[List[float]]) -> List[float]:
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.array(X)
        return (self.intercept_ + X @ self.coef_).tolist()
