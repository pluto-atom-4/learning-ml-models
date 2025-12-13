"""
Mean Squared Error (MSE) Metric Explained Step-by-Step
-------------------------------------------------------
Mean Squared Error (MSE) is a fundamental loss function and evaluation metric for regression problems.
It measures the average of squared differences between predicted and actual values, with quadratic
penalty for large errors. MSE is widely used in machine learning for training and evaluating
regression models, optimization objectives, and comparing different models.

Here is how MSE works:

1. **Error Calculation**: Compute the prediction error for each sample.
   - For each sample i: error_i = y_true_i - y_pred_i
   - Error can be positive (underprediction) or negative (overprediction)
   - Each error represents how far prediction deviates from truth

2. **Squaring the Errors**: Amplify larger errors quadratically.
   - Square each error: squared_error_i = error_i²
   - Squaring ensures all errors contribute positively (no cancellation)
   - Large errors are penalized much more than small errors
   - Example: error of 10 → squared error of 100 (very high penalty)

3. **Averaging Across All Samples**: Normalize by sample count.
   - MSE = (1/n) * Σ(y_true_i - y_pred_i)² for n samples
   - Averaging makes MSE independent of dataset size
   - Allows comparing models trained on datasets of different sizes
   - Average indicates typical squared error per sample

4. **Properties and Interpretation**:
   - MSE ≥ 0 always (squared values are non-negative)
   - MSE = 0 only when predictions are perfect (all errors = 0)
   - Units: MSE is in squared units of target variable
   - Example: predicting price in dollars → MSE in dollars²

5. **Why Square Instead of Absolute Value?**:
   - Squared errors are differentiable (important for gradient descent)
   - Quadratic penalty heavily penalizes outliers
   - Mathematical convenience for closed-form solutions
   - Compare: MAE (mean absolute error) = (1/n) * Σ|error_i| - also valid but less commonly used

6. **Example: House Price Prediction**
   - Actual prices: [300000, 250000, 350000]
   - Predicted prices: [310000, 240000, 345000]
   - Errors: [10000, -10000, -5000]
   - Squared errors: [100000000, 100000000, 25000000]
   - MSE = (100000000 + 100000000 + 25000000) / 3 = 75000000

7. **Relationship with Other Metrics**:
   - RMSE (Root Mean Squared Error) = √MSE (brings units back to original)
   - RMSE is more interpretable (same units as target variable)
   - Example: RMSE ≈ 8660 for house price above (in dollars, like prices)
   - Lower MSE/RMSE indicates better predictions

Time Complexity: O(n) for n samples (single pass through data)
Space Complexity: O(1) if computed iteratively, O(n) if storing all errors

MSE in Practice:

1. **Model Training**: Most regression algorithms minimize MSE
   - Linear Regression minimizes MSE by solving normal equation
   - Gradient descent algorithms compute MSE gradient
   - Loss function during training to guide weight updates

2. **Model Evaluation**: Assess quality on test/validation set
   - Training MSE: error on data used for fitting
   - Validation MSE: error on held-out data (shows generalization)
   - Overfitting detected if validation MSE >> training MSE

3. **Hyperparameter Selection**: Compare models via MSE
   - Different regularization strengths (α in Ridge/Lasso)
   - Different model complexities
   - Cross-validation to find best hyperparameters

Advantages of MSE:
- Mathematically convenient and differentiable
- Heavily penalizes large errors (good for some applications)
- Standard metric in machine learning
- Direct relationship to optimization in regression

Limitations of MSE:
- Sensitive to outliers (squared penalty amplifies them)
- Not robust to large prediction errors
- Hard to interpret (squared units)
- May not reflect business impact of errors

Alternatives to MSE:
- MAE (Mean Absolute Error): More robust to outliers
- MAPE (Mean Absolute Percentage Error): Good for percentage-based interpretation
- Huber Loss: Combines MSE and MAE benefits
- Custom loss functions: Tailored to specific problem needs

Key Interview Points:
1. Why minimize MSE? Closed-form solution exists, differentiable for gradient descent
2. Why not MAE? Less mathematical convenience, harder optimization
3. How does MSE relate to variance? Large MSE indicates high prediction variance
4. When to use alternatives? MSE assumes outliers are important; MAE when not

This metric is fundamental to regression modeling and essential for
understanding model performance and the learning process in machine learning.
"""

from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Compute Mean Squared Error (MSE).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))
