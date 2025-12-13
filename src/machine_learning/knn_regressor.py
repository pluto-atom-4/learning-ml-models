"""
k-Nearest Neighbors (kNN) Regressor Algorithm Explained Step-by-Step
---------------------------------------------------------------------
k-Nearest Neighbors is a fundamental non-parametric machine learning algorithm used for regression and
classification tasks. It predicts target values based on the average (or weighted average) of k nearest
training samples. kNN is intuitive, lazy-learning (no explicit training phase), and widely used as a
baseline algorithm. It demonstrates core ML concepts: distance metrics, hyperparameter tuning, and
prediction strategies that appear frequently in machine learning interviews.

Here is how the process works:

1. **Lazy Learning (No Training Phase)**: Store training data without explicit model learning.
   - Unlike parametric models (linear regression, neural networks), kNN simply memorizes training data
   - No parameters to learn or optimize during fit() phase
   - Training is O(1) - just store X_train and y_train arrays
   - All computation happens during prediction (hence "lazy")

2. **Distance Metric Computation**: Measure similarity between query point and training samples.
   - Euclidean Distance: sqrt((x1-x2)² + (y1-y2)² + ... + (zn-z2)²)
   - Also called L2 norm, most common for continuous features
   - Manhattan Distance (L1): |x1-x2| + |y1-y2| + ... (alternative)
   - Cosine Similarity (for high-dimensional data or text)
   - Choose metric based on feature types and domain knowledge

3. **Finding k Nearest Neighbors**: Identify k closest training samples.
   - Compute distances from query point to all training samples
   - Sort distances in ascending order
   - Select indices of k smallest distances
   - Time complexity: O(n) where n = number of training samples
   - Trade-off: larger k = smoother predictions but slower, smaller k = noisier but faster

4. **Aggregation Strategy**: Combine k neighbors' values to make prediction.
   - For Regression: Average (mean) of k neighbors' target values
   - Alternative: Weighted average using inverse distance as weights (closer points have more influence)
   - For Classification: Majority voting among k neighbors' class labels
   - Handles outliers better when using weighted average

5. **Prediction Pipeline**: For each query sample, repeat distance-find-aggregate.
   - Input: Query sample(s) X, trained model with stored X_train and y_train
   - For each query point: compute distances, find k neighbors, average their y values
   - Output: Predicted values for all query samples
   - Total time: O(n * d * m) where n=training samples, d=features, m=query samples

6. **Hyperparameter: k Selection**: Critical choice affecting model performance.
   - k=1: Overfitting risk, too sensitive to individual training samples
   - k=n (all samples): Underfitting risk, loses local patterns, predicts overall mean
   - Typical: k=3, 5, 10, or sqrt(n); choose via cross-validation
   - Odd k for classification helps avoid ties; flexibility for regression

Example: Predicting house prices with 5 nearest neighbors
- Training: Store 1000 house data points (features: size, location, age; target: price)
- Query: New house with size=2000, location=downtown, age=10 years
- Find: 5 nearest houses by Euclidean distance in feature space
- Predict: Average price of those 5 neighbors, e.g., ($300k + $310k + $295k + $305k + $298k)/5 = $301.6k

Advantages:
- Simple to understand and implement, no training required
- Works for both regression and classification
- Adapts to local patterns in data
- No assumptions about data distribution

Disadvantages:
- Slow prediction (O(n) per query) - must compute all distances
- Memory intensive - stores entire dataset
- Sensitive to feature scaling - features with large ranges dominate distance calculations
- Curse of dimensionality - performance degrades in high dimensions
- Sensitive to irrelevant features and outliers

Time Complexity:
- Fit/Training: O(n*d) to store n samples with d features
- Predict: O(k*m*n*d) where k=neighbors, m=query samples, n=training samples, d=features
  - O(n*d) to compute all distances per query sample
  - O(n*log(k)) to find k smallest distances efficiently

Space Complexity: O(n*d) to store all training data and features

Interview Tips:
- Mention distance metric scaling: normalize/standardize features before using kNN
- Discuss k selection as hyperparameter tuning problem
- Compare with decision trees, SVM, or neural networks for trade-offs
- Explain why large k causes underfitting, small k causes overfitting
- Know lazy learning vs eager learning trade-offs
- Discuss optimization techniques: KD-trees, Ball trees for faster nearest neighbor search
- Recognize curse of dimensionality in high-dimensional spaces
"""

from typing import List

import numpy as np


class KNeighborsRegressor:
    """
    A simple implementation of k-Nearest Neighbors Regressor.
    """

    def __init__(self, n_neighbors: int = 5):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        """
        Store the training data.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Predict regression values for given input samples.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.array(X)
        predictions = []

        for x in X:
            # Compute Euclidean distances to all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Get indices of k nearest neighbors
            nn_indices = np.argsort(distances)[:self.n_neighbors]
            # Average their target values
            pred = np.mean(self.y_train[nn_indices])
            predictions.append(pred)

        return predictions
