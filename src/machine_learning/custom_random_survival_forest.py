from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from lifelines.datasets import load_waltons

# Re-export for convenience
__all__ = [
    'load_waltons_dataset',
    'km_estimator',
    'SurvivalTreeNode',
    'SurvivalTree',
    'RandomSurvivalForest',
]


def load_waltons_dataset():
    """Load Waltons dataset from lifelines.

    Returns a tuple of (X, T, E, data) where:
    - X: Feature matrix (one-hot encoded group)
    - T: Time to event
    - E: Event indicator (1 = observed, 0 = censored)
    - data: Original DataFrame
    """
    data = load_waltons()

    # Extract T (time) and E (event) columns
    T = data['T'].values
    E = data['E'].values

    # One-hot encode the 'group' column as features
    groups = data['group'].values
    unique_groups = np.unique(groups)
    X = np.zeros((len(groups), len(unique_groups)))

    for i, group in enumerate(unique_groups):
        X[groups == group, i] = 1

    return X, T, E, data

# ---------------------------------------------------------
# Utility: Kaplan–Meier estimator for a node
# ---------------------------------------------------------
def km_estimator(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a simple Kaplan–Meier survival curve."""
    order = np.argsort(times)
    t = times[order]
    e = events[order]

    unique_times = np.unique(t)
    survival = []

    n_at_risk = len(t)
    s = 1.0

    idx = 0
    for ut in unique_times:
        # events at this time
        d = np.sum((t == ut) & (e == 1))
        # number at risk
        n = np.sum(t >= ut)

        if n > 0:
            s *= (1 - d / n)
        survival.append(s)

    return unique_times, np.array(survival)


# ---------------------------------------------------------
# Survival Tree Node
# ---------------------------------------------------------
@dataclass
class SurvivalTreeNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["SurvivalTreeNode"] = None
    right: Optional["SurvivalTreeNode"] = None
    times: Optional[np.ndarray] = None
    events: Optional[np.ndarray] = None
    km_times: Optional[np.ndarray] = None
    km_survival: Optional[np.ndarray] = None


# ---------------------------------------------------------
# Survival Tree
# ---------------------------------------------------------
class SurvivalTree:
    def __init__(self, max_depth=5, min_samples_split=10, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root: Optional[SurvivalTreeNode] = None

    def fit(self, X, T, E):
        self.root = self._build_tree(X, T, E, depth=0)

    def _best_split(self, X, T, E):
        n_samples, n_features = X.shape
        features = np.arange(n_features)

        if self.max_features is not None:
            features = np.random.choice(features, self.max_features, replace=False)

        best_feature = None
        best_threshold = None
        best_score = -np.inf

        for f in features:
            thresholds = np.unique(X[:, f])
            for thr in thresholds:
                left = X[:, f] <= thr
                right = ~left

                if left.sum() < self.min_samples_split or right.sum() < self.min_samples_split:
                    continue

                # log-rank style score
                score = self._logrank_score(T[left], E[left], T[right], E[right])

                if score > best_score:
                    best_score = score
                    best_feature = f
                    best_threshold = thr

        return best_feature, best_threshold, best_score

    def _logrank_score(self, T_left, E_left, T_right, E_right):
        """Simple log-rank statistic."""
        if len(T_left) == 0 or len(T_right) == 0:
            return -np.inf

        t = np.unique(np.concatenate([T_left, T_right]))
        score = 0.0

        for ti in t:
            dl = np.sum((T_left == ti) & (E_left == 1))
            dr = np.sum((T_right == ti) & (E_right == 1))
            nl = np.sum(T_left >= ti)
            nr = np.sum(T_right >= ti)

            if nl + nr > 0:
                expected_l = (nl / (nl + nr)) * (dl + dr)
                score += (dl - expected_l)

        return abs(score)

    def _build_tree(self, X, T, E, depth):
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            node = SurvivalTreeNode(times=T, events=E)
            node.km_times, node.km_survival = km_estimator(T, E)
            return node

        feature, threshold, score = self._best_split(X, T, E)

        if feature is None:
            node = SurvivalTreeNode(times=T, events=E)
            node.km_times, node.km_survival = km_estimator(T, E)
            return node

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return SurvivalTreeNode(
            feature_index=feature,
            threshold=threshold,
            left=self._build_tree(X[left_mask], T[left_mask], E[left_mask], depth + 1),
            right=self._build_tree(X[right_mask], T[right_mask], E[right_mask], depth + 1),
        )

    def predict_survival(self, x: np.ndarray):
        node = self.root
        while node.left is not None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.km_times, node.km_survival


# ---------------------------------------------------------
# Random Survival Forest
# ---------------------------------------------------------
class RandomSurvivalForest:
    def __init__(self, n_estimators=50, max_depth=5, min_samples_split=10, max_features=None):
        self.n_estimators = n_estimators
        self.trees: List[SurvivalTree] = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def fit(self, X, T, E):
        n = len(X)
        self.trees = []

        for _ in range(self.n_estimators):
            idx = np.random.choice(n, n, replace=True)
            tree = SurvivalTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
            )
            tree.fit(X[idx], T[idx], E[idx])
            self.trees.append(tree)

    def predict_survival_function(self, x: np.ndarray):
        survs = []
        all_times = []

        for tree in self.trees:
            t, s = tree.predict_survival(x)
            survs.append((t, s))
            all_times.extend(t)

        # Create a common time grid from all unique times
        common_times = np.unique(np.concatenate([t for t, s in survs]))

        # Interpolate all survival curves to the common time grid
        interpolated_survs = []
        for t, s in survs:
            # Use left-continuous step function interpolation for survival curves
            interp_s = np.interp(common_times, t, s, left=1.0, right=s[-1])
            interpolated_survs.append(interp_s)

        # Average survival curves
        mean_survival = np.mean(interpolated_survs, axis=0)

        return common_times, mean_survival
