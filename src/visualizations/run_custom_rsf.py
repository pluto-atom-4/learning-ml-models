from src.machine_learning.custom_random_survival_forest import (
    load_waltons_dataset,
    RandomSurvivalForest,
)
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path to enable src imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    X, T, E, df = load_waltons_dataset()

    model = RandomSurvivalForest(
        n_estimators=20,
        max_depth=4,
        min_samples_split=10,
        max_features=1,
    )
    model.fit(X, T, E)

    # Features correspond to one-hot encoded groups: [miR-137, control]
    x_group0 = np.array([1.0, 0.0])  # miR-137 group
    x_group1 = np.array([0.0, 1.0])  # control group

    t0, s0 = model.predict_survival_function(x_group0)
    t1, s1 = model.predict_survival_function(x_group1)

    plt.figure(figsize=(10, 6))
    plt.step(t0, s0, where="post", label="RSF miR-137 group")
    plt.step(t1, s1, where="post", label="RSF control group")
    plt.title("Custom Random Survival Forest")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
