# Real-World Examples

## Linear Regression Visualization

This example is from the actual project implementation.

### Dataset: Advertising-three-media.csv
- **Features**: TV, Radio, Newspaper (advertising spending in $1000s)
- **Target**: Sales (in $1000s)
- **Rows**: ~200 observations
- **Use Case**: Predict sales from TV advertising budget

## Implementation Pattern

### Plot Module (`linear_regression_plot.py`)

Key features:
- `load_advertising_data()`: CSV data loading with column-specific names
- `plot_regression()`: Scatter plot + regression line with optional save
- Type hints with `tuple[np.ndarray, np.ndarray]` syntax
- Path handling with `pathlib.Path`

### Runner Module (`linear_regression_plot_runner.py`)

Key features:
- `sys.path.insert()` for src imports
- `Path(__file__).parent.parent.parent` for data path resolution
- Model fitting and prediction workflow
- Parameter printing for model interpretation

## Pattern Summary

```python
# Runner: sys.path setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Runner: Data path resolution
csv_path = Path(__file__).parent.parent.parent / "generated/data/raw/file.csv"

# Plot: Modern type hints
def load_data(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:

def plot_regression(
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    title: str = "...",
    save_path: str | Path | None = None,
) -> None:
```

## Testing Your Implementation

Run the runner:
```bash
python -m src.visualizations.linear_regression_plot_runner
```

Expected output:
```
Intercept (β₀): 7.0326
Coefficient (β₁): 0.0475

Regression equation: Sales = 7.0326 + 0.0475 × TV
[matplotlib plot displays]
```

## Common Variations

### Multiple Features
```python
X = np.array([[tv, radio, newspaper] for each sample])
model.fit(X, y)  # X is already 2D
```

### Saving to File
```python
plot_regression(x, y, y_pred, save_path="output/plot.png")
```

### Custom Visualization
Modify plot_regression to add more visual elements based on your needs.
