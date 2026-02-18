---
name: viz-ml-script
title: Machine Learning Visualization Scripts
description: Create professional, modular visualization scripts for machine learning models using modern Python patterns with two-module architecture (plot module + runner module).
version: 1.0.0
author: Claude Code
tags:
  - python
  - machine-learning
  - visualization
  - matplotlib
  - regression
  - refactoring
keywords:
  - ML visualization
  - regression plots
  - matplotlib
  - Python patterns
  - modular design
  - type hints
category: Machine Learning
created: 2025-02-18
updated: 2025-02-18
project: learning-ml-models
---

# viz-ml-script: Machine Learning Visualization Scripts

A skill for creating professional, modular visualization scripts for machine learning models using modern Python patterns.

## Purpose

This skill guides the creation of two-module visualization systems:
1. **Plot Module** (`*_plot.py`) - Pure plotting functions with data loading utilities
2. **Runner Module** (`*_plot_runner.py`) - Orchestration script that fits models and calls plotting functions

Perfect for visualizing regression models, classification results, and model evaluations.

## Key Patterns

### Module Structure
```
src/visualizations/
├── linear_regression_plot.py         # Plotting functions only
└── linear_regression_plot_runner.py  # Data loading → Model fitting → Plotting
```

### Plot Module (`*_plot.py`)
- Contains data loading function(s)
- Contains plotting function(s)
- No model fitting logic
- Type-hinted with modern Python (3.11+) syntax
- Uses `pathlib.Path` for file handling
- Optional `save_path` parameter for saving figures

Example functions:
```python
def load_advertising_data(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load data from CSV."""

def plot_regression(
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Default Title",
    save_path: str | Path | None = None,
) -> None:
    """Plot actual data and predictions."""
```

### Runner Module (`*_plot_runner.py`)
- Adds parent directory to sys.path for imports
- Defines data path relative to script location using `Path(__file__)`
- Loads data using plot module functions
- Instantiates and fits the model
- Makes predictions
- Calls plot functions from plot module
- Optional: prints model parameters/statistics

Essential pattern:
```python
from pathlib import Path
import sys

# Enable src imports from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Resolve data path relative to script location
csv_path = Path(__file__).parent.parent.parent / "generated/data/raw/filename.csv"
```

## Best Practices

### Type Hints
- Use `from __future__ import annotations` for forward compatibility
- Use modern union syntax: `str | Path` instead of `Union[str, Path]`
- Type all function parameters and return values

### Imports
- Always add sys.path setup in runner before model imports
- Import plot functions from plot module
- Use relative imports: `from src.visualizations.module import function`

### Path Handling
- Use `Path(__file__).parent.parent.parent` to resolve to project root
- Works regardless of where the script is executed from
- Better than hardcoded relative paths like `"generated/data/raw/file.csv"`

### Plotting
- Use matplotlib with consistent figure sizes: `figsize=(10, 6)` or `(8, 5)`
- Add labels, title, grid, and legend
- Support optional `save_path` for automated testing
- Use `plt.show()` if no save_path provided

### Model Integration
- Fit models in runner, not in plot functions
- Plot functions should be model-agnostic (accept x, y, y_pred arrays)
- Print model parameters/statistics in runner's main()

## References

- LinearRegression class: `src/machine_learning/linear_regression.py`
- Similar patterns: `src/visualizations/logistic_plot_example.py`, `plot_polynomial_models.py`
- Dataset location: `generated/data/raw/`
