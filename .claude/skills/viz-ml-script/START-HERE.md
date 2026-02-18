# Quick Start: Create ML Visualization Scripts

## 30-Second Summary

Need to visualize a machine learning model? Use this skill to create two Python scripts:

1. **`*_plot.py`** - Pure plotting functions (no model logic)
2. **`*_plot_runner.py`** - Load data → Fit model → Plot results

## Key Checklist

- [ ] **Plot module** has data loading function(s)
- [ ] **Plot module** has plotting function(s) that accept `x`, `y`, `y_pred`
- [ ] **Plot module** uses type hints and docstrings
- [ ] **Runner module** has `sys.path.insert()` at the top
- [ ] **Runner module** uses `Path(__file__).parent.parent.parent` for data path
- [ ] Both modules follow PEP 8 and use modern Python (3.11+) syntax
- [ ] Functions have parameter type hints and return type hints
- [ ] Plotting function has optional `save_path` parameter

## Real Example

See the actual implementation in this project:
- `src/visualizations/linear_regression_plot.py` - Plot module
- `src/visualizations/linear_regression_plot_runner.py` - Runner module

## Common Issues

**"ModuleNotFoundError: No module named 'src'"**
→ Runner must have: `sys.path.insert(0, str(Path(__file__).parent.parent.parent))`

**"FileNotFoundError: ...csv file"**
→ Use: `Path(__file__).parent.parent.parent / "generated/data/raw/filename.csv"`
→ NOT: `"generated/data/raw/filename.csv"`

**Plot doesn't show**
→ Check: `if save_path: ... else: plt.show()`

## See Also

- SKILL.md - Complete guide with patterns and best practices
- TEMPLATES/ - Template files for quick setup
- references/EXAMPLES.md - Real-world examples from this project
