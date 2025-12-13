# learning-ml-models

A comprehensive machine learning project for learning and understanding ML algorithms through both educational custom implementations and scikit-learn comparisons. This repository follows the Microsoft Learn path "Create machine learning models" and contains starter notebooks, educational code examples, performance benchmarks, and unit tests for each module.

## 📚 Learning Modules

Each module contains Jupyter notebooks with hands-on exercises and explanations:

- **01-explore-analyze-data** - Explore and analyze data with Python
  - `Explore_and_analyze_data_with_Python.ipynb`
  - `Explore_data_distributions_and_comparisons.ipynb`
  - `Exercise_Visualize_data_with_Matplotlib.ipynb`

- **02-train-evaluate-regression** - Train and evaluate regression models
  - `Get_started_with_regression.ipynb`
  - `Train_and_evaluate_regression_models.ipynb`
  - `Optimize_regression_models.ipynb`
  - `Experiment_with_regression_models.ipynb`

- **03-train-evaluate-classification** - Train and evaluate classification models

- **04-train-evaluate-clustering** - Train and evaluate clustering models

- **05-train-evaluate-deep-learning** - Train and evaluate deep learning models

## 📦 Core Package: `src/machine_learning`

The `machine_learning` package contains all educational ML implementations and scikit-learn comparison wrappers. This namespace-organized package follows PEP 517/518 standards.

**For detailed information on all modules, custom implementations, and import examples, see [src/quick-reference.md](src/quick-reference.md#-core-package-srcmachine_learning)**

### Quick Overview

The package includes:
- **Custom Implementations:** KNN, Linear Regression, Lasso, Ridge, Silhouette Score, MSE
- **Scikit-Learn Wrappers:** Pre-configured models for performance comparison
- **Educational Design:** Clear, readable code prioritizing learning over performance

### Basic Import Example

```python
from machine_learning.knn_regressor import KNeighborsRegressor
from machine_learning.knn_regressor_sklearn import KNeighborsRegressorSklearn
```

---

## 🧪 Unit Tests: `tests/` Directory

The `tests/` directory contains 15 comprehensive unit tests for all machine learning implementations.

### Quick Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=machine_learning

# Run specific test file
pytest tests/test_knn_regressor.py -v
```

### Test Results

All 15 tests pass in ~0.34 seconds:

```
tests/test_knn_regressor.py ............ [ 26%]
tests/test_lasso_regression.py ......... [ 33%]
tests/test_linear_regression.py ........ [ 46%]
tests/test_mse.py ....................... [ 60%]
tests/test_ridge_regression.py ......... [ 66%]
tests/test_silhouette_score.py ......... [ 93%]

======================== 15 passed in 0.34s ========================
```

**For detailed test information, running instructions, and best practices, see [src/quick-reference.md#-unit-tests-tests-directory](src/quick-reference.md#-unit-tests-tests-directory)**

---

## 📊 Performance Benchmarks: `benchmarks/` Directory

The `benchmarks/` directory contains performance comparison scripts that measure custom implementations against scikit-learn.

### Quick Benchmark Commands

```bash
# Run KNN regressor benchmark
python benchmarks/benchmark_knn_regressor.py

# Run linear regression benchmark
python benchmarks/benchmark_linear_regression.py

# Run silhouette score benchmark
python benchmarks/benchmark_silhouette_score.py
```

### Example Output

```
Implementation            Avg        Min        Max
Custom                    0.04115  0.03469  0.05081
Sklearn                   0.02447  0.01508  0.04301
```

This shows scikit-learn is ~1.7x faster due to C/Fortran optimizations.

**For detailed benchmark information, interpretation guide, and how to create benchmarks, see [src/quick-reference.md#-performance-benchmarks-benchmarks-directory](src/quick-reference.md#-performance-benchmarks-benchmarks-directory)**

## 🚀 Quick Start (Git Bash on Windows)

### 1. Create and Activate Virtual Environment

```bash
# create venv
python -m venv .venv

# activate (Git Bash on Windows)
source .venv/Scripts/activate
```

### 2. Install Project and Dependencies

```bash
# install project in editable mode
pip install -e .

# optional: install dev or notebook extras
pip install -e ".[dev]"
pip install -e ".[notebook]"
```

### 3. Run Tests and Benchmarks

```bash
# run all unit tests
pytest tests/ -v

# run a benchmark
python benchmarks/benchmark_knn_regressor.py
```

### 4. Start Jupyter Lab

```bash
# start Jupyter Lab
jupyter lab --port 8888 --no-browser
```

Navigate to `http://localhost:8888` in your browser.

## 📋 Project Structure

```
learning-ml-models/
├── .github/
│   └── copilot-instructions.md      # GitHub Copilot configuration
├── benchmarks/                       # Performance benchmark scripts
├── generated/                        # Generated outputs
│   ├── docs-copilot/                # Copilot-generated documentation
│   ├── data/                         # Generated data files
│   ├── images/                       # Generated visualizations
│   └── models/                       # Trained model artifacts
├── learning_modules/                 # Learning notebooks and modules
│   ├── 01-explore-analyze-data/
│   ├── 02-train-evaluate-regression/
│   ├── 03-train-evaluate-classification/
│   ├── 04-train-evaluate-clustering/
│   └── 05-train-evaluate-deep-learning/
├── scripts/                          # Utility scripts
├── src/
│   ├── machine_learning/             # ML model implementations and wrappers
│   ├── quick-reference.md            # Comprehensive reference guide
│   └── learning_ml_models.egg-info/
├── tests/                            # Unit tests (15 tests total)
├── pyproject.toml                    # Project configuration (PEP 517/518)
├── README.md                         # This file
└── .venv/                            # Virtual environment (git-ignored)
```

## 📝 Notes

- This project uses **Git Bash on Windows** for all example commands
- Configuration details are in `.github/copilot-instructions.md`
- Jupyter notebooks are stored in `learning_modules/*/`
- All custom implementations include unit tests in `tests/`
- Performance benchmarks are available in `benchmarks/`

---

**Last Updated:** December 13, 2025  
**Status:** ✅ Production Ready
