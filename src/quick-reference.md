# Quick Reference Guide: learning-ml-models

**Last Updated:** December 13, 2025

---

## 📦 Core Package: `src/machine_learning`

The `machine_learning` package contains all educational ML implementations and scikit-learn comparison wrappers. This namespace-organized package follows PEP 517/518 standards.

### Custom Implementations

Educational implementations of classic ML algorithms designed for learning:

- **`knn_regressor.py`** – Custom k-Nearest Neighbors Regressor
  - Manual implementation without scikit-learn
  - Demonstrates core KNN concepts (distance calculation, neighbor selection, averaging)
  - Perfect for understanding how KNN works under the hood

- **`linear_regression.py`** – Custom Linear Regression
  - Normal equation-based implementation
  - Shows matrix operations and optimization concepts
  - Educational reference for understanding linear regression

- **`lasso_regression.py`** – Custom Lasso Regression with L1 Regularization
  - Demonstrates regularization techniques
  - Includes coefficient shrinkage and sparsity concepts

- **`ridge_regression.py`** – Custom Ridge Regression with L2 Regularization
  - L2 regularization implementation
  - Shows how regularization prevents overfitting

- **`silhouette_score.py`** – Custom Silhouette Score Calculator
  - Cluster evaluation metric implementation
  - Measures cohesion and separation of clusters
  - Educational implementation of clustering quality assessment

- **`mse.py`** – Mean Squared Error Utility
  - Basic MSE calculation for regression evaluation
  - Foundation for understanding regression metrics

### Scikit-Learn Wrapper Modules

Pre-configured scikit-learn implementations for performance comparison:

- **`knn_regressor_sklearn.py`** – Scikit-learn KNN Regressor wrapper
- **`linear_regression_sklearn.py`** – Scikit-learn Linear Regression wrapper
- **`silhouette_score_sklearn.py`** – Scikit-learn Silhouette Score wrapper

These wrappers provide a consistent interface for benchmarking custom implementations against industry-standard scikit-learn models.

### Importing from machine_learning

All modules are imported from the `machine_learning` package:

```python
# Custom implementations
from machine_learning.knn_regressor import KNeighborsRegressor
from machine_learning.linear_regression import LinearRegression
from machine_learning.lasso_regression import LassoRegression
from machine_learning.ridge_regression import RidgeRegression
from machine_learning.silhouette_score import silhouette_score
from machine_learning.mse import mse

# Scikit-learn wrappers
from machine_learning.knn_regressor_sklearn import KNeighborsRegressorSklearn
from machine_learning.linear_regression_sklearn import LinearRegressionSklearn
from machine_learning.silhouette_score_sklearn import silhouette_score_sklearn
```

---

## 🧪 Unit Tests: `tests/` Directory

The `tests/` directory contains comprehensive unit tests for all machine learning implementations. Tests verify correctness, edge cases, and consistency with scikit-learn.

### Test Files Overview

- **`test_knn_regressor.py`** – Tests for custom KNN implementation
  - Basic functionality with multiple neighbors
  - Single neighbor edge case
  - Invalid input validation
  - Not-fitted error handling

- **`test_linear_regression.py`** – Tests for custom linear regression
  - Fitting and prediction accuracy
  - Not-fitted state error handling

- **`test_lasso_regression.py`** – Tests for Lasso regression
  - L1 regularization effects
  - Coefficient sparsity

- **`test_ridge_regression.py`** – Tests for Ridge regression
  - L2 regularization effects
  - Overfitting prevention

- **`test_silhouette_score.py`** – Tests for silhouette score calculation
  - Simple cluster separation
  - Single cluster edge case
  - Perfect separation scenarios
  - Mixed/overlapping clusters
  - Empty input handling

- **`test_mse.py`** – Tests for MSE calculation
  - Basic MSE computation
  - Various error magnitudes

### Running Unit Tests

#### Run All Tests

```bash
pytest tests/
```

All tests should run and pass within ~0.34 seconds.

#### Run Tests with Verbose Output

Shows each individual test as it runs:

```bash
pytest tests/ -v
```

#### Run Tests with Coverage Report

Generate a code coverage report to see which lines of code are tested:

```bash
pytest tests/ --cov=machine_learning
```

#### Run Specific Test File

Run only tests from a single file:

```bash
pytest tests/test_knn_regressor.py -v
```

#### Run Specific Test Function

Run a single test function by name:

```bash
pytest tests/test_knn_regressor.py::test_knn_regressor_basic -v
```

#### Run Tests Matching a Pattern

Run tests matching a keyword pattern:

```bash
pytest tests/ -k "knn" -v
```

#### Run Tests and Stop on First Failure

Useful for debugging - stops immediately when a test fails:

```bash
pytest tests/ -x
```

#### Run Tests with Maximum Detail

Shows verbose output including print statements:

```bash
pytest tests/ -vv
```

### Test Results

All 15 tests should pass:

```
tests/test_knn_regressor.py ............ [ 26%]
tests/test_lasso_regression.py ......... [ 33%]
tests/test_linear_regression.py ........ [ 46%]
tests/test_mse.py ....................... [ 60%]
tests/test_ridge_regression.py ......... [ 66%]
tests/test_silhouette_score.py ......... [ 93%]

======================== 15 passed in 0.34s ========================
```

### Best Practices for Testing

1. **Run tests frequently** – Before committing changes
2. **Check coverage** – Aim for high coverage on critical code
3. **Add new tests** – When adding new features or fixing bugs
4. **Keep tests isolated** – Each test should be independent
5. **Use descriptive names** – Test function names should describe what they test

---

## 📊 Performance Benchmarks: `benchmarks/` Directory

The `benchmarks/` directory contains scripts that compare the performance of custom implementations against their scikit-learn counterparts. These benchmarks help measure efficiency and identify optimization opportunities.

### Benchmark Files Overview

- **`benchmark_knn_regressor.py`** – Compares custom KNN vs. scikit-learn KNN
  - Tests performance on realistic dataset sizes
  - Measures average, minimum, and maximum execution times
  - Demonstrates the efficiency trade-off between educational and optimized implementations

- **`benchmark_linear_regression.py`** – Benchmarks linear regression implementations
  - Compares custom normal equation vs. scikit-learn solver
  - Tests on various data dimensions

- **`benchmark_silhouette_score.py`** – Benchmarks clustering evaluation metrics
  - Compares custom implementation with scikit-learn
  - Measures performance on different cluster configurations

### Running Benchmarks

#### Run Individual Benchmark

Run a specific benchmark script:

```bash
python benchmarks/benchmark_knn_regressor.py
```

#### Run All Benchmarks

Run each benchmark sequentially:

```bash
python benchmarks/benchmark_knn_regressor.py
python benchmarks/benchmark_linear_regression.py
python benchmarks/benchmark_silhouette_score.py
```

#### Run Benchmarks with Custom Parameters

Modify benchmark scripts to test with different dataset sizes or iterations:

```bash
# Edit the benchmark file to adjust:
# - Number of samples
# - Number of features
# - Number of iterations
# - Algorithm parameters
```

### Typical Benchmark Output

Example output from `benchmark_knn_regressor.py`:

```
Implementation            Avg        Min        Max
Custom                    0.04115  0.03469  0.05081
Sklearn                   0.02447  0.01508  0.04301
```

**Interpreting the Output:**

- **Custom Implementation:** ~0.041 seconds average
  - Educational implementation with clear, readable code
  - Prioritizes clarity over performance
  - Good for learning and understanding the algorithm

- **Scikit-learn Implementation:** ~0.024 seconds average
  - Optimized implementation with C/Fortran backend
  - Production-ready code
  - ~1.7x faster than custom implementation

- **Performance Ratio:** scikit-learn is approximately 1.7x faster due to:
  - C/Fortran optimizations
  - BLAS/LAPACK libraries
  - Vectorized operations
  - Compiled code vs. Python

### Understanding Benchmark Results

Benchmarks help you understand:

1. **Performance Trade-offs**
   - Custom implementations prioritize clarity over speed
   - Production implementations prioritize performance
   - Educational value vs. computational cost

2. **Scalability**
   - How algorithms perform as dataset size increases
   - Time complexity in practice
   - Memory usage patterns

3. **Optimization Targets**
   - Where custom implementations could be optimized
   - Which operations are bottlenecks
   - Opportunities for vectorization

4. **Learning Value**
   - The practical cost of educational implementations
   - Real-world performance implications
   - Why production libraries matter

### Benchmark Best Practices

1. **Run multiple times** – Get average across multiple runs
2. **Use realistic data** – Test with data similar to your use case
3. **Control variables** – Change one parameter at a time
4. **Document results** – Keep track of results over time
5. **Compare fairly** – Use same dataset/parameters for all implementations

### Creating Your Own Benchmarks

Template for creating a new benchmark:

```python
import time
import numpy as np
from machine_learning.your_module import YourImplementation
from machine_learning.your_module_sklearn import YourImplementationSklearn

# Generate test data
X = np.random.randn(1000, 10)
y = np.random.randn(1000)

# Benchmark custom implementation
times_custom = []
for _ in range(10):
    start = time.time()
    model = YourImplementation()
    model.fit(X, y)
    model.predict(X)
    times_custom.append(time.time() - start)

# Benchmark sklearn implementation
times_sklearn = []
for _ in range(10):
    start = time.time()
    model = YourImplementationSklearn()
    model.fit(X, y)
    model.predict(X)
    times_sklearn.append(time.time() - start)

# Print results
print(f"Custom:  {np.mean(times_custom):.5f}s ± {np.std(times_custom):.5f}s")
print(f"Sklearn: {np.mean(times_sklearn):.5f}s ± {np.std(times_sklearn):.5f}s")
print(f"Ratio:   {np.mean(times_custom) / np.mean(times_sklearn):.2f}x")
```

---

## 🚀 Complete Workflow Examples

### Example 1: Develop and Test a New Implementation

```bash
# 1. Create your implementation in src/machine_learning/my_algorithm.py
# 2. Create corresponding tests in tests/test_my_algorithm.py
# 3. Run tests to verify correctness
pytest tests/test_my_algorithm.py -v

# 4. Run all tests to ensure no regressions
pytest tests/ -v

# 5. Create benchmark in benchmarks/benchmark_my_algorithm.py
# 6. Run benchmark to measure performance
python benchmarks/benchmark_my_algorithm.py

# 7. Check overall coverage
pytest tests/ --cov=machine_learning
```

### Example 2: Debug a Failing Test

```bash
# 1. Run tests to see what's failing
pytest tests/ -v

# 2. Run specific failing test with extra verbosity
pytest tests/test_module.py::test_function -vv

# 3. Stop on first failure to see the error clearly
pytest tests/test_module.py::test_function -x

# 4. Run with print statements visible
pytest tests/test_module.py::test_function -vv -s
```

### Example 3: Performance Investigation

```bash
# 1. Run the benchmark
python benchmarks/benchmark_algorithm.py

# 2. If custom is slower than expected:
#    - Check implementation for inefficiencies
#    - Look for unnecessary loops or copies
#    - Consider vectorization with numpy

# 3. Profile the code (requires cProfile)
python -m cProfile -s cumulative benchmarks/benchmark_algorithm.py

# 4. Re-run benchmark after optimization
python benchmarks/benchmark_algorithm.py
```

---

## 📚 Common Commands Reference

| Task | Command |
|------|---------|
| Run all tests | `pytest tests/` |
| Run tests verbose | `pytest tests/ -v` |
| Run with coverage | `pytest tests/ --cov=machine_learning` |
| Run specific test | `pytest tests/test_file.py::test_name -v` |
| Run matching pattern | `pytest tests/ -k "pattern" -v` |
| Run benchmark | `python benchmarks/benchmark_name.py` |
| Stop on failure | `pytest tests/ -x` |
| Detailed output | `pytest tests/ -vv -s` |

---

## 📝 Tips & Tricks

### Organizing Custom Implementations

```python
# Good: Clear structure with helper functions
def fit(self, X, y):
    """Fit the model to training data."""
    self._validate_input(X, y)
    self._compute_coefficients(X, y)
    self.is_fitted_ = True
    return self

def _validate_input(self, X, y):
    """Validate input data shapes and types."""
    ...

def _compute_coefficients(self, X, y):
    """Compute model coefficients."""
    ...
```

### Writing Effective Tests

```python
# Good: Descriptive names and clear assertions
def test_knn_regressor_basic():
    """Test KNN with basic usage."""
    model = KNeighborsRegressor(n_neighbors=3)
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([1, 2, 3])
    
    model.fit(X, y)
    predictions = model.predict(np.array([[2, 3]]))
    
    assert predictions is not None
    assert len(predictions) == 1
```

### Iterating on Performance

1. **Measure first** – Run benchmark to establish baseline
2. **Profile second** – Identify bottlenecks
3. **Optimize third** – Focus on high-impact areas
4. **Verify fourth** – Re-run benchmark to confirm improvement
5. **Test always** – Ensure correctness throughout

---

## 🔗 Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Python timeit Module](https://docs.python.org/3/library/timeit.html)
- [NumPy Performance Tips](https://numpy.org/doc/stable/reference/routines.array-creation.html)
- [PEP 517: Build System](https://peps.python.org/pep-0517/)
- [PEP 518: Dependency Specification](https://peps.python.org/pep-0518/)

