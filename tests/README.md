# PyGRANSO Tests

This directory contains tests for PyGRANSO functionality.

## Running Tests

### Install Test Dependencies

```bash
pip install -r tests/requirements-test.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test

```bash
# Run only eigenvalue optimization tests
pytest tests/test_eigenval_optimization.py

# Run only CPU vs CUDA consistency test
pytest tests/test_eigenval_optimization.py::TestEigenvalOptimization::test_cpu_cuda_consistency

# Run only CPU tests (skip CUDA)
pytest tests/test_eigenval_optimization.py -m "not cuda"
```

### Run with Coverage

```bash
pytest --cov=pygranso tests/
```

## Test Structure

- `conftest.py` - Pytest configuration and common fixtures
- `test_eigenval_optimization.py` - Tests for eigenvalue optimization example
- `requirements-test.txt` - Test dependencies

## Test Categories

### Eigenvalue Optimization Tests

These tests verify that the eigenvalue optimization example works correctly on both CPU and CUDA devices:

- `test_cpu_cuda_consistency` - Ensures CPU and CUDA produce consistent results
- `test_cpu_only` - Verifies CPU optimization works
- `test_cuda_only` - Verifies CUDA optimization works (if CUDA available)
- `test_solution_structure` - Checks solution object structure

## Fixtures

- `cuda_available` - Boolean indicating if CUDA is available
- `test_devices` - List of available test devices (CPU + CUDA if available)
- `tolerance` - Default tolerance for numerical comparisons (1e-6)
- `max_iterations` - Default max iterations for optimization tests (50)

## Notes

- Tests automatically skip CUDA tests if CUDA is not available
- The eigenvalue optimization test uses a reduced number of iterations (50) for faster testing
- All tests use the data file from `examples/data/spec_radius_opt_data.mat`
