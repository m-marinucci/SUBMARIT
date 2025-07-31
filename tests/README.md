# SUBMARIT Test Suite

This directory contains a comprehensive test suite for the SUBMARIT (SUBMARket Identification and Testing) package, ensuring correctness, performance, and reliability of all components.

## Test Categories

### 1. Unit Tests
- **Location**: Various `test_*.py` files
- **Purpose**: Test individual functions and classes in isolation
- **Coverage**: Core algorithms, data structures, utilities

### 2. Integration Tests (`test_integration.py`)
- **Purpose**: Test complete workflows and component interactions
- **Coverage**: End-to-end pipelines, multi-component scenarios
- **Key Tests**:
  - Full pipeline from consumer data to clustering results
  - Constrained clustering workflows
  - Validation and evaluation pipelines

### 3. Property-Based Tests (`test_property_based.py`)
- **Framework**: Hypothesis
- **Purpose**: Test algorithm properties with randomly generated inputs
- **Coverage**: 
  - Clustering invariants
  - Matrix properties preservation
  - Algorithm consistency
  - Numerical stability

### 4. Regression Tests (`test_regression.py`)
- **Purpose**: Ensure consistent behavior across versions
- **Coverage**:
  - Deterministic outputs with fixed seeds
  - Expected quality on synthetic problems
  - MATLAB compatibility
  - Numerical accuracy

### 5. Performance Benchmarks (`test_performance.py`)
- **Purpose**: Measure and track performance characteristics
- **Metrics**:
  - Execution time vs matrix size
  - Memory usage
  - Scalability analysis
  - Algorithm comparison

### 6. Edge Cases (`test_edge_cases.py`)
- **Purpose**: Test robustness with unusual inputs
- **Coverage**:
  - Minimum/maximum sizes
  - Sparse/dense matrices
  - Numerical edge cases
  - Error conditions

## Test Fixtures (`test_fixtures.py`)

Common test data and utilities:
- Sample substitution matrices (small, medium, large)
- Clustered data with known structure
- Consumer-product data generators
- MATLAB reference data
- Assertion helpers

## Running Tests

### Quick Start
```bash
# Run all fast tests
pytest

# Run with coverage
pytest --cov=submarit --cov-report=html

# Run specific test category
pytest -m integration
pytest -m benchmark
pytest -m slow
```

### Comprehensive Test Runner
```bash
# Run all tests with detailed reporting
python run_all_tests.py --all

# Run specific test categories
python run_all_tests.py --unit --integration
python run_all_tests.py --performance
python run_all_tests.py --coverage
```

### Test Markers
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.benchmark` - Performance benchmarks
- `@pytest.mark.slow` - Tests that take > 5 seconds
- `@pytest.mark.parametrize` - Parameterized tests

## Coverage Requirements

- **Target**: 80% overall coverage
- **Critical paths**: Must have 100% coverage
  - Core clustering algorithms
  - Substitution matrix creation
  - Evaluation metrics

View coverage report: `open htmlcov/index.html`

## Performance Benchmarks

### Running Benchmarks
```bash
# Run all benchmarks
pytest -m benchmark tests/test_performance.py

# Generate benchmark visualizations
python benchmarks/visualize_benchmarks.py
```

### Benchmark Metrics
- Algorithm timing vs matrix size
- Memory usage profiling
- Scalability analysis (O(n) complexity)
- Convergence characteristics

## Property-Based Testing

Using Hypothesis for generative testing:
- Random substitution matrices
- Consumer-product data
- Clustering parameters
- Edge case generation

### Key Properties Tested
1. **Clustering Invariants**
   - All items assigned
   - Number of clusters preserved
   - Minimum items constraint

2. **Matrix Properties**
   - Symmetry preservation
   - Normalization
   - Zero diagonal

3. **Algorithm Properties**
   - Determinism with fixed seed
   - Objective improvement
   - Constraint satisfaction

## Continuous Integration

### Pre-commit Checks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### CI Pipeline
1. Linting (flake8, black, isort)
2. Type checking (mypy)
3. Unit tests
4. Integration tests
5. Coverage check (>80%)
6. Performance regression

## Adding New Tests

### Test Structure
```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_basic_functionality(self):
        """Test basic use case."""
        # Arrange
        data = create_test_data()
        
        # Act
        result = new_feature(data)
        
        # Assert
        assert result.is_valid()
    
    @pytest.mark.parametrize("param", [1, 2, 3])
    def test_parameterized(self, param):
        """Test with different parameters."""
        result = new_feature(param=param)
        assert result > 0
```

### Best Practices
1. Use descriptive test names
2. Follow Arrange-Act-Assert pattern
3. Use fixtures for common setup
4. Test both success and failure cases
5. Include edge cases
6. Add performance tests for new algorithms

## Debugging Failed Tests

### Verbose Output
```bash
pytest -vv tests/test_specific.py::TestClass::test_method
```

### Debug Mode
```bash
pytest --pdb  # Drop into debugger on failure
pytest -x     # Stop on first failure
pytest --lf   # Run last failed tests
```

### Test Isolation
```bash
pytest tests/test_specific.py -k "test_name"
```

## Test Data

### Synthetic Data
- Generated programmatically
- Reproducible with seeds
- Known properties

### Reference Data
- MATLAB outputs (when available)
- Published benchmarks
- Validated results

## Performance Monitoring

### Tracking Performance
Results saved in `benchmark_results/`:
- Timing data (JSON)
- Memory profiles
- Scalability plots

### Regression Detection
- Compare against baseline
- Alert on >10% degradation
- Track trends over time

## Contributing Tests

1. **New Features**: Add unit and integration tests
2. **Bug Fixes**: Add regression test
3. **Performance**: Add benchmark if algorithmic
4. **Edge Cases**: Add to `test_edge_cases.py`

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Slow Tests**
   ```bash
   pytest -m "not slow"
   ```

3. **Memory Issues**
   ```bash
   pytest --no-cov  # Disable coverage
   ```

4. **Platform-Specific**
   - Check `sys.platform` conditions
   - Use `pytest.mark.skipif`

### Getting Help
- Check test output carefully
- Review fixture data
- Use debugger for complex issues
- Check CI logs for environment issues