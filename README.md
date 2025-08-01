# SUBMARIT - SUBMARket Identification and Testing

A Python implementation of submarket clustering algorithms for analyzing product substitution patterns.

## Overview

SUBMARIT is a comprehensive toolkit for identifying and analyzing submarkets based on product substitution patterns. This Python implementation provides:

- Efficient clustering algorithms for submarket identification
- Statistical evaluation methods
- Validation techniques including k-fold cross-validation
- Support for large-scale data analysis
- MATLAB compatibility layer for seamless migration

## Installation

### From Source

```bash
git clone https://github.com/yourusername/submarit.git
cd submarit
pip install -e .
```

### For Development

```bash
pip install -e ".[dev]"
pre-commit install
```

## Quick Start

```python
import submarit
import numpy as np

# Load substitution matrix
data = submarit.load_substitution_data("data.csv")
matrix = submarit.SubstitutionMatrix(data)

# Run clustering
clusterer = submarit.LocalSearch(n_clusters=5)
labels = clusterer.fit_predict(matrix)

# Evaluate results
evaluator = submarit.ClusterEvaluator()
metrics = evaluator.evaluate(matrix, labels)
print(f"Log-likelihood: {metrics.log_likelihood}")
print(f"Z-score: {metrics.z_score}")
```

## Features

- **Core Algorithms**
  - Local search optimization (quick approximation and direct log-likelihood)
  - Constrained clustering with fixed assignments
  - Multiple initialization strategies

- **Evaluation Metrics**
  - Log-likelihood calculations
  - Z-value computations
  - GAP statistic for optimal cluster selection
  - Entropy-based comparisons

- **Validation**
  - K-fold cross-validation
  - Empirical distribution generation
  - Rand index calculations
  - P-value computations

- **Performance**
  - Optimized NumPy operations
  - Optional Numba JIT compilation
  - Parallel processing support
  - Memory-efficient sparse matrix handling

## Documentation

### 📚 Online Documentation
Full documentation is available at [https://submarit.readthedocs.io](https://submarit.readthedocs.io)

### 📖 Documentation Contents
- [Installation Guide](docs/source/installation.rst) - Platform-specific installation instructions
- [Quick Start Tutorial](docs/source/quickstart.rst) - Get started with SUBMARIT in minutes
- [API Reference](docs/source/api.rst) - Complete API documentation with examples
- [Algorithm Theory](docs/source/algorithms.rst) - Mathematical foundations and implementation details
- [Performance Guide](docs/source/performance.rst) - Optimization strategies and benchmarks
- [FAQ](docs/source/faq.rst) - Frequently asked questions

### 🔄 Migration from MATLAB
- [Migration Guide](docs/source/migration_guide.rst) - Comprehensive guide for MATLAB users
- [Function Mapping](docs/source/migration_guide.rst#function-mapping) - 1-to-1 MATLAB to Python function reference
- [Migration Examples](examples/05_matlab_migration.ipynb) - Jupyter notebook with practical examples

### 📓 Example Notebooks
- [Getting Started](examples/01_getting_started.ipynb) - Basic introduction to SUBMARIT
- [Advanced Clustering](examples/02_advanced_clustering.ipynb) - Advanced techniques and algorithms
- [Performance Optimization](examples/03_performance_optimization.ipynb) - Tips for optimal performance
- [Visualization Gallery](examples/04_visualization_gallery.ipynb) - Beautiful visualizations
- [MATLAB Migration](examples/05_matlab_migration.ipynb) - Examples for MATLAB users

### 🧪 Testing
- [Test Suite Documentation](tests/README.md) - Guide to running tests
- [Benchmarks](benchmarks/README.md) - Performance benchmark results

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SUBMARIT in your research, please cite:

```bibtex
@software{submarit,
  title = {SUBMARIT: SUBMARket Identification and Testing},
  year = {2024},
  url = {https://github.com/yourusername/submarit}
}
```

## Acknowledgments

This is a Python implementation of the original MATLAB SUBMARIT package. The original MATLAB files are preserved in the `matlab_original/` directory for reference and validation purposes.

### Original MATLAB Implementation Credits
The MATLAB implementation includes contributions from:
- Stephen France, Mississippi State University (RandIndex4.m, 2012)
- Additional contributors (names unknown)

The methodology is based on submarket identification research from marketing science literature, including:
- Rand (1971) - Rand Index for clustering similarity
- Hubert and Arabie (1985) - Adjusted Rand Index
- Urban, Johnson, and Hauser - Z-value calculations
- Tibshirani, Walther, and Hastie (2001) - GAP statistic for optimal cluster selection