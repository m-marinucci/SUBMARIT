# Original MATLAB Implementation

This directory contains the original MATLAB implementation of SUBMARIT (SUBMARket Identification and Testing).

## ⚠️ Important Notice

**The Python implementation in the parent directory is now the primary and actively maintained version of SUBMARIT.**

These MATLAB files are preserved for:
- Historical reference
- Validation of Python implementation
- Users who still require MATLAB version
- Understanding original algorithm implementations

## 📁 Contents

- **Core Algorithms**
  - `kSMLocalSearch.m`, `kSMLocalSearch2.m` - Main clustering algorithms
  - `kSMLocalSearchConstrained.m`, `kSMLocalSearchConstrained2.m` - Constrained versions

- **Data Processing**
  - `CreateSubstitutionMatrix.m` - Create substitution matrices from sales data

- **Evaluation Functions**
  - `kSMEvaluateClustering.m`, `kEvaluateClustering.m` - Cluster evaluation
  - `GAPStatisticUniform.m` - GAP statistic for optimal k
  - `kSMEntropy.m` - Entropy-based clustering

- **Validation Functions**
  - `kSMNFold.m`, `kSMNFold2.m` - K-fold cross-validation
  - `kSMCreateDist.m`, `kSMEmpiricalP.m` - Empirical distributions
  - `RandIndex4.m`, `RandCreateDist.m`, `RandEmpiricalP.m` - Rand index calculations

- **Runner Functions**
  - `RunClusters.m`, `RunClusters2.m` - Multiple run execution
  - `RunClustersTopk.m`, `RunClustersTopk2.m` - Top-k analysis

- **Documentation**
  - `READMEFIRST.txt` - Original MATLAB documentation

## 🔄 Using for Validation

To validate Python results against MATLAB:

```matlab
% MATLAB code
load('test_data.mat');
[FSWM, PIndexes, PCount] = CreateSubstitutionMatrix(X, 1, 0, 0);
[Items, Clusters, P, PHat, LL, DItems, DLL] = kSMLocalSearch(FSWM, 5);
```

```python
# Equivalent Python code
import submarit
from submarit.io import load_mat

data = load_mat('test_data.mat')
fswm, p_indexes, p_count = submarit.create_substitution_matrix(
    data['X'], normalize=True, weight=0, diag=False
)
result = submarit.KSMLocalSearch(n_clusters=5).fit(fswm)
```

## 📊 Numerical Differences

Small numerical differences between MATLAB and Python implementations are expected due to:
- Different BLAS/LAPACK implementations
- Floating-point precision handling
- Random number generation

Typical differences are within 1e-10 relative tolerance.

## 📚 Migration Guide

For detailed migration instructions, see:
- [Python Migration Guide](../docs/source/migration_guide.rst)
- [Function Mapping Table](../docs/source/migration_guide.rst#function-mapping)
- [Migration Examples](../examples/05_matlab_migration.ipynb)

## 🤝 Support

- For MATLAB-specific questions: Refer to original documentation in READMEFIRST.txt
- For Python implementation: See main project README.md
- For migration help: Open an issue in the main repository

## ⚖️ License

The MATLAB implementation maintains its original license. See the main project LICENSE file for details.