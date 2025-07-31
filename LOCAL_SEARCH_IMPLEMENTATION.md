# k-Submarket Local Search Implementation

## Overview

This document summarizes the successful conversion of four MATLAB k-submarket local search algorithms to Python. The implementation is located in `/Users/numinate/PY/SubmarketIdentificationTesting/src/submarit/algorithms/local_search.py`.

## Implemented Algorithms

### 1. KSMLocalSearch
- **Optimization**: Maximizes the difference (PHat - P)
- **MATLAB file**: `kSMLocalSearch.m`
- **Description**: Uses a quick approximation approach where items are moved to clusters that maximize the difference between observed switching proportions (PHat) and expected proportions under independence (P).

### 2. KSMLocalSearch2
- **Optimization**: Directly optimizes log-likelihood
- **MATLAB file**: `kSMLocalSearch2.m`
- **Description**: More sophisticated approach that tracks and incrementally updates log-likelihood values for efficiency.

### 3. KSMLocalSearchConstrained
- **Optimization**: Maximizes (PHat - P) with constraints
- **MATLAB file**: `kSMLocalSearchConstrained.m`
- **Description**: Allows fixing some items to specific clusters while optimizing the assignment of free items.

### 4. KSMLocalSearchConstrained2
- **Optimization**: Log-likelihood with constraints
- **MATLAB file**: `kSMLocalSearchConstrained2.m`
- **Description**: Combines constrained assignments with direct log-likelihood optimization.

## Key Features

### MATLAB Compatibility
- Uses 1-based indexing internally (converted to 0-based for NumPy operations)
- Maintains MATLAB's random number generation patterns via `MatlabRandom` class
- Ensures float64 precision throughout
- Result structure matches MATLAB output exactly

### Algorithm Components
1. **Initialization**: Random assignment ensuring minimum items per cluster
2. **Local Search**: Iterative improvement by moving items between clusters
3. **Objective Functions**:
   - Version 1 & Constrained: Maximize sum(PHat - P)
   - Version 2 & Constrained2: Minimize negative log-likelihood
4. **Convergence**: Stops when no items change clusters or max iterations reached

### Result Structure
The `LocalSearchResult` class contains:
- `Assign`: Cluster assignments (1-based)
- `Indexes`: Dictionary of item indices per cluster
- `Count`: Number of items per cluster
- `Diff`, `ItemDiff`, `ScaledDiff`: Various difference metrics
- `LogLH`, `LogLH2`: Log-likelihood values
- `ZValue`: Z-statistic for cluster quality
- `Var`, `SDComp`, `SDiff`: Variance components

## Usage Examples

### Basic Usage
```python
from submarit.algorithms import KSMLocalSearch

# Create model
model = KSMLocalSearch(n_clusters=3, min_items=2, random_state=42)

# Fit to switching matrix
model.fit(switching_matrix)

# Get results
result = model.get_result()
print(f"Cluster assignments: {result.Assign}")
print(f"Log-likelihood: {result.LogLH}")
```

### Constrained Usage
```python
from submarit.algorithms import KSMLocalSearchConstrained

# Define constraints
fixed_assign = np.array([1, 2, 3])  # Fix items to clusters
assign_indexes = np.array([1, 2, 3])  # Which items to fix
free_indexes = np.array([4, 5, 6, 7, 8])  # Which items are free

# Create and fit model
model = KSMLocalSearchConstrained(n_clusters=3, min_items=2)
model.fit_constrained(switching_matrix, fixed_assign, assign_indexes, free_indexes)
```

### Functional Interface (MATLAB-style)
```python
from submarit.algorithms import k_sm_local_search

# Direct function call
result = k_sm_local_search(switching_matrix, n_clusters=3, min_items=2)
```

## Testing

A comprehensive test suite (`test_local_search.py`) validates:
1. Basic algorithm functionality
2. Constrained versions
3. Edge cases (minimum items, zero sales)
4. Result structure completeness
5. MATLAB compatibility

All tests pass successfully, confirming accurate algorithm conversion.

## Integration

The algorithms are fully integrated into the SUBMARIT package:
- Base class inheritance from `BaseClusterer`
- Compatible with sklearn-style fit/predict interface
- Proper error handling and validation
- Comprehensive docstrings

## Performance Notes

- Algorithms converge quickly (typically 2-5 iterations on test data)
- Numerical stability is maintained through careful handling of edge cases
- Memory efficient with in-place updates where possible