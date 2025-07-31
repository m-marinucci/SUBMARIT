"""Validation and empirical distribution utilities for SUBMARIT.

This module provides comprehensive validation tools including:
- K-fold cross-validation
- Rand index calculations
- Empirical distribution generation
- P-value calculations
- Multiple runs optimization
"""

# K-fold validation
from submarit.validation.kfold import (
    KFoldValidator,
    KFoldResult,
    k_fold_validate
)

# Rand index
from submarit.validation.rand_index import (
    RandIndex,
    RandIndexResult,
    RandEmpiricalDistribution,
    create_rand_empirical_distribution,
    rand_empirical_p
)

# Empirical distributions
from submarit.validation.empirical_distributions import (
    SwitchingMatrixEmpiricalDistribution,
    create_switching_matrix_distribution,
    k_sm_empirical_p,
    create_bootstrap_distribution
)

# Multiple runs
from submarit.validation.multiple_runs import (
    run_clusters,
    run_clusters_constrained,
    compare_multiple_runs,
    evaluate_clustering_stability
)

# Top-k analysis
from submarit.validation.topk_analysis import (
    TopKResult,
    run_clusters_topk,
    analyze_solution_stability
)

# P-value utilities
from submarit.validation.p_values import (
    PValueResult,
    calculate_empirical_p_value,
    calculate_p_value_range,
    multiple_testing_correction,
    combine_p_values,
    permutation_test
)

__all__ = [
    # K-fold validation
    "KFoldValidator",
    "KFoldResult", 
    "k_fold_validate",
    
    # Rand index
    "RandIndex",
    "RandIndexResult",
    "RandEmpiricalDistribution",
    "create_rand_empirical_distribution",
    "rand_empirical_p",
    
    # Empirical distributions
    "SwitchingMatrixEmpiricalDistribution",
    "create_switching_matrix_distribution",
    "k_sm_empirical_p",
    "create_bootstrap_distribution",
    
    # Multiple runs
    "run_clusters",
    "run_clusters_constrained",
    "compare_multiple_runs",
    "evaluate_clustering_stability",
    
    # Top-k analysis
    "TopKResult",
    "run_clusters_topk", 
    "analyze_solution_stability",
    
    # P-values
    "PValueResult",
    "calculate_empirical_p_value",
    "calculate_p_value_range",
    "multiple_testing_correction",
    "combine_p_values",
    "permutation_test"
]