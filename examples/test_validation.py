"""Test script for SUBMARIT validation functions.

This script demonstrates the usage of all validation functions including:
- K-fold cross-validation
- Rand index calculations
- Empirical distribution generation
- P-value calculations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from submarit.validation import (
    # K-fold validation
    KFoldValidator,
    k_fold_validate,
    
    # Rand index
    RandIndex,
    create_rand_empirical_distribution,
    rand_empirical_p,
    
    # Empirical distributions
    create_switching_matrix_distribution,
    k_sm_empirical_p,
    create_bootstrap_distribution,
    
    # Multiple runs
    run_clusters,
    run_clusters_constrained,
    compare_multiple_runs,
    evaluate_clustering_stability,
    
    # P-values
    calculate_empirical_p_value,
    calculate_p_value_range,
    multiple_testing_correction,
    combine_p_values
)

from submarit.core import SubstitutionMatrix
from submarit.io import load_matlab_matrix


def test_rand_index():
    """Test Rand index calculations."""
    print("\n" + "="*60)
    print("Testing Rand Index Calculations")
    print("="*60)
    
    # Create two example clusterings
    n_items = 20
    clusters1 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])
    clusters2 = np.array([1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 5, 5, 5, 5])
    
    # Calculate Rand index
    rand_calc = RandIndex()
    rand, adj_rand = rand_calc.calculate(clusters1, clusters2)
    
    print(f"Clustering 1: {clusters1}")
    print(f"Clustering 2: {clusters2}")
    print(f"Rand index: {rand:.4f}")
    print(f"Adjusted Rand index: {adj_rand:.4f}")
    
    # Get detailed results
    detailed = rand_calc.calculate_detailed(clusters1, clusters2)
    print(f"\nDetailed results:")
    print(detailed.summary())
    
    # Create empirical distribution
    print("\nCreating empirical distribution for Rand index...")
    emp_dist = create_rand_empirical_distribution(
        n_items=20,
        n_clusters=5,
        n_points=1000,
        min_items=2,
        random_state=42
    )
    
    # Calculate p-values
    p_values = emp_dist.calculate_p_values(rand, adj_rand)
    print(f"\nP-values from empirical distribution:")
    print(f"Rand index p-value: {p_values['rand'][0]:.4f}")
    print(f"Adjusted Rand p-value: {p_values['adj_rand'][0]:.4f}")
    
    # Get percentiles
    percentiles = emp_dist.get_percentiles()
    print(f"\nEmpirical distribution percentiles:")
    print(f"Rand 95% CI: [{percentiles['rand'][1]:.4f}, {percentiles['rand'][7]:.4f}]")
    print(f"Adj Rand 95% CI: [{percentiles['adj_rand'][1]:.4f}, {percentiles['adj_rand'][7]:.4f}]")


def test_multiple_runs():
    """Test multiple runs functionality."""
    print("\n" + "="*60)
    print("Testing Multiple Runs")
    print("="*60)
    
    # Create a simple switching matrix
    np.random.seed(42)
    n_items = 30
    swm = np.random.rand(n_items, n_items) * 100
    swm = (swm + swm.T) / 2  # Make symmetric
    np.fill_diagonal(swm, 0)
    
    print(f"Testing with {n_items}x{n_items} switching matrix")
    
    # Run clustering multiple times
    print("\nRunning clustering 10 times...")
    result = run_clusters(
        swm,
        n_clusters=3,
        min_items=2,
        n_runs=10,
        random_state=42,
        algorithm='v1'
    )
    
    print(f"Best result:")
    print(f"  Objective (Diff): {result.Diff:.6f}")
    print(f"  Log-likelihood: {result.LogLH:.6f}")
    print(f"  Z-value: {result.ZValue:.4f}")
    print(f"  Iterations: {result.Iter}")
    
    # Test stability
    print("\nEvaluating clustering stability...")
    stability = evaluate_clustering_stability(
        swm,
        n_clusters=3,
        min_items=2,
        n_runs=50,
        random_state=42,
        algorithm='v1'
    )
    
    print(f"Stability analysis results:")
    print(f"  Mean objective: {stability['objective_mean']:.6f}")
    print(f"  Std objective: {stability['objective_std']:.6f}")
    print(f"  Overall stability: {stability['overall_stability']:.4f}")
    
    # Show most stable items
    most_stable = np.argsort(stability['item_stability'])[:5]
    print(f"  Most stable items: {most_stable}")


def test_empirical_distributions():
    """Test empirical distribution generation."""
    print("\n" + "="*60)
    print("Testing Empirical Distributions")
    print("="*60)
    
    # Create switching matrix
    np.random.seed(42)
    n_items = 25
    swm = np.random.rand(n_items, n_items) * 100
    swm = (swm + swm.T) / 2
    np.fill_diagonal(swm, 0)
    
    print(f"Creating empirical distribution with {n_items} items...")
    
    # Generate empirical distribution
    emp_dist = create_switching_matrix_distribution(
        swm,
        n_clusters=3,
        n_points=500,
        min_items=2,
        random_state=42,
        n_jobs=1
    )
    
    print(f"Generated {emp_dist.n_points} random clusterings")
    
    # Get percentiles
    percentiles = emp_dist.get_percentiles()
    print(f"\nEmpirical distribution percentiles:")
    print(f"  Z-value 95% CI: [{percentiles['z'][1]:.4f}, {percentiles['z'][7]:.4f}]")
    print(f"  Log-LH 95% CI: [{percentiles['ll'][7]:.4f}, {percentiles['ll'][1]:.4f}]")
    print(f"  Diff 95% CI: [{percentiles['diff'][1]:.6f}, {percentiles['diff'][7]:.6f}]")
    
    # Test with actual clustering
    print("\nRunning actual clustering for comparison...")
    cluster_result = run_clusters(
        swm, n_clusters=3, min_items=2, n_runs=5, random_state=42
    )
    
    # Calculate p-values
    p_values = emp_dist.calculate_p_values(
        cluster_result.ZValue,
        cluster_result.LogLH,
        cluster_result.Diff
    )
    
    print(f"\nP-values for clustering result:")
    print(f"  Z-value: {cluster_result.ZValue:.4f}, p={p_values['z'][0]:.4f}")
    print(f"  Log-LH: {cluster_result.LogLH:.4f}, p={p_values['ll'][0]:.4f}")
    print(f"  Diff: {cluster_result.Diff:.6f}, p={p_values['diff'][0]:.4f}")


def test_kfold_validation():
    """Test k-fold cross-validation."""
    print("\n" + "="*60)
    print("Testing K-Fold Cross-Validation")
    print("="*60)
    
    # Create switching matrix with clear cluster structure
    np.random.seed(42)
    n_items = 40
    n_clusters = 4
    
    # Create block-diagonal structure
    swm = np.zeros((n_items, n_items))
    items_per_cluster = n_items // n_clusters
    
    for i in range(n_clusters):
        start = i * items_per_cluster
        end = (i + 1) * items_per_cluster
        block = np.random.rand(items_per_cluster, items_per_cluster) * 100
        block = (block + block.T) / 2
        np.fill_diagonal(block, 0)
        swm[start:end, start:end] = block
        
    # Add some noise between clusters
    noise = np.random.rand(n_items, n_items) * 10
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0)
    swm += noise
    
    print(f"Testing with {n_items}x{n_items} switching matrix")
    print(f"True number of clusters: {n_clusters}")
    
    # Run k-fold validation
    print("\nRunning 5-fold cross-validation...")
    validator = KFoldValidator(
        n_folds=5,
        n_random=100,  # Small number for testing
        random_state=42,
        algorithm='v1'
    )
    
    kfold_result = validator.validate(
        swm,
        n_clusters=n_clusters,
        min_items=2,
        n_runs=5
    )
    
    print(kfold_result.summary())
    
    # Test significance
    if kfold_result.av_rand_dist is not None:
        p_value = np.mean(kfold_result.av_rand_dist >= kfold_result.av_rand)
        print(f"\nStatistical significance:")
        print(f"  P-value for average Rand index: {p_value:.4f}")


def test_p_values():
    """Test p-value calculation utilities."""
    print("\n" + "="*60)
    print("Testing P-Value Calculations")
    print("="*60)
    
    # Create example null distribution
    np.random.seed(42)
    null_dist = np.random.normal(0, 1, 10000)
    null_dist.sort()
    
    # Test different observed values
    test_values = [-2, -1, 0, 1, 2, 3]
    
    print("Testing empirical p-value calculations:")
    print(f"Null distribution: N(0,1), n={len(null_dist)}")
    print("\nObserved | Direction | P-value | Percentile")
    print("-" * 50)
    
    for obs in test_values:
        result = calculate_empirical_p_value(
            obs, null_dist, direction='greater'
        )
        print(f"{obs:8.1f} | greater   | {result.p_value:.4f} | {result.percentile:6.1f}%")
        
    # Test multiple testing correction
    print("\n\nTesting multiple testing correction:")
    p_values = [0.001, 0.01, 0.03, 0.05, 0.10, 0.20]
    
    corrections = ['bonferroni', 'holm', 'fdr_bh']
    
    print("\nOriginal p-values:", p_values)
    print("\nMethod      | Corrected p-values")
    print("-" * 60)
    
    for method in corrections:
        corrected = multiple_testing_correction(p_values, method=method)
        print(f"{method:11} | {corrected['corrected']}")
        
    # Test p-value combination
    print("\n\nTesting p-value combination:")
    p_vals = [0.01, 0.05, 0.10]
    
    methods = ['fisher', 'stouffer', 'min']
    print(f"\nCombining p-values: {p_vals}")
    print("\nMethod   | Combined p-value")
    print("-" * 30)
    
    for method in methods:
        combined = combine_p_values(p_vals, method=method)
        print(f"{method:8} | {combined['p_value']:.4f}")


def test_constrained_clustering():
    """Test constrained clustering functionality."""
    print("\n" + "="*60)
    print("Testing Constrained Clustering")
    print("="*60)
    
    # Create switching matrix
    np.random.seed(42)
    n_items = 30
    swm = np.random.rand(n_items, n_items) * 100
    swm = (swm + swm.T) / 2
    np.fill_diagonal(swm, 0)
    
    # Define constraints
    n_fixed = 10
    fixed_indexes = np.arange(1, n_fixed + 1)  # 1-based
    fixed_assign = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    free_indexes = np.arange(n_fixed + 1, n_items + 1)  # 1-based
    
    print(f"Testing with {n_items} items:")
    print(f"  Fixed items: {n_fixed}")
    print(f"  Free items: {len(free_indexes)}")
    print(f"  Fixed assignments: {fixed_assign}")
    
    # Run constrained clustering
    result = run_clusters_constrained(
        swm,
        n_clusters=3,
        min_items=2,
        fixed_assign=fixed_assign,
        assign_indexes=fixed_indexes,
        free_indexes=free_indexes,
        n_runs=5,
        random_state=42,
        algorithm='v1'
    )
    
    print(f"\nConstrained clustering result:")
    print(f"  Objective (Diff): {result.Diff:.6f}")
    print(f"  Log-likelihood: {result.LogLH:.6f}")
    print(f"  Iterations: {result.Iter}")
    
    # Verify constraints were maintained
    final_assign = result.Assign
    constraints_maintained = np.all(
        final_assign[fixed_indexes - 1] == fixed_assign
    )
    print(f"  Constraints maintained: {constraints_maintained}")


def main():
    """Run all validation tests."""
    print("SUBMARIT Validation Module Tests")
    print("================================")
    
    # Run all tests
    test_rand_index()
    test_multiple_runs()
    test_empirical_distributions()
    test_kfold_validation()
    test_p_values()
    test_constrained_clustering()
    
    print("\n" + "="*60)
    print("All validation tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()