"""Regression tests with reference outputs for SUBMARIT."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from submarit.algorithms.local_search import (
    KSMLocalSearch,
    KSMLocalSearch2,
    KSMLocalSearchConstrained,
    KSMLocalSearchConstrained2,
)
from submarit.core.create_substitution_matrix import create_substitution_matrix
from submarit.evaluation.cluster_evaluator import ClusterEvaluator
from submarit.evaluation.gap_statistic import GAPStatistic
from submarit.utils.matlab_compat import MatlabRandom
from submarit.validation.empirical_distributions import EmpiricalDistribution
from submarit.validation.rand_index import rand_index


class TestRegressionWithReferenceOutputs:
    """Regression tests comparing against reference outputs."""
    
    @pytest.fixture
    def reference_data_dir(self, test_data_dir):
        """Get reference data directory."""
        ref_dir = test_data_dir / "reference"
        ref_dir.mkdir(exist_ok=True)
        return ref_dir
    
    @pytest.fixture
    def synthetic_reference_data(self):
        """Generate synthetic reference data with known properties."""
        # Create a simple block-diagonal matrix with known clustering
        n_items = 12
        matrix = np.zeros((n_items, n_items))
        
        # Block 1: items 0-3
        matrix[0:4, 0:4] = 0.8
        # Block 2: items 4-7
        matrix[4:8, 4:8] = 0.8
        # Block 3: items 8-11
        matrix[8:12, 8:12] = 0.8
        
        # Add small between-block connections
        matrix[0:4, 4:8] = matrix[4:8, 0:4] = 0.1
        matrix[0:4, 8:12] = matrix[8:12, 0:4] = 0.1
        matrix[4:8, 8:12] = matrix[8:12, 4:8] = 0.1
        
        # Zero diagonal and normalize
        np.fill_diagonal(matrix, 0)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        # Known optimal clustering
        true_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        return {
            "matrix": matrix,
            "true_labels": true_labels,
            "n_clusters": 3,
            "expected_diff_range": (0.4, 0.6),  # Expected range for Diff metric
            "expected_ri": 1.0,  # Perfect clustering should achieve RI=1
        }
    
    def test_local_search_deterministic_output(self, synthetic_reference_data):
        """Test that LocalSearch produces deterministic output with fixed seed."""
        matrix = synthetic_reference_data["matrix"]
        n_clusters = synthetic_reference_data["n_clusters"]
        
        # Run algorithm multiple times with same seed
        results = []
        for _ in range(5):
            clusterer = KSMLocalSearch(n_clusters=n_clusters, random_state=42)
            clusterer.fit(matrix)
            results.append(clusterer.get_result())
        
        # All results should be identical
        reference = results[0]
        for result in results[1:]:
            assert np.array_equal(result.Assign, reference.Assign)
            assert np.isclose(result.Diff, reference.Diff)
            assert np.isclose(result.LogLH, reference.LogLH)
            assert result.Iter == reference.Iter
    
    def test_local_search_expected_output_quality(self, synthetic_reference_data):
        """Test that LocalSearch achieves expected quality on synthetic data."""
        matrix = synthetic_reference_data["matrix"]
        true_labels = synthetic_reference_data["true_labels"]
        n_clusters = synthetic_reference_data["n_clusters"]
        expected_diff_range = synthetic_reference_data["expected_diff_range"]
        
        # Run clustering
        clusterer = KSMLocalSearch(n_clusters=n_clusters, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Check objective value is in expected range
        assert expected_diff_range[0] <= result.Diff <= expected_diff_range[1], \
            f"Diff {result.Diff} outside expected range {expected_diff_range}"
        
        # Check clustering quality
        predicted_labels = result.Assign - 1  # Convert to 0-based
        ri = rand_index(true_labels, predicted_labels)
        
        # Should achieve near-perfect clustering on this simple data
        assert ri > 0.9, f"Poor clustering quality: RI={ri}"
    
    def test_local_search2_convergence_behavior(self, synthetic_reference_data):
        """Test LocalSearch2 convergence behavior."""
        matrix = synthetic_reference_data["matrix"]
        n_clusters = synthetic_reference_data["n_clusters"]
        
        # Run with different max_iter settings
        iter_settings = [5, 10, 20, 50]
        log_likelihoods = []
        
        for max_iter in iter_settings:
            clusterer = KSMLocalSearch2(
                n_clusters=n_clusters,
                max_iter=max_iter,
                random_state=42
            )
            clusterer.fit(matrix)
            result = clusterer.get_result()
            log_likelihoods.append(result.LogLH)
        
        # Log-likelihood should improve (become less negative) with more iterations
        for i in range(len(log_likelihoods) - 1):
            assert log_likelihoods[i + 1] >= log_likelihoods[i] - 1e-6, \
                "Log-likelihood decreased with more iterations"
    
    def test_constrained_clustering_reference_behavior(self, synthetic_reference_data):
        """Test constrained clustering with known constraints."""
        matrix = synthetic_reference_data["matrix"]
        n_clusters = synthetic_reference_data["n_clusters"]
        
        # Fix some items to correct clusters
        fixed_indices = np.array([1, 5, 9])  # One from each true cluster
        fixed_assignments = np.array([1, 2, 3])
        free_indices = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12])  # 1-based
        
        # Run constrained clustering
        clusterer = KSMLocalSearchConstrained(
            n_clusters=n_clusters,
            random_state=42
        )
        clusterer.fit_constrained(
            matrix,
            fixed_assignments,
            fixed_indices,
            free_indices
        )
        result = clusterer.get_result()
        
        # Check constraints are satisfied
        assert result.Assign[0] == 1  # Item 1 -> cluster 1
        assert result.Assign[4] == 2  # Item 5 -> cluster 2
        assert result.Assign[8] == 3  # Item 9 -> cluster 3
        
        # Should still achieve good clustering
        true_labels = synthetic_reference_data["true_labels"]
        predicted_labels = result.Assign - 1
        ri = rand_index(true_labels, predicted_labels)
        assert ri > 0.8, f"Poor constrained clustering: RI={ri}"
    
    def test_substitution_matrix_creation_regression(self):
        """Test substitution matrix creation against reference behavior."""
        # Create simple consumer-product data
        np.random.seed(42)
        n_consumers = 100
        n_products = 5
        
        # Each consumer buys 2 products
        consumer_data = np.zeros((n_consumers, n_products), dtype=np.int64)
        for i in range(n_consumers):
            products = np.random.choice(n_products, 2, replace=False)
            consumer_data[i, products] = 1
        
        # Create substitution matrix
        matrix, indexes, count = create_substitution_matrix(
            consumer_data,
            normalize=True,
            weight=0,
            diag=False
        )
        
        # Check properties
        assert matrix.shape == (n_products, n_products)
        assert np.allclose(np.diag(matrix), 0)
        assert count == n_products
        
        # Check that products bought together have higher substitution
        # (This is a weak test - mainly checking it runs)
        assert matrix.max() > matrix.mean()
    
    def test_gap_statistic_reference_behavior(self, synthetic_reference_data):
        """Test GAP statistic identifies correct number of clusters."""
        matrix = synthetic_reference_data["matrix"]
        true_k = synthetic_reference_data["n_clusters"]
        
        # Run GAP statistic
        gap = GAPStatistic(n_refs=20, random_state=42)
        results = gap.compute(
            matrix,
            max_clusters=5,
            clusterer_class=KSMLocalSearch
        )
        
        # Should identify correct number of clusters
        optimal_k = results["optimal_k"]
        assert optimal_k == true_k, f"GAP statistic chose k={optimal_k}, expected k={true_k}"
        
        # GAP values should peak at true k
        gap_values = results["gap_values"]
        assert np.argmax(gap_values) + 2 == true_k  # +2 because we start from k=2
    
    def test_empirical_distribution_regression(self, synthetic_reference_data):
        """Test empirical distribution generation consistency."""
        matrix = synthetic_reference_data["matrix"]
        n_clusters = synthetic_reference_data["n_clusters"]
        
        # Create clusterer
        clusterer = KSMLocalSearch(n_clusters=n_clusters, random_state=42)
        clusterer.fit(matrix)
        observed_diff = clusterer.get_result().Diff
        
        # Generate empirical distribution
        emp_dist = EmpiricalDistribution(n_iterations=100, random_state=42)
        dist_results = emp_dist.generate(
            matrix,
            clusterer,
            statistic="diff"
        )
        
        # Check distribution properties
        distribution = dist_results["distribution"]
        assert len(distribution) == 100
        
        # Observed value should be extreme compared to null distribution
        p_value = emp_dist.compute_p_value(observed_diff, distribution)
        assert p_value < 0.05, "Structured data not detected as significant"
        
        # Distribution should be consistent across runs
        dist_results2 = emp_dist.generate(
            matrix,
            clusterer,
            statistic="diff"
        )
        
        # Means should be very close
        assert np.isclose(
            np.mean(distribution),
            np.mean(dist_results2["distribution"]),
            rtol=0.1
        )
    
    def test_numerical_stability_regression(self):
        """Test numerical stability with edge cases."""
        # Test with very small values
        small_matrix = np.ones((10, 10)) * 1e-10
        np.fill_diagonal(small_matrix, 0)
        row_sums = small_matrix.sum(axis=1)
        small_matrix = small_matrix / row_sums[:, np.newaxis]
        
        clusterer = KSMLocalSearch(n_clusters=2, random_state=42)
        clusterer.fit(small_matrix)
        result = clusterer.get_result()
        
        # Should not produce NaN or Inf
        assert np.isfinite(result.Diff)
        assert np.isfinite(result.LogLH)
        
        # Test with very large values
        large_matrix = np.ones((10, 10)) * 1e6
        np.fill_diagonal(large_matrix, 0)
        row_sums = large_matrix.sum(axis=1)
        large_matrix = large_matrix / row_sums[:, np.newaxis]
        
        clusterer2 = KSMLocalSearch(n_clusters=2, random_state=42)
        clusterer2.fit(large_matrix)
        result2 = clusterer2.get_result()
        
        assert np.isfinite(result2.Diff)
        assert np.isfinite(result2.LogLH)
    
    def test_matlab_random_compatibility(self):
        """Test MATLAB random number generator compatibility."""
        # Test basic operations
        rng1 = MatlabRandom(seed=42)
        rng2 = MatlabRandom(seed=42)
        
        # Should produce same sequence
        for _ in range(10):
            assert rng1.rand() == rng2.rand()
        
        # Test randperm
        perm1 = rng1.randperm(10)
        rng2 = MatlabRandom(seed=42)
        for _ in range(10):
            rng2.rand()  # Advance to same state
        perm2 = rng2.randperm(10)
        assert np.array_equal(perm1, perm2)
    
    @pytest.mark.parametrize("algorithm", ["local_search", "local_search2"])
    def test_algorithm_regression_on_standard_problems(self, algorithm):
        """Test algorithms on standard test problems."""
        # Create standard test problems
        problems = []
        
        # Problem 1: Clear 2-cluster structure
        matrix1 = np.array([
            [0, 0.9, 0.8, 0.1, 0.1],
            [0.9, 0, 0.8, 0.1, 0.1],
            [0.8, 0.8, 0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0, 0.9],
            [0.1, 0.1, 0.1, 0.9, 0]
        ])
        row_sums = matrix1.sum(axis=1)
        matrix1 = matrix1 / row_sums[:, np.newaxis]
        problems.append({
            "matrix": matrix1,
            "expected_clusters": 2,
            "expected_assignment": [1, 1, 1, 2, 2]  # or [2, 2, 2, 1, 1]
        })
        
        # Problem 2: 3-cluster ring structure
        n = 9
        matrix2 = np.zeros((n, n))
        for i in range(n):
            # Strong connections within groups of 3
            group = i // 3
            for j in range(n):
                if j // 3 == group and i != j:
                    matrix2[i, j] = 0.8
                elif abs(j // 3 - group) == 1 or (group == 0 and j // 3 == 2) or (group == 2 and j // 3 == 0):
                    matrix2[i, j] = 0.2
        row_sums = matrix2.sum(axis=1)
        matrix2 = matrix2 / row_sums[:, np.newaxis]
        problems.append({
            "matrix": matrix2,
            "expected_clusters": 3,
            "expected_assignment": [1, 1, 1, 2, 2, 2, 3, 3, 3]
        })
        
        # Test each problem
        for i, problem in enumerate(problems):
            if algorithm == "local_search":
                clusterer = KSMLocalSearch(
                    n_clusters=problem["expected_clusters"],
                    random_state=42
                )
            else:
                clusterer = KSMLocalSearch2(
                    n_clusters=problem["expected_clusters"],
                    random_state=42
                )
            
            clusterer.fit(problem["matrix"])
            result = clusterer.get_result()
            
            # Check that it finds the expected structure
            # (allowing for label permutation)
            expected = np.array(problem["expected_assignment"])
            predicted = result.Assign
            
            # Compute Rand index
            ri = rand_index(expected, predicted)
            assert ri > 0.8, f"Problem {i+1}: Poor clustering with {algorithm}, RI={ri}"