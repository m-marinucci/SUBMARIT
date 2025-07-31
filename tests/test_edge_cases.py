"""Edge case and error condition tests for SUBMARIT."""

import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from submarit.algorithms.local_search import (
    KSMLocalSearch,
    KSMLocalSearch2,
    KSMLocalSearchConstrained,
    KSMLocalSearchConstrained2,
)
from submarit.core.substitution_matrix import SubstitutionMatrix
from submarit.evaluation.cluster_evaluator import ClusterEvaluator
from submarit.evaluation.entropy_evaluator import EntropyEvaluator
from submarit.evaluation.gap_statistic import GAPStatistic
from submarit.validation.empirical_distributions import EmpiricalDistribution
from submarit.validation.kfold import KFoldValidator
from submarit.validation.rand_index import adjusted_rand_score, rand_index


class TestEdgeCases:
    """Test edge cases for robustness."""
    
    def test_minimum_size_matrix(self):
        """Test with minimum viable matrix size."""
        # 2x2 matrix - smallest possible
        matrix = np.array([[0, 1], [1, 0]], dtype=np.float64)
        
        # Should work with 2 clusters
        clusterer = KSMLocalSearch(n_clusters=2, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        assert result.NoItems == 2
        assert len(np.unique(result.Assign)) == 2
    
    def test_single_cluster_edge_case(self):
        """Test clustering with n_clusters=1."""
        matrix = np.random.rand(10, 10)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        
        # Single cluster - all items should be in one cluster
        clusterer = KSMLocalSearch(n_clusters=1, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        assert len(np.unique(result.Assign)) == 1
        assert np.all(result.Assign == 1)
    
    def test_all_zeros_matrix(self):
        """Test with matrix of all zeros."""
        matrix = np.zeros((10, 10))
        
        clusterer = KSMLocalSearch(n_clusters=3, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Should still produce valid clustering
        assert result.NoItems == 10
        assert len(np.unique(result.Assign)) == 3
        
        # Objective values might be zero or NaN
        assert np.isfinite(result.Diff) or result.Diff == 0
    
    def test_identity_matrix(self):
        """Test with identity matrix (no connections)."""
        matrix = np.eye(10)
        np.fill_diagonal(matrix, 0)  # Zero diagonal
        
        clusterer = KSMLocalSearch(n_clusters=3, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Should produce some clustering even with no connections
        assert len(np.unique(result.Assign)) == 3
    
    def test_single_connected_component(self):
        """Test matrix with disconnected components."""
        # Create block diagonal matrix with disconnected parts
        matrix = np.zeros((12, 12))
        # Component 1
        matrix[0:4, 0:4] = 1
        # Component 2
        matrix[4:8, 4:8] = 1
        # Component 3
        matrix[8:12, 8:12] = 1
        
        np.fill_diagonal(matrix, 0)
        row_sums = matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        matrix = matrix / row_sums[:, np.newaxis]
        
        clusterer = KSMLocalSearch(n_clusters=3, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Should find the natural components
        assert len(np.unique(result.Assign)) == 3
    
    def test_highly_imbalanced_clusters(self):
        """Test when natural clusters are highly imbalanced."""
        # Create matrix with one large and one small cluster
        matrix = np.zeros((20, 20))
        # Large cluster (18 items)
        matrix[0:18, 0:18] = 0.8
        # Small cluster (2 items)
        matrix[18:20, 18:20] = 0.9
        # Weak connections between
        matrix[0:18, 18:20] = matrix[18:20, 0:18] = 0.1
        
        np.fill_diagonal(matrix, 0)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        clusterer = KSMLocalSearch(n_clusters=2, min_items=1, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Should handle imbalanced clusters
        assert len(np.unique(result.Assign)) == 2
        assert min(result.Count.values()) >= 1
    
    def test_near_singular_matrix(self):
        """Test with near-singular substitution matrix."""
        # Create matrix with very small eigenvalues
        n = 10
        matrix = np.ones((n, n)) * 1e-10
        # Add small random noise
        matrix += np.random.rand(n, n) * 1e-12
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        
        # Normalize
        row_sums = matrix.sum(axis=1)
        row_sums[row_sums < 1e-15] = 1
        matrix = matrix / row_sums[:, np.newaxis]
        
        clusterer = KSMLocalSearch(n_clusters=2, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Should not crash and produce valid result
        assert np.isfinite(result.LogLH) or np.isnan(result.LogLH)
        assert len(result.Assign) == n
    
    def test_extreme_values(self):
        """Test with extreme values in matrix."""
        # Mix of very large and very small values
        matrix = np.random.rand(10, 10)
        matrix[0:5, 0:5] *= 1e6  # Very large
        matrix[5:10, 5:10] *= 1e-6  # Very small
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        
        # Normalize
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        clusterer = KSMLocalSearch(n_clusters=2, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Should handle extreme values gracefully
        assert result.NoItems == 10
        assert not np.any(np.isinf(result.Assign))
    
    def test_constrained_with_all_fixed(self):
        """Test constrained clustering with all items fixed."""
        matrix = np.random.rand(10, 10)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        
        # Fix all items
        fixed_assign = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
        assign_indexes = np.arange(1, 11)  # All items
        free_indexes = np.array([], dtype=np.int64)  # No free items
        
        clusterer = KSMLocalSearchConstrained(n_clusters=3, random_state=42)
        
        # Should handle edge case of no free items
        with pytest.raises(ValueError):
            clusterer.fit_constrained(
                matrix,
                fixed_assign,
                assign_indexes,
                free_indexes
            )
    
    def test_constrained_with_conflicting_requirements(self):
        """Test constrained clustering with conflicting min_items requirement."""
        matrix = np.random.rand(10, 10)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        
        # Try to force a configuration that violates min_items
        fixed_assign = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # 8 items in cluster 1
        assign_indexes = np.arange(1, 9)
        free_indexes = np.array([9, 10])  # Only 2 free items for 2 other clusters
        
        clusterer = KSMLocalSearchConstrained(
            n_clusters=3,
            min_items=2,  # Requires 2 items per cluster
            random_state=42
        )
        
        # Cannot satisfy min_items constraint
        with pytest.raises(ValueError):
            clusterer.fit_constrained(
                matrix,
                fixed_assign,
                assign_indexes,
                free_indexes
            )


class TestErrorConditions:
    """Test error handling and invalid inputs."""
    
    def test_invalid_matrix_shape(self):
        """Test with non-square matrix."""
        matrix = np.random.rand(10, 5)
        
        clusterer = KSMLocalSearch(n_clusters=2)
        with pytest.raises(ValueError, match="square"):
            clusterer.fit(matrix)
    
    def test_invalid_n_clusters(self):
        """Test with invalid number of clusters."""
        matrix = np.random.rand(5, 5)
        
        # More clusters than items
        clusterer = KSMLocalSearch(n_clusters=10)
        with pytest.raises(ValueError):
            clusterer.fit(matrix)
        
        # Zero clusters
        with pytest.raises(ValueError):
            KSMLocalSearch(n_clusters=0)
        
        # Negative clusters
        with pytest.raises(ValueError):
            KSMLocalSearch(n_clusters=-1)
    
    def test_invalid_min_items(self):
        """Test with invalid min_items constraint."""
        matrix = np.random.rand(10, 10)
        
        # min_items too large
        clusterer = KSMLocalSearch(n_clusters=3, min_items=5)
        with pytest.raises(ValueError):
            clusterer.fit(matrix)
    
    def test_substitution_matrix_invalid_data(self):
        """Test SubstitutionMatrix with invalid inputs."""
        sub_matrix = SubstitutionMatrix()
        
        # Non-2D data
        with pytest.raises(ValueError):
            sub_matrix.set_data(np.array([1, 2, 3]))
        
        # Non-square data
        with pytest.raises(ValueError):
            sub_matrix.set_data(np.random.rand(5, 3))
        
        # Get matrix before setting data
        with pytest.raises(ValueError):
            sub_matrix.get_matrix()
    
    def test_invalid_consumer_data(self):
        """Test invalid consumer-product data."""
        sub_matrix = SubstitutionMatrix()
        
        # Empty data
        empty_data = np.array([], dtype=np.int64).reshape(0, 0)
        with pytest.raises(Exception):  # Might raise various exceptions
            sub_matrix.create_from_consumer_product_data(empty_data)
        
        # All zeros (no purchases)
        zero_data = np.zeros((100, 10), dtype=np.int64)
        indexes, count = sub_matrix.create_from_consumer_product_data(zero_data)
        # Should handle gracefully, possibly returning empty or minimal result
        assert count >= 0
    
    def test_evaluation_with_mismatched_dimensions(self):
        """Test evaluation with mismatched matrix and label dimensions."""
        matrix = np.random.rand(10, 10)
        labels = np.array([0, 1, 2])  # Wrong size
        
        evaluator = ClusterEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate(matrix, labels)
    
    def test_rand_index_with_different_lengths(self):
        """Test Rand index with different length inputs."""
        labels1 = np.array([0, 0, 1, 1])
        labels2 = np.array([0, 1, 0])
        
        with pytest.raises(ValueError):
            rand_index(labels1, labels2)
    
    def test_gap_statistic_with_invalid_parameters(self):
        """Test GAP statistic with invalid parameters."""
        matrix = np.random.rand(10, 10)
        
        # max_clusters larger than n_items
        gap = GAPStatistic()
        with pytest.raises(ValueError):
            gap.compute(matrix, max_clusters=20)
        
        # max_clusters < 2
        with pytest.raises(ValueError):
            gap.compute(matrix, max_clusters=1)
    
    def test_kfold_with_too_many_folds(self):
        """Test k-fold validation with more folds than items."""
        matrix = np.random.rand(5, 5)
        
        validator = KFoldValidator(n_folds=10)  # More folds than items
        clusterer = KSMLocalSearch(n_clusters=2)
        
        with pytest.raises(ValueError):
            validator.validate(matrix, clusterer)
    
    def test_empirical_distribution_with_invalid_statistic(self):
        """Test empirical distribution with invalid statistic."""
        matrix = np.random.rand(10, 10)
        clusterer = KSMLocalSearch(n_clusters=2)
        
        emp_dist = EmpiricalDistribution()
        with pytest.raises(ValueError):
            emp_dist.generate(matrix, clusterer, statistic="invalid_stat")


class TestNumericalStability:
    """Test numerical stability in edge cases."""
    
    def test_zero_variance_clusters(self):
        """Test when clusters have zero variance."""
        # Create matrix where all items in a cluster are identical
        matrix = np.zeros((9, 9))
        # Cluster 1: identical connections
        matrix[0:3, 0:3] = 0.5
        matrix[3:6, 3:6] = 0.5
        matrix[6:9, 6:9] = 0.5
        
        np.fill_diagonal(matrix, 0)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        clusterer = KSMLocalSearch(n_clusters=3, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Should handle zero variance without crashing
        assert np.isfinite(result.LogLH) or np.isnan(result.LogLH)
    
    def test_underflow_in_calculations(self):
        """Test handling of numerical underflow."""
        # Create matrix with very small probabilities
        matrix = np.full((10, 10), 1e-300)
        np.fill_diagonal(matrix, 0)
        
        # This might cause underflow in calculations
        clusterer = KSMLocalSearch(n_clusters=2, random_state=42)
        
        # Should either handle gracefully or raise appropriate error
        try:
            clusterer.fit(matrix)
            result = clusterer.get_result()
            # If it succeeds, check for reasonable output
            assert len(result.Assign) == 10
        except (ValueError, RuntimeWarning):
            # Acceptable to fail with very extreme values
            pass
    
    def test_overflow_in_calculations(self):
        """Test handling of numerical overflow."""
        # Create matrix that might cause overflow
        matrix = np.random.rand(10, 10)
        matrix = matrix * 1e200  # Very large values
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        
        # Normalize (this should bring values back to [0,1])
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        clusterer = KSMLocalSearch(n_clusters=2, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Should produce valid clustering
        assert len(np.unique(result.Assign)) == 2
    
    def test_entropy_with_zero_probabilities(self):
        """Test entropy calculation with zero probabilities."""
        matrix = np.zeros((10, 10))
        # Add a few non-zero entries
        matrix[0, 1] = matrix[1, 0] = 1
        matrix[5, 6] = matrix[6, 5] = 1
        
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        evaluator = EntropyEvaluator()
        
        # Should handle zero probabilities in entropy calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore log(0) warnings
            within_entropy = evaluator.within_cluster_entropy(matrix, labels)
            between_entropy = evaluator.between_cluster_entropy(matrix, labels)
        
        assert np.isfinite(within_entropy) or within_entropy == 0
        assert np.isfinite(between_entropy) or between_entropy == 0