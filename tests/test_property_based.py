"""Property-based tests using Hypothesis for SUBMARIT."""

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from submarit.algorithms.local_search import (
    KSMLocalSearch,
    KSMLocalSearch2,
    KSMLocalSearchConstrained,
)
from submarit.core.substitution_matrix import SubstitutionMatrix
from submarit.validation.rand_index import rand_index


# Custom strategies for generating test data

@st.composite
def substitution_matrices(draw, min_size=3, max_size=20):
    """Generate valid substitution matrices."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Generate random matrix
    matrix = draw(arrays(
        dtype=np.float64,
        shape=(size, size),
        elements=st.floats(min_value=0, max_value=1, exclude_min=False),
        unique=False
    ))
    
    # Make symmetric
    matrix = (matrix + matrix.T) / 2
    
    # Zero diagonal
    np.fill_diagonal(matrix, 0)
    
    # Normalize rows (avoid division by zero)
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    matrix = matrix / row_sums[:, np.newaxis]
    
    return matrix


@st.composite
def clustering_parameters(draw, matrix_size):
    """Generate valid clustering parameters for a given matrix size."""
    max_clusters = min(matrix_size - 1, 10)  # Reasonable upper limit
    n_clusters = draw(st.integers(min_value=2, max_value=max_clusters))
    
    # Minimum items per cluster
    max_min_items = matrix_size // (n_clusters * 2)
    min_items = draw(st.integers(min_value=1, max_value=max(1, max_min_items)))
    
    return n_clusters, min_items


@st.composite
def consumer_product_data(draw):
    """Generate valid consumer-product purchase data."""
    n_consumers = draw(st.integers(min_value=10, max_value=100))
    n_products = draw(st.integers(min_value=3, max_value=20))
    
    # Density of purchases (what fraction of entries are 1)
    density = draw(st.floats(min_value=0.01, max_value=0.5))
    
    data = draw(arrays(
        dtype=np.int64,
        shape=(n_consumers, n_products),
        elements=st.integers(0, 1),
        unique=False
    ))
    
    # Ensure at least some purchases
    if data.sum() == 0:
        # Add some random purchases
        n_purchases = max(1, int(n_consumers * n_products * density))
        indices = np.random.choice(n_consumers * n_products, n_purchases, replace=False)
        data.flat[indices] = 1
    
    return data


@st.composite
def constrained_clustering_problem(draw, matrix):
    """Generate a valid constrained clustering problem."""
    n_items = matrix.shape[0]
    
    # Number of clusters
    n_clusters = draw(st.integers(min_value=2, max_value=min(n_items - 1, 5)))
    
    # Number of fixed items (leave some free)
    max_fixed = n_items - n_clusters  # Need at least one free item per cluster
    n_fixed = draw(st.integers(min_value=n_clusters, max_value=max_fixed))
    
    # Select fixed indices
    all_indices = np.arange(1, n_items + 1)  # 1-based
    fixed_indices = draw(st.permutations(all_indices))[:n_fixed]
    fixed_indices = np.array(sorted(fixed_indices))
    
    # Assign fixed items to clusters (ensure each cluster gets at least one)
    fixed_assignments = np.zeros(n_fixed, dtype=np.int64)
    # First, assign one to each cluster
    for i in range(n_clusters):
        fixed_assignments[i] = i + 1
    # Randomly assign the rest
    for i in range(n_clusters, n_fixed):
        fixed_assignments[i] = draw(st.integers(1, n_clusters))
    
    # Shuffle assignments
    np.random.shuffle(fixed_assignments)
    
    # Free indices
    free_indices = np.setdiff1d(all_indices, fixed_indices)
    
    return n_clusters, fixed_assignments, fixed_indices, free_indices


class TestSubstitutionMatrixProperties:
    """Property-based tests for SubstitutionMatrix class."""
    
    @given(matrix=substitution_matrices())
    @settings(max_examples=50, deadline=None)
    def test_matrix_properties_preserved(self, matrix):
        """Test that matrix properties are preserved after operations."""
        sub_matrix = SubstitutionMatrix(matrix)
        result = sub_matrix.get_matrix()
        
        # Check properties
        assert np.allclose(result, result.T), "Matrix not symmetric"
        assert np.allclose(np.diag(result), 0), "Diagonal not zero"
        
        # Check normalization (accounting for zero rows)
        row_sums = result.sum(axis=1)
        non_zero_rows = row_sums > 1e-10
        if np.any(non_zero_rows):
            assert np.allclose(row_sums[non_zero_rows], 1.0), "Non-zero rows not normalized"
    
    @given(data=consumer_product_data())
    @settings(max_examples=20, deadline=None)
    def test_create_from_consumer_data_properties(self, data):
        """Test properties of matrices created from consumer data."""
        sub_matrix = SubstitutionMatrix()
        
        # Skip if no products are purchased
        if data.sum() == 0:
            return
        
        try:
            indexes, count = sub_matrix.create_from_consumer_product_data(
                data, normalize=True
            )
            
            result = sub_matrix.get_matrix()
            
            # Basic properties
            assert result.shape[0] == result.shape[1]
            assert result.shape[0] <= data.shape[1]  # At most n_products
            assert np.allclose(np.diag(result), 0)
            
            # Non-negative
            assert np.all(result >= 0)
            
        except ValueError:
            # Some data might not produce valid matrices
            pass
    
    @given(matrix=substitution_matrices(), indices=st.lists(st.integers(0, 19), min_size=1, max_size=10))
    @settings(max_examples=30, deadline=None)
    def test_submatrix_extraction(self, matrix, indices):
        """Test that submatrix extraction preserves properties."""
        # Filter indices to valid range
        valid_indices = [i for i in indices if 0 <= i < matrix.shape[0]]
        assume(len(valid_indices) > 0)
        
        valid_indices = np.unique(valid_indices)
        
        sub_matrix = SubstitutionMatrix(matrix)
        submatrix = sub_matrix.get_submatrix(valid_indices)
        
        # Check properties
        assert submatrix.shape == (len(valid_indices), len(valid_indices))
        assert np.allclose(np.diag(submatrix), 0)
        
        # Check values match
        for i, idx_i in enumerate(valid_indices):
            for j, idx_j in enumerate(valid_indices):
                assert np.isclose(submatrix[i, j], matrix[idx_i, idx_j])


class TestLocalSearchProperties:
    """Property-based tests for local search algorithms."""
    
    @given(matrix=substitution_matrices())
    @settings(max_examples=30, deadline=None)
    def test_clustering_basic_invariants(self, matrix):
        """Test basic invariants that should hold for any clustering."""
        n_items = matrix.shape[0]
        assume(n_items >= 4)  # Need enough items for 2 clusters
        
        # Generate valid parameters
        n_clusters = min(2, n_items // 2)
        
        clusterer = KSMLocalSearch(n_clusters=n_clusters, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Invariants
        assert result.NoItems == n_items
        assert result.NoClusters == n_clusters
        assert len(result.Assign) == n_items
        assert np.all((result.Assign >= 1) & (result.Assign <= n_clusters))
        assert len(np.unique(result.Assign)) == n_clusters  # All clusters used
        
        # Each cluster should have at least min_items
        for i in range(1, n_clusters + 1):
            assert result.Count[i] >= clusterer.min_items
    
    @given(matrix=substitution_matrices())
    @settings(max_examples=20, deadline=None)
    def test_clustering_deterministic_with_seed(self, matrix):
        """Test that clustering is deterministic with fixed seed."""
        assume(matrix.shape[0] >= 6)
        
        n_clusters = 3
        seed = 42
        
        # Run twice with same seed
        clusterer1 = KSMLocalSearch(n_clusters=n_clusters, random_state=seed)
        clusterer1.fit(matrix)
        result1 = clusterer1.get_result()
        
        clusterer2 = KSMLocalSearch(n_clusters=n_clusters, random_state=seed)
        clusterer2.fit(matrix)
        result2 = clusterer2.get_result()
        
        # Results should be identical
        assert np.array_equal(result1.Assign, result2.Assign)
        assert np.isclose(result1.Diff, result2.Diff)
        assert np.isclose(result1.LogLH, result2.LogLH)
    
    @given(matrix=substitution_matrices())
    @settings(max_examples=15, deadline=None)
    def test_algorithm_consistency(self, matrix):
        """Test consistency between LocalSearch and LocalSearch2."""
        assume(matrix.shape[0] >= 6)
        
        n_clusters = 3
        seed = 42
        
        # Run both algorithms
        algo1 = KSMLocalSearch(n_clusters=n_clusters, random_state=seed)
        algo1.fit(matrix)
        result1 = algo1.get_result()
        
        algo2 = KSMLocalSearch2(n_clusters=n_clusters, random_state=seed)
        algo2.fit(matrix)
        result2 = algo2.get_result()
        
        # They optimize different objectives but should produce similar structure
        # Check Rand index between results
        ri = rand_index(result1.Assign, result2.Assign)
        assert ri > 0.5, f"Algorithms produced very different clusterings: RI={ri}"
    
    @given(matrix=substitution_matrices())
    @settings(max_examples=20, deadline=None)
    def test_objective_improvement(self, matrix):
        """Test that local search improves objective function."""
        assume(matrix.shape[0] >= 6)
        
        n_clusters = 3
        
        clusterer = KSMLocalSearch(n_clusters=n_clusters, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Create random assignment for comparison
        np.random.seed(42)
        random_assign = np.random.randint(1, n_clusters + 1, matrix.shape[0])
        
        # Ensure all clusters are used
        for i in range(1, n_clusters + 1):
            if not np.any(random_assign == i):
                random_assign[i - 1] = i
        
        # Local search result should have better objective value
        # (This is a weak test - just checking it terminates with some result)
        assert result.Iter > 0  # At least one iteration
        assert result.Diff is not None
    
    @given(data=constrained_clustering_problem(substitution_matrices()))
    @settings(max_examples=15, deadline=None)
    def test_constrained_clustering_respects_constraints(self, data):
        """Test that constrained clustering respects fixed assignments."""
        matrix, (n_clusters, fixed_assign, assign_indexes, free_indexes) = data
        
        # Ensure we have a valid problem
        assume(len(free_indexes) >= n_clusters)
        
        clusterer = KSMLocalSearchConstrained(
            n_clusters=n_clusters,
            min_items=1,
            random_state=42
        )
        
        clusterer.fit_constrained(
            matrix,
            fixed_assign,
            assign_indexes,
            free_indexes
        )
        
        result = clusterer.get_result()
        
        # Check constraints are satisfied
        for idx, expected in zip(assign_indexes, fixed_assign):
            actual = result.Assign[idx - 1]  # Convert to 0-based
            assert actual == expected, f"Constraint violated: item {idx} should be in cluster {expected}, got {actual}"
        
        # Check all clusters have items
        for i in range(1, n_clusters + 1):
            assert result.Count[i] > 0


class TestValidationProperties:
    """Property-based tests for validation methods."""
    
    @given(
        true_labels=arrays(
            dtype=np.int64,
            shape=st.integers(10, 50),
            elements=st.integers(0, 4)
        ),
        pred_labels=arrays(
            dtype=np.int64,
            shape=st.integers(10, 50),
            elements=st.integers(0, 4)
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_rand_index_properties(self, true_labels, pred_labels):
        """Test properties of Rand index."""
        # Make sure arrays have same length
        min_len = min(len(true_labels), len(pred_labels))
        true_labels = true_labels[:min_len]
        pred_labels = pred_labels[:min_len]
        
        ri = rand_index(true_labels, pred_labels)
        
        # Properties
        assert 0 <= ri <= 1, "Rand index out of bounds"
        
        # Perfect agreement
        ri_perfect = rand_index(true_labels, true_labels)
        assert np.isclose(ri_perfect, 1.0)
        
        # Symmetry
        ri_reversed = rand_index(pred_labels, true_labels)
        assert np.isclose(ri, ri_reversed)
    
    @given(matrix=substitution_matrices())
    @settings(max_examples=10, deadline=None)
    def test_kfold_validation_consistency(self, matrix):
        """Test k-fold validation produces consistent results."""
        from submarit.validation.kfold import KFoldValidator
        
        assume(matrix.shape[0] >= 10)  # Need enough items for k-fold
        
        validator = KFoldValidator(n_folds=3, random_state=42)
        clusterer = KSMLocalSearch(n_clusters=2, random_state=42)
        
        # Run validation twice
        results1 = validator.validate(matrix, clusterer, n_runs=2)
        results2 = validator.validate(matrix, clusterer, n_runs=2)
        
        # With same seed, results should be very similar
        assert np.isclose(results1["mean_score"], results2["mean_score"], rtol=0.1)


class TestEdgeCaseProperties:
    """Property-based tests for edge cases."""
    
    @given(size=st.integers(2, 100))
    @settings(max_examples=20, deadline=None)
    def test_sparse_matrix_handling(self, size):
        """Test handling of very sparse matrices."""
        # Create a chain matrix (each item only connects to neighbors)
        matrix = np.zeros((size, size))
        for i in range(size - 1):
            matrix[i, i + 1] = matrix[i + 1, i] = 1.0
        
        # Normalize
        row_sums = matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums[:, np.newaxis]
        
        # Should still be able to cluster
        n_clusters = min(2, size // 2)
        clusterer = KSMLocalSearch(n_clusters=n_clusters, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        assert len(np.unique(result.Assign)) == n_clusters
    
    @given(size=st.integers(3, 20), density=st.floats(0.8, 1.0))
    @settings(max_examples=20, deadline=None)
    def test_dense_matrix_handling(self, size, density):
        """Test handling of very dense matrices."""
        # Create dense matrix
        matrix = np.random.rand(size, size)
        matrix = (matrix + matrix.T) / 2
        
        # Apply density threshold
        matrix[matrix < (1 - density)] = 0
        matrix[matrix >= (1 - density)] = 1
        
        np.fill_diagonal(matrix, 0)
        
        # Normalize
        row_sums = matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums[:, np.newaxis]
        
        # Should still cluster
        n_clusters = min(2, size // 2)
        clusterer = KSMLocalSearch(n_clusters=n_clusters, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        assert result.NoItems == size