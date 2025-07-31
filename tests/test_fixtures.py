"""Common test fixtures and utilities for SUBMARIT tests."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from submarit.utils.matlab_compat import MatlabRandom


# --- Data Generation Fixtures ---

@pytest.fixture
def small_substitution_matrix() -> NDArray[np.float64]:
    """Generate a small 5x5 substitution matrix for testing."""
    np.random.seed(42)
    n = 5
    matrix = np.random.rand(n, n)
    # Make symmetric
    matrix = (matrix + matrix.T) / 2
    # Zero diagonal
    np.fill_diagonal(matrix, 0)
    # Normalize rows
    row_sums = matrix.sum(axis=1)
    matrix = matrix / row_sums[:, np.newaxis]
    return matrix


@pytest.fixture
def medium_substitution_matrix() -> NDArray[np.float64]:
    """Generate a medium 20x20 substitution matrix for testing."""
    np.random.seed(42)
    n = 20
    matrix = np.random.rand(n, n)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    row_sums = matrix.sum(axis=1)
    matrix = matrix / row_sums[:, np.newaxis]
    return matrix


@pytest.fixture
def large_substitution_matrix() -> NDArray[np.float64]:
    """Generate a large 100x100 substitution matrix for testing."""
    np.random.seed(42)
    n = 100
    matrix = np.random.rand(n, n)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    row_sums = matrix.sum(axis=1)
    matrix = matrix / row_sums[:, np.newaxis]
    return matrix


@pytest.fixture
def clustered_substitution_matrix() -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate a substitution matrix with known cluster structure.
    
    Returns:
        Tuple of (matrix, true_labels)
    """
    np.random.seed(42)
    n_clusters = 3
    cluster_sizes = [15, 20, 15]
    n_total = sum(cluster_sizes)
    
    # Create true labels
    true_labels = np.zeros(n_total, dtype=np.int64)
    start = 0
    for i, size in enumerate(cluster_sizes):
        true_labels[start:start + size] = i
        start += size
    
    # Create block-structured matrix
    matrix = np.zeros((n_total, n_total))
    
    # Within-cluster substitution rates (high)
    within_rate = 0.8
    # Between-cluster substitution rates (low)
    between_rate = 0.2
    
    start_i = 0
    for i, size_i in enumerate(cluster_sizes):
        end_i = start_i + size_i
        start_j = 0
        for j, size_j in enumerate(cluster_sizes):
            end_j = start_j + size_j
            if i == j:
                # Within cluster
                block = np.random.rand(size_i, size_j) * within_rate
                np.fill_diagonal(block, 0)  # Zero diagonal for this block
            else:
                # Between clusters
                block = np.random.rand(size_i, size_j) * between_rate
            matrix[start_i:end_i, start_j:end_j] = block
            start_j = end_j
        start_i = end_i
    
    # Make symmetric
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    
    # Normalize
    row_sums = matrix.sum(axis=1)
    matrix = matrix / row_sums[:, np.newaxis]
    
    return matrix, true_labels


@pytest.fixture
def consumer_product_data() -> NDArray[np.int64]:
    """Generate sample consumer-product purchase data."""
    np.random.seed(42)
    n_consumers = 1000
    n_products = 20
    
    # Create purchase probabilities with some products more popular
    product_probs = np.random.dirichlet(np.ones(n_products) * 2)
    
    # Generate purchases
    data = np.zeros((n_consumers, n_products), dtype=np.int64)
    for i in range(n_consumers):
        # Each consumer buys 1-5 products
        n_purchases = np.random.randint(1, 6)
        purchased = np.random.choice(n_products, n_purchases, p=product_probs, replace=False)
        data[i, purchased] = 1
    
    return data


@pytest.fixture
def sales_time_series() -> NDArray[np.float64]:
    """Generate sample sales time series data."""
    np.random.seed(42)
    n_products = 10
    n_periods = 52  # Weekly data for a year
    
    # Base sales levels
    base_sales = np.random.uniform(100, 1000, n_products)
    
    # Generate time series with trends and seasonality
    data = np.zeros((n_products, n_periods))
    for i in range(n_products):
        # Trend
        trend = np.linspace(0, 0.2, n_periods) * base_sales[i]
        # Seasonality
        seasonal = 0.1 * base_sales[i] * np.sin(2 * np.pi * np.arange(n_periods) / 13)
        # Noise
        noise = np.random.normal(0, 0.05 * base_sales[i], n_periods)
        # Combine
        data[i] = base_sales[i] + trend + seasonal + noise
        # Ensure non-negative
        data[i] = np.maximum(data[i], 0)
    
    return data


# --- Matlab Compatibility Fixtures ---

@pytest.fixture
def matlab_reference_data() -> Dict[str, NDArray]:
    """Load or generate MATLAB reference data for validation.
    
    This would normally load .mat files with pre-computed results.
    For testing, we generate synthetic reference data.
    """
    np.random.seed(123)  # Different seed for reference
    
    # Reference switching matrix
    n = 10
    swm = np.random.rand(n, n)
    swm = (swm + swm.T) / 2
    np.fill_diagonal(swm, 0)
    
    # Reference cluster assignment (simulated MATLAB output)
    # Using 1-based indexing like MATLAB
    assign = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1])
    
    # Reference statistics
    diff = 0.1234
    log_lh = -123.456
    
    return {
        "swm": swm,
        "assign": assign,
        "diff": diff,
        "log_lh": log_lh,
        "n_clusters": 3,
        "min_items": 1,
    }


@pytest.fixture
def matlab_random_state():
    """Create a MATLAB-compatible random state for reproducible tests."""
    return MatlabRandom(seed=42)


# --- Performance Testing Fixtures ---

@pytest.fixture
def performance_matrices() -> Dict[str, NDArray[np.float64]]:
    """Generate matrices of various sizes for performance testing."""
    np.random.seed(42)
    sizes = [10, 50, 100, 200, 500]
    matrices = {}
    
    for size in sizes:
        matrix = np.random.rand(size, size)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        matrices[f"size_{size}"] = matrix
    
    return matrices


# --- Edge Case Fixtures ---

@pytest.fixture
def edge_case_matrices() -> Dict[str, NDArray[np.float64]]:
    """Generate matrices with edge cases for testing."""
    matrices = {}
    
    # Sparse matrix
    sparse = np.zeros((20, 20))
    # Add a few non-zero entries
    for i in range(20):
        if i < 19:
            sparse[i, i + 1] = sparse[i + 1, i] = 0.5
    np.fill_diagonal(sparse, 0)
    matrices["sparse"] = sparse
    
    # Dense matrix (all entries non-zero)
    dense = np.ones((10, 10))
    np.fill_diagonal(dense, 0)
    row_sums = dense.sum(axis=1)
    dense = dense / row_sums[:, np.newaxis]
    matrices["dense"] = dense
    
    # Matrix with isolated nodes
    isolated = np.eye(15)
    # Connect some nodes
    for i in range(10):
        if i < 9:
            isolated[i, i + 1] = isolated[i + 1, i] = 0.5
    np.fill_diagonal(isolated, 0)
    matrices["isolated"] = isolated
    
    # Matrix with one dominant cluster
    dominant = np.ones((20, 20)) * 0.01
    dominant[:10, :10] = 0.9
    np.fill_diagonal(dominant, 0)
    row_sums = dominant.sum(axis=1)
    dominant = dominant / row_sums[:, np.newaxis]
    matrices["dominant"] = dominant
    
    return matrices


# --- Test Data Directory ---

@pytest.fixture
def test_data_dir() -> Path:
    """Get the test data directory path."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# --- Clustering Results Fixtures ---

@pytest.fixture
def sample_clustering_result():
    """Generate a sample clustering result for testing evaluation metrics."""
    from submarit.algorithms.local_search import LocalSearchResult
    
    result = LocalSearchResult()
    result.NoClusters = 3
    result.NoItems = 10
    result.Assign = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1])
    result.Diff = 0.5
    result.LogLH = -100.0
    result.Iter = 10
    
    return result


# --- Benchmark Configuration ---

@pytest.fixture
def benchmark_config() -> Dict:
    """Configuration for benchmark tests."""
    return {
        "matrix_sizes": [10, 20, 50, 100],
        "n_clusters_list": [2, 3, 5],
        "n_runs": 10,
        "algorithms": ["local_search", "local_search2"],
        "time_limit": 60.0,  # seconds
    }


# --- Test Utilities ---

def assert_matrix_properties(matrix: NDArray[np.float64], 
                           symmetric: bool = True,
                           normalized: bool = True,
                           zero_diagonal: bool = True,
                           tol: float = 1e-10) -> None:
    """Assert common properties of substitution matrices."""
    # Check square
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    
    # Check symmetric
    if symmetric:
        assert np.allclose(matrix, matrix.T, atol=tol), "Matrix not symmetric"
    
    # Check diagonal
    if zero_diagonal:
        assert np.allclose(np.diag(matrix), 0, atol=tol), "Diagonal not zero"
    
    # Check normalized
    if normalized:
        row_sums = matrix.sum(axis=1)
        expected_sums = np.ones_like(row_sums)
        # Account for rows that might be all zeros
        mask = row_sums > tol
        assert np.allclose(row_sums[mask], expected_sums[mask], atol=tol), "Rows not normalized"


def generate_constrained_problem(n_items: int = 20, 
                               n_fixed: int = 5,
                               n_clusters: int = 3) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Generate a constrained clustering problem."""
    np.random.seed(42)
    
    # Randomly select fixed items
    all_indices = np.arange(n_items) + 1  # 1-based
    fixed_indices = np.random.choice(all_indices, n_fixed, replace=False)
    free_indices = np.setdiff1d(all_indices, fixed_indices)
    
    # Assign fixed items to clusters
    fixed_assignments = np.random.randint(1, n_clusters + 1, n_fixed)
    
    return fixed_assignments, fixed_indices, free_indices