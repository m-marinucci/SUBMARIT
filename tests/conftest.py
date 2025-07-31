"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_substitution_matrix():
    """Generate a sample substitution matrix for testing."""
    np.random.seed(42)
    n_products = 20
    matrix = np.random.rand(n_products, n_products)
    # Make it symmetric
    matrix = (matrix + matrix.T) / 2
    # Zero diagonal
    np.fill_diagonal(matrix, 0)
    return matrix


@pytest.fixture
def sample_clusters():
    """Generate sample cluster assignments."""
    return np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])


@pytest.fixture
def matlab_test_data():
    """Load test data from MATLAB for validation."""
    # This would load .mat files with known outputs
    # For now, return mock data
    return {
        "input_matrix": np.random.rand(10, 10),
        "expected_clusters": np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0]),
        "expected_likelihood": -123.45,
    }