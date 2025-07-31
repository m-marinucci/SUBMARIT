"""Tests for substitution matrix functionality."""

import numpy as np
import pytest

from submarit.core.create_substitution_matrix import create_substitution_matrix
from submarit.core.substitution_matrix import SubstitutionMatrix


class TestCreateSubstitutionMatrix:
    """Test the create_substitution_matrix function."""
    
    def test_basic_functionality(self):
        """Test basic matrix creation."""
        # Create simple consumer-product matrix
        # 4 consumers, 3 products
        X = np.array([
            [1, 0, 2],  # Consumer 1: buys product 1 and 3
            [1, 1, 0],  # Consumer 2: buys product 1 and 2
            [0, 1, 1],  # Consumer 3: buys product 2 and 3
            [2, 1, 1],  # Consumer 4: buys all products
        ], dtype=np.float64)
        
        FSWM, indexes, count = create_substitution_matrix(X, normalize=True, weight=0)
        
        # Check output dimensions
        assert FSWM.shape == (3, 3)
        assert count == 3
        assert len(indexes) == 3
        assert np.array_equal(indexes, [1, 2, 3])  # 1-based indexing
        
        # Check diagonal is zero
        assert np.allclose(np.diag(FSWM), 0)
        
        # Check matrix is non-negative
        assert np.all(FSWM >= 0)
    
    def test_filtering_products(self):
        """Test that products with insufficient data are filtered."""
        # Create matrix where product 3 is only bought by one consumer
        X = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 0],  # Consumer 4 buys nothing
        ], dtype=np.float64)
        
        FSWM, indexes, count = create_substitution_matrix(X)
        
        # Product 3 should be filtered out
        assert count == 2
        assert np.array_equal(indexes, [1, 2])
        assert FSWM.shape == (2, 2)
    
    def test_filtering_consumers(self):
        """Test that consumers with insufficient products are filtered."""
        # Create matrix where consumer 1 only buys one product
        X = np.array([
            [1, 0, 0],  # Only buys product 1
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ], dtype=np.float64)
        
        FSWM, indexes, count = create_substitution_matrix(X)
        
        # All products should remain (each bought by at least 2 consumers)
        # But the resulting matrix should reflect the filtering
        assert count == 3
    
    def test_normalization(self):
        """Test normalization options."""
        X = np.array([
            [1, 2, 0],
            [2, 1, 3],
            [0, 3, 1],
            [1, 1, 2],
        ], dtype=np.float64)
        
        # Test with normalization
        FSWM_norm, _, _ = create_substitution_matrix(X, normalize=True)
        
        # Test without normalization
        FSWM_no_norm, _, _ = create_substitution_matrix(X, normalize=False)
        
        # Normalized and non-normalized should be different
        assert not np.allclose(FSWM_norm, FSWM_no_norm)
    
    def test_weight_options(self):
        """Test different weighting options."""
        X = np.array([
            [1, 2, 0],
            [2, 1, 3],
            [0, 3, 1],
            [1, 1, 2],
        ], dtype=np.float64)
        
        # Test weight by consumers (0)
        FSWM_w0, _, _ = create_substitution_matrix(X, weight=0)
        
        # Test weight by sales (1)
        FSWM_w1, _, _ = create_substitution_matrix(X, weight=1)
        
        # Results should be different
        assert not np.allclose(FSWM_w0, FSWM_w1)
    
    def test_diagonal_option(self):
        """Test diagonal inclusion option."""
        X = np.array([
            [1, 2, 1],
            [2, 1, 3],
            [1, 3, 1],
            [1, 1, 2],
        ], dtype=np.float64)
        
        # Test without diagonal
        FSWM_no_diag, _, _ = create_substitution_matrix(X, diag=False)
        assert np.allclose(np.diag(FSWM_no_diag), 0)
        
        # Test with diagonal
        FSWM_diag, _, _ = create_substitution_matrix(X, diag=True)
        # Diagonal should have non-zero values
        assert not np.allclose(np.diag(FSWM_diag), 0)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty matrix
        with pytest.raises(ValueError):
            create_substitution_matrix(np.array([]))
        
        # 1D array
        with pytest.raises(ValueError):
            create_substitution_matrix(np.array([1, 2, 3]))
        
        # All zeros
        X = np.zeros((4, 3))
        FSWM, indexes, count = create_substitution_matrix(X)
        assert count == 0
        assert len(indexes) == 0


class TestSubstitutionMatrix:
    """Test the SubstitutionMatrix class."""
    
    def test_initialization(self):
        """Test matrix initialization."""
        # Initialize empty
        sm = SubstitutionMatrix()
        assert sm.shape == (0, 0)
        assert sm.n_products == 0
        
        # Initialize with data
        data = np.array([[0, 0.5, 0.5], [0.3, 0, 0.7], [0.4, 0.6, 0]])
        sm = SubstitutionMatrix(data)
        assert sm.shape == (3, 3)
        assert sm.n_products == 3
    
    def test_set_data(self):
        """Test setting matrix data."""
        sm = SubstitutionMatrix()
        
        data = np.array([[0, 0.5, 0.5], [0.3, 0, 0.7], [0.4, 0.6, 0]])
        sm.set_data(data)
        
        assert sm.shape == (3, 3)
        assert np.allclose(np.diag(sm.get_matrix()), 0)
    
    def test_symmetry_check(self):
        """Test symmetry checking and enforcement."""
        # Asymmetric matrix
        data = np.array([[0, 0.5, 0.3], [0.7, 0, 0.6], [0.4, 0.5, 0]])
        sm = SubstitutionMatrix(data, check_symmetry=True)
        
        # Should be made symmetric
        assert sm.is_symmetric()
        
        # Verify symmetry
        matrix = sm.get_matrix()
        assert np.allclose(matrix, matrix.T)
    
    def test_normalization(self):
        """Test matrix normalization."""
        data = np.array([[0, 2, 3], [1, 0, 4], [2, 1, 0]])
        sm = SubstitutionMatrix(data, normalize=True)
        
        matrix = sm.get_matrix()
        # Check that rows sum to 1 (excluding diagonal)
        for i in range(3):
            row_sum = matrix[i, :].sum() - matrix[i, i]
            assert np.isclose(row_sum, 1.0)
    
    def test_create_from_consumer_product_data(self):
        """Test creating matrix from consumer-product data."""
        sm = SubstitutionMatrix()
        
        # Consumer-product data
        cp_data = np.array([
            [1, 0, 2],
            [1, 1, 0],
            [0, 1, 1],
            [2, 1, 1],
        ])
        
        indexes, count = sm.create_from_consumer_product_data(cp_data)
        
        assert count == 3
        assert sm.shape == (3, 3)
        assert np.allclose(np.diag(sm.get_matrix()), 0)
    
    def test_create_from_sales_data(self):
        """Test creating matrix from sales time series."""
        sm = SubstitutionMatrix()
        
        # Sales data (products × time periods)
        sales_data = np.array([
            [10, 12, 15, 11, 13],  # Product 1 sales
            [8, 9, 7, 10, 8],      # Product 2 sales
            [5, 6, 7, 6, 5],       # Product 3 sales
        ])
        
        sm.create_from_sales_data(sales_data, method='correlation')
        
        assert sm.shape == (3, 3)
        assert np.allclose(np.diag(sm.get_matrix()), 0)
        assert sm.is_symmetric()
    
    def test_submatrix_extraction(self):
        """Test extracting submatrices."""
        data = np.array([
            [0, 0.1, 0.2, 0.3],
            [0.1, 0, 0.4, 0.5],
            [0.2, 0.4, 0, 0.6],
            [0.3, 0.5, 0.6, 0],
        ])
        sm = SubstitutionMatrix(data)
        
        # Extract submatrix for products 0 and 2
        sub = sm.get_submatrix([0, 2])
        
        assert sub.shape == (2, 2)
        assert np.allclose(sub, [[0, 0.2], [0.2, 0]])
    
    def test_cluster_substitution(self):
        """Test inter and intra cluster substitution calculations."""
        data = np.array([
            [0, 0.8, 0.1, 0.1],
            [0.8, 0, 0.1, 0.1],
            [0.1, 0.1, 0, 0.8],
            [0.1, 0.1, 0.8, 0],
        ])
        sm = SubstitutionMatrix(data)
        
        # Products 0,1 in cluster 0; products 2,3 in cluster 1
        labels = np.array([0, 0, 1, 1])
        
        # Test inter-cluster substitution
        inter = sm.get_inter_cluster_substitution(labels)
        assert inter.shape == (2, 2)
        assert inter[0, 0] == 0  # No self-substitution
        assert inter[1, 1] == 0
        assert inter[0, 1] > 0  # Substitution from cluster 0 to 1
        assert inter[1, 0] > 0  # Substitution from cluster 1 to 0
        
        # Test intra-cluster substitution
        intra = sm.get_intra_cluster_substitution(labels)
        assert len(intra) == 2
        assert intra[0] > 0  # High within cluster 0
        assert intra[1] > 0  # High within cluster 1
    
    def test_indexing(self):
        """Test matrix indexing."""
        data = np.array([[0, 0.5, 0.5], [0.3, 0, 0.7], [0.4, 0.6, 0]])
        sm = SubstitutionMatrix(data)
        
        # Test single element access
        assert sm[0, 1] == 0.5
        
        # Test row access
        assert np.array_equal(sm[1, :], [0.3, 0, 0.7])
        
        # Test slicing
        assert sm[0:2, 1:3].shape == (2, 2)