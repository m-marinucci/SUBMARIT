"""Tests for MATLAB compatibility utilities."""

import numpy as np
import pytest

from submarit.utils.matlab_compat import (
    IndexConverter,
    MatlabCompatibilityError,
    MatlabRandom,
    ensure_matlab_compatibility,
    matlab_compatible_random,
    matlab_find,
    matlab_reshape,
    matlab_size,
    matlab_to_python_index,
    python_to_matlab_index,
)


class TestIndexConversion:
    """Test index conversion functions."""
    
    def test_matlab_to_python_scalar(self):
        """Test converting scalar MATLAB indices to Python."""
        assert matlab_to_python_index(1) == 0
        assert matlab_to_python_index(5) == 4
        assert matlab_to_python_index(100) == 99
    
    def test_python_to_matlab_scalar(self):
        """Test converting scalar Python indices to MATLAB."""
        assert python_to_matlab_index(0) == 1
        assert python_to_matlab_index(4) == 5
        assert python_to_matlab_index(99) == 100
    
    def test_matlab_to_python_array(self):
        """Test converting MATLAB index arrays to Python."""
        matlab_idx = np.array([1, 2, 3, 4, 5])
        python_idx = matlab_to_python_index(matlab_idx)
        assert np.array_equal(python_idx, [0, 1, 2, 3, 4])
    
    def test_python_to_matlab_array(self):
        """Test converting Python index arrays to MATLAB."""
        python_idx = np.array([0, 1, 2, 3, 4])
        matlab_idx = python_to_matlab_index(python_idx)
        assert np.array_equal(matlab_idx, [1, 2, 3, 4, 5])
    
    def test_invalid_indices(self):
        """Test error handling for invalid indices."""
        # MATLAB indices must be >= 1
        with pytest.raises(MatlabCompatibilityError):
            matlab_to_python_index(0)
        
        with pytest.raises(MatlabCompatibilityError):
            matlab_to_python_index(-1)
        
        # Python indices must be >= 0
        with pytest.raises(MatlabCompatibilityError):
            python_to_matlab_index(-1)
        
        # Arrays with invalid indices
        with pytest.raises(MatlabCompatibilityError):
            matlab_to_python_index(np.array([1, 0, 2]))
        
        with pytest.raises(MatlabCompatibilityError):
            python_to_matlab_index(np.array([0, -1, 2]))


class TestIndexConverter:
    """Test the IndexConverter context manager."""
    
    def test_matlab_style_converter(self):
        """Test converter in MATLAB style mode."""
        converter = IndexConverter(matlab_style=True)
        
        # Test input conversion (MATLAB to Python)
        assert converter.convert_in(1) == 0
        assert converter.convert_in(5) == 4
        
        # Test output conversion (Python to MATLAB)
        assert converter.convert_out(0) == 1
        assert converter.convert_out(4) == 5
    
    def test_python_style_converter(self):
        """Test converter in Python style mode."""
        converter = IndexConverter(matlab_style=False)
        
        # No conversion should occur
        assert converter.convert_in(0) == 0
        assert converter.convert_in(4) == 4
        assert converter.convert_out(0) == 0
        assert converter.convert_out(4) == 4


class TestMatlabRandom:
    """Test MATLAB-compatible random number generation."""
    
    def test_rand(self):
        """Test uniform random number generation."""
        rng = MatlabRandom(seed=42)
        
        # Scalar
        scalar = rng.rand()
        assert isinstance(scalar, (float, np.floating))
        assert 0 <= scalar < 1
        
        # Vector
        vec = rng.rand(5)
        assert vec.shape == (5,)
        assert np.all((0 <= vec) & (vec < 1))
        
        # Matrix
        mat = rng.rand(3, 4)
        assert mat.shape == (3, 4)
        assert np.all((0 <= mat) & (mat < 1))
    
    def test_randn(self):
        """Test normal random number generation."""
        rng = MatlabRandom(seed=42)
        
        # Scalar
        scalar = rng.randn()
        assert isinstance(scalar, (float, np.floating))
        
        # Vector
        vec = rng.randn(5)
        assert vec.shape == (5,)
        
        # Matrix
        mat = rng.randn(3, 4)
        assert mat.shape == (3, 4)
        
        # Check approximate normal distribution properties
        large_sample = rng.randn(10000)
        assert -0.1 < np.mean(large_sample) < 0.1
        assert 0.9 < np.std(large_sample) < 1.1
    
    def test_randperm(self):
        """Test random permutation generation."""
        rng = MatlabRandom(seed=42)
        
        perm = rng.randperm(5)
        assert len(perm) == 5
        # Check that it's a permutation of 1:5 (MATLAB style)
        assert set(perm) == {1, 2, 3, 4, 5}
        
        # Test larger permutation
        perm = rng.randperm(100)
        assert len(perm) == 100
        assert set(perm) == set(range(1, 101))
    
    def test_randi(self):
        """Test random integer generation."""
        rng = MatlabRandom(seed=42)
        
        # Scalar
        scalar = rng.randi(10)
        assert 1 <= scalar <= 10
        
        # Vector
        vec = rng.randi(10, 5)
        assert vec.shape == (5,)
        assert np.all((1 <= vec) & (vec <= 10))
        
        # Matrix
        mat = rng.randi(10, 3, 4)
        assert mat.shape == (3, 4)
        assert np.all((1 <= mat) & (mat <= 10))
    
    def test_reproducibility(self):
        """Test that seeding produces reproducible results."""
        rng1 = MatlabRandom(seed=123)
        rng2 = MatlabRandom(seed=123)
        
        # Should produce same results
        assert np.array_equal(rng1.rand(5), rng2.rand(5))
        assert np.array_equal(rng1.randn(5), rng2.randn(5))
        assert np.array_equal(rng1.randperm(10), rng2.randperm(10))
        assert np.array_equal(rng1.randi(100, 5), rng2.randi(100, 5))


class TestMatlabCompatibleRandom:
    """Test the factory function for MATLAB-compatible RNG."""
    
    def test_factory_function(self):
        """Test that factory creates correct object."""
        rng = matlab_compatible_random(seed=42)
        assert isinstance(rng, MatlabRandom)
        
        # Test it works
        vals = rng.rand(5)
        assert len(vals) == 5


class TestEnsureMatlabCompatibility:
    """Test the ensure_matlab_compatibility decorator."""
    
    @ensure_matlab_compatibility
    def sample_function(self, x, y):
        """Sample function for testing."""
        return x + y
    
    def test_float64_conversion(self):
        """Test that arrays are converted to float64."""
        # Integer input
        result = self.sample_function([1, 2, 3], [4, 5, 6])
        assert result.dtype == np.float64
        
        # Float32 input
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = self.sample_function(x, y)
        assert result.dtype == np.float64
    
    def test_non_numeric_passthrough(self):
        """Test that non-numeric arguments pass through."""
        result = self.sample_function("hello", " world")
        assert result == "hello world"


class TestMatlabFunctions:
    """Test MATLAB-compatible utility functions."""
    
    def test_matlab_size(self):
        """Test MATLAB-style size function."""
        # Scalar
        assert matlab_size(5) == (1, 1)
        
        # Vector
        assert matlab_size([1, 2, 3, 4, 5]) == (5, 1)
        assert matlab_size(np.array([1, 2, 3, 4, 5])) == (5, 1)
        
        # Matrix
        assert matlab_size(np.zeros((3, 4))) == (3, 4)
        
        # 3D array
        assert matlab_size(np.zeros((2, 3, 4))) == (2, 3, 4)
    
    def test_matlab_reshape(self):
        """Test MATLAB-style reshape (column-major order)."""
        # Create a simple matrix
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        
        # MATLAB reshape uses column-major order
        reshaped = matlab_reshape(mat, 3, 2)
        expected = np.array([[1, 5], [4, 3], [2, 6]])
        assert np.array_equal(reshaped, expected)
        
        # Compare with NumPy default (row-major)
        numpy_reshaped = mat.reshape(3, 2)
        assert not np.array_equal(reshaped, numpy_reshaped)
    
    def test_matlab_find(self):
        """Test MATLAB-style find function."""
        # Boolean array
        arr = np.array([False, True, False, True, True, False])
        indices = matlab_find(arr)
        assert np.array_equal(indices, [2, 4, 5])  # 1-based indices
        
        # Condition on array
        arr = np.array([1, 5, 3, 7, 2, 8])
        indices = matlab_find(arr > 4)
        assert np.array_equal(indices, [2, 4, 6])  # 1-based indices
        
        # Empty result
        arr = np.array([1, 2, 3])
        indices = matlab_find(arr > 10)
        assert len(indices) == 0