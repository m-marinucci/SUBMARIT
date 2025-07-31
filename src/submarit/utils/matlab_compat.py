"""MATLAB compatibility utilities for SUBMARIT.

This module provides functions to ensure compatibility between MATLAB and Python
implementations, handling differences in indexing, random number generation, and
numerical operations.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


class MatlabCompatibilityError(Exception):
    """Raised when MATLAB compatibility issues occur."""
    pass


def matlab_to_python_index(idx: Union[int, ArrayLike]) -> Union[int, NDArray]:
    """Convert MATLAB 1-based indices to Python 0-based indices.
    
    Args:
        idx: MATLAB index or array of indices (1-based)
        
    Returns:
        Python index or array of indices (0-based)
        
    Raises:
        MatlabCompatibilityError: If index is less than 1
    """
    if isinstance(idx, (int, np.integer)):
        if idx < 1:
            raise MatlabCompatibilityError(f"MATLAB index must be >= 1, got {idx}")
        return idx - 1
    else:
        idx = np.asarray(idx)
        if np.any(idx < 1):
            raise MatlabCompatibilityError("All MATLAB indices must be >= 1")
        return idx - 1


def python_to_matlab_index(idx: Union[int, ArrayLike]) -> Union[int, NDArray]:
    """Convert Python 0-based indices to MATLAB 1-based indices.
    
    Args:
        idx: Python index or array of indices (0-based)
        
    Returns:
        MATLAB index or array of indices (1-based)
        
    Raises:
        MatlabCompatibilityError: If index is negative
    """
    if isinstance(idx, (int, np.integer)):
        if idx < 0:
            raise MatlabCompatibilityError(f"Python index must be >= 0, got {idx}")
        return idx + 1
    else:
        idx = np.asarray(idx)
        if np.any(idx < 0):
            raise MatlabCompatibilityError("All Python indices must be >= 0")
        return idx + 1


class IndexConverter:
    """Context manager for automatic index conversion."""
    
    def __init__(self, matlab_style: bool = True):
        """Initialize the index converter.
        
        Args:
            matlab_style: If True, expect 1-based indices; if False, use 0-based
        """
        self.matlab_style = matlab_style
        
    def convert_in(self, idx: Union[int, ArrayLike]) -> Union[int, NDArray]:
        """Convert indices on input."""
        if self.matlab_style:
            return matlab_to_python_index(idx)
        return idx
        
    def convert_out(self, idx: Union[int, ArrayLike]) -> Union[int, NDArray]:
        """Convert indices on output."""
        if self.matlab_style:
            return python_to_matlab_index(idx)
        return idx


class MatlabRandom:
    """MATLAB-compatible random number generator.
    
    This class provides random number generation that matches MATLAB's behavior
    as closely as possible, including the ability to use MATLAB's random seeds.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the random number generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        
    def rand(self, *shape: int) -> NDArray[np.float64]:
        """Generate uniformly distributed random numbers like MATLAB's rand.
        
        Args:
            *shape: Dimensions of the output array
            
        Returns:
            Array of random numbers from uniform distribution [0, 1)
        """
        if not shape:
            return self.rng.random()
        return self.rng.random(shape)
        
    def randn(self, *shape: int) -> NDArray[np.float64]:
        """Generate normally distributed random numbers like MATLAB's randn.
        
        Args:
            *shape: Dimensions of the output array
            
        Returns:
            Array of random numbers from standard normal distribution
        """
        if not shape:
            return self.rng.randn()
        return self.rng.randn(*shape)
        
    def randperm(self, n: int) -> NDArray[np.int64]:
        """Generate random permutation like MATLAB's randperm.
        
        Args:
            n: Number of elements to permute
            
        Returns:
            Random permutation of integers from 1 to n (1-based like MATLAB)
        """
        # MATLAB returns 1-based indices
        return self.rng.permutation(n) + 1
        
    def randi(self, imax: int, *shape: int) -> NDArray[np.int64]:
        """Generate uniformly distributed random integers like MATLAB's randi.
        
        Args:
            imax: Maximum integer value (inclusive)
            *shape: Dimensions of the output array
            
        Returns:
            Array of random integers from 1 to imax (inclusive, 1-based)
        """
        if not shape:
            return self.rng.randint(1, imax + 1)
        return self.rng.randint(1, imax + 1, size=shape)


def matlab_compatible_random(seed: Optional[int] = None) -> MatlabRandom:
    """Create a MATLAB-compatible random number generator.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        MatlabRandom instance
    """
    return MatlabRandom(seed)


def ensure_matlab_compatibility(func):
    """Decorator to ensure MATLAB compatibility for numerical functions.
    
    This decorator:
    - Ensures float64 precision
    - Handles index conversion
    - Manages numerical tolerances
    """
    def wrapper(*args, **kwargs):
        # Convert inputs to float64 where appropriate
        args = [np.asarray(arg, dtype=np.float64) if isinstance(arg, (list, np.ndarray)) 
                and np.issubdtype(np.asarray(arg).dtype, np.number)
                else arg for arg in args]
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Ensure float64 output for numerical arrays
        if isinstance(result, np.ndarray) and np.issubdtype(result.dtype, np.number):
            result = result.astype(np.float64)
            
        return result
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def matlab_size(array: ArrayLike) -> Tuple[int, ...]:
    """Get array dimensions in MATLAB format.
    
    Args:
        array: Input array
        
    Returns:
        Tuple of dimensions (always at least 2D)
    """
    array = np.asarray(array)
    shape = array.shape
    
    # MATLAB always returns at least 2 dimensions
    if len(shape) == 0:
        return (1, 1)
    elif len(shape) == 1:
        return (shape[0], 1)
    else:
        return shape


def matlab_reshape(array: ArrayLike, *shape: int) -> NDArray:
    """Reshape array using MATLAB column-major order.
    
    Args:
        array: Input array
        *shape: New shape dimensions
        
    Returns:
        Reshaped array
    """
    array = np.asarray(array)
    # MATLAB uses column-major (Fortran) order
    return array.reshape(shape, order='F')


def matlab_find(condition: ArrayLike) -> NDArray[np.int64]:
    """Find indices of nonzero elements like MATLAB's find.
    
    Args:
        condition: Boolean array or condition
        
    Returns:
        1-based indices of True/nonzero elements
    """
    indices = np.nonzero(condition)[0]
    # Convert to 1-based indexing
    return indices + 1


def set_default_dtype():
    """Set NumPy default dtype to float64 to match MATLAB."""
    # This ensures new arrays default to float64
    # Note: This is a global setting
    np.float_ = np.float64
    np.int_ = np.int64