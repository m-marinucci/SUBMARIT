"""Python implementation of CreateSubstitutionMatrix from MATLAB."""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from submarit.utils.matlab_compat import ensure_matlab_compatibility


@ensure_matlab_compatibility
def create_substitution_matrix(
    X: NDArray[np.float64],
    normalize: bool = True,
    weight: int = 0,
    diag: bool = False
) -> Tuple[NDArray[np.float64], NDArray[np.int64], int]:
    """Create a forced substitution matrix between products.
    
    For each row i, the columns give the value of other products
    purchased by purchasers of product i.
    
    Args:
        X: A consumer × product substitution matrix
        normalize: Whether to normalize rows to sum to 1
        weight: 0 = weight by number of consumers, 1 = weight by product sales
                (weight=1 and normalize=0 is equivalent to the forced substitution 
                matrix given in UJH)
        diag: Whether to include diagonal self substitution
        
    Returns:
        Tuple containing:
            - FSWM: A product × product forced substitution matrix
            - PIndexes: The indexes of the products that are included (1-based)
            - PCount: The number of products that are included
    """
    # Ensure float64
    X = np.asarray(X, dtype=np.float64)
    
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    
    CCount, PCount = X.shape
    PIndexes = np.arange(1, PCount + 1)  # 1-based indexing for MATLAB compatibility
    
    # Iteratively filter products and consumers
    Continue = True
    while Continue:
        # Ensure that consumer has sales for at least two products
        # Sort each row in descending order
        Sorted = np.sort(X, axis=1)[:, ::-1]  # Descending order
        # Find rows where at least 2 products have positive sales
        CurIndexes = np.where(Sorted[:, 1] > 0)[0]
        X = X[CurIndexes, :]
        
        # Now find products that are bought by at least two consumers
        # Sort each column in descending order
        Sorted = np.sort(X, axis=0)[::-1, :]  # Descending order
        # Find columns where at least 2 consumers have positive purchases
        CurIndexes = np.where(Sorted[1, :] > 0)[0]
        X = X[:, CurIndexes]
        PIndexes = PIndexes[CurIndexes]
        
        OldPCount = PCount
        PCount = len(PIndexes)
        Continue = (PCount != OldPCount)
    
    # Calculate consumer sales totals
    CSales = X.sum(axis=1)
    
    if weight == 0:
        # Normalize by consumer sales (rows sum to 1)
        X = X / CSales[:, np.newaxis]
    
    # Initialize forced substitution matrix
    FSWM = np.zeros((PCount, PCount), dtype=np.float64)
    
    # Calculate forced substitution matrix
    for i in range(PCount):
        # Calculate ni(1 to number of products)
        if weight == 0:
            # Weighting by consumers - all rows add up to 1
            XMinus = 1 - X[:, i:i+1]  # Keep as column vector
        else:
            # Weight by product sales
            XMinus = (CSales - X[:, i])[:, np.newaxis]
        
        # Avoid division by zero
        XMinus = np.where(XMinus == 0, 1e-10, XMinus)
        
        # Calculate product ratios
        XProd = X / XMinus
        
        # Weight by consumers who chose product i
        Choosei = X[:, i:i+1]  # Keep as column vector
        
        # Sum weighted product choices
        FSWM[i, :] = np.sum(XProd * Choosei, axis=0)
        
        if normalize:
            # Normalize by total choosers of product i
            sum_choosei = np.sum(Choosei)
            if sum_choosei > 0:
                FSWM[i, :] = FSWM[i, :] / sum_choosei
    
    # Remove diagonal if requested
    if not diag:
        np.fill_diagonal(FSWM, 0)
    
    return FSWM, PIndexes, PCount


def create_substitution_matrix_from_data(
    consumer_product_data: NDArray[np.float64],
    **kwargs
) -> Tuple[NDArray[np.float64], NDArray[np.int64], int]:
    """Convenience function to create substitution matrix from consumer-product data.
    
    This is a wrapper around create_substitution_matrix with more intuitive naming.
    
    Args:
        consumer_product_data: Matrix where rows are consumers and columns are products
        **kwargs: Additional arguments passed to create_substitution_matrix
        
    Returns:
        Same as create_substitution_matrix
    """
    return create_substitution_matrix(consumer_product_data, **kwargs)