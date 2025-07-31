"""Substitution matrix implementation for SUBMARIT."""

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from submarit.utils.matlab_compat import ensure_matlab_compatibility


class SubstitutionMatrix:
    """Represents a product substitution matrix.
    
    This class handles the creation and manipulation of substitution matrices
    used in submarket identification. The matrix represents substitution patterns
    between products based on sales or other data.
    """
    
    def __init__(
        self,
        data: Union[ArrayLike, None] = None,
        normalize: bool = True,
        check_symmetry: bool = True,
        tol: float = 1e-10
    ):
        """Initialize the substitution matrix.
        
        Args:
            data: Input data (can be raw sales data or pre-computed matrix)
            normalize: Whether to normalize the matrix
            check_symmetry: Whether to check and enforce symmetry
            tol: Tolerance for numerical operations
        """
        self.tol = tol
        self._matrix = None
        self._normalized = False
        
        if data is not None:
            self.set_data(data, normalize, check_symmetry)
    
    def set_data(
        self,
        data: ArrayLike,
        normalize: bool = True,
        check_symmetry: bool = True
    ) -> None:
        """Set the substitution matrix data.
        
        Args:
            data: Input data
            normalize: Whether to normalize the matrix
            check_symmetry: Whether to check and enforce symmetry
        """
        data = np.asarray(data, dtype=np.float64)
        
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")
        
        if data.shape[0] != data.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {data.shape}")
        
        self._matrix = data.copy()
        
        # Ensure diagonal is zero
        np.fill_diagonal(self._matrix, 0)
        
        # Check and enforce symmetry if requested
        if check_symmetry:
            if not self.is_symmetric():
                self._matrix = (self._matrix + self._matrix.T) / 2
        
        # Normalize if requested
        if normalize:
            self.normalize()
    
    def create_from_consumer_product_data(
        self,
        consumer_product_data: ArrayLike,
        normalize: bool = True,
        weight: int = 0,
        diag: bool = False
    ) -> Tuple[NDArray[np.int64], int]:
        """Create substitution matrix from consumer-product data.
        
        This method implements the logic from CreateSubstitutionMatrix.m
        
        Args:
            consumer_product_data: Consumer × product data matrix
            normalize: Whether to normalize rows to sum to 1
            weight: 0 = weight by number of consumers, 1 = weight by product sales
            diag: Whether to include diagonal self substitution
            
        Returns:
            Tuple of (product_indexes, product_count)
        """
        from submarit.core.create_substitution_matrix import create_substitution_matrix
        
        matrix, indexes, count = create_substitution_matrix(
            consumer_product_data, normalize, weight, diag
        )
        
        self._matrix = matrix
        self._normalized = normalize
        self._product_indexes = indexes
        
        return indexes, count
    
    @ensure_matlab_compatibility
    def create_from_sales_data(
        self,
        sales_data: ArrayLike,
        method: str = "correlation"
    ) -> None:
        """Create substitution matrix from sales data time series.
        
        Args:
            sales_data: Sales data matrix (products × time periods)
            method: Method for computing substitution ('correlation', 'covariance')
        """
        sales_data = np.asarray(sales_data, dtype=np.float64)
        
        if sales_data.ndim != 2:
            raise ValueError("Sales data must be 2D (products × time periods)")
        
        n_products = sales_data.shape[0]
        
        if method == "correlation":
            # Compute correlation matrix
            self._matrix = np.corrcoef(sales_data)
        elif method == "covariance":
            # Compute covariance matrix
            self._matrix = np.cov(sales_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Ensure non-negative values
        self._matrix = np.maximum(self._matrix, 0)
        
        # Zero diagonal
        np.fill_diagonal(self._matrix, 0)
        
        # Normalize
        self.normalize()
    
    def normalize(self) -> None:
        """Normalize the substitution matrix.
        
        Ensures that each row sums to 1 (excluding diagonal).
        """
        if self._matrix is None:
            raise ValueError("No data set")
        
        # Calculate row sums excluding diagonal
        row_sums = self._matrix.sum(axis=1)
        
        # Avoid division by zero
        row_sums[row_sums < self.tol] = 1.0
        
        # Normalize rows
        self._matrix = self._matrix / row_sums[:, np.newaxis]
        
        # Ensure diagonal remains zero
        np.fill_diagonal(self._matrix, 0)
        
        self._normalized = True
    
    def is_symmetric(self, tol: Optional[float] = None) -> bool:
        """Check if the matrix is symmetric.
        
        Args:
            tol: Tolerance for symmetry check
            
        Returns:
            True if symmetric within tolerance
        """
        if self._matrix is None:
            return True
        
        tol = tol or self.tol
        return np.allclose(self._matrix, self._matrix.T, rtol=tol, atol=tol)
    
    def get_matrix(self) -> NDArray[np.float64]:
        """Get the substitution matrix.
        
        Returns:
            The substitution matrix
        """
        if self._matrix is None:
            raise ValueError("No data set")
        return self._matrix.copy()
    
    def get_submatrix(
        self,
        indices: ArrayLike
    ) -> NDArray[np.float64]:
        """Extract a submatrix for given indices.
        
        Args:
            indices: Indices of products to include
            
        Returns:
            Submatrix for the specified products
        """
        if self._matrix is None:
            raise ValueError("No data set")
        
        indices = np.asarray(indices)
        return self._matrix[np.ix_(indices, indices)]
    
    def get_inter_cluster_substitution(
        self,
        labels: ArrayLike
    ) -> NDArray[np.float64]:
        """Compute inter-cluster substitution matrix.
        
        Args:
            labels: Cluster assignments for each product
            
        Returns:
            Matrix of substitution rates between clusters
        """
        if self._matrix is None:
            raise ValueError("No data set")
        
        labels = np.asarray(labels)
        n_clusters = len(np.unique(labels))
        
        inter_cluster = np.zeros((n_clusters, n_clusters))
        
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    mask_i = labels == i
                    mask_j = labels == j
                    inter_cluster[i, j] = self._matrix[np.ix_(mask_i, mask_j)].sum()
        
        return inter_cluster
    
    def get_intra_cluster_substitution(
        self,
        labels: ArrayLike
    ) -> NDArray[np.float64]:
        """Compute average intra-cluster substitution for each cluster.
        
        Args:
            labels: Cluster assignments for each product
            
        Returns:
            Array of average intra-cluster substitution rates
        """
        if self._matrix is None:
            raise ValueError("No data set")
        
        labels = np.asarray(labels)
        n_clusters = len(np.unique(labels))
        
        intra_cluster = np.zeros(n_clusters)
        
        for i in range(n_clusters):
            mask = labels == i
            cluster_size = mask.sum()
            if cluster_size > 1:
                submatrix = self._matrix[np.ix_(mask, mask)]
                # Average over non-diagonal elements
                intra_cluster[i] = submatrix.sum() / (cluster_size * (cluster_size - 1))
        
        return intra_cluster
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the substitution matrix."""
        if self._matrix is None:
            return (0, 0)
        return self._matrix.shape
    
    @property
    def n_products(self) -> int:
        """Get the number of products."""
        return self.shape[0]
    
    @property
    def is_normalized(self) -> bool:
        """Check if the matrix is normalized."""
        return self._normalized
    
    def __repr__(self) -> str:
        """String representation."""
        if self._matrix is None:
            return "SubstitutionMatrix(no data)"
        return f"SubstitutionMatrix(shape={self.shape}, normalized={self._normalized})"
    
    def __getitem__(self, key):
        """Enable indexing."""
        if self._matrix is None:
            raise ValueError("No data set")
        return self._matrix[key]