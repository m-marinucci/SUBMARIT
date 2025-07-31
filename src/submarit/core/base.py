"""Base classes for SUBMARIT core functionality."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class BaseEstimator(ABC):
    """Base class for all estimators in SUBMARIT.
    
    All estimators should specify all the parameters that can be set
    at the class level in their __init__ as explicit keyword arguments.
    """
    
    def __init__(self):
        """Initialize the base estimator."""
        pass
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.
        
        Args:
            deep: If True, will return parameters for sub-objects
            
        Returns:
            Parameter names mapped to their values
        """
        params = {}
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                params[key] = getattr(self, key)
        return params
    
    def set_params(self, **params) -> "BaseEstimator":
        """Set parameters for this estimator.
        
        Args:
            **params: Estimator parameters
            
        Returns:
            Self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class BaseClusterer(BaseEstimator):
    """Base class for clustering algorithms."""
    
    def __init__(self, n_clusters: int = 2, random_state: Optional[int] = None):
        """Initialize the base clusterer.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._labels = None
        self._n_iter = None
        
    @abstractmethod
    def fit(self, X: NDArray[np.float64]) -> "BaseClusterer":
        """Fit the clustering model.
        
        Args:
            X: Input data matrix
            
        Returns:
            Self
        """
        pass
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict cluster labels.
        
        Args:
            X: Input data matrix
            
        Returns:
            Cluster labels
        """
        if self._labels is None:
            raise ValueError("Model must be fitted before prediction")
        return self._labels
    
    def fit_predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Fit the model and predict cluster labels.
        
        Args:
            X: Input data matrix
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        return self._labels
    
    @property
    def labels_(self) -> Optional[NDArray[np.int64]]:
        """Get cluster labels."""
        return self._labels
    
    @property
    def n_iter_(self) -> Optional[int]:
        """Get number of iterations."""
        return self._n_iter


class BaseEvaluator(BaseEstimator):
    """Base class for cluster evaluation metrics."""
    
    @abstractmethod
    def evaluate(
        self, 
        X: NDArray[np.float64], 
        labels: NDArray[np.int64]
    ) -> Dict[str, float]:
        """Evaluate clustering results.
        
        Args:
            X: Input data matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass


class BaseValidator(BaseEstimator):
    """Base class for validation methods."""
    
    @abstractmethod
    def validate(
        self,
        X: NDArray[np.float64],
        clusterer: BaseClusterer,
        **kwargs
    ) -> Dict[str, Any]:
        """Validate clustering results.
        
        Args:
            X: Input data matrix
            clusterer: Clustering algorithm instance
            **kwargs: Additional validation parameters
            
        Returns:
            Validation results
        """
        pass


class ClusteringResult:
    """Container for clustering results."""
    
    def __init__(
        self,
        labels: NDArray[np.int64],
        log_likelihood: float,
        n_iter: int,
        converged: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize clustering result.
        
        Args:
            labels: Cluster assignments
            log_likelihood: Final log-likelihood value
            n_iter: Number of iterations
            converged: Whether algorithm converged
            metadata: Additional metadata
        """
        self.labels = labels
        self.log_likelihood = log_likelihood
        self.n_iter = n_iter
        self.converged = converged
        self.metadata = metadata or {}
        
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ClusteringResult(n_clusters={len(np.unique(self.labels))}, "
            f"log_likelihood={self.log_likelihood:.4f}, "
            f"n_iter={self.n_iter}, converged={self.converged})"
        )


class EvaluationResult:
    """Container for evaluation results."""
    
    def __init__(
        self,
        log_likelihood: float,
        z_score: float,
        diff_value: float,
        p_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize evaluation result.
        
        Args:
            log_likelihood: Log-likelihood value
            z_score: Z-score statistic
            diff_value: Difference metric
            p_value: P-value if available
            metadata: Additional metadata
        """
        self.log_likelihood = log_likelihood
        self.z_score = z_score
        self.diff_value = diff_value
        self.p_value = p_value
        self.metadata = metadata or {}
        
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EvaluationResult(log_likelihood={self.log_likelihood:.4f}, "
            f"z_score={self.z_score:.4f}, diff_value={self.diff_value:.4f})"
        )