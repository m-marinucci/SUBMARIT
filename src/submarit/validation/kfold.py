"""K-fold cross-validation for SUBMARIT clustering.

This module implements k-fold cross-validation for SUBMARIT clustering algorithms,
following the methodology from kSMNFold.m and kSMNFold2.m. It supports:

1. Standard k-fold validation with train/test splitting
2. Empirical distribution generation for Rand index
3. Constrained clustering for test set assignment
4. Multiple clustering algorithm support
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from submarit.core.base import BaseValidator
from submarit.algorithms.local_search import (
    KSMLocalSearch, KSMLocalSearch2,
    KSMLocalSearchConstrained, KSMLocalSearchConstrained2
)
from submarit.validation.rand_index import RandIndex
from submarit.validation.multiple_runs import run_clusters, run_clusters_constrained


@dataclass
class KFoldResult:
    """Container for k-fold validation results.
    
    Attributes:
        av_rand: Average Rand index across all fold pairs
        av_adj_rand: Average adjusted Rand index across all fold pairs
        av_rand_dist: Empirical distribution for average Rand index (if computed)
        av_adj_rand_dist: Empirical distribution for average adjusted Rand index
        fold_results: List of clustering results for each fold
        rand_pairs: Rand index values for each fold pair
        adj_rand_pairs: Adjusted Rand index values for each fold pair
        n_folds: Number of folds used
        n_random: Number of random permutations for empirical distribution
    """
    av_rand: float
    av_adj_rand: float
    av_rand_dist: Optional[NDArray[np.float64]] = None
    av_adj_rand_dist: Optional[NDArray[np.float64]] = None
    fold_results: Optional[List[Dict]] = None
    rand_pairs: Optional[List[Tuple[int, int, float]]] = None
    adj_rand_pairs: Optional[List[Tuple[int, int, float]]] = None
    n_folds: int = 0
    n_random: int = 0
    
    def summary(self) -> str:
        """Generate summary of k-fold validation results."""
        lines = [
            "K-Fold Cross-Validation Results",
            "=" * 40,
            f"Number of folds: {self.n_folds}",
            f"Average Rand index: {self.av_rand:.4f}",
            f"Average adjusted Rand index: {self.av_adj_rand:.4f}",
        ]
        
        if self.av_rand_dist is not None:
            lines.extend([
                "",
                "Empirical distribution statistics:",
                f"  Random permutations: {self.n_random}",
                f"  Rand index 95% CI: [{np.percentile(self.av_rand_dist, 2.5):.4f}, "
                f"{np.percentile(self.av_rand_dist, 97.5):.4f}]",
                f"  Adjusted Rand 95% CI: [{np.percentile(self.av_adj_rand_dist, 2.5):.4f}, "
                f"{np.percentile(self.av_adj_rand_dist, 97.5):.4f}]",
            ])
            
        return "\n".join(lines)


class KFoldValidator(BaseValidator):
    """K-fold cross-validation for SUBMARIT clustering.
    
    This validator implements the k-fold validation procedure from kSMNFold.m,
    where data is split into k folds, each fold is held out as test data,
    and the model is trained on the remaining k-1 folds. Test items are then
    assigned using constrained clustering.
    
    Parameters
    ----------
    n_folds : int, default=5
        Number of folds for cross-validation
    n_random : int, default=0
        Number of random permutations for empirical distribution.
        If 0, no empirical distribution is computed.
    max_fold_run : int, optional
        Maximum number of folds to actually run (defaults to n_folds).
        Useful for testing with subset of folds.
    random_state : int, optional
        Random seed for reproducibility
    n_jobs : int, default=1
        Number of parallel jobs for empirical distribution generation.
        -1 means using all processors.
    algorithm : {'v1', 'v2'}, default='v1'
        Which clustering algorithm to use:
        - 'v1': Uses KSMLocalSearch (PHat-P optimization)
        - 'v2': Uses KSMLocalSearch2 (log-likelihood optimization)
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        n_random: int = 0,
        max_fold_run: Optional[int] = None,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        algorithm: str = 'v1'
    ):
        """Initialize the k-fold validator."""
        super().__init__()
        self.n_folds = n_folds
        self.n_random = n_random
        self.max_fold_run = max_fold_run or n_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        
        # Validate parameters
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.max_fold_run > self.n_folds:
            raise ValueError("max_fold_run cannot exceed n_folds")
        if self.algorithm not in ['v1', 'v2']:
            raise ValueError("algorithm must be 'v1' or 'v2'")
            
    def validate(
        self,
        swm: NDArray[np.float64],
        n_clusters: int,
        min_items: int = 1,
        n_runs: int = 10,
        **kwargs
    ) -> KFoldResult:
        """Perform k-fold cross-validation on switching matrix.
        
        Parameters
        ----------
        swm : ndarray of shape (n_items, n_items)
            Product x product switching matrix
        n_clusters : int
            Number of clusters
        min_items : int, default=1
            Minimum number of items per cluster
        n_runs : int, default=10
            Number of SUBMARIT runs for each optimization
        **kwargs : dict
            Additional parameters passed to clustering algorithm
            
        Returns
        -------
        result : KFoldResult
            Cross-validation results including average Rand indices
            and optional empirical distributions
        """
        n_items = swm.shape[0]
        
        # Generate random permutation of items
        rng = np.random.RandomState(self.random_state)
        col_perm = rng.permutation(n_items)
        
        # Storage for fold results
        fold_clusters = []
        fold_assignments = []
        random_assignments = [] if self.n_random > 0 else None
        
        # Process each fold
        for i in range(self.max_fold_run):
            # Calculate fold boundaries
            start = int(np.round(i * n_items / self.n_folds))
            end = int(np.round((i + 1) * n_items / self.n_folds))
            
            # Split into test and train indices
            test_indexes = col_perm[start:end]
            train_indexes = np.setdiff1d(np.arange(n_items), test_indexes)
            
            # Check for zero rows/columns after removing test items
            sub_swm = swm[np.ix_(train_indexes, train_indexes)]
            col_sum = np.sum(sub_swm, axis=0)
            row_sum = np.sum(sub_swm, axis=1)
            
            # Find items that would create zero rows/columns
            chk_items = (col_sum == 0) | (row_sum == 0)
            rem_indexes = train_indexes[chk_items]
            
            # Move problematic items to test set
            if len(rem_indexes) > 0:
                test_indexes = np.sort(np.concatenate([test_indexes, rem_indexes]))
                train_indexes = np.setdiff1d(train_indexes, rem_indexes)
                sub_swm = swm[np.ix_(train_indexes, train_indexes)]
                
            # Run clustering on training data
            train_result = run_clusters(
                sub_swm, n_clusters, min_items, n_runs,
                random_state=self.random_state,
                algorithm=self.algorithm
            )
            
            # Assign test items using constrained algorithm
            # Need to convert train assignments to full array
            full_assign = np.zeros(n_items, dtype=np.int64)
            full_assign[train_indexes] = train_result.Assign
            
            # Run constrained clustering
            fold_result = run_clusters_constrained(
                swm, n_clusters, min_items,
                full_assign[train_indexes],
                train_indexes + 1,  # Convert to 1-based
                test_indexes + 1,   # Convert to 1-based
                n_runs,
                random_state=self.random_state,
                algorithm=self.algorithm
            )
            
            fold_clusters.append(fold_result)
            fold_assignments.append(fold_result.Assign)
            
            # Generate random assignments for empirical distribution
            if self.n_random > 0:
                random_fold = []
                temp_col = np.zeros(n_items, dtype=np.int64)
                temp_col[train_indexes] = train_result.Assign
                
                for j in range(self.n_random):
                    # Random assignment for test items
                    temp_col[test_indexes] = rng.randint(1, n_clusters + 1, 
                                                         size=len(test_indexes))
                    random_fold.append(temp_col.copy())
                    
                random_assignments.append(np.column_stack(random_fold))
                
        # Calculate Rand indices for all fold pairs
        rand_calc = RandIndex()
        sum_rand = 0.0
        sum_adj_rand = 0.0
        rand_pairs = []
        adj_rand_pairs = []
        
        # Calculate pairwise Rand indices
        n_pairs = 0
        for i in range(self.max_fold_run - 1):
            for j in range(i + 1, self.max_fold_run):
                rand, adj_rand = rand_calc.calculate(
                    fold_assignments[i],
                    fold_assignments[j]
                )
                sum_rand += rand
                sum_adj_rand += adj_rand
                n_pairs += 1
                
                rand_pairs.append((i, j, rand))
                adj_rand_pairs.append((i, j, adj_rand))
                
        # Calculate averages
        av_rand = sum_rand / n_pairs if n_pairs > 0 else 0.0
        av_adj_rand = sum_adj_rand / n_pairs if n_pairs > 0 else 0.0
        
        # Generate empirical distribution if requested
        av_rand_dist = None
        av_adj_rand_dist = None
        
        if self.n_random > 0:
            av_rand_dist, av_adj_rand_dist = self._generate_empirical_distribution(
                random_assignments, rand_calc
            )
            
        return KFoldResult(
            av_rand=av_rand,
            av_adj_rand=av_adj_rand,
            av_rand_dist=av_rand_dist,
            av_adj_rand_dist=av_adj_rand_dist,
            fold_results=fold_clusters,
            rand_pairs=rand_pairs,
            adj_rand_pairs=adj_rand_pairs,
            n_folds=self.n_folds,
            n_random=self.n_random
        )
        
    def _generate_empirical_distribution(
        self,
        random_assignments: List[NDArray[np.int64]],
        rand_calc: RandIndex
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Generate empirical distribution for random assignments.
        
        Parameters
        ----------
        random_assignments : list of ndarray
            Random cluster assignments for each fold
        rand_calc : RandIndex
            Rand index calculator
            
        Returns
        -------
        av_rand_dist : ndarray
            Sorted distribution of average Rand indices
        av_adj_rand_dist : ndarray
            Sorted distribution of average adjusted Rand indices
        """
        n_random = random_assignments[0].shape[1]
        av_rand_dist = np.zeros(n_random)
        av_adj_rand_dist = np.zeros(n_random)
        
        # Process in parallel if requested
        if self.n_jobs != 1:
            av_rand_dist, av_adj_rand_dist = self._parallel_empirical_dist(
                random_assignments, rand_calc
            )
        else:
            # Sequential processing
            for r_count in range(n_random):
                sum_rand = 0.0
                sum_adj_rand = 0.0
                n_pairs = 0
                
                for i in range(self.max_fold_run - 1):
                    for j in range(i + 1, self.max_fold_run):
                        rand, adj_rand = rand_calc.calculate(
                            random_assignments[i][:, r_count],
                            random_assignments[j][:, r_count]
                        )
                        sum_rand += rand
                        sum_adj_rand += adj_rand
                        n_pairs += 1
                        
                av_rand_dist[r_count] = sum_rand / n_pairs
                av_adj_rand_dist[r_count] = sum_adj_rand / n_pairs
                
        # Sort distributions
        return np.sort(av_rand_dist), np.sort(av_adj_rand_dist)
        
    def _parallel_empirical_dist(
        self,
        random_assignments: List[NDArray[np.int64]],
        rand_calc: RandIndex
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Generate empirical distribution in parallel."""
        n_random = random_assignments[0].shape[1]
        av_rand_dist = np.zeros(n_random)
        av_adj_rand_dist = np.zeros(n_random)
        
        # Determine number of workers
        n_workers = self.n_jobs
        if n_workers == -1:
            n_workers = None  # Use all available CPUs
            
        def compute_random_sample(r_idx: int) -> Tuple[float, float]:
            """Compute Rand indices for one random sample."""
            sum_rand = 0.0
            sum_adj_rand = 0.0
            n_pairs = 0
            
            for i in range(self.max_fold_run - 1):
                for j in range(i + 1, self.max_fold_run):
                    rand, adj_rand = rand_calc.calculate(
                        random_assignments[i][:, r_idx],
                        random_assignments[j][:, r_idx]
                    )
                    sum_rand += rand
                    sum_adj_rand += adj_rand
                    n_pairs += 1
                    
            return sum_rand / n_pairs, sum_adj_rand / n_pairs
            
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(compute_random_sample, i): i
                for i in range(n_random)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                av_rand, av_adj_rand = future.result()
                av_rand_dist[idx] = av_rand
                av_adj_rand_dist[idx] = av_adj_rand
                
        return np.sort(av_rand_dist), np.sort(av_adj_rand_dist)


def k_fold_validate(
    swm: NDArray[np.float64],
    n_clusters: int,
    min_items: int = 1,
    n_runs: int = 10,
    n_folds: int = 5,
    n_random: int = 0,
    max_fold_run: Optional[int] = None,
    random_state: Optional[int] = None,
    algorithm: str = 'v1'
) -> KFoldResult:
    """Convenience function for k-fold validation (MATLAB-compatible interface).
    
    Parameters
    ----------
    swm : ndarray of shape (n_items, n_items)
        Product x product switching matrix
    n_clusters : int
        Number of clusters
    min_items : int, default=1
        Minimum number of items per cluster
    n_runs : int, default=10
        Number of SUBMARIT runs for each optimization
    n_folds : int, default=5
        Number of folds
    n_random : int, default=0
        Number of random permutations for empirical distribution
    max_fold_run : int, optional
        Maximum number of folds to run
    random_state : int, optional
        Random seed
    algorithm : {'v1', 'v2'}, default='v1'
        Which algorithm version to use
        
    Returns
    -------
    result : KFoldResult
        Cross-validation results
    """
    validator = KFoldValidator(
        n_folds=n_folds,
        n_random=n_random,
        max_fold_run=max_fold_run,
        random_state=random_state,
        algorithm=algorithm
    )
    
    return validator.validate(swm, n_clusters, min_items, n_runs)