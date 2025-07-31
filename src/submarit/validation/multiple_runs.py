"""Multiple runs of clustering algorithms for robustness.

This module provides wrappers to run SUBMARIT clustering algorithms multiple
times and select the best result, implementing the functionality from
RunClusters.m and RunClustersConstrained.m.
"""

from typing import Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray
import warnings

from submarit.algorithms.local_search import (
    LocalSearchResult,
    KSMLocalSearch,
    KSMLocalSearch2,
    KSMLocalSearchConstrained,
    KSMLocalSearchConstrained2
)
from submarit.evaluation.cluster_evaluator import ClusterEvaluator


def run_clusters(
    swm: NDArray[np.float64],
    n_clusters: int,
    min_items: int = 1,
    n_runs: int = 10,
    max_iter: int = 100,
    random_state: Optional[int] = None,
    algorithm: str = 'v1',
    return_all: bool = False
) -> Union[LocalSearchResult, List[LocalSearchResult]]:
    """Run SUBMARIT clustering multiple times and return best result.
    
    This function implements the functionality of RunClusters.m and
    RunClusters2.m, running the clustering algorithm multiple times
    with different random initializations and returning the best result
    based on the objective function.
    
    Parameters
    ----------
    swm : ndarray of shape (n_items, n_items)
        Product x product switching matrix
    n_clusters : int
        Number of clusters
    min_items : int, default=1
        Minimum number of items per cluster
    n_runs : int, default=10
        Number of runs with different random initializations
    max_iter : int, default=100
        Maximum iterations per run
    random_state : int, optional
        Random seed for reproducibility. If provided, each run
        uses sequential seeds starting from this value.
    algorithm : {'v1', 'v2'}, default='v1'
        Which algorithm version to use:
        - 'v1': KSMLocalSearch (PHat-P optimization)
        - 'v2': KSMLocalSearch2 (log-likelihood optimization)
    return_all : bool, default=False
        If True, return all results; if False, return only best
        
    Returns
    -------
    result : LocalSearchResult or list of LocalSearchResult
        Best clustering result (or all results if return_all=True)
    """
    if algorithm not in ['v1', 'v2']:
        raise ValueError("algorithm must be 'v1' or 'v2'")
        
    # Select algorithm class
    if algorithm == 'v1':
        clusterer_class = KSMLocalSearch
    else:
        clusterer_class = KSMLocalSearch2
        
    # Run clustering multiple times
    results = []
    best_result = None
    best_objective = -np.inf if algorithm == 'v1' else np.inf
    
    for i in range(n_runs):
        # Set random seed for this run
        if random_state is not None:
            run_seed = random_state + i
        else:
            run_seed = None
            
        # Create and run clusterer
        clusterer = clusterer_class(
            n_clusters=n_clusters,
            min_items=min_items,
            max_iter=max_iter,
            random_state=run_seed
        )
        
        try:
            clusterer.fit(swm)
            result = clusterer.get_result()
            results.append(result)
            
            # Check if this is the best result
            if algorithm == 'v1':
                # Maximize Diff (PHat - P)
                if result.Diff > best_objective:
                    best_objective = result.Diff
                    best_result = result
            else:
                # Maximize LogLH (minimize negative LogLH)
                if result.LogLH > best_objective:
                    best_objective = result.LogLH
                    best_result = result
                    
        except Exception as e:
            warnings.warn(f"Run {i+1} failed: {str(e)}")
            continue
            
    if best_result is None:
        raise RuntimeError("All clustering runs failed")
        
    if return_all:
        return results
    else:
        return best_result


def run_clusters_constrained(
    swm: NDArray[np.float64],
    n_clusters: int,
    min_items: int,
    fixed_assign: NDArray[np.int64],
    assign_indexes: NDArray[np.int64],
    free_indexes: NDArray[np.int64],
    n_runs: int = 10,
    max_iter: int = 100,
    random_state: Optional[int] = None,
    algorithm: str = 'v1',
    return_all: bool = False
) -> Union[LocalSearchResult, List[LocalSearchResult]]:
    """Run constrained SUBMARIT clustering multiple times.
    
    This function implements the functionality of RunClustersConstrained.m
    and RunClustersConstrained2.m, running the constrained clustering
    algorithm multiple times with different random initializations for
    the free items.
    
    Parameters
    ----------
    swm : ndarray of shape (n_items, n_items)
        Product x product switching matrix
    n_clusters : int
        Number of clusters
    min_items : int
        Minimum number of items per cluster
    fixed_assign : ndarray of shape (n_fixed,)
        Cluster assignments for fixed items (1-based)
    assign_indexes : ndarray of shape (n_fixed,)
        Indices of fixed items (1-based)
    free_indexes : ndarray of shape (n_free,)
        Indices of free items that can be reassigned (1-based)
    n_runs : int, default=10
        Number of runs with different random initializations
    max_iter : int, default=100
        Maximum iterations per run
    random_state : int, optional
        Random seed for reproducibility
    algorithm : {'v1', 'v2'}, default='v1'
        Which algorithm version to use
    return_all : bool, default=False
        If True, return all results; if False, return only best
        
    Returns
    -------
    result : LocalSearchResult or list of LocalSearchResult
        Best clustering result (or all results if return_all=True)
    """
    if algorithm not in ['v1', 'v2']:
        raise ValueError("algorithm must be 'v1' or 'v2'")
        
    # Select algorithm class
    if algorithm == 'v1':
        clusterer_class = KSMLocalSearchConstrained
    else:
        clusterer_class = KSMLocalSearchConstrained2
        
    # Run constrained clustering multiple times
    results = []
    best_result = None
    best_objective = -np.inf if algorithm == 'v1' else np.inf
    
    for i in range(n_runs):
        # Set random seed for this run
        if random_state is not None:
            run_seed = random_state + i
        else:
            run_seed = None
            
        # Create and run constrained clusterer
        clusterer = clusterer_class(
            n_clusters=n_clusters,
            min_items=min_items,
            max_iter=max_iter,
            random_state=run_seed
        )
        
        try:
            clusterer.fit_constrained(
                swm, fixed_assign, assign_indexes, free_indexes
            )
            result = clusterer.get_result()
            results.append(result)
            
            # Check if this is the best result
            if algorithm == 'v1':
                # Maximize Diff
                if result.Diff > best_objective:
                    best_objective = result.Diff
                    best_result = result
            else:
                # Maximize LogLH
                if result.LogLH > best_objective:
                    best_objective = result.LogLH
                    best_result = result
                    
        except Exception as e:
            warnings.warn(f"Run {i+1} failed: {str(e)}")
            continue
            
    if best_result is None:
        raise RuntimeError("All constrained clustering runs failed")
        
    if return_all:
        return results
    else:
        return best_result


def compare_multiple_runs(
    results: List[LocalSearchResult],
    criterion: str = 'diff'
) -> Dict[str, Union[float, int, LocalSearchResult]]:
    """Compare results from multiple clustering runs.
    
    Parameters
    ----------
    results : list of LocalSearchResult
        Results from multiple runs
    criterion : {'diff', 'log_lh', 'z_value'}, default='diff'
        Criterion for comparison
        
    Returns
    -------
    comparison : dict
        Dictionary containing:
        - 'best_idx': Index of best result
        - 'best_value': Best objective value
        - 'best_result': Best LocalSearchResult
        - 'mean': Mean of criterion across runs
        - 'std': Standard deviation of criterion
        - 'values': Array of all criterion values
    """
    if not results:
        raise ValueError("No results to compare")
        
    # Extract criterion values
    if criterion == 'diff':
        values = np.array([r.Diff for r in results])
        maximize = True
    elif criterion == 'log_lh':
        values = np.array([r.LogLH for r in results])
        maximize = True
    elif criterion == 'z_value':
        values = np.array([r.ZValue for r in results])
        maximize = True
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
        
    # Find best
    if maximize:
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)
        
    return {
        'best_idx': best_idx,
        'best_value': values[best_idx],
        'best_result': results[best_idx],
        'mean': np.mean(values),
        'std': np.std(values),
        'values': values
    }


def evaluate_clustering_stability(
    swm: NDArray[np.float64],
    n_clusters: int,
    min_items: int = 1,
    n_runs: int = 100,
    random_state: Optional[int] = None,
    algorithm: str = 'v1'
) -> Dict[str, Union[float, NDArray]]:
    """Evaluate clustering stability across multiple runs.
    
    This function runs the clustering algorithm many times and analyzes
    the stability of the results, including the distribution of objective
    values and cluster assignments.
    
    Parameters
    ----------
    swm : ndarray of shape (n_items, n_items)
        Product x product switching matrix
    n_clusters : int
        Number of clusters
    min_items : int, default=1
        Minimum number of items per cluster
    n_runs : int, default=100
        Number of runs for stability analysis
    random_state : int, optional
        Random seed for reproducibility
    algorithm : {'v1', 'v2'}, default='v1'
        Which algorithm version to use
        
    Returns
    -------
    stability : dict
        Dictionary containing:
        - 'objective_mean': Mean objective value
        - 'objective_std': Standard deviation of objective
        - 'objective_values': All objective values
        - 'consensus_matrix': Item x item matrix of co-clustering frequency
        - 'item_stability': Stability score for each item
        - 'overall_stability': Overall clustering stability (0-1)
    """
    # Run clustering multiple times
    results = run_clusters(
        swm, n_clusters, min_items, n_runs,
        random_state=random_state,
        algorithm=algorithm,
        return_all=True
    )
    
    n_items = swm.shape[0]
    
    # Extract objective values
    if algorithm == 'v1':
        objective_values = np.array([r.Diff for r in results])
    else:
        objective_values = np.array([r.LogLH for r in results])
        
    # Build consensus matrix
    consensus_matrix = np.zeros((n_items, n_items))
    
    for result in results:
        assignments = result.Assign
        for i in range(n_items):
            for j in range(i + 1, n_items):
                if assignments[i] == assignments[j]:
                    consensus_matrix[i, j] += 1
                    consensus_matrix[j, i] += 1
                    
    # Normalize consensus matrix
    consensus_matrix /= n_runs
    
    # Calculate item stability (how consistently each item is clustered)
    item_stability = np.zeros(n_items)
    for i in range(n_items):
        # Get co-clustering frequencies for this item
        co_cluster_freq = consensus_matrix[i, :]
        # Stability is high if frequencies are close to 0 or 1
        item_stability[i] = np.mean(np.minimum(co_cluster_freq, 1 - co_cluster_freq)) * 2
        
    # Overall stability
    overall_stability = 1 - np.mean(item_stability)
    
    return {
        'objective_mean': np.mean(objective_values),
        'objective_std': np.std(objective_values),
        'objective_values': objective_values,
        'consensus_matrix': consensus_matrix,
        'item_stability': item_stability,
        'overall_stability': overall_stability,
        'n_runs': n_runs
    }