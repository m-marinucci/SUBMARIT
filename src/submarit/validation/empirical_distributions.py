"""Empirical distribution generation for SUBMARIT clustering validation.

This module implements empirical distribution generation for clustering
validation, following the methodology from kSMCreateDist.m. It creates
null distributions for various clustering quality metrics by generating
random clusterings.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from submarit.evaluation.cluster_evaluator import ClusterEvaluator
from submarit.utils.matlab_compat import MatlabRandom


@dataclass
class SwitchingMatrixEmpiricalDistribution:
    """Container for switching matrix empirical distribution results.
    
    Attributes:
        z_dist: Sorted distribution of z-values (ascending)
        ll_dist: Sorted distribution of log-likelihood values (descending)
        diff_dist: Sorted distribution of (PHat - P) values (ascending)
        n_points: Number of points in the distribution
        n_items: Number of items in switching matrix
        n_clusters: Number of clusters
        min_items: Minimum items per cluster constraint
    """
    z_dist: NDArray[np.float64]
    ll_dist: NDArray[np.float64]
    diff_dist: NDArray[np.float64]
    n_points: int
    n_items: int
    n_clusters: int
    min_items: int = 2
    
    def get_percentiles(
        self,
        percentiles: Optional[NDArray[np.float64]] = None
    ) -> Dict[str, NDArray[np.float64]]:
        """Get percentile values from the distributions.
        
        Parameters
        ----------
        percentiles : array-like, optional
            Percentiles to compute (0-100). Defaults to standard values.
            
        Returns
        -------
        dict
            Dictionary with 'z', 'll', and 'diff' percentile values
        """
        if percentiles is None:
            percentiles = np.array([0.5, 2.5, 5, 25, 50, 75, 95, 97.5, 99.5])
            
        indices = np.round(percentiles / 100 * self.n_points).astype(int)
        indices = np.clip(indices, 0, self.n_points - 1)
        
        # Note: ll_dist is sorted descending, others ascending
        return {
            'z': self.z_dist[indices],
            'll': self.ll_dist[indices],
            'diff': self.diff_dist[indices],
            'percentiles': percentiles
        }
        
    def calculate_p_values(
        self,
        z_value: float,
        log_lh: float,
        diff_value: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate p-values for observed statistics.
        
        Parameters
        ----------
        z_value : float
            Observed z-value
        log_lh : float
            Observed log-likelihood
        diff_value : float
            Observed (PHat - P) difference
            
        Returns
        -------
        dict
            Dictionary with p-values: {metric: (upper_p, lower_p)}
        """
        results = {}
        
        # Z-value (ascending, higher is better)
        gr_idx = np.where(self.z_dist > z_value)[0]
        if len(gr_idx) == 0:
            results['z'] = (0.0, 1.0)
        else:
            below = (gr_idx[0] - 1) / self.n_points
            results['z'] = (1 - below, below)
            
        # Log-likelihood (descending, higher is better)
        gr_idx = np.where(self.ll_dist < log_lh)[0]
        if len(gr_idx) == 0:
            results['ll'] = (0.0, 1.0)
        else:
            below = (gr_idx[0] - 1) / self.n_points
            results['ll'] = (1 - below, below)
            
        # Diff value (ascending, higher is better)
        gr_idx = np.where(self.diff_dist > diff_value)[0]
        if len(gr_idx) == 0:
            results['diff'] = (0.0, 1.0)
        else:
            below = (gr_idx[0] - 1) / self.n_points
            results['diff'] = (1 - below, below)
            
        return results


def create_switching_matrix_distribution(
    swm: NDArray[np.float64],
    n_clusters: int,
    n_points: int,
    min_items: int = 2,
    random_state: Optional[int] = None,
    n_jobs: int = 1
) -> SwitchingMatrixEmpiricalDistribution:
    """Create empirical distribution for switching matrix clustering metrics.
    
    This function implements the methodology from kSMCreateDist.m, generating
    random clusterings and computing their evaluation metrics to build
    empirical null distributions.
    
    Parameters
    ----------
    swm : ndarray of shape (n_items, n_items)
        Product x product switching matrix
    n_clusters : int
        Number of clusters
    n_points : int
        Number of points in empirical distribution
    min_items : int, default=2
        Minimum number of items required per cluster
    random_state : int, optional
        Random seed for reproducibility
    n_jobs : int, default=1
        Number of parallel jobs. -1 means using all processors.
        
    Returns
    -------
    distribution : SwitchingMatrixEmpiricalDistribution
        Empirical distribution of clustering metrics
    """
    n_items = swm.shape[0]
    
    # Validate parameters
    if n_items < n_clusters * min_items:
        raise ValueError(
            f"Not enough items ({n_items}) for {n_clusters} clusters "
            f"with minimum {min_items} items each"
        )
        
    # Initialize arrays
    z_dist = np.zeros(n_points)
    ll_dist = np.zeros(n_points)
    diff_dist = np.zeros(n_points)
    
    # Create evaluator
    evaluator = ClusterEvaluator()
    
    if n_jobs == 1:
        # Sequential processing
        rng = MatlabRandom(random_state)
        
        for i_point in range(n_points):
            # Generate random clustering with constraints
            cluster_assign = _generate_constrained_clustering(
                n_items, n_clusters, min_items, rng
            )
            
            # Evaluate clustering
            eval_result = evaluator.evaluate(swm, n_clusters, cluster_assign)
            
            # Store metrics
            z_dist[i_point] = eval_result.z_value
            ll_dist[i_point] = eval_result.log_lh
            diff_dist[i_point] = eval_result.diff
            
    else:
        # Parallel processing
        z_dist, ll_dist, diff_dist = _parallel_distribution_generation(
            swm, n_clusters, n_points, min_items, random_state, n_jobs
        )
        
    # Sort distributions appropriately
    z_dist.sort()  # Ascending (higher is better)
    ll_dist[::-1].sort()  # Descending (higher is better)
    diff_dist.sort()  # Ascending (higher is better)
    
    return SwitchingMatrixEmpiricalDistribution(
        z_dist=z_dist,
        ll_dist=ll_dist,
        diff_dist=diff_dist,
        n_points=n_points,
        n_items=n_items,
        n_clusters=n_clusters,
        min_items=min_items
    )


def _generate_constrained_clustering(
    n_items: int,
    n_clusters: int,
    min_items: int,
    rng: MatlabRandom
) -> NDArray[np.int64]:
    """Generate random clustering with minimum items constraint.
    
    Parameters
    ----------
    n_items : int
        Number of items
    n_clusters : int
        Number of clusters
    min_items : int
        Minimum items per cluster
    rng : MatlabRandom
        Random number generator
        
    Returns
    -------
    clusters : ndarray
        Random cluster assignments (1-based)
    """
    min_item_count = 0
    max_attempts = 1000
    attempts = 0
    
    while min_item_count < min_items and attempts < max_attempts:
        # Generate random assignments (1-based)
        clusters = rng.randi(n_clusters, n_items)
        
        # Check minimum item count
        min_item_count = n_items
        for i in range(1, n_clusters + 1):
            count = np.sum(clusters == i)
            min_item_count = min(count, min_item_count)
            
        attempts += 1
        
    if attempts >= max_attempts:
        # Fall back to deterministic assignment
        warnings.warn(
            f"Could not generate random clustering meeting constraints after "
            f"{max_attempts} attempts. Using deterministic assignment."
        )
        clusters = np.zeros(n_items, dtype=np.int64)
        items_per_cluster = n_items // n_clusters
        for i in range(n_clusters):
            start = i * items_per_cluster
            end = (i + 1) * items_per_cluster if i < n_clusters - 1 else n_items
            clusters[start:end] = i + 1
            
    return clusters


def _parallel_distribution_generation(
    swm: NDArray[np.float64],
    n_clusters: int,
    n_points: int,
    min_items: int,
    random_state: Optional[int],
    n_jobs: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate empirical distribution in parallel.
    
    Returns
    -------
    z_dist, ll_dist, diff_dist : tuple of ndarray
        Arrays of distribution values
    """
    n_workers = n_jobs if n_jobs > 0 else None
    
    # Generate unique seeds for each worker
    if random_state is not None:
        base_rng = np.random.RandomState(random_state)
        seeds = base_rng.randint(0, 2**31, size=n_points)
    else:
        seeds = np.random.randint(0, 2**31, size=n_points)
        
    def compute_sample(seed: int) -> Tuple[float, float, float]:
        """Compute one sample for the distribution."""
        local_rng = MatlabRandom(seed)
        n_items = swm.shape[0]
        
        # Generate random clustering
        cluster_assign = _generate_constrained_clustering(
            n_items, n_clusters, min_items, local_rng
        )
        
        # Evaluate
        evaluator = ClusterEvaluator()
        eval_result = evaluator.evaluate(swm, n_clusters, cluster_assign)
        
        return eval_result.z_value, eval_result.log_lh, eval_result.diff
        
    # Initialize arrays
    z_dist = np.zeros(n_points)
    ll_dist = np.zeros(n_points)
    diff_dist = np.zeros(n_points)
    
    # Execute in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(compute_sample, seeds[i]): i
            for i in range(n_points)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            z_dist[idx], ll_dist[idx], diff_dist[idx] = future.result()
            
    return z_dist, ll_dist, diff_dist


def k_sm_empirical_p(
    emp_dist: SwitchingMatrixEmpiricalDistribution,
    cluster_result: Dict[str, Union[float, NDArray]]
) -> Dict[str, Union[float, Tuple[float, float]]]:
    """Calculate p-values from empirical distribution (MATLAB-compatible).
    
    This function provides a MATLAB-compatible interface matching
    kSMEmpiricalP.m for calculating p-values from empirical distributions.
    
    Parameters
    ----------
    emp_dist : SwitchingMatrixEmpiricalDistribution
        Empirical distribution
    cluster_result : dict
        Clustering result containing 'ZValue', 'LogLH', and 'Diff'
        
    Returns
    -------
    p_struct : dict
        Dictionary containing:
        - 'Zp': [higher_p, lower_p] for z-value
        - 'LLp': [higher_p, lower_p] for log-likelihood
        - 'Diffp': [higher_p, lower_p] for difference
    """
    # Extract values from cluster result
    z_value = cluster_result.get('ZValue', cluster_result.get('z_value'))
    log_lh = cluster_result.get('LogLH', cluster_result.get('log_lh'))
    diff = cluster_result.get('Diff', cluster_result.get('diff'))
    
    if z_value is None or log_lh is None or diff is None:
        raise ValueError(
            "cluster_result must contain ZValue/z_value, LogLH/log_lh, "
            "and Diff/diff"
        )
        
    # Calculate p-values
    p_values = emp_dist.calculate_p_values(z_value, log_lh, diff)
    
    # Format for MATLAB compatibility
    return {
        'Zp': p_values['z'],
        'LLp': p_values['ll'],
        'Diffp': p_values['diff']
    }


def create_bootstrap_distribution(
    swm: NDArray[np.float64],
    cluster_assign: NDArray[np.int64],
    n_clusters: int,
    n_bootstrap: int = 1000,
    sample_fraction: float = 0.8,
    random_state: Optional[int] = None
) -> Dict[str, NDArray[np.float64]]:
    """Create bootstrap distribution for clustering stability.
    
    This function creates a bootstrap distribution by resampling items
    and recalculating clustering metrics, useful for assessing the
    stability of a particular clustering solution.
    
    Parameters
    ----------
    swm : ndarray of shape (n_items, n_items)
        Product x product switching matrix
    cluster_assign : ndarray of shape (n_items,)
        Cluster assignments to evaluate (1-based)
    n_clusters : int
        Number of clusters
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    sample_fraction : float, default=0.8
        Fraction of items to sample in each bootstrap
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    bootstrap_dist : dict
        Dictionary containing bootstrap distributions for:
        - 'z_values': Z-value distribution
        - 'log_lh': Log-likelihood distribution
        - 'diff': (PHat - P) distribution
        - 'cluster_sizes': Cluster size distributions
    """
    n_items = swm.shape[0]
    n_sample = int(n_items * sample_fraction)
    
    # Initialize arrays
    z_values = np.zeros(n_bootstrap)
    log_lh_values = np.zeros(n_bootstrap)
    diff_values = np.zeros(n_bootstrap)
    cluster_sizes = np.zeros((n_bootstrap, n_clusters))
    
    # Random number generator
    rng = np.random.RandomState(random_state)
    evaluator = ClusterEvaluator()
    
    for i in range(n_bootstrap):
        # Sample items with replacement
        sample_idx = rng.choice(n_items, size=n_sample, replace=True)
        sample_idx = np.unique(sample_idx)  # Remove duplicates
        
        # Extract submatrix and assignments
        sub_swm = swm[np.ix_(sample_idx, sample_idx)]
        sub_assign = cluster_assign[sample_idx]
        
        # Renumber clusters if any are missing
        unique_clusters = np.unique(sub_assign)
        if len(unique_clusters) < n_clusters:
            # Create mapping to consecutive clusters
            cluster_map = {old: new for new, old in enumerate(unique_clusters, 1)}
            sub_assign = np.array([cluster_map[c] for c in sub_assign])
            n_clus_sample = len(unique_clusters)
        else:
            n_clus_sample = n_clusters
            
        # Evaluate subsampled clustering
        try:
            eval_result = evaluator.evaluate(sub_swm, n_clus_sample, sub_assign)
            z_values[i] = eval_result.z_value
            log_lh_values[i] = eval_result.log_lh
            diff_values[i] = eval_result.diff
            
            # Store cluster sizes
            for j in range(1, n_clus_sample + 1):
                cluster_sizes[i, j-1] = np.sum(sub_assign == j)
                
        except Exception as e:
            warnings.warn(f"Bootstrap sample {i} failed: {str(e)}")
            z_values[i] = np.nan
            log_lh_values[i] = np.nan
            diff_values[i] = np.nan
            
    # Remove failed samples
    valid_idx = ~np.isnan(z_values)
    
    return {
        'z_values': z_values[valid_idx],
        'log_lh': log_lh_values[valid_idx],
        'diff': diff_values[valid_idx],
        'cluster_sizes': cluster_sizes[valid_idx],
        'n_bootstrap': np.sum(valid_idx),
        'sample_fraction': sample_fraction
    }