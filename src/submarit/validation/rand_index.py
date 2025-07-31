"""Rand index calculations and empirical distributions.

This module implements the Rand index and adjusted Rand index for comparing
clustering solutions, following the methodology from RandIndex4.m. It also
provides functions for creating empirical distributions of Rand indices.
"""

from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ProcessPoolExecutor, as_completed

from submarit.utils.matlab_compat import MatlabRandom


@dataclass
class RandIndexResult:
    """Container for Rand index calculation results.
    
    Attributes:
        rand: Rand index value (0 to 1)
        adj_rand: Adjusted Rand index value
        a: Number of pairs in same cluster in both clusterings
        b: Number of pairs in different clusters in both clusterings
        c: Number of pairs in same cluster in C1, different in C2
        d: Number of pairs in different cluster in C1, same in C2
        n_items: Number of items compared
        n_pairs: Total number of pairs
    """
    rand: float
    adj_rand: float
    a: int
    b: int
    c: int
    d: int
    n_items: int
    n_pairs: int
    
    def summary(self) -> str:
        """Generate summary of Rand index results."""
        return (
            f"Rand Index Results\n"
            f"==================\n"
            f"Rand index: {self.rand:.4f}\n"
            f"Adjusted Rand index: {self.adj_rand:.4f}\n"
            f"Number of items: {self.n_items}\n"
            f"Total pairs: {self.n_pairs}\n"
            f"Contingency table:\n"
            f"  a (agree same): {self.a}\n"
            f"  b (agree different): {self.b}\n"
            f"  c (disagree C1 same): {self.c}\n"
            f"  d (disagree C1 diff): {self.d}"
        )


@dataclass
class RandEmpiricalDistribution:
    """Container for Rand index empirical distribution.
    
    Attributes:
        rand_dist: Sorted array of Rand index values
        adj_rand_dist: Sorted array of adjusted Rand index values
        n_points: Number of points in distribution
        n_items: Number of items in clusterings
        n_clusters: Number of clusters
        min_items: Minimum items per cluster constraint
    """
    rand_dist: NDArray[np.float64]
    adj_rand_dist: NDArray[np.float64]
    n_points: int
    n_items: int
    n_clusters: int
    min_items: int = 1
    
    def get_percentiles(self, percentiles: Optional[NDArray[np.float64]] = None) -> Dict[str, NDArray[np.float64]]:
        """Get percentile values from the distribution.
        
        Parameters
        ----------
        percentiles : array-like, optional
            Percentiles to compute (0-100). Defaults to standard confidence intervals.
            
        Returns
        -------
        dict
            Dictionary with 'rand' and 'adj_rand' percentile values
        """
        if percentiles is None:
            percentiles = np.array([0.5, 2.5, 5, 25, 50, 75, 95, 97.5, 99.5])
            
        indices = np.round(percentiles / 100 * self.n_points).astype(int)
        indices = np.clip(indices, 0, self.n_points - 1)
        
        return {
            'rand': self.rand_dist[indices],
            'adj_rand': self.adj_rand_dist[indices],
            'percentiles': percentiles
        }
        
    def calculate_p_values(self, rand_value: float, adj_rand_value: float) -> Dict[str, Tuple[float, float]]:
        """Calculate p-values for given Rand index values.
        
        Parameters
        ----------
        rand_value : float
            Observed Rand index
        adj_rand_value : float
            Observed adjusted Rand index
            
        Returns
        -------
        dict
            Dictionary with p-values: {'rand': (upper_p, lower_p), 'adj_rand': (upper_p, lower_p)}
        """
        # Find indices with higher values
        rand_gr_idx = np.where(self.rand_dist > rand_value)[0]
        adj_rand_gr_idx = np.where(self.adj_rand_dist > adj_rand_value)[0]
        
        # Calculate p-values
        if len(rand_gr_idx) == 0:
            rand_p = (0.0, 1.0)  # Better than all values
        else:
            below = (rand_gr_idx[0] - 1) / self.n_points
            rand_p = (1 - below, below)
            
        if len(adj_rand_gr_idx) == 0:
            adj_rand_p = (0.0, 1.0)  # Better than all values
        else:
            below = (adj_rand_gr_idx[0] - 1) / self.n_points
            adj_rand_p = (1 - below, below)
            
        return {
            'rand': rand_p,
            'adj_rand': adj_rand_p
        }


class RandIndex:
    """Calculator for Rand index and adjusted Rand index.
    
    The Rand index measures the similarity between two clusterings by considering
    all pairs of items and checking whether they are grouped consistently.
    
    The adjusted Rand index corrects for chance agreement and is preferred
    for comparing clusterings with different numbers of clusters.
    """
    
    def calculate(
        self,
        clusters1: NDArray[np.int64],
        clusters2: NDArray[np.int64]
    ) -> Tuple[float, float]:
        """Calculate Rand index and adjusted Rand index.
        
        This implements the methodology from RandIndex4.m, computing both
        the standard Rand index (Rand, 1971) and the adjusted Rand index
        (Hubert and Arabie, 1985).
        
        Parameters
        ----------
        clusters1 : ndarray of shape (n_items,)
            First clustering assignment (1-based cluster labels)
        clusters2 : ndarray of shape (n_items,)
            Second clustering assignment (1-based cluster labels)
            
        Returns
        -------
        rand : float
            Rand index (0 to 1, where 1 indicates perfect agreement)
        adj_rand : float
            Adjusted Rand index (can be negative, 1 indicates perfect agreement)
        """
        # Validate inputs
        if len(clusters1) != len(clusters2):
            raise ValueError("Clusterings must have the same number of items")
            
        n = len(clusters1)
        if n < 2:
            raise ValueError("At least 2 items required for Rand index")
            
        # Total number of pairs
        N = n * (n - 1) // 2
        
        # Get maximum cluster label to size contingency table
        max_label = max(np.max(clusters1), np.max(clusters2))
        
        # Create indicator matrices (n x max_clusters)
        # CA1[i,j] = 1 if item i is in cluster j+1
        CA1 = np.zeros((n, max_label), dtype=int)
        CA2 = np.zeros((n, max_label), dtype=int)
        
        # Fill indicator matrices (convert to 0-based indexing)
        CA1[np.arange(n), clusters1 - 1] = 1
        CA2[np.arange(n), clusters2 - 1] = 1
        
        # Compute contingency table
        match = CA1.T @ CA2  # Clusters1 x Clusters2 contingency table
        
        # Row and column sums
        row_sums = np.sum(match, axis=1)
        col_sums = np.sum(match, axis=0)
        
        # Calculate intermediate values
        # P: pairs from same cluster in clustering 1
        P = np.sum(row_sums * (row_sums - 1) // 2)
        # Q: pairs from same cluster in clustering 2  
        Q = np.sum(col_sums * (col_sums - 1) // 2)
        # T: pairs in same cluster in both clusterings
        T = (np.sum(match ** 2) - n) // 2
        
        # Rand index: (a + b) / (n choose 2)
        # where a = T (agree same), b = N - P - Q + T (agree different)
        rand = (N + 2 * T - P - Q) / N
        
        # Adjusted Rand index
        if N * (P + Q) - 2 * P * Q == 0:
            # Handle edge case where denominator is 0
            adj_rand = 0.0
        else:
            adj_rand = 2 * (N * T - P * Q) / (N * (P + Q) - 2 * P * Q)
            
        return rand, adj_rand
        
    def calculate_detailed(
        self,
        clusters1: NDArray[np.int64],
        clusters2: NDArray[np.int64]
    ) -> RandIndexResult:
        """Calculate Rand index with detailed contingency information.
        
        Parameters
        ----------
        clusters1 : ndarray of shape (n_items,)
            First clustering assignment (1-based cluster labels)
        clusters2 : ndarray of shape (n_items,)
            Second clustering assignment (1-based cluster labels)
            
        Returns
        -------
        result : RandIndexResult
            Detailed results including contingency table counts
        """
        n = len(clusters1)
        N = n * (n - 1) // 2
        
        # Get basic Rand indices
        rand, adj_rand = self.calculate(clusters1, clusters2)
        
        # Calculate detailed contingency counts
        # a: pairs in same cluster in both
        # b: pairs in different clusters in both
        # c: pairs in same cluster in C1, different in C2
        # d: pairs in different cluster in C1, same in C2
        
        a = 0  # agree same
        b = 0  # agree different
        c = 0  # disagree (C1 same, C2 diff)
        d = 0  # disagree (C1 diff, C2 same)
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                same_c1 = clusters1[i] == clusters1[j]
                same_c2 = clusters2[i] == clusters2[j]
                
                if same_c1 and same_c2:
                    a += 1
                elif not same_c1 and not same_c2:
                    b += 1
                elif same_c1 and not same_c2:
                    c += 1
                else:  # not same_c1 and same_c2
                    d += 1
                    
        return RandIndexResult(
            rand=rand,
            adj_rand=adj_rand,
            a=a,
            b=b,
            c=c,
            d=d,
            n_items=n,
            n_pairs=N
        )


def create_rand_empirical_distribution(
    n_items: int,
    n_clusters: int,
    n_points: int,
    min_items: int = 1,
    random_state: Optional[int] = None,
    n_jobs: int = 1
) -> RandEmpiricalDistribution:
    """Create empirical distribution for Rand index under random clustering.
    
    This function implements the methodology from RandCreateDist.m, generating
    random clustering pairs and computing their Rand indices to build an
    empirical null distribution.
    
    Parameters
    ----------
    n_items : int
        Number of items to cluster
    n_clusters : int
        Number of clusters
    n_points : int
        Number of points in empirical distribution
    min_items : int, default=1
        Minimum number of items required per cluster
    random_state : int, optional
        Random seed for reproducibility
    n_jobs : int, default=1
        Number of parallel jobs. -1 means using all processors.
        
    Returns
    -------
    distribution : RandEmpiricalDistribution
        Empirical distribution of Rand indices
    """
    # Initialize arrays
    rand_dist = np.zeros(n_points)
    adj_rand_dist = np.zeros(n_points)
    
    # Random number generator
    rng = MatlabRandom(random_state)
    rand_calc = RandIndex()
    
    if n_jobs == 1:
        # Sequential processing
        for i_point in range(n_points):
            # Generate two random clusterings with min_items constraint
            clusters1 = _generate_random_clustering(n_items, n_clusters, min_items, rng)
            clusters2 = _generate_random_clustering(n_items, n_clusters, min_items, rng)
            
            # Calculate Rand indices
            rand_dist[i_point], adj_rand_dist[i_point] = rand_calc.calculate(
                clusters1, clusters2
            )
    else:
        # Parallel processing
        n_workers = n_jobs if n_jobs > 0 else None
        
        def compute_sample(seed: int) -> Tuple[float, float]:
            """Compute one sample for the distribution."""
            local_rng = MatlabRandom(seed)
            clusters1 = _generate_random_clustering(n_items, n_clusters, min_items, local_rng)
            clusters2 = _generate_random_clustering(n_items, n_clusters, min_items, local_rng)
            return rand_calc.calculate(clusters1, clusters2)
            
        # Generate unique seeds for each worker
        if random_state is not None:
            base_rng = np.random.RandomState(random_state)
            seeds = base_rng.randint(0, 2**31, size=n_points)
        else:
            seeds = np.random.randint(0, 2**31, size=n_points)
            
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(compute_sample, seeds[i]): i
                for i in range(n_points)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                rand_dist[idx], adj_rand_dist[idx] = future.result()
                
    # Sort distributions
    rand_dist.sort()
    adj_rand_dist.sort()
    
    return RandEmpiricalDistribution(
        rand_dist=rand_dist,
        adj_rand_dist=adj_rand_dist,
        n_points=n_points,
        n_items=n_items,
        n_clusters=n_clusters,
        min_items=min_items
    )


def _generate_random_clustering(
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
    
    while min_item_count < min_items:
        # Generate random assignments (1-based)
        clusters = rng.randi(n_clusters, n_items)
        
        # Check minimum item count
        min_item_count = n_items
        for i in range(1, n_clusters + 1):
            count = np.sum(clusters == i)
            min_item_count = min(count, min_item_count)
            
    return clusters


def rand_empirical_p(
    rand_dist: NDArray[np.float64],
    adj_rand_dist: NDArray[np.float64],
    rand: float,
    adj_rand: float,
    create_confidence: bool = True
) -> Dict[str, Union[float, Tuple[float, float], NDArray[np.float64]]]:
    """Calculate p-values from empirical distribution.
    
    This implements the functionality from RandEmpiricalP.m, computing
    p-values and confidence intervals from empirical distributions.
    
    Parameters
    ----------
    rand_dist : ndarray
        Sorted empirical distribution of Rand indices
    adj_rand_dist : ndarray
        Sorted empirical distribution of adjusted Rand indices
    rand : float
        Observed Rand index
    adj_rand : float
        Observed adjusted Rand index
    create_confidence : bool, default=True
        Whether to compute confidence interval values
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'rand': Observed Rand index
        - 'adj_rand': Observed adjusted Rand index  
        - 'rand_p': (upper_p, lower_p) tuple for Rand index
        - 'adj_rand_p': (upper_p, lower_p) tuple for adjusted Rand
        - 'rand_conf': Confidence interval values (if requested)
        - 'adj_rand_conf': Adjusted Rand CI values (if requested)
    """
    n_points = len(rand_dist)
    result = {
        'rand': rand,
        'adj_rand': adj_rand
    }
    
    # Calculate p-values for Rand index
    gr_indexes = np.where(rand_dist > rand)[0]
    if len(gr_indexes) == 0:
        result['rand_p'] = (0.0, 1.0)  # Better than all values
    else:
        below = (gr_indexes[0] - 1) / n_points
        result['rand_p'] = (1 - below, below)
        
    # Calculate p-values for adjusted Rand index  
    gr_indexes = np.where(adj_rand_dist > adj_rand)[0]
    if len(gr_indexes) == 0:
        result['adj_rand_p'] = (0.0, 1.0)  # Better than all values
    else:
        below = (gr_indexes[0] - 1) / n_points
        result['adj_rand_p'] = (1 - below, below)
        
    # Create confidence interval values if requested
    if create_confidence:
        # Standard confidence levels
        conf_levels = np.array([0.005, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.995])
        indices = np.round(conf_levels * n_points).astype(int)
        indices = np.clip(indices, 0, n_points - 1)
        
        result['rand_conf'] = rand_dist[indices]
        result['adj_rand_conf'] = adj_rand_dist[indices]
        result['conf_levels'] = conf_levels
        
    return result