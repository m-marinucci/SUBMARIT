"""Top-k clustering solution analysis.

This module implements functionality from RunClustersTopk.m and RunClustersTopk2.m
for analyzing agreement among top-k clustering solutions.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from submarit.algorithms.local_search import LocalSearchResult
from submarit.validation.multiple_runs import run_clusters
from submarit.validation.rand_index import RandIndex, create_rand_empirical_distribution, rand_empirical_p
from submarit.utils.matlab_compat import MatlabRandom


@dataclass
class TopKResult:
    """Container for top-k clustering analysis results.
    
    Attributes:
        best_solutions: List of top-k best clustering solutions
        avg_rand: Average Rand index among top-k solutions
        avg_adj_rand: Average adjusted Rand index among top-k solutions
        rand_p: P-value for Rand index
        adj_rand_p: P-value for adjusted Rand index
        rand_conf: Confidence interval values for Rand index
        adj_rand_conf: Confidence interval values for adjusted Rand
        pairwise_rand: Matrix of pairwise Rand indices
        pairwise_adj_rand: Matrix of pairwise adjusted Rand indices
        empirical_dist: Empirical distribution used for p-values
    """
    best_solutions: List[LocalSearchResult]
    avg_rand: float
    avg_adj_rand: float
    rand_p: Tuple[float, float]
    adj_rand_p: Tuple[float, float]
    rand_conf: Optional[NDArray[np.float64]] = None
    adj_rand_conf: Optional[NDArray[np.float64]] = None
    pairwise_rand: Optional[NDArray[np.float64]] = None
    pairwise_adj_rand: Optional[NDArray[np.float64]] = None
    empirical_dist: Optional[Dict] = None
    
    def summary(self) -> str:
        """Generate summary of top-k analysis results."""
        k = len(self.best_solutions)
        return (
            f"Top-{k} Clustering Analysis\n"
            f"===========================\n"
            f"Average Rand index: {self.avg_rand:.4f}\n"
            f"Average adjusted Rand index: {self.avg_adj_rand:.4f}\n"
            f"Rand p-value: {self.rand_p[0]:.4f}\n"
            f"Adjusted Rand p-value: {self.adj_rand_p[0]:.4f}\n"
            f"\nBest solution statistics:\n"
            f"Log-likelihood: {self.best_solutions[0].LogLH:.4f}\n"
            f"Diff value: {self.best_solutions[0].Diff:.4f}\n"
            f"Z-value: {self.best_solutions[0].ZValue:.4f}"
        )


def run_clusters_topk(
    swm: NDArray[np.float64],
    n_clusters: int,
    min_items: int,
    n_runs: int,
    top_k: int,
    n_random: int = 10000,
    algorithm: str = 'v1',
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> TopKResult:
    """Run multiple clusterings and analyze agreement among top-k solutions.
    
    This function implements the functionality of RunClustersTopk.m and
    RunClustersTopk2.m, finding the top-k best clustering solutions and
    analyzing their agreement using Rand indices.
    
    Parameters
    ----------
    swm : ndarray of shape (n_items, n_items)
        Product x product switching matrix
    n_clusters : int
        Number of clusters
    min_items : int
        Minimum number of items per cluster
    n_runs : int
        Number of experimental runs
    top_k : int
        Number of top solutions to return
    n_random : int, default=10000
        Number of random values for empirical distributions
    algorithm : {'v1', 'v2'}, default='v1'
        Which algorithm version to use:
        - 'v1': KSMLocalSearch (PHat-P optimization)
        - 'v2': KSMLocalSearch2 (log-likelihood optimization)
    random_state : int, optional
        Random seed for reproducibility
    n_jobs : int, default=1
        Number of parallel jobs (-1 for all processors)
    verbose : bool, default=False
        Whether to print progress messages
        
    Returns
    -------
    result : TopKResult
        Analysis results including p-values and confidence intervals
    """
    n_items = swm.shape[0]
    
    if verbose:
        print(f"Running {n_runs} clustering attempts to find top-{top_k} solutions...")
    
    # Initialize storage for best solutions
    best_ll = np.full(top_k, 1e7 if algorithm == 'v2' else -1e7)
    best_solutions = [None] * top_k
    
    # Run clustering multiple times
    for i in range(n_runs):
        if verbose and (i + 1) % max(1, n_runs // 10) == 0:
            print(f"  Progress: {i + 1}/{n_runs} runs completed")
            
        # Set random seed for this run
        if random_state is not None:
            run_seed = random_state + i
        else:
            run_seed = None
            
        try:
            # Run clustering
            result = run_clusters(
                swm, n_clusters, min_items, 
                n_runs=1,  # Single run
                random_state=run_seed,
                algorithm=algorithm
            )
            
            # Get objective value
            if algorithm == 'v1':
                obj_value = result.Diff  # Maximize
            else:
                obj_value = result.LogLH  # Maximize (note: stored as positive)
                
            # Check if this belongs in top-k
            for j in range(top_k):
                if algorithm == 'v1':
                    # For v1, we maximize Diff
                    if obj_value > best_ll[j]:
                        # Shift worse solutions down
                        for k in range(top_k - 1, j, -1):
                            best_ll[k] = best_ll[k - 1]
                            best_solutions[k] = best_solutions[k - 1]
                        # Insert new solution
                        best_ll[j] = obj_value
                        best_solutions[j] = result
                        break
                else:
                    # For v2, we maximize LogLH (less negative is better)
                    if obj_value > best_ll[j]:
                        # Shift worse solutions down
                        for k in range(top_k - 1, j, -1):
                            best_ll[k] = best_ll[k - 1]
                            best_solutions[k] = best_solutions[k - 1]
                        # Insert new solution
                        best_ll[j] = obj_value
                        best_solutions[j] = result
                        break
                        
        except Exception as e:
            if verbose:
                warnings.warn(f"Run {i + 1} failed: {str(e)}")
            continue
    
    # Filter out None values (in case we didn't find enough solutions)
    best_solutions = [s for s in best_solutions if s is not None]
    if len(best_solutions) < top_k:
        warnings.warn(f"Only found {len(best_solutions)} valid solutions out of {top_k} requested")
        top_k = len(best_solutions)
    
    if verbose:
        print(f"Computing pairwise Rand indices for top-{top_k} solutions...")
    
    # Calculate pairwise Rand indices
    rand_calc = RandIndex()
    pairwise_rand = np.zeros((top_k, top_k))
    pairwise_adj_rand = np.zeros((top_k, top_k))
    
    sum_rand = 0.0
    sum_adj_rand = 0.0
    n_pairs = 0
    
    for i in range(top_k - 1):
        for j in range(i + 1, top_k):
            rand_idx, adj_rand_idx = rand_calc.calculate(
                best_solutions[i].Assign,
                best_solutions[j].Assign
            )
            pairwise_rand[i, j] = rand_idx
            pairwise_rand[j, i] = rand_idx
            pairwise_adj_rand[i, j] = adj_rand_idx
            pairwise_adj_rand[j, i] = adj_rand_idx
            
            sum_rand += rand_idx
            sum_adj_rand += adj_rand_idx
            n_pairs += 1
    
    # Fill diagonal
    np.fill_diagonal(pairwise_rand, 1.0)
    np.fill_diagonal(pairwise_adj_rand, 1.0)
    
    # Calculate average Rand indices
    avg_rand = sum_rand / n_pairs if n_pairs > 0 else 0.0
    avg_adj_rand = sum_adj_rand / n_pairs if n_pairs > 0 else 0.0
    
    if verbose:
        print(f"Creating empirical distribution with {n_random} random samples...")
    
    # Create empirical distribution for p-values
    if n_jobs == 1:
        # Sequential version
        avg_rand_dist = np.zeros(n_random)
        avg_adj_rand_dist = np.zeros(n_random)
        
        rng = MatlabRandom(random_state)
        
        for k in range(n_random):
            if verbose and (k + 1) % max(1, n_random // 10) == 0:
                print(f"  Progress: {k + 1}/{n_random} samples generated")
                
            # Generate top_k random assignments
            rand_assign = np.zeros((n_items, top_k), dtype=np.int64)
            for t in range(top_k):
                rand_assign[:, t] = rng.randi(n_clusters, n_items)
            
            # Calculate pairwise Rand indices
            k_sum_rand = 0.0
            k_sum_adj_rand = 0.0
            
            for i in range(top_k - 1):
                for j in range(i + 1, top_k):
                    rand_idx, adj_rand_idx = rand_calc.calculate(
                        rand_assign[:, i],
                        rand_assign[:, j]
                    )
                    k_sum_rand += rand_idx
                    k_sum_adj_rand += adj_rand_idx
            
            avg_rand_dist[k] = k_sum_rand / n_pairs if n_pairs > 0 else 0.0
            avg_adj_rand_dist[k] = k_sum_adj_rand / n_pairs if n_pairs > 0 else 0.0
    else:
        # Parallel version
        avg_rand_dist, avg_adj_rand_dist = _parallel_empirical_distribution(
            n_items, n_clusters, top_k, n_random, random_state, n_jobs, verbose
        )
    
    # Sort distributions
    avg_rand_dist.sort()
    avg_adj_rand_dist.sort()
    
    if verbose:
        print("Computing p-values and confidence intervals...")
    
    # Calculate p-values using empirical distribution
    p_struct = rand_empirical_p(
        avg_rand_dist,
        avg_adj_rand_dist,
        avg_rand,
        avg_adj_rand,
        create_confidence=True
    )
    
    return TopKResult(
        best_solutions=best_solutions,
        avg_rand=avg_rand,
        avg_adj_rand=avg_adj_rand,
        rand_p=p_struct['rand_p'],
        adj_rand_p=p_struct['adj_rand_p'],
        rand_conf=p_struct.get('rand_conf'),
        adj_rand_conf=p_struct.get('adj_rand_conf'),
        pairwise_rand=pairwise_rand,
        pairwise_adj_rand=pairwise_adj_rand,
        empirical_dist={
            'avg_rand_dist': avg_rand_dist,
            'avg_adj_rand_dist': avg_adj_rand_dist,
            'n_random': n_random
        }
    )


def _parallel_empirical_distribution(
    n_items: int,
    n_clusters: int,
    top_k: int,
    n_random: int,
    random_state: Optional[int],
    n_jobs: int,
    verbose: bool
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate empirical distribution in parallel."""
    n_workers = n_jobs if n_jobs > 0 else None
    
    def compute_sample(seed: int) -> Tuple[float, float]:
        """Compute one sample for the distribution."""
        local_rng = MatlabRandom(seed)
        rand_calc = RandIndex()
        
        # Generate top_k random assignments
        rand_assign = np.zeros((n_items, top_k), dtype=np.int64)
        for t in range(top_k):
            rand_assign[:, t] = local_rng.randi(n_clusters, n_items)
        
        # Calculate pairwise Rand indices
        sum_rand = 0.0
        sum_adj_rand = 0.0
        n_pairs = 0
        
        for i in range(top_k - 1):
            for j in range(i + 1, top_k):
                rand_idx, adj_rand_idx = rand_calc.calculate(
                    rand_assign[:, i],
                    rand_assign[:, j]
                )
                sum_rand += rand_idx
                sum_adj_rand += adj_rand_idx
                n_pairs += 1
        
        avg_rand = sum_rand / n_pairs if n_pairs > 0 else 0.0
        avg_adj_rand = sum_adj_rand / n_pairs if n_pairs > 0 else 0.0
        
        return avg_rand, avg_adj_rand
    
    # Generate unique seeds
    if random_state is not None:
        base_rng = np.random.RandomState(random_state)
        seeds = base_rng.randint(0, 2**31, size=n_random)
    else:
        seeds = np.random.randint(0, 2**31, size=n_random)
    
    # Initialize arrays
    avg_rand_dist = np.zeros(n_random)
    avg_adj_rand_dist = np.zeros(n_random)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(compute_sample, seeds[i]): i
            for i in range(n_random)
        }
        
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            avg_rand_dist[idx], avg_adj_rand_dist[idx] = future.result()
            
            completed += 1
            if verbose and completed % max(1, n_random // 10) == 0:
                print(f"  Progress: {completed}/{n_random} samples generated")
    
    return avg_rand_dist, avg_adj_rand_dist


def analyze_solution_stability(
    solutions: List[LocalSearchResult],
    verbose: bool = False
) -> Dict[str, Union[float, NDArray[np.float64]]]:
    """Analyze stability across multiple clustering solutions.
    
    Parameters
    ----------
    solutions : list of LocalSearchResult
        Clustering solutions to analyze
    verbose : bool, default=False
        Whether to print analysis details
        
    Returns
    -------
    analysis : dict
        Dictionary containing:
        - 'consensus_matrix': Item x item co-clustering frequency
        - 'item_stability': Stability score for each item
        - 'cluster_stability': Stability of each cluster
        - 'overall_stability': Overall stability score
    """
    if not solutions:
        raise ValueError("No solutions to analyze")
    
    n_items = len(solutions[0].Assign)
    n_solutions = len(solutions)
    
    # Build consensus matrix
    consensus_matrix = np.zeros((n_items, n_items))
    
    for solution in solutions:
        assignments = solution.Assign
        for i in range(n_items):
            for j in range(i + 1, n_items):
                if assignments[i] == assignments[j]:
                    consensus_matrix[i, j] += 1
                    consensus_matrix[j, i] += 1
    
    # Normalize by number of solutions
    consensus_matrix /= n_solutions
    np.fill_diagonal(consensus_matrix, 1.0)
    
    # Calculate item stability
    item_stability = np.zeros(n_items)
    for i in range(n_items):
        # Stability is high if co-clustering frequencies are close to 0 or 1
        co_cluster_freq = consensus_matrix[i, :]
        item_stability[i] = np.mean(np.minimum(co_cluster_freq, 1 - co_cluster_freq)) * 2
    
    # Overall stability
    overall_stability = 1 - np.mean(item_stability)
    
    # Analyze cluster-level stability
    # Find most common cluster configuration
    from collections import Counter
    config_counts = Counter()
    
    for solution in solutions:
        # Create canonical representation of clustering
        # (to handle label permutations)
        config = tuple(sorted(Counter(solution.Assign).values()))
        config_counts[config] += 1
    
    most_common_config = config_counts.most_common(1)[0]
    config_stability = most_common_config[1] / n_solutions
    
    if verbose:
        print(f"Overall stability: {overall_stability:.3f}")
        print(f"Most common configuration appears in {config_stability:.1%} of solutions")
        print(f"Configuration: {most_common_config[0]}")
    
    return {
        'consensus_matrix': consensus_matrix,
        'item_stability': item_stability,
        'overall_stability': overall_stability,
        'config_stability': config_stability,
        'most_common_config': most_common_config[0]
    }