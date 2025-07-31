"""Statistical tests and utilities for SUBMARIT evaluation."""

from typing import Tuple, Optional, Dict, List
import numpy as np
from numpy.typing import NDArray
from scipy import stats
import warnings

from submarit.evaluation.cluster_evaluator import ClusterEvaluationResult


class StatisticalTests:
    """Statistical testing utilities for cluster evaluation."""
    
    @staticmethod
    def permutation_test(
        swm: NDArray[np.float64],
        assign1: NDArray[np.int64],
        assign2: NDArray[np.int64],
        n_permutations: int = 1000,
        metric: str = "z_value",
        random_state: Optional[int] = None
    ) -> Dict[str, float]:
        """Perform permutation test to compare two clustering solutions.
        
        Args:
            swm: Switching matrix
            assign1: First cluster assignment
            assign2: Second cluster assignment
            n_permutations: Number of permutations
            metric: Metric to compare ('z_value', 'diff', 'log_lh')
            random_state: Random seed
            
        Returns:
            Dictionary with test statistics and p-value
        """
        from submarit.evaluation.cluster_evaluator import ClusterEvaluator
        
        if random_state is not None:
            np.random.seed(random_state)
            
        evaluator = ClusterEvaluator()
        n_clusters1 = len(np.unique(assign1))
        n_clusters2 = len(np.unique(assign2))
        
        # Evaluate original assignments
        result1 = evaluator.evaluate(swm, n_clusters1, assign1)
        result2 = evaluator.evaluate(swm, n_clusters2, assign2)
        
        # Get original difference
        orig_diff = getattr(result1, metric) - getattr(result2, metric)
        
        # Perform permutations
        perm_diffs = []
        n_items = len(assign1)
        
        for _ in range(n_permutations):
            # Randomly swap assignments between solutions
            swap_mask = np.random.rand(n_items) < 0.5
            perm_assign1 = assign1.copy()
            perm_assign2 = assign2.copy()
            
            perm_assign1[swap_mask] = assign2[swap_mask]
            perm_assign2[swap_mask] = assign1[swap_mask]
            
            # Evaluate permuted assignments
            try:
                perm_result1 = evaluator.evaluate(swm, n_clusters1, perm_assign1)
                perm_result2 = evaluator.evaluate(swm, n_clusters2, perm_assign2)
                perm_diff = getattr(perm_result1, metric) - getattr(perm_result2, metric)
                perm_diffs.append(perm_diff)
            except:
                # Skip invalid permutations
                continue
                
        perm_diffs = np.array(perm_diffs)
        
        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(orig_diff))
        
        return {
            "original_diff": orig_diff,
            "p_value": p_value,
            "n_permutations": len(perm_diffs),
            "metric": metric,
            "result1_value": getattr(result1, metric),
            "result2_value": getattr(result2, metric),
            "perm_mean": np.mean(perm_diffs),
            "perm_std": np.std(perm_diffs)
        }
    
    @staticmethod
    def bootstrap_confidence_interval(
        swm: NDArray[np.float64],
        cluster_assign: NDArray[np.int64],
        metric: str = "z_value",
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_state: Optional[int] = None
    ) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals for evaluation metrics.
        
        Args:
            swm: Switching matrix
            cluster_assign: Cluster assignments
            metric: Metric to analyze
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            random_state: Random seed
            
        Returns:
            Dictionary with confidence intervals and statistics
        """
        from submarit.evaluation.cluster_evaluator import ClusterEvaluator
        
        if random_state is not None:
            np.random.seed(random_state)
            
        evaluator = ClusterEvaluator()
        n_clusters = len(np.unique(cluster_assign))
        n_items = swm.shape[0]
        
        # Original evaluation
        orig_result = evaluator.evaluate(swm, n_clusters, cluster_assign)
        orig_value = getattr(orig_result, metric)
        
        # Bootstrap samples
        bootstrap_values = []
        
        for _ in range(n_bootstrap):
            # Resample items with replacement
            boot_indices = np.random.choice(n_items, size=n_items, replace=True)
            
            # Create bootstrap switching matrix
            boot_swm = swm[np.ix_(boot_indices, boot_indices)]
            
            # Map cluster assignments
            boot_assign = cluster_assign[boot_indices]
            
            # Ensure all clusters are represented
            unique_clusters = np.unique(boot_assign)
            if len(unique_clusters) < n_clusters:
                continue
                
            # Remap to consecutive integers
            remap = {old: new+1 for new, old in enumerate(unique_clusters)}
            boot_assign = np.array([remap[x] for x in boot_assign])
            
            try:
                boot_result = evaluator.evaluate(boot_swm, len(unique_clusters), boot_assign)
                bootstrap_values.append(getattr(boot_result, metric))
            except:
                continue
                
        bootstrap_values = np.array(bootstrap_values)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_values, lower_percentile)
        ci_upper = np.percentile(bootstrap_values, upper_percentile)
        
        return {
            "metric": metric,
            "original_value": orig_value,
            "mean": np.mean(bootstrap_values),
            "std": np.std(bootstrap_values),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence": confidence,
            "n_bootstrap": len(bootstrap_values)
        }
    
    @staticmethod
    def stability_analysis(
        swm: NDArray[np.float64],
        cluster_func,
        n_clusters: int,
        n_runs: int = 10,
        min_items: int = 1
    ) -> Dict[str, float]:
        """Analyze stability of clustering results across multiple runs.
        
        Args:
            swm: Switching matrix
            cluster_func: Clustering function
            n_clusters: Number of clusters
            n_runs: Number of runs to perform
            min_items: Minimum items per cluster
            
        Returns:
            Dictionary with stability metrics
        """
        from submarit.evaluation.cluster_evaluator import ClusterEvaluator
        
        evaluator = ClusterEvaluator()
        
        # Store results from multiple runs
        z_values = []
        log_lh_values = []
        diff_values = []
        assignments = []
        
        for i in range(n_runs):
            # Run clustering
            cluster_result = cluster_func(swm, n_clusters, min_items, 1)
            
            # Evaluate
            eval_result = evaluator.evaluate(swm, n_clusters, cluster_result.assign)
            
            z_values.append(eval_result.z_value)
            log_lh_values.append(eval_result.log_lh)
            diff_values.append(eval_result.diff)
            assignments.append(cluster_result.assign)
            
        # Calculate stability metrics
        z_values = np.array(z_values)
        log_lh_values = np.array(log_lh_values)
        diff_values = np.array(diff_values)
        
        # Calculate pairwise agreement between assignments
        n_items = len(assignments[0])
        agreement_scores = []
        
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                # Calculate Rand index or similar
                agreement = StatisticalTests._calculate_agreement(
                    assignments[i], assignments[j]
                )
                agreement_scores.append(agreement)
                
        return {
            "z_value_mean": np.mean(z_values),
            "z_value_std": np.std(z_values),
            "z_value_cv": np.std(z_values) / (np.mean(z_values) + 1e-10),
            "log_lh_mean": np.mean(log_lh_values),
            "log_lh_std": np.std(log_lh_values),
            "diff_mean": np.mean(diff_values),
            "diff_std": np.std(diff_values),
            "mean_agreement": np.mean(agreement_scores) if agreement_scores else 0,
            "min_agreement": np.min(agreement_scores) if agreement_scores else 0,
            "n_runs": n_runs
        }
    
    @staticmethod
    def _calculate_agreement(assign1: NDArray[np.int64], assign2: NDArray[np.int64]) -> float:
        """Calculate agreement between two cluster assignments (simplified Rand index)."""
        n = len(assign1)
        agreements = 0
        
        for i in range(n):
            for j in range(i+1, n):
                # Check if pair is in same cluster in both assignments
                same1 = assign1[i] == assign1[j]
                same2 = assign2[i] == assign2[j]
                if same1 == same2:
                    agreements += 1
                    
        total_pairs = n * (n - 1) / 2
        return agreements / total_pairs if total_pairs > 0 else 0
    
    @staticmethod
    def cluster_validity_indices(
        swm: NDArray[np.float64],
        cluster_assign: NDArray[np.int64]
    ) -> Dict[str, float]:
        """Calculate various cluster validity indices.
        
        Args:
            swm: Switching matrix
            cluster_assign: Cluster assignments
            
        Returns:
            Dictionary with validity indices
        """
        n_items = swm.shape[0]
        n_clusters = len(np.unique(cluster_assign))
        
        # Calculate within-cluster and between-cluster switching
        within_switching = 0
        between_switching = 0
        
        for i in range(n_items):
            for j in range(i+1, n_items):
                if cluster_assign[i] == cluster_assign[j]:
                    within_switching += swm[i, j]
                else:
                    between_switching += swm[i, j]
                    
        total_switching = within_switching + between_switching
        
        # Silhouette-like coefficient for switching data
        silhouette_values = []
        
        for i in range(n_items):
            # Average switching to items in same cluster
            same_cluster = cluster_assign == cluster_assign[i]
            same_cluster[i] = False  # Exclude self
            
            if np.sum(same_cluster) > 0:
                a_i = np.mean(swm[i, same_cluster])
            else:
                a_i = 0
                
            # Average switching to items in other clusters
            b_values = []
            for c in range(1, n_clusters + 1):
                if c != cluster_assign[i]:
                    other_cluster = cluster_assign == c
                    if np.sum(other_cluster) > 0:
                        b_values.append(np.mean(swm[i, other_cluster]))
                        
            b_i = np.min(b_values) if b_values else 0
            
            # Silhouette coefficient
            s_i = (b_i - a_i) / (max(a_i, b_i) + 1e-10)
            silhouette_values.append(s_i)
            
        # Dunn-like index
        min_between = np.inf
        max_within = 0
        
        for c1 in range(1, n_clusters + 1):
            cluster1 = cluster_assign == c1
            
            # Within-cluster maximum
            if np.sum(cluster1) > 1:
                within_swm = swm[np.ix_(cluster1, cluster1)]
                max_within = max(max_within, np.max(within_swm))
                
            # Between-cluster minimum
            for c2 in range(c1 + 1, n_clusters + 1):
                cluster2 = cluster_assign == c2
                if np.sum(cluster1) > 0 and np.sum(cluster2) > 0:
                    between_swm = swm[np.ix_(cluster1, cluster2)]
                    min_between = min(min_between, np.min(between_swm[between_swm > 0]))
                    
        dunn_index = min_between / (max_within + 1e-10) if max_within > 0 else 0
        
        return {
            "within_switching_ratio": within_switching / (total_switching + 1e-10),
            "between_switching_ratio": between_switching / (total_switching + 1e-10),
            "silhouette_coefficient": np.mean(silhouette_values),
            "dunn_index": dunn_index,
            "n_clusters": n_clusters,
            "n_items": n_items
        }