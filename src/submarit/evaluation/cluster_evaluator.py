"""Cluster evaluation metrics for SUBMARIT clustering."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from submarit.core.base import BaseEvaluator


@dataclass
class ClusterEvaluationResult:
    """Container for cluster evaluation results.
    
    Attributes:
        assign: Cluster assignments (1 to n_clusters)
        indexes: List of arrays containing product indices for each cluster
        counts: Number of products in each cluster
        diff: Sum of (PHat - P)
        item_diff: Sum of (PHat - P) / n_items
        scaled_diff: Sum of (PHat - P) / P
        z_value: Z-value statistic as per Urban and Hauser
        max_obj: Objective function value (same as diff)
        log_lh: Log-likelihood from maximum likelihood model
        diff_sq: Sum of (PHat - P)^2
        p_hat: Estimated probabilities
        p: Theoretical probabilities
        var: Variance values for each cluster
        n_clusters: Number of clusters
        n_items: Number of items
    """
    assign: NDArray[np.int64]
    indexes: List[NDArray[np.int64]]
    counts: List[int]
    diff: float
    item_diff: float
    scaled_diff: float
    z_value: float
    max_obj: float
    log_lh: float
    diff_sq: float
    p_hat: NDArray[np.float64]
    p: NDArray[np.float64]
    var: List[NDArray[np.float64]]
    n_clusters: int
    n_items: int
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ClusterEvaluationResult("
            f"n_clusters={self.n_clusters}, "
            f"n_items={self.n_items}, "
            f"diff={self.diff:.6f}, "
            f"z_value={self.z_value:.4f}, "
            f"log_lh={self.log_lh:.4f})"
        )
    
    def summary(self) -> str:
        """Generate detailed summary."""
        lines = [
            f"Cluster Evaluation Results",
            f"=" * 40,
            f"Number of clusters: {self.n_clusters}",
            f"Number of items: {self.n_items}",
            f"",
            f"Cluster sizes:",
        ]
        
        for i, count in enumerate(self.counts):
            lines.append(f"  Cluster {i+1}: {count} items")
        
        lines.extend([
            f"",
            f"Evaluation metrics:",
            f"  Diff (PHat - P): {self.diff:.6f}",
            f"  Item diff: {self.item_diff:.6f}",
            f"  Scaled diff: {self.scaled_diff:.6f}",
            f"  Z-value: {self.z_value:.4f}",
            f"  Log-likelihood: {self.log_lh:.4f}",
            f"  Squared diff: {self.diff_sq:.6f}",
        ])
        
        return "\n".join(lines)


class ClusterEvaluator(BaseEvaluator):
    """Evaluates SUBMARIT clustering solutions using multiple criteria.
    
    This class implements the evaluation methodology from kSMEvaluateClustering.m,
    computing various statistics to assess clustering quality including:
    - Difference between estimated and theoretical probabilities
    - Z-value statistics
    - Log-likelihood values
    """
    
    def __init__(self):
        """Initialize the cluster evaluator."""
        super().__init__()
    
    def evaluate(
        self,
        swm: NDArray[np.float64],
        n_clusters: int,
        cluster_assign: NDArray[np.int64]
    ) -> ClusterEvaluationResult:
        """Evaluate a SUBMARIT clustering solution.
        
        Args:
            swm: Product x product switching matrix
            n_clusters: Number of clusters
            cluster_assign: Product cluster assignments (1 to n_clusters)
            
        Returns:
            ClusterEvaluationResult containing all evaluation metrics
        """
        n_items = swm.shape[0]
        
        # Calculate sales and proportions
        p_sales = np.sum(swm, axis=1)
        pswm = swm / (p_sales[:, np.newaxis] + 1e-10)  # Add small value to avoid division by zero
        pp_sales = p_sales / np.sum(p_sales)
        
        # Initialize cluster information
        indexes = []
        counts = []
        for i in range(1, n_clusters + 1):
            cur_indexes = np.where(cluster_assign == i)[0]
            indexes.append(cur_indexes)
            counts.append(len(cur_indexes))
        
        # Setup P and PHat arrays
        p_hat = np.zeros(n_items)
        p = np.zeros(n_items)
        var_list = []
        log_lh = 0.0
        
        # Calculate values for each cluster
        for i_clus in range(n_clusters):
            idx = indexes[i_clus]
            if len(idx) == 0:
                var_list.append(np.array([]))
                continue
                
            # Extract submatrix for current cluster
            sub_swm = pswm[np.ix_(idx, idx)]
            
            # Sum for each item to get PHat value
            p_hat[idx] = np.sum(sub_swm, axis=1)
            
            # Calculate P values
            spp_sales = pp_sales[idx]
            props = np.outer(np.ones(counts[i_clus]), spp_sales) - np.diag(spp_sales)
            p[idx] = np.sum(props, axis=1) / (1 - spp_sales + 1e-10)
            
            # Calculate variance and log-likelihood components
            var = p[idx] * (1 - p[idx]) * p_sales[idx]
            var_list.append(var)
            
            sd_comp = np.log(1 / (np.sqrt(np.sum(var) * 2 * np.pi) + 1e-10))
            s_diff = np.sum(p_hat[idx] * p_sales[idx]) - np.sum(p[idx] * p_sales[idx])
            
            # Update log-likelihood
            log_lh += sd_comp - (np.sign(s_diff) * (s_diff ** 2)) / (2 * np.sum(var) + 1e-10)
        
        # Calculate differences
        diff = np.sum(p_hat - p)
        diff_sq = np.sum((p_hat - p) ** 2)
        item_diff = diff / n_items
        
        # Scaled differences
        valid_ix = (~np.isinf(p)) & (~np.isnan(p)) & (np.abs(p) > 1e-10)
        scaled_diff = np.sum((p_hat[valid_ix] - p[valid_ix]) / p[valid_ix])
        
        # Z-value as per Urban and Hauser
        m_p_hat = np.sum(p_hat * p_sales)
        m_p = np.sum(p * p_sales)
        denominator = np.sqrt(np.sum(p_hat * (1 - p_hat) * p_sales) + 1e-10)
        z_value = (m_p_hat - m_p) / denominator
        
        return ClusterEvaluationResult(
            assign=cluster_assign,
            indexes=indexes,
            counts=counts,
            diff=diff,
            item_diff=item_diff,
            scaled_diff=scaled_diff,
            z_value=z_value,
            max_obj=diff,
            log_lh=log_lh,
            diff_sq=diff_sq,
            p_hat=p_hat,
            p=p,
            var=var_list,
            n_clusters=n_clusters,
            n_items=n_items
        )
    
    def evaluate_legacy(
        self,
        swm: NDArray[np.float64],
        n_clusters: int,
        cluster_assign: NDArray[np.int64]
    ) -> Dict[str, Union[float, NDArray[np.float64]]]:
        """Legacy evaluation function matching kEvaluateClustering.m.
        
        This is a simplified version without some of the advanced statistics.
        
        Args:
            swm: Product x product switching matrix
            n_clusters: Number of clusters
            cluster_assign: Product cluster assignments (1 to n_clusters)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        n_items = swm.shape[0]
        
        # Calculate sales and proportions
        p_sales = np.sum(swm, axis=1)
        pswm = swm / (p_sales[:, np.newaxis] + 1e-10)
        pp_sales = p_sales / np.sum(p_sales)
        
        # Initialize arrays
        p_hat = np.zeros(n_items)
        p = np.zeros(n_items)
        
        # Calculate values for each cluster
        for i_clus in range(1, n_clusters + 1):
            idx = np.where(cluster_assign == i_clus)[0]
            if len(idx) == 0:
                continue
                
            # Extract submatrix
            sub_swm = pswm[np.ix_(idx, idx)]
            
            # PHat values
            p_hat[idx] = np.sum(sub_swm, axis=1)
            
            # P values
            spp_sales = pp_sales[idx]
            n_idx = len(idx)
            props = np.outer(np.ones(n_idx), spp_sales) - np.diag(spp_sales)
            p[idx] = np.sum(props, axis=1) / (1 - spp_sales + 1e-10)
        
        # Calculate metrics
        diff = np.sum(p_hat - p)
        item_diff = diff / n_items
        
        # Scaled differences
        valid_ix = (~np.isinf(p)) & (~np.isnan(p)) & (np.abs(p) > 1e-10)
        n_valid = np.sum(valid_ix)
        scaled_diff = np.sum((p_hat[valid_ix] - p[valid_ix]) / p[valid_ix])
        
        # Z-value
        m_p_hat = np.mean(p_hat)
        m_p = np.mean(p)
        all_sales = np.sum(p_sales)
        z_value = (m_p_hat - m_p) / np.sqrt(m_p * (1 - m_p) / all_sales + 1e-10)
        
        # Log-likelihood (simplified version)
        var = p[valid_ix] * (1 - p[valid_ix]) / (p_sales[valid_ix] + 1e-10)
        part1 = n_valid * np.log(2 * np.pi)
        part2 = np.sum(var)
        part3 = np.sum(((p_hat[valid_ix] - p[valid_ix]) ** 2) / np.sqrt(var + 1e-10))
        log_lh = -0.5 * (part1 + part2 + part3)
        
        return {
            "diff": diff,
            "item_diff": item_diff,
            "scaled_diff": scaled_diff,
            "z_value": z_value,
            "log_lh": log_lh,
            "p": p,
            "p_hat": p_hat,
            "max_obj": diff
        }