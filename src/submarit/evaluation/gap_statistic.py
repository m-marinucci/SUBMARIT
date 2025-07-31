"""GAP statistic implementation for optimal cluster number selection."""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import warnings

from submarit.core.base import BaseEvaluator
from submarit.evaluation.cluster_evaluator import ClusterEvaluator


@dataclass
class GAPStatisticResult:
    """Container for GAP statistic results.
    
    Attributes:
        k_values: Array of tested cluster numbers
        observed_values: Observed criterion values for each k
        uniform_values: Average criterion values for uniform data
        gap_values: GAP statistic values (difference)
        best_k: Optimal number of clusters
        criterion_name: Name of the criterion used
        detailed_results: Dictionary with results for each criterion type
    """
    k_values: NDArray[np.int64]
    observed_values: NDArray[np.float64]
    uniform_values: NDArray[np.float64]
    gap_values: NDArray[np.float64]
    best_k: int
    criterion_name: str
    detailed_results: Dict[str, NDArray[np.float64]]
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GAPStatisticResult("
            f"k_range=[{self.k_values[0]}, {self.k_values[-1]}], "
            f"best_k={self.best_k}, "
            f"criterion='{self.criterion_name}')"
        )
    
    def summary(self) -> str:
        """Generate detailed summary."""
        lines = [
            f"GAP Statistic Results",
            f"=" * 50,
            f"Criterion: {self.criterion_name}",
            f"Optimal k: {self.best_k}",
            f"",
            f"{'k':>3} {'Observed':>12} {'Uniform':>12} {'GAP':>12}",
            f"{'-'*3} {'-'*12} {'-'*12} {'-'*12}",
        ]
        
        for i, k in enumerate(self.k_values):
            lines.append(
                f"{k:3d} {self.observed_values[i]:12.6f} "
                f"{self.uniform_values[i]:12.6f} {self.gap_values[i]:12.6f}"
            )
        
        return "\n".join(lines)


class GAPStatistic(BaseEvaluator):
    """GAP statistic for determining optimal number of clusters.
    
    Implements the GAP statistic described in Tibshirani, Walther, and Hastie (2001)
    adapted for the SUBMARIT procedure. The GAP statistic compares the clustering
    criterion for the observed data with that expected under a null reference
    distribution (uniform random data).
    """
    
    def __init__(
        self,
        criterion: str = "diff",
        n_uniform: int = 10,
        verbose: bool = False
    ):
        """Initialize GAP statistic evaluator.
        
        Args:
            criterion: Criterion to optimize ('diff', 'diff_sq', 'log_lh', 'z_value')
            n_uniform: Number of uniform random datasets to generate
            verbose: Whether to print progress information
        """
        super().__init__()
        self.criterion = criterion
        self.n_uniform = n_uniform
        self.verbose = verbose
        self._evaluator = ClusterEvaluator()
        
    def evaluate(
        self,
        swm: NDArray[np.float64],
        data_matrix: NDArray[np.float64],
        min_k: int,
        max_k: int,
        min_items: int,
        cluster_func: Callable,
        n_runs: int = 10
    ) -> GAPStatisticResult:
        """Compute GAP statistic for a range of cluster numbers.
        
        Args:
            swm: Switching matrix
            data_matrix: Original data matrix (customers x products)
            min_k: Minimum number of clusters to test
            max_k: Maximum number of clusters to test
            min_items: Minimum number of items per cluster
            cluster_func: Function to perform clustering (e.g., RunClusters)
            n_runs: Number of optimization runs per solution
            
        Returns:
            GAPStatisticResult containing optimal k and detailed results
        """
        n_items = swm.shape[0]
        n_customers, n_products = data_matrix.shape
        
        # Get data range for uniform generation
        max_data = np.max(data_matrix)
        min_data = np.min(data_matrix)
        
        # Initialize result arrays
        k_range = np.arange(min_k, max_k + 1)
        n_k = len(k_range)
        
        results = {
            "diff": np.zeros((n_k, 4)),
            "diff_sq": np.zeros((n_k, 4)),
            "log_lh": np.zeros((n_k, 4)),
            "z_value": np.zeros((n_k, 4))
        }
        
        # First column is k value
        for key in results:
            results[key][:, 0] = k_range
        
        best_values = {
            "diff": -1e10,
            "diff_sq": -1e10,
            "log_lh": -1e10,
            "z_value": -1e10
        }
        
        best_k = {key: min_k for key in results.keys()}
        
        # Test each k value
        for i, n_clusters in enumerate(k_range):
            if self.verbose:
                print(f"Testing k={n_clusters}...")
            
            # Run clustering on observed data
            cluster_result = cluster_func(swm, n_clusters, min_items, n_runs)
            eval_result = self._evaluator.evaluate(swm, n_clusters, cluster_result.assign)
            
            # Store observed values
            results["diff"][i, 1] = eval_result.diff
            results["diff_sq"][i, 1] = eval_result.diff_sq
            results["log_lh"][i, 1] = eval_result.log_lh
            results["z_value"][i, 1] = eval_result.z_value
            
            # Generate uniform reference distributions
            sum_diff = 0.0
            sum_diff_sq = 0.0
            sum_log_lh = 0.0
            sum_z_value = 0.0
            
            for j in range(self.n_uniform):
                # Generate uniform random data
                x_uniform = min_data + np.random.rand(n_customers, n_products) * (max_data - min_data)
                
                # Create switching matrix from uniform data
                # This would require the CreateForcedSwitching2 function
                # For now, we'll create a simple approximation
                uniform_swm = self._create_uniform_switching_matrix(x_uniform, n_items)
                
                # Run clustering on uniform data
                try:
                    uniform_cluster = cluster_func(uniform_swm, n_clusters, min_items, n_runs)
                    uniform_eval = self._evaluator.evaluate(
                        uniform_swm, n_clusters, uniform_cluster.assign
                    )
                    
                    sum_diff += uniform_eval.diff
                    sum_diff_sq += uniform_eval.diff_sq
                    sum_log_lh += uniform_eval.log_lh
                    sum_z_value += uniform_eval.z_value
                    
                except Exception as e:
                    warnings.warn(f"Failed to cluster uniform data: {e}")
                    continue
            
            # Calculate average uniform values
            results["diff"][i, 2] = sum_diff / self.n_uniform
            results["diff_sq"][i, 2] = sum_diff_sq / self.n_uniform
            results["log_lh"][i, 2] = sum_log_lh / self.n_uniform
            results["z_value"][i, 2] = sum_z_value / self.n_uniform
            
            # Calculate GAP values (ensure positive is good)
            results["diff"][i, 3] = results["diff"][i, 1] - results["diff"][i, 2]
            results["diff_sq"][i, 3] = results["diff_sq"][i, 1] - results["diff_sq"][i, 2]
            results["log_lh"][i, 3] = results["log_lh"][i, 2] - results["log_lh"][i, 1]
            results["z_value"][i, 3] = results["z_value"][i, 1] - results["z_value"][i, 2]
            
            # Update best k for each criterion
            for key in results.keys():
                if results[key][i, 3] > best_values[key]:
                    best_values[key] = results[key][i, 3]
                    best_k[key] = n_clusters
        
        # Return results for specified criterion
        criterion_idx = {
            "diff": "diff",
            "diff_sq": "diff_sq", 
            "log_lh": "log_lh",
            "z_value": "z_value"
        }.get(self.criterion, "diff")
        
        return GAPStatisticResult(
            k_values=k_range,
            observed_values=results[criterion_idx][:, 1],
            uniform_values=results[criterion_idx][:, 2],
            gap_values=results[criterion_idx][:, 3],
            best_k=best_k[criterion_idx],
            criterion_name=self.criterion,
            detailed_results=results
        )
    
    def _create_uniform_switching_matrix(
        self,
        data_matrix: NDArray[np.float64],
        n_items: int
    ) -> NDArray[np.float64]:
        """Create switching matrix from uniform random data.
        
        This is a simplified version that creates a random switching matrix
        with properties similar to what would be generated from uniform data.
        
        Args:
            data_matrix: Uniform random data matrix
            n_items: Number of items (products) to include
            
        Returns:
            Switching matrix
        """
        # Simple approximation: create random switching matrix
        # with row sums similar to typical switching data
        n = min(n_items, data_matrix.shape[1])
        
        # Generate random switching matrix
        swm = np.random.rand(n, n) * 100
        
        # Make it symmetric (undirected switching)
        swm = (swm + swm.T) / 2
        
        # Zero out diagonal (no self-switching)
        np.fill_diagonal(swm, 0)
        
        # Make it sparse (most products don't switch)
        swm[swm < np.percentile(swm, 70)] = 0
        
        return swm
    
    def plot_gap_statistic(
        self,
        result: GAPStatisticResult,
        save_path: Optional[str] = None
    ) -> None:
        """Plot GAP statistic results.
        
        Args:
            result: GAP statistic results
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot observed vs uniform values
        ax1.plot(result.k_values, result.observed_values, 'b-o', label='Observed')
        ax1.plot(result.k_values, result.uniform_values, 'r--s', label='Uniform')
        ax1.set_xlabel('Number of clusters (k)')
        ax1.set_ylabel(f'{result.criterion_name} value')
        ax1.set_title('Observed vs Uniform Reference Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot GAP statistic
        ax2.plot(result.k_values, result.gap_values, 'g-^', linewidth=2)
        ax2.axvline(x=result.best_k, color='red', linestyle='--', 
                   label=f'Best k = {result.best_k}')
        ax2.set_xlabel('Number of clusters (k)')
        ax2.set_ylabel('GAP statistic')
        ax2.set_title(f'GAP Statistic ({result.criterion_name})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()