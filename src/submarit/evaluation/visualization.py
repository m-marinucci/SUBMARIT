"""Visualization utilities for evaluation results."""

from typing import List, Optional, Union, Dict, Any
import numpy as np
from numpy.typing import NDArray
import warnings

from submarit.evaluation.cluster_evaluator import ClusterEvaluationResult
from submarit.evaluation.gap_statistic import GAPStatisticResult
from submarit.evaluation.entropy_evaluator import EntropyClusterResult


class EvaluationVisualizer:
    """Visualization utilities for SUBMARIT evaluation results."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self._check_matplotlib()
        
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            return True
        except ImportError:
            warnings.warn("matplotlib not available for visualization")
            self.plt = None
            return False
    
    def plot_cluster_comparison(
        self,
        results: List[ClusterEvaluationResult],
        labels: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Compare multiple clustering results.
        
        Args:
            results: List of evaluation results to compare
            labels: Labels for each result
            metrics: Metrics to plot (default: all)
            save_path: Optional path to save the plot
        """
        if not self.plt:
            return
            
        if metrics is None:
            metrics = ["diff", "z_value", "log_lh", "scaled_diff"]
            
        if labels is None:
            labels = [f"Result {i+1}" for i in range(len(results))]
            
        n_metrics = len(metrics)
        fig, axes = self.plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
            
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [getattr(r, metric) for r in results]
            
            bars = ax.bar(range(len(results)), values)
            ax.set_xlabel("Clustering")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
            ax.set_xticks(range(len(results)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Color best result
            best_idx = np.argmax(values) if metric != "scaled_diff" else np.argmin(np.abs(values))
            bars[best_idx].set_color('green')
            
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            self.plt.show()
            
        self.plt.close()
    
    def plot_cluster_sizes(
        self,
        result: Union[ClusterEvaluationResult, EntropyClusterResult],
        save_path: Optional[str] = None
    ) -> None:
        """Plot cluster size distribution.
        
        Args:
            result: Clustering result
            save_path: Optional path to save the plot
        """
        if not self.plt:
            return
            
        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of cluster sizes
        cluster_labels = [f"Cluster {i+1}" for i in range(result.n_clusters)]
        ax1.bar(cluster_labels, result.counts)
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Number of Items")
        ax1.set_title("Cluster Sizes")
        ax1.axhline(y=np.mean(result.counts), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(result.counts):.1f}')
        ax1.legend()
        
        # Pie chart
        ax2.pie(result.counts, labels=cluster_labels, autopct='%1.1f%%')
        ax2.set_title("Cluster Size Distribution")
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            self.plt.show()
            
        self.plt.close()
    
    def plot_probability_comparison(
        self,
        result: ClusterEvaluationResult,
        cluster_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot P vs PHat comparison.
        
        Args:
            result: Evaluation result
            cluster_idx: Optional specific cluster to highlight
            save_path: Optional path to save the plot
        """
        if not self.plt:
            return
            
        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot of P vs PHat
        ax1.scatter(result.p, result.p_hat, alpha=0.6)
        ax1.plot([0, 1], [0, 1], 'r--', label='y=x')
        ax1.set_xlabel('P (Theoretical)')
        ax1.set_ylabel('PHat (Estimated)')
        ax1.set_title('P vs PHat Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Difference histogram
        differences = result.p_hat - result.p
        ax2.hist(differences, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', label='Zero difference')
        ax2.set_xlabel('PHat - P')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Differences')
        ax2.legend()
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            self.plt.show()
            
        self.plt.close()
    
    def plot_entropy_evolution(
        self,
        entropy_values: List[float],
        save_path: Optional[str] = None
    ) -> None:
        """Plot entropy values over iterations.
        
        Args:
            entropy_values: List of entropy values per iteration
            save_path: Optional path to save the plot
        """
        if not self.plt:
            return
            
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        iterations = range(len(entropy_values))
        ax.plot(iterations, entropy_values, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy Evolution During Clustering')
        ax.grid(True, alpha=0.3)
        
        # Mark convergence
        if len(entropy_values) > 1:
            final_value = entropy_values[-1]
            ax.axhline(y=final_value, color='r', linestyle='--', 
                      label=f'Final: {final_value:.4f}')
            ax.legend()
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            self.plt.show()
            
        self.plt.close()
    
    def create_evaluation_report(
        self,
        results: Dict[str, Any],
        save_path: str
    ) -> None:
        """Create a comprehensive evaluation report.
        
        Args:
            results: Dictionary containing various evaluation results
            save_path: Path to save the report (PDF or image)
        """
        if not self.plt:
            return
            
        # Create figure with subplots
        fig = self.plt.figure(figsize=(16, 12))
        
        # Title
        fig.suptitle('SUBMARIT Clustering Evaluation Report', fontsize=16, fontweight='bold')
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Add various plots based on available results
        plot_idx = 0
        
        if 'cluster_eval' in results:
            eval_result = results['cluster_eval']
            
            # Cluster sizes
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.bar(range(len(eval_result.counts)), eval_result.counts)
            ax1.set_title('Cluster Sizes')
            ax1.set_xlabel('Cluster')
            ax1.set_ylabel('Count')
            
            # P vs PHat
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.scatter(eval_result.p, eval_result.p_hat, alpha=0.5)
            ax2.plot([0, max(eval_result.p)], [0, max(eval_result.p)], 'r--')
            ax2.set_title('P vs PHat')
            ax2.set_xlabel('P')
            ax2.set_ylabel('PHat')
            
            # Metrics summary
            ax3 = fig.add_subplot(gs[0, 2])
            metrics = {
                'Diff': eval_result.diff,
                'Z-value': eval_result.z_value,
                'Log-LH': eval_result.log_lh,
                'Scaled Diff': eval_result.scaled_diff
            }
            ax3.bar(metrics.keys(), metrics.values())
            ax3.set_title('Evaluation Metrics')
            ax3.tick_params(axis='x', rotation=45)
            
        if 'gap_result' in results:
            gap_result = results['gap_result']
            
            # GAP statistic plot
            ax4 = fig.add_subplot(gs[1, :2])
            ax4.plot(gap_result.k_values, gap_result.gap_values, 'g-^', linewidth=2)
            ax4.axvline(x=gap_result.best_k, color='r', linestyle='--',
                       label=f'Best k={gap_result.best_k}')
            ax4.set_title('GAP Statistic')
            ax4.set_xlabel('Number of Clusters (k)')
            ax4.set_ylabel('GAP Value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
        if 'entropy_result' in results:
            entropy_result = results['entropy_result']
            
            # Entropy metrics
            ax5 = fig.add_subplot(gs[1, 2])
            entropy_metrics = {
                'Total': entropy_result.entropy,
                'Normalized': entropy_result.entropy_norm,
                'Size-scaled': entropy_result.entropy_norm2
            }
            ax5.bar(entropy_metrics.keys(), entropy_metrics.values())
            ax5.set_title('Entropy Metrics')
            ax5.set_ylabel('Entropy')
            
        # Add text summary
        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis('off')
        
        summary_text = self._generate_text_summary(results)
        ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        self.plt.tight_layout()
        self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.plt.close()
    
    def _generate_text_summary(self, results: Dict[str, Any]) -> str:
        """Generate text summary for the report."""
        lines = ["SUMMARY", "=" * 50]
        
        if 'cluster_eval' in results:
            eval_result = results['cluster_eval']
            lines.extend([
                f"Number of clusters: {eval_result.n_clusters}",
                f"Number of items: {eval_result.n_items}",
                f"Z-value: {eval_result.z_value:.4f}",
                f"Log-likelihood: {eval_result.log_lh:.4f}",
                f"Total difference (PHat-P): {eval_result.diff:.6f}",
                ""
            ])
            
        if 'gap_result' in results:
            gap_result = results['gap_result']
            lines.extend([
                f"GAP Statistic Analysis:",
                f"  Optimal k: {gap_result.best_k}",
                f"  Criterion: {gap_result.criterion_name}",
                f"  K range tested: [{gap_result.k_values[0]}, {gap_result.k_values[-1]}]",
                ""
            ])
            
        if 'entropy_result' in results:
            entropy_result = results['entropy_result']
            lines.extend([
                f"Entropy Clustering:",
                f"  Iterations: {entropy_result.n_iter}",
                f"  Total entropy: {entropy_result.entropy:.6f}",
                f"  Max objective: {entropy_result.max_obj:.6f}",
                ""
            ])
            
        return "\n".join(lines)