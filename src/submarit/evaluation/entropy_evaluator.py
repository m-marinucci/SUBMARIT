"""Entropy-based clustering evaluation and optimization."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import warnings

from submarit.core.base import BaseEvaluator, BaseClusterer


@dataclass
class EntropyClusterResult:
    """Container for entropy clustering results.
    
    Attributes:
        assign: Cluster assignments (1 to n_clusters)
        indexes: List of arrays containing product indices for each cluster  
        counts: Number of products in each cluster
        entropy: Overall cluster entropy
        entropy_norm: Normalized entropy
        entropy_norm2: Normalized entropy scaled by cluster size
        n_iter: Number of iterations
        n_clusters: Number of clusters
        n_items: Number of items
        sub_criteria: List of sub-criteria values for each cluster
        sales: Sales data for each cluster
        all_sales: Total sales for each cluster
        max_obj: Objective function value
    """
    assign: NDArray[np.int64]
    indexes: List[NDArray[np.int64]]
    counts: List[int]
    entropy: float
    entropy_norm: float
    entropy_norm2: float
    n_iter: int
    n_clusters: int
    n_items: int
    sub_criteria: List[float]
    sales: List[NDArray[np.float64]]
    all_sales: List[float]
    max_obj: float
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EntropyClusterResult("
            f"n_clusters={self.n_clusters}, "
            f"n_items={self.n_items}, "
            f"entropy={self.entropy:.6f}, "
            f"entropy_norm={self.entropy_norm:.6f}, "
            f"n_iter={self.n_iter})"
        )
    
    def summary(self) -> str:
        """Generate detailed summary."""
        lines = [
            f"Entropy Clustering Results",
            f"=" * 40,
            f"Number of clusters: {self.n_clusters}",
            f"Number of items: {self.n_items}",
            f"Iterations: {self.n_iter}",
            f"",
            f"Cluster sizes:",
        ]
        
        for i, count in enumerate(self.counts):
            lines.append(f"  Cluster {i+1}: {count} items")
        
        lines.extend([
            f"",
            f"Entropy metrics:",
            f"  Total entropy: {self.entropy:.6f}",
            f"  Normalized entropy: {self.entropy_norm:.6f}",
            f"  Size-scaled normalized: {self.entropy_norm2:.6f}",
            f"  Max objective: {self.max_obj:.6f}",
        ])
        
        return "\n".join(lines)


class EntropyClusterer(BaseClusterer):
    """Entropy-based clustering algorithm for SUBMARIT.
    
    This implements the entropy clustering method where at each stage
    items are assigned to clusters to optimize entropy-based criteria.
    """
    
    def __init__(
        self,
        n_clusters: int = 2,
        min_items: int = 1,
        opt_mode: int = 1,
        max_iter: int = 1000,
        random_state: Optional[int] = None
    ):
        """Initialize entropy clusterer.
        
        Args:
            n_clusters: Number of clusters
            min_items: Minimum number of items per cluster
            opt_mode: Optimization mode
                1 - Optimize ENT (total entropy)
                2 - Optimize ENTNorm (normalized entropy)
                3 - Optimize ENTNorm2 (size-scaled normalized entropy)
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.min_items = min_items
        self.opt_mode = opt_mode
        self.max_iter = max_iter
        self._result = None
        
    def fit(
        self,
        swm: NDArray[np.float64],
        initial_assign: Optional[NDArray[np.int64]] = None
    ) -> "EntropyClusterer":
        """Fit the entropy clustering model.
        
        Args:
            swm: Product x product switching matrix
            initial_assign: Optional initial cluster assignments
            
        Returns:
            Self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_items = swm.shape[0]
        
        # Check minimum items requirement
        total_min_items = max(self.min_items * self.n_clusters * 2, 1)
        if n_items < total_min_items:
            raise ValueError(
                f"Need at least {total_min_items} items for "
                f"{self.n_clusters} clusters with min {self.min_items} items each"
            )
        
        # Calculate sales
        sales = np.sum(swm, axis=1)
        
        # Initialize cluster assignments
        if initial_assign is None:
            assign = self._initialize_assignments(n_items)
        else:
            assign = initial_assign.copy()
            
        # Initialize cluster information
        clusters = self._update_cluster_info(assign, swm, sales)
        
        # Iterative optimization
        n_iter = 0
        changes = 1
        
        while changes > 0 and n_iter < self.max_iter:
            n_iter += 1
            old_assign = assign.copy()
            
            # Random order for item updates
            rand_items = np.random.permutation(n_items)
            
            for i_item in rand_items:
                # Find current cluster
                old_cluster = assign[i_item] - 1  # Convert to 0-based
                
                # Skip if removing would violate min items constraint
                if clusters['counts'][old_cluster] <= self.min_items:
                    continue
                
                # Calculate objective change for each possible cluster
                obj_changes = np.zeros(self.n_clusters)
                
                for j_clus in range(self.n_clusters):
                    # Calculate objective change
                    obj_change = self._calculate_objective_change(
                        i_item, j_clus, old_cluster, clusters, swm, sales
                    )
                    obj_changes[j_clus] = obj_change
                
                # Choose best cluster
                new_cluster = np.argmax(obj_changes)
                
                # Update if beneficial
                if new_cluster != old_cluster and obj_changes[new_cluster] > 0:
                    # Update assignment
                    assign[i_item] = new_cluster + 1  # Convert to 1-based
                    
                    # Update cluster information efficiently
                    self._update_cluster_assignment(
                        i_item, old_cluster, new_cluster, clusters, swm, sales,
                        obj_changes[new_cluster], obj_changes[old_cluster]
                    )
            
            # Count changes
            changes = np.sum(old_assign != assign)
        
        # Calculate final entropy values
        self._result = self._calculate_final_entropy(assign, swm, sales, n_iter)
        self._labels = assign
        self._n_iter = n_iter
        
        return self
        
    def _initialize_assignments(self, n_items: int) -> NDArray[np.int64]:
        """Initialize random cluster assignments ensuring minimum items."""
        min_item_count = 0
        
        while min_item_count < self.min_items:
            # Random assignments (1-based)
            assign = np.random.randint(1, self.n_clusters + 1, size=n_items)
            
            # Check minimum count
            min_item_count = n_items
            for i in range(1, self.n_clusters + 1):
                count = np.sum(assign == i)
                min_item_count = min(count, min_item_count)
                
        return assign
    
    def _update_cluster_info(
        self,
        assign: NDArray[np.int64],
        swm: NDArray[np.float64],
        sales: NDArray[np.float64]
    ) -> Dict:
        """Update cluster information based on current assignments."""
        clusters = {
            'indexes': [],
            'counts': [],
            'sales': [],
            'all_sales': [],
            'sub_criteria': []
        }
        
        for i in range(self.n_clusters):
            # Get indices for cluster (1-based assignments)
            idx = np.where(assign == i + 1)[0]
            clusters['indexes'].append(idx)
            clusters['counts'].append(len(idx))
            
            if len(idx) > 0:
                cluster_sales = sales[idx]
                clusters['sales'].append(cluster_sales)
                clusters['all_sales'].append(np.sum(cluster_sales))
                
                # Calculate entropy for this cluster
                p = cluster_sales / clusters['all_sales'][i]
                p_cond = swm[np.ix_(idx, idx)] / (cluster_sales[:, np.newaxis] + 1e-10)
                
                # Calculate entropy (handle log(0) = -inf)
                temp = p_cond * np.log(p_cond + 1e-10)
                temp = temp * p[:, np.newaxis]
                
                # Set inf/nan values to 0
                h_all = np.where(np.isfinite(temp), temp, 0)
                ent_sum = -np.sum(h_all)
                
                # Calculate criterion based on mode
                if self.opt_mode == 1:
                    criterion = ent_sum
                elif self.opt_mode == 2:
                    criterion = ent_sum / (np.log(len(idx)) + 1e-10)
                else:  # opt_mode == 3
                    criterion = len(idx) * ent_sum / (np.log(len(idx)) + 1e-10)
                    
                clusters['sub_criteria'].append(criterion)
            else:
                clusters['sales'].append(np.array([]))
                clusters['all_sales'].append(0)
                clusters['sub_criteria'].append(0)
                
        return clusters
    
    def _calculate_objective_change(
        self,
        item: int,
        new_cluster: int,
        old_cluster: int,
        clusters: Dict,
        swm: NDArray[np.float64],
        sales: NDArray[np.float64]
    ) -> float:
        """Calculate change in objective function from moving an item."""
        # Get indices excluding current item
        if new_cluster == old_cluster:
            ex_indexes = clusters['indexes'][new_cluster][
                clusters['indexes'][new_cluster] != item
            ]
        else:
            ex_indexes = np.append(clusters['indexes'][new_cluster], item)
            
        ex_indexes = np.sort(ex_indexes)
        ex_count = len(ex_indexes)
        
        if ex_count == 0:
            return -np.inf
            
        # Calculate new entropy
        new_sales = np.sum(swm[np.ix_(ex_indexes, ex_indexes)], axis=1)
        new_all_sales = np.sum(new_sales)
        
        if new_all_sales == 0:
            return -np.inf
            
        new_p = new_sales / new_all_sales
        new_p_cond = swm[np.ix_(ex_indexes, ex_indexes)] / (new_sales[:, np.newaxis] + 1e-10)
        
        # Calculate entropy
        temp = new_p_cond * np.log(new_p_cond + 1e-10)
        temp = temp * new_p[:, np.newaxis]
        h_all = np.where(np.isfinite(temp), temp, 0)
        ent_sum = -np.sum(h_all)
        
        # Calculate new criterion
        if self.opt_mode == 1:
            new_ent = ent_sum
        elif self.opt_mode == 2:
            new_ent = ent_sum / (np.log(ex_count) + 1e-10)
        else:  # opt_mode == 3
            new_ent = ex_count * ent_sum / (np.log(ex_count) + 1e-10)
        
        # Calculate change
        if ex_count < clusters['counts'][new_cluster]:
            # Removing item
            obj_change = clusters['sub_criteria'][new_cluster] - new_ent
        else:
            # Adding item
            obj_change = new_ent - clusters['sub_criteria'][new_cluster]
            
        return obj_change
    
    def _update_cluster_assignment(
        self,
        item: int,
        old_cluster: int,
        new_cluster: int,
        clusters: Dict,
        swm: NDArray[np.float64],
        sales: NDArray[np.float64],
        new_obj_change: float,
        old_obj_change: float
    ) -> None:
        """Update cluster information after moving an item."""
        # Update new cluster
        clusters['indexes'][new_cluster] = np.sort(
            np.append(clusters['indexes'][new_cluster], item)
        )
        clusters['counts'][new_cluster] += 1
        clusters['sub_criteria'][new_cluster] += new_obj_change
        
        # Update old cluster
        clusters['indexes'][old_cluster] = clusters['indexes'][old_cluster][
            clusters['indexes'][old_cluster] != item
        ]
        clusters['counts'][old_cluster] -= 1
        clusters['sub_criteria'][old_cluster] -= old_obj_change
        
    def _calculate_final_entropy(
        self,
        assign: NDArray[np.int64],
        swm: NDArray[np.float64],
        sales: NDArray[np.float64],
        n_iter: int
    ) -> EntropyClusterResult:
        """Calculate final entropy values for all clusters."""
        n_items = swm.shape[0]
        
        # Initialize results
        indexes = []
        counts = []
        cluster_sales = []
        all_sales = []
        sub_criteria = []
        
        total_ent = 0.0
        total_ent_norm = 0.0
        total_ent_norm2 = 0.0
        
        for i in range(1, self.n_clusters + 1):
            idx = np.where(assign == i)[0]
            indexes.append(idx)
            sn = len(idx)
            counts.append(sn)
            
            if sn > 0:
                # Extract submatrix
                x = swm[np.ix_(idx, idx)]
                
                # Calculate market share
                sales_i = np.sum(x, axis=1)
                all_sales_i = np.sum(sales_i)
                cluster_sales.append(sales_i)
                all_sales.append(all_sales_i)
                
                if all_sales_i > 0:
                    p = sales_i / all_sales_i
                    p_cond = x / (sales_i[:, np.newaxis] + 1e-10)
                    
                    # Calculate entropy
                    temp = p_cond * np.log(p_cond + 1e-10)
                    temp = temp * p[:, np.newaxis]
                    h_all = np.where(np.isfinite(temp), temp, 0)
                    
                    ent = -np.sum(h_all)
                    total_ent += ent
                    
                    if sn > 1:
                        ent_norm = ent / np.log(sn)
                        total_ent_norm += ent_norm
                        total_ent_norm2 += sn * ent_norm
                    
                    # Store sub-criterion
                    if self.opt_mode == 1:
                        sub_criteria.append(ent)
                    elif self.opt_mode == 2:
                        sub_criteria.append(ent / (np.log(sn) + 1e-10))
                    else:
                        sub_criteria.append(sn * ent / (np.log(sn) + 1e-10))
                else:
                    cluster_sales.append(sales_i)
                    all_sales.append(0)
                    sub_criteria.append(0)
            else:
                cluster_sales.append(np.array([]))
                all_sales.append(0)
                sub_criteria.append(0)
        
        # Normalize
        entropy_norm = total_ent_norm / self.n_clusters
        entropy_norm2 = total_ent_norm2 / n_items
        
        # Determine max objective
        if self.opt_mode == 1:
            max_obj = total_ent
        elif self.opt_mode == 2:
            max_obj = entropy_norm
        else:
            max_obj = entropy_norm2
            
        return EntropyClusterResult(
            assign=assign,
            indexes=indexes,
            counts=counts,
            entropy=total_ent,
            entropy_norm=entropy_norm,
            entropy_norm2=entropy_norm2,
            n_iter=n_iter,
            n_clusters=self.n_clusters,
            n_items=n_items,
            sub_criteria=sub_criteria,
            sales=cluster_sales,
            all_sales=all_sales,
            max_obj=max_obj
        )
    
    def get_result(self) -> Optional[EntropyClusterResult]:
        """Get the detailed clustering result."""
        return self._result