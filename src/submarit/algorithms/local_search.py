"""k-Submarket Local Search Clustering Algorithms.

This module implements local search algorithms for k-submarket clustering based on
switching matrices. It includes both standard and constrained versions, with two
optimization approaches:

1. kSMLocalSearch/kSMLocalSearchConstrained: Uses a quick approximation based on
   maximizing (PHat - P) difference
2. kSMLocalSearch2/kSMLocalSearchConstrained2: Direct optimization of log-likelihood

The algorithms identify submarkets (clusters) of products based on customer switching
behavior, where products within a submarket have higher substitution rates than
products across submarkets.
"""

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ..core.base import BaseClusterer
from ..utils.matlab_compat import MatlabRandom, ensure_matlab_compatibility


class LocalSearchResult:
    """Container for local search clustering results.
    
    This class stores all outputs from the k-submarket local search algorithms,
    maintaining compatibility with the MATLAB output structure.
    """
    
    def __init__(self):
        """Initialize empty result container."""
        self.SWM: Optional[NDArray[np.float64]] = None
        self.NoClusters: Optional[int] = None
        self.NoItems: Optional[int] = None
        self.Assign: Optional[NDArray[np.int64]] = None
        self.Indexes: Dict[int, NDArray[np.int64]] = {}
        self.Count: Dict[int, int] = {}
        self.Diff: Optional[float] = None
        self.DiffSq: Optional[float] = None
        self.ItemDiff: Optional[float] = None
        self.ScaledDiff: Optional[float] = None
        self.ZValue: Optional[float] = None
        self.MaxObj: Optional[float] = None
        self.LogLH: Optional[float] = None
        self.LogLH2: Optional[float] = None
        self.Iter: Optional[int] = None
        self.Var: Dict[int, NDArray[np.float64]] = {}
        self.SDComp: Dict[int, float] = {}
        self.SDiff: Dict[int, float] = {}
        self.SLogLH: Dict[int, float] = {}  # For version 2 algorithms
        
    def to_dict(self) -> Dict:
        """Convert result to dictionary format."""
        return {
            'SWM': self.SWM,
            'NoClusters': self.NoClusters,
            'NoItems': self.NoItems,
            'Assign': self.Assign,
            'Indexes': self.Indexes,
            'Count': self.Count,
            'Diff': self.Diff,
            'DiffSq': self.DiffSq,
            'ItemDiff': self.ItemDiff,
            'ScaledDiff': self.ScaledDiff,
            'ZValue': self.ZValue,
            'MaxObj': self.MaxObj,
            'LogLH': self.LogLH,
            'LogLH2': self.LogLH2,
            'Iter': self.Iter,
            'Var': self.Var,
            'SDComp': self.SDComp,
            'SDiff': self.SDiff
        }


class KSMLocalSearch(BaseClusterer):
    """k-Submarket Local Search clustering algorithm.
    
    This algorithm clusters products based on a switching matrix using local search
    optimization. At each iteration, items are moved to clusters that maximize the
    difference between observed switching proportions (PHat) and expected proportions
    under independence (P).
    
    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters (submarkets) to find
    min_items : int, default=1
        Minimum number of items required in each cluster
    max_iter : int, default=100
        Maximum number of iterations
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_clusters: int = 2,
        min_items: int = 1,
        max_iter: int = 100,
        random_state: Optional[int] = None
    ):
        """Initialize the local search clusterer."""
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.min_items = min_items
        self.max_iter = max_iter
        self.result_ = None
        
    @ensure_matlab_compatibility
    def fit(self, X: NDArray[np.float64]) -> "KSMLocalSearch":
        """Fit the k-submarket local search model.
        
        Parameters
        ----------
        X : ndarray of shape (n_items, n_items)
            Switching matrix where X[i,j] represents switches from item i to item j
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate input
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("Switching matrix must be square")
            
        n_items = X.shape[0]
        
        # Check minimum items requirement
        total_min_items = max(self.min_items * self.n_clusters * 2, self.min_items * self.n_clusters)
        if n_items < total_min_items:
            raise ValueError(
                f"A minimum of {total_min_items} items is required for "
                f"{self.n_clusters} clusters with {self.min_items} minimum items per cluster"
            )
            
        # Initialize result
        result = LocalSearchResult()
        result.SWM = X.copy()
        result.NoClusters = self.n_clusters
        result.NoItems = n_items
        
        # Calculate switching proportions
        PSales = np.sum(X, axis=1)  # Row sums
        # Handle zero sales
        PSales_safe = np.where(PSales > 0, PSales, 1)
        PSWM = X / PSales_safe[:, np.newaxis]
        PPSales = PSales / np.sum(PSales)
        
        # Initialize random assignments ensuring minimum items per cluster
        rng = MatlabRandom(self.random_state)
        min_item_count = 0
        
        while min_item_count < self.min_items:
            # Random assignment (1-based like MATLAB)
            new_assign = rng.randi(self.n_clusters, n_items)
            
            # Check minimum item count
            min_item_count = n_items
            for i in range(1, self.n_clusters + 1):
                count = np.sum(new_assign == i)
                min_item_count = min(count, min_item_count)
                
        result.Assign = new_assign
        
        # Initialize cluster information
        for i in range(1, self.n_clusters + 1):
            cur_indexes = np.where(result.Assign == i)[0] + 1  # 1-based indices
            result.Indexes[i] = cur_indexes
            result.Count[i] = len(cur_indexes)
            
        # Local search iterations
        iter_count = 0
        as_change = 1
        
        while as_change > 0 and iter_count < self.max_iter:
            iter_count += 1
            old_assign = result.Assign.copy()
            
            # Random order for items (0-based for iteration)
            rand_items = rng.randperm(n_items) - 1  # Convert to 0-based
            
            for i_item in rand_items:
                phat_add = np.zeros(self.n_clusters)
                p_add = np.zeros(self.n_clusters)
                
                for j_clus in range(1, self.n_clusters + 1):
                    # Find indexes excluding current item (convert to 0-based)
                    ex_indexes = result.Indexes[j_clus][result.Indexes[j_clus] != (i_item + 1)] - 1
                    ex_count = len(ex_indexes)
                    
                    if ex_count > 0:
                        # Calculate PHat addition
                        cluster_idx_0based = result.Indexes[j_clus] - 1
                        phat_add[j_clus - 1] = (
                            np.sum(PSWM[i_item, cluster_idx_0based]) +
                            np.sum(PSWM[cluster_idx_0based, i_item])
                        )
                        
                        # Calculate P addition
                        p_term1 = np.sum(PPSales[ex_indexes]) / (1 - PPSales[i_item])
                        p_term2 = np.sum(PPSales[i_item] / (1 - PPSales[ex_indexes]))
                        p_add[j_clus - 1] = p_term1 + p_term2
                        
                # Choose best cluster
                group_diff = phat_add - p_add
                new_cluster = np.argmax(group_diff) + 1  # 1-based
                old_cluster = old_assign[i_item]
                
                # Move item if beneficial and maintains minimum items
                if new_cluster != old_cluster and result.Count[old_cluster] > self.min_items:
                    # Update new cluster
                    result.Indexes[new_cluster] = np.sort(
                        np.append(result.Indexes[new_cluster], i_item + 1)
                    )
                    result.Count[new_cluster] += 1
                    
                    # Update old cluster
                    result.Indexes[old_cluster] = result.Indexes[old_cluster][
                        result.Indexes[old_cluster] != (i_item + 1)
                    ]
                    result.Count[old_cluster] -= 1
                    
                    # Update assignment
                    result.Assign[i_item] = new_cluster
                    
            # Check for changes
            ch_assign = old_assign != result.Assign
            as_change = np.sum(ch_assign)
            
        # Calculate final objective function values
        PHat = np.zeros(n_items)
        P = np.zeros(n_items)
        result.LogLH = 0.0
        result.LogLH2 = 0.0
        
        for i_clus in range(1, self.n_clusters + 1):
            indexes = result.Indexes[i_clus] - 1  # 0-based for indexing
            sub_swm = PSWM[np.ix_(indexes, indexes)]
            
            # PHat values
            PHat[indexes] = np.sum(sub_swm, axis=1)
            
            # P values
            spp_sales = PPSales[indexes]
            props = np.outer(np.ones(result.Count[i_clus]), spp_sales) - np.diag(spp_sales)
            P[indexes] = np.sum(props, axis=1) / (1 - spp_sales)
            
            # Variance and log-likelihood components
            result.Var[i_clus] = P[indexes] * (1 - P[indexes]) * PSales[indexes]
            var_sum = np.sum(result.Var[i_clus])
            
            result.SDComp[i_clus] = np.log(1 / (np.sqrt(var_sum * 2 * np.pi)))
            result.SDiff[i_clus] = (
                np.sum(PHat[indexes] * PSales[indexes]) -
                np.sum(P[indexes] * PSales[indexes])
            )
            
            # Log-likelihood calculation
            if var_sum > 0:
                result.LogLH += (
                    result.SDComp[i_clus] -
                    (np.sign(result.SDiff[i_clus]) * result.SDiff[i_clus]**2) / (2 * var_sum)
                )
                result.LogLH2 += (
                    result.SDComp[i_clus] -
                    (result.SDiff[i_clus]**2) / (2 * var_sum)
                )
                
        # Calculate summary statistics
        result.Diff = np.sum(PHat - P)
        result.DiffSq = np.sum((PHat - P)**2)
        result.ItemDiff = result.Diff / n_items
        
        # Scaled differences
        valid_ix = (~np.isinf(P)) & (~np.isnan(P)) & (P != 0)
        result.ScaledDiff = np.sum((PHat[valid_ix] - P[valid_ix]) / P[valid_ix])
        
        # Z-value calculation
        mphat = np.sum(PHat * PSales)
        mp = np.sum(P * PSales)
        denom = np.sqrt(np.sum(PHat * (1 - PHat) * PSales))
        if denom > 0:
            result.ZValue = (mphat - mp) / denom
        else:
            result.ZValue = 0.0
            
        result.MaxObj = result.Diff
        result.Iter = iter_count
        
        # Store results
        self.result_ = result
        self._labels = result.Assign - 1  # Convert to 0-based for sklearn compatibility
        self._n_iter = iter_count
        
        return self
        
    def get_result(self) -> LocalSearchResult:
        """Get the detailed clustering result.
        
        Returns
        -------
        result : LocalSearchResult
            Complete clustering result with all statistics
        """
        if self.result_ is None:
            raise ValueError("Model must be fitted before getting results")
        return self.result_


class KSMLocalSearch2(KSMLocalSearch):
    """k-Submarket Local Search with log-likelihood optimization.
    
    This variant directly optimizes the log-likelihood objective function
    rather than using the (PHat - P) approximation. It maintains incremental
    log-likelihood values for efficiency.
    
    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters (submarkets) to find
    min_items : int, default=1
        Minimum number of items required in each cluster
    max_iter : int, default=100
        Maximum number of iterations
    random_state : int, optional
        Random seed for reproducibility
    """
    
    @ensure_matlab_compatibility
    def fit(self, X: NDArray[np.float64]) -> "KSMLocalSearch2":
        """Fit the k-submarket local search model with log-likelihood optimization.
        
        Parameters
        ----------
        X : ndarray of shape (n_items, n_items)
            Switching matrix where X[i,j] represents switches from item i to item j
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate input
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("Switching matrix must be square")
            
        n_items = X.shape[0]
        
        # Check minimum items requirement
        total_min_items = max(self.min_items * self.n_clusters * 2, self.min_items * self.n_clusters)
        if n_items < total_min_items:
            raise ValueError(
                f"A minimum of {total_min_items} items is required for "
                f"{self.n_clusters} clusters with {self.min_items} minimum items per cluster"
            )
            
        # Initialize result
        result = LocalSearchResult()
        result.SWM = X.copy()
        result.NoClusters = self.n_clusters
        result.NoItems = n_items
        
        # Calculate switching proportions
        PSales = np.sum(X, axis=1)
        PSales_safe = np.where(PSales > 0, PSales, 1)
        PSWM = X / PSales_safe[:, np.newaxis]
        PPSales = PSales / np.sum(PSales)
        
        # Initialize random assignments ensuring minimum items per cluster
        rng = MatlabRandom(self.random_state)
        min_item_count = 0
        
        while min_item_count < self.min_items:
            new_assign = rng.randi(self.n_clusters, n_items)
            min_item_count = n_items
            for i in range(1, self.n_clusters + 1):
                count = np.sum(new_assign == i)
                min_item_count = min(count, min_item_count)
                
        result.Assign = new_assign
        
        # Initialize cluster information and log-likelihoods
        PHat = np.zeros(n_items)
        P = np.zeros(n_items)
        
        for i_clus in range(1, self.n_clusters + 1):
            cur_indexes = np.where(result.Assign == i_clus)[0] + 1
            result.Indexes[i_clus] = cur_indexes
            result.Count[i_clus] = len(cur_indexes)
            
            # Calculate initial log-likelihood for each cluster
            indexes = cur_indexes - 1  # 0-based
            sub_swm = PSWM[np.ix_(indexes, indexes)]
            PHat[indexes] = np.sum(sub_swm, axis=1)
            
            spp_sales = PPSales[indexes]
            props = np.outer(np.ones(result.Count[i_clus]), spp_sales) - np.diag(spp_sales)
            P[indexes] = np.sum(props, axis=1) / (1 - spp_sales)
            
            result.Var[i_clus] = P[indexes] * (1 - P[indexes]) * PSales[indexes]
            var_sum = np.sum(result.Var[i_clus])
            result.SDComp[i_clus] = np.log(1 / (np.sqrt(var_sum * 2 * np.pi)))
            result.SDiff[i_clus] = (
                np.sum(PHat[indexes] * PSales[indexes]) -
                np.sum(P[indexes] * PSales[indexes])
            )
            
            if var_sum > 0:
                result.SLogLH[i_clus] = (
                    result.SDComp[i_clus] -
                    (np.sign(result.SDiff[i_clus]) * result.SDiff[i_clus]**2) / (2 * var_sum)
                )
            else:
                result.SLogLH[i_clus] = result.SDComp[i_clus]
                
        # Local search iterations
        iter_count = 0
        as_change = 1
        
        while as_change > 0 and iter_count < self.max_iter:
            iter_count += 1
            old_assign = result.Assign.copy()
            rand_items = rng.randperm(n_items) - 1
            
            for i_item in rand_items:
                obj_change = np.zeros(self.n_clusters)
                old_cluster = None
                
                for j_clus in range(1, self.n_clusters + 1):
                    ex_indexes = result.Indexes[j_clus][result.Indexes[j_clus] != (i_item + 1)] - 1
                    ex_count = len(ex_indexes)
                    
                    if ex_count == result.Count[j_clus]:
                        # Adding item to cluster
                        ex_indexes = np.sort(np.append(ex_indexes, i_item))
                        ex_count += 1
                    else:
                        # Item is in this cluster
                        old_cluster = j_clus
                        
                    # Calculate log-likelihood with this configuration
                    sub_swm = PSWM[np.ix_(ex_indexes, ex_indexes)]
                    PHat[ex_indexes] = np.sum(sub_swm, axis=1)
                    spp_sales = PPSales[ex_indexes]
                    props = np.outer(np.ones(ex_count), spp_sales) - np.diag(spp_sales)
                    P[ex_indexes] = np.sum(props, axis=1) / (1 - spp_sales)
                    
                    s_var = P[ex_indexes] * (1 - P[ex_indexes]) * PSales[ex_indexes]
                    var_sum = np.sum(s_var)
                    sd_comp = np.log(1 / (np.sqrt(var_sum * 2 * np.pi)))
                    s_diff = (
                        np.sum(PHat[ex_indexes] * PSales[ex_indexes]) -
                        np.sum(P[ex_indexes] * PSales[ex_indexes])
                    )
                    
                    if var_sum > 0:
                        new_log_lh = sd_comp - (np.sign(s_diff) * s_diff**2) / (2 * var_sum)
                    else:
                        new_log_lh = sd_comp
                        
                    if ex_count < result.Count[j_clus]:
                        # Removing item
                        obj_change[j_clus - 1] = result.SLogLH[j_clus] - new_log_lh
                    else:
                        # Adding item
                        obj_change[j_clus - 1] = new_log_lh - result.SLogLH[j_clus]
                        
                # Choose best cluster (minimize objective change)
                new_cluster = np.argmin(obj_change) + 1
                
                if new_cluster != old_cluster and result.Count[old_cluster] > self.min_items:
                    # Update clusters
                    result.Indexes[new_cluster] = np.sort(
                        np.append(result.Indexes[new_cluster], i_item + 1)
                    )
                    result.Count[new_cluster] += 1
                    result.Indexes[old_cluster] = result.Indexes[old_cluster][
                        result.Indexes[old_cluster] != (i_item + 1)
                    ]
                    result.Count[old_cluster] -= 1
                    result.Assign[i_item] = new_cluster
                    
                    # Update log-likelihoods
                    result.SLogLH[new_cluster] += obj_change[new_cluster - 1]
                    result.SLogLH[old_cluster] -= obj_change[old_cluster - 1]
                    
            ch_assign = old_assign != result.Assign
            as_change = np.sum(ch_assign)
            
        # Calculate final statistics
        PHat = np.zeros(n_items)
        P = np.zeros(n_items)
        result.LogLH = 0.0
        
        for i_clus in range(1, self.n_clusters + 1):
            indexes = result.Indexes[i_clus] - 1
            sub_swm = PSWM[np.ix_(indexes, indexes)]
            PHat[indexes] = np.sum(sub_swm, axis=1)
            
            spp_sales = PPSales[indexes]
            props = np.outer(np.ones(result.Count[i_clus]), spp_sales) - np.diag(spp_sales)
            P[indexes] = np.sum(props, axis=1) / (1 - spp_sales)
            
            result.Var[i_clus] = P[indexes] * (1 - P[indexes]) * PSales[indexes]
            var_sum = np.sum(result.Var[i_clus])
            result.SDComp[i_clus] = np.log(1 / (np.sqrt(var_sum * 2 * np.pi)))
            result.SDiff[i_clus] = (
                np.sum(PHat[indexes] * PSales[indexes]) -
                np.sum(P[indexes] * PSales[indexes])
            )
            
            if var_sum > 0:
                result.LogLH += (
                    result.SDComp[i_clus] -
                    (np.sign(result.SDiff[i_clus]) * result.SDiff[i_clus]**2) / (2 * var_sum)
                )
                
        # Summary statistics
        result.Diff = np.sum(PHat - P)
        result.ItemDiff = result.Diff / n_items
        
        valid_ix = (~np.isinf(P)) & (~np.isnan(P)) & (P != 0)
        result.ScaledDiff = np.sum((PHat[valid_ix] - P[valid_ix]) / P[valid_ix])
        
        mphat = np.sum(PHat * PSales)
        mp = np.sum(P * PSales)
        denom = np.sqrt(np.sum(PHat * (1 - PHat) * PSales))
        if denom > 0:
            result.ZValue = (mphat - mp) / denom
        else:
            result.ZValue = 0.0
            
        result.MaxObj = -result.LogLH  # Negative for minimization
        result.Iter = iter_count
        
        self.result_ = result
        self._labels = result.Assign - 1
        self._n_iter = iter_count
        
        return self


class KSMLocalSearchConstrained(KSMLocalSearch):
    """k-Submarket Local Search with constrained assignments.
    
    This variant allows fixing some items to specific clusters while optimizing
    the assignment of free items. Uses the (PHat - P) difference objective.
    
    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters (submarkets) to find
    min_items : int, default=1
        Minimum number of items required in each cluster
    max_iter : int, default=100
        Maximum number of iterations
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def fit_constrained(
        self,
        X: NDArray[np.float64],
        fixed_assign: NDArray[np.int64],
        assign_indexes: NDArray[np.int64],
        free_indexes: NDArray[np.int64]
    ) -> "KSMLocalSearchConstrained":
        """Fit the constrained k-submarket local search model.
        
        Parameters
        ----------
        X : ndarray of shape (n_items, n_items)
            Switching matrix
        fixed_assign : ndarray of shape (n_fixed,)
            Cluster assignments for fixed items (1-based)
        assign_indexes : ndarray of shape (n_fixed,)
            Indices of fixed items (1-based)
        free_indexes : ndarray of shape (n_free,)
            Indices of free items that can be reassigned (1-based)
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate input
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("Switching matrix must be square")
            
        n_items = X.shape[0]
        n_assign = len(assign_indexes)
        n_free = len(free_indexes)
        
        # Validate constraints
        if len(fixed_assign) != n_assign:
            raise ValueError(
                "fixed_assign must have same length as assign_indexes"
            )
        if n_free + n_assign != n_items:
            raise ValueError(
                "Number of free + assigned indexes must equal number of items"
            )
            
        # Check minimum items
        total_min_items = max(self.min_items * self.n_clusters * 2, self.min_items * self.n_clusters)
        if n_items < total_min_items:
            raise ValueError(
                f"A minimum of {total_min_items} items is required"
            )
            
        # Initialize result
        result = LocalSearchResult()
        result.SWM = X.copy()
        result.NoClusters = self.n_clusters
        result.NoItems = n_items
        
        # Calculate switching proportions
        PSales = np.sum(X, axis=1)
        PSales_safe = np.where(PSales > 0, PSales, 1)
        PSWM = X / PSales_safe[:, np.newaxis]
        PPSales = PSales / np.sum(PSales)
        
        # Initialize assignments with constraints
        initial_assign = np.zeros(n_items, dtype=np.int64)
        initial_assign[assign_indexes - 1] = fixed_assign  # Convert to 0-based indexing
        
        # Random assignment for free items
        rng = MatlabRandom(self.random_state)
        min_item_count = 0
        
        while min_item_count < self.min_items:
            initial_assign[free_indexes - 1] = rng.randi(self.n_clusters, n_free)
            min_item_count = n_items
            for i in range(1, self.n_clusters + 1):
                count = np.sum(initial_assign == i)
                min_item_count = min(count, min_item_count)
                
        result.Assign = initial_assign
        
        # Initialize cluster information
        for i in range(1, self.n_clusters + 1):
            cur_indexes = np.where(result.Assign == i)[0] + 1
            result.Indexes[i] = cur_indexes
            result.Count[i] = len(cur_indexes)
            
        # Local search iterations (only on free items)
        iter_count = 0
        as_change = 1
        
        while as_change > 0 and iter_count < self.max_iter:
            iter_count += 1
            old_assign = result.Assign.copy()
            
            # Random order for free items
            rand_items = free_indexes[rng.randperm(n_free) - 1] - 1  # Convert to 0-based
            
            for i_item in rand_items:
                phat_add = np.zeros(self.n_clusters)
                p_add = np.zeros(self.n_clusters)
                
                for j_clus in range(1, self.n_clusters + 1):
                    ex_indexes = result.Indexes[j_clus][result.Indexes[j_clus] != (i_item + 1)] - 1
                    ex_count = len(ex_indexes)
                    
                    if ex_count > 0:
                        cluster_idx_0based = result.Indexes[j_clus] - 1
                        phat_add[j_clus - 1] = (
                            np.sum(PSWM[i_item, cluster_idx_0based]) +
                            np.sum(PSWM[cluster_idx_0based, i_item])
                        )
                        
                        p_term1 = np.sum(PPSales[ex_indexes]) / (1 - PPSales[i_item])
                        p_term2 = np.sum(PPSales[i_item] / (1 - PPSales[ex_indexes]))
                        p_add[j_clus - 1] = p_term1 + p_term2
                        
                group_diff = phat_add - p_add
                new_cluster = np.argmax(group_diff) + 1
                old_cluster = old_assign[i_item]
                
                if new_cluster != old_cluster and result.Count[old_cluster] > self.min_items:
                    result.Indexes[new_cluster] = np.sort(
                        np.append(result.Indexes[new_cluster], i_item + 1)
                    )
                    result.Count[new_cluster] += 1
                    result.Indexes[old_cluster] = result.Indexes[old_cluster][
                        result.Indexes[old_cluster] != (i_item + 1)
                    ]
                    result.Count[old_cluster] -= 1
                    result.Assign[i_item] = new_cluster
                    
            ch_assign = old_assign != result.Assign
            as_change = np.sum(ch_assign)
            
        # Calculate final statistics (same as base class)
        self._calculate_final_statistics(result, PSWM, PPSales, PSales, n_items)
        result.Iter = iter_count
        
        self.result_ = result
        self._labels = result.Assign - 1
        self._n_iter = iter_count
        
        return self
        
    def _calculate_final_statistics(
        self,
        result: LocalSearchResult,
        PSWM: NDArray[np.float64],
        PPSales: NDArray[np.float64],
        PSales: NDArray[np.float64],
        n_items: int
    ) -> None:
        """Calculate final objective function values and statistics."""
        PHat = np.zeros(n_items)
        P = np.zeros(n_items)
        result.LogLH = 0.0
        result.LogLH2 = 0.0
        
        for i_clus in range(1, self.n_clusters + 1):
            indexes = result.Indexes[i_clus] - 1
            sub_swm = PSWM[np.ix_(indexes, indexes)]
            PHat[indexes] = np.sum(sub_swm, axis=1)
            
            spp_sales = PPSales[indexes]
            props = np.outer(np.ones(result.Count[i_clus]), spp_sales) - np.diag(spp_sales)
            P[indexes] = np.sum(props, axis=1) / (1 - spp_sales)
            
            result.Var[i_clus] = P[indexes] * (1 - P[indexes]) * PSales[indexes]
            var_sum = np.sum(result.Var[i_clus])
            result.SDComp[i_clus] = np.log(1 / (np.sqrt(var_sum * 2 * np.pi)))
            result.SDiff[i_clus] = (
                np.sum(PHat[indexes] * PSales[indexes]) -
                np.sum(P[indexes] * PSales[indexes])
            )
            
            if var_sum > 0:
                result.LogLH += (
                    result.SDComp[i_clus] -
                    (np.sign(result.SDiff[i_clus]) * result.SDiff[i_clus]**2) / (2 * var_sum)
                )
                result.LogLH2 += (
                    result.SDComp[i_clus] -
                    (result.SDiff[i_clus]**2) / (2 * var_sum)
                )
                
        result.Diff = np.sum(PHat - P)
        result.DiffSq = np.sum((PHat - P)**2)
        result.ItemDiff = result.Diff / n_items
        
        valid_ix = (~np.isinf(P)) & (~np.isnan(P)) & (P != 0)
        result.ScaledDiff = np.sum((PHat[valid_ix] - P[valid_ix]) / P[valid_ix])
        
        mphat = np.sum(PHat * PSales)
        mp = np.sum(P * PSales)
        denom = np.sqrt(np.sum(PHat * (1 - PHat) * PSales))
        if denom > 0:
            result.ZValue = (mphat - mp) / denom
        else:
            result.ZValue = 0.0
            
        result.MaxObj = result.Diff


class KSMLocalSearchConstrained2(KSMLocalSearchConstrained):
    """k-Submarket Local Search with constrained assignments and log-likelihood optimization.
    
    This variant combines constrained assignments with direct log-likelihood
    optimization, allowing fixed cluster assignments while optimizing free items.
    
    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters (submarkets) to find
    min_items : int, default=1
        Minimum number of items required in each cluster
    max_iter : int, default=100
        Maximum number of iterations
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def fit_constrained(
        self,
        X: NDArray[np.float64],
        fixed_assign: NDArray[np.int64],
        assign_indexes: NDArray[np.int64],
        free_indexes: NDArray[np.int64]
    ) -> "KSMLocalSearchConstrained2":
        """Fit the constrained k-submarket local search model with log-likelihood optimization.
        
        Parameters
        ----------
        X : ndarray of shape (n_items, n_items)
            Switching matrix
        fixed_assign : ndarray of shape (n_fixed,)
            Cluster assignments for fixed items (1-based)
        assign_indexes : ndarray of shape (n_fixed,)
            Indices of fixed items (1-based)
        free_indexes : ndarray of shape (n_free,)
            Indices of free items that can be reassigned (1-based)
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate input
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("Switching matrix must be square")
            
        n_items = X.shape[0]
        n_assign = len(assign_indexes)
        n_free = len(free_indexes)
        
        # Validate constraints
        if len(fixed_assign) != n_assign:
            raise ValueError("fixed_assign must have same length as assign_indexes")
        if n_free + n_assign != n_items:
            raise ValueError("Number of free + assigned indexes must equal number of items")
            
        # Check minimum items
        total_min_items = max(self.min_items * self.n_clusters * 2, self.min_items * self.n_clusters)
        if n_items < total_min_items:
            raise ValueError(f"A minimum of {total_min_items} items is required")
            
        # Initialize result
        result = LocalSearchResult()
        result.SWM = X.copy()
        result.NoClusters = self.n_clusters
        result.NoItems = n_items
        
        # Calculate switching proportions
        PSales = np.sum(X, axis=1)
        PSales_safe = np.where(PSales > 0, PSales, 1)
        PSWM = X / PSales_safe[:, np.newaxis]
        PPSales = PSales / np.sum(PSales)
        
        # Initialize assignments with constraints
        initial_assign = np.zeros(n_items, dtype=np.int64)
        initial_assign[assign_indexes - 1] = fixed_assign
        
        # Random assignment for free items
        rng = MatlabRandom(self.random_state)
        min_item_count = 0
        
        while min_item_count < self.min_items:
            initial_assign[free_indexes - 1] = rng.randi(self.n_clusters, n_free)
            min_item_count = n_items
            for i in range(1, self.n_clusters + 1):
                count = np.sum(initial_assign == i)
                min_item_count = min(count, min_item_count)
                
        result.Assign = initial_assign
        
        # Initialize clusters with log-likelihood tracking
        for i in range(1, self.n_clusters + 1):
            cur_indexes = np.where(result.Assign == i)[0] + 1
            result.Indexes[i] = cur_indexes
            result.Count[i] = len(cur_indexes)
            
        # Initialize log-likelihoods (same as KSMLocalSearch2)
        PHat = np.zeros(n_items)
        P = np.zeros(n_items)
        
        for i_clus in range(1, self.n_clusters + 1):
            indexes = result.Indexes[i_clus] - 1
            sub_swm = PSWM[np.ix_(indexes, indexes)]
            PHat[indexes] = np.sum(sub_swm, axis=1)
            
            spp_sales = PPSales[indexes]
            props = np.outer(np.ones(result.Count[i_clus]), spp_sales) - np.diag(spp_sales)
            P[indexes] = np.sum(props, axis=1) / (1 - spp_sales)
            
            result.Var[i_clus] = P[indexes] * (1 - P[indexes]) * PSales[indexes]
            var_sum = np.sum(result.Var[i_clus])
            result.SDComp[i_clus] = np.log(1 / (np.sqrt(var_sum * 2 * np.pi)))
            result.SDiff[i_clus] = (
                np.sum(PHat[indexes] * PSales[indexes]) -
                np.sum(P[indexes] * PSales[indexes])
            )
            
            if var_sum > 0:
                result.SLogLH[i_clus] = (
                    result.SDComp[i_clus] -
                    (np.sign(result.SDiff[i_clus]) * result.SDiff[i_clus]**2) / (2 * var_sum)
                )
            else:
                result.SLogLH[i_clus] = result.SDComp[i_clus]
                
        # Local search iterations (only on free items)
        iter_count = 0
        as_change = 1
        
        while as_change > 0 and iter_count < self.max_iter:
            iter_count += 1
            old_assign = result.Assign.copy()
            rand_items = free_indexes[rng.randperm(n_free) - 1] - 1
            
            for i_item in rand_items:
                obj_change = np.zeros(self.n_clusters)
                old_cluster = None
                
                for j_clus in range(1, self.n_clusters + 1):
                    ex_indexes = result.Indexes[j_clus][result.Indexes[j_clus] != (i_item + 1)] - 1
                    ex_count = len(ex_indexes)
                    
                    if ex_count == result.Count[j_clus]:
                        ex_indexes = np.sort(np.append(ex_indexes, i_item))
                        ex_count += 1
                    else:
                        old_cluster = j_clus
                        
                    # Calculate log-likelihood change
                    sub_swm = PSWM[np.ix_(ex_indexes, ex_indexes)]
                    PHat[ex_indexes] = np.sum(sub_swm, axis=1)
                    spp_sales = PPSales[ex_indexes]
                    props = np.outer(np.ones(ex_count), spp_sales) - np.diag(spp_sales)
                    P[ex_indexes] = np.sum(props, axis=1) / (1 - spp_sales)
                    
                    s_var = P[ex_indexes] * (1 - P[ex_indexes]) * PSales[ex_indexes]
                    var_sum = np.sum(s_var)
                    sd_comp = np.log(1 / (np.sqrt(var_sum * 2 * np.pi)))
                    s_diff = (
                        np.sum(PHat[ex_indexes] * PSales[ex_indexes]) -
                        np.sum(P[ex_indexes] * PSales[ex_indexes])
                    )
                    
                    if var_sum > 0:
                        new_log_lh = sd_comp - (np.sign(s_diff) * s_diff**2) / (2 * var_sum)
                    else:
                        new_log_lh = sd_comp
                        
                    if ex_count < result.Count[j_clus]:
                        obj_change[j_clus - 1] = result.SLogLH[j_clus] - new_log_lh
                    else:
                        obj_change[j_clus - 1] = new_log_lh - result.SLogLH[j_clus]
                        
                new_cluster = np.argmin(obj_change) + 1
                
                if new_cluster != old_cluster and result.Count[old_cluster] > self.min_items:
                    result.Indexes[new_cluster] = np.sort(
                        np.append(result.Indexes[new_cluster], i_item + 1)
                    )
                    result.Count[new_cluster] += 1
                    result.Indexes[old_cluster] = result.Indexes[old_cluster][
                        result.Indexes[old_cluster] != (i_item + 1)
                    ]
                    result.Count[old_cluster] -= 1
                    result.Assign[i_item] = new_cluster
                    
                    result.SLogLH[new_cluster] += obj_change[new_cluster - 1]
                    result.SLogLH[old_cluster] -= obj_change[old_cluster - 1]
                    
            ch_assign = old_assign != result.Assign
            as_change = np.sum(ch_assign)
            
        # Calculate final statistics
        self._calculate_final_statistics(result, PSWM, PPSales, PSales, n_items)
        result.MaxObj = result.Diff  # Note: MATLAB version has this as Diff, not -LogLH
        result.Iter = iter_count
        
        self.result_ = result
        self._labels = result.Assign - 1
        self._n_iter = iter_count
        
        return self


# Convenience functions for MATLAB-style interface
def k_sm_local_search(
    swm: NDArray[np.float64],
    n_clusters: int,
    min_items: int = 1,
    random_state: Optional[int] = None
) -> LocalSearchResult:
    """Run k-submarket local search clustering (MATLAB-compatible interface).
    
    Parameters
    ----------
    swm : ndarray of shape (n_items, n_items)
        Switching matrix
    n_clusters : int
        Number of clusters
    min_items : int, default=1
        Minimum items per cluster
    random_state : int, optional
        Random seed
        
    Returns
    -------
    result : LocalSearchResult
        Clustering result
    """
    model = KSMLocalSearch(
        n_clusters=n_clusters,
        min_items=min_items,
        random_state=random_state
    )
    model.fit(swm)
    return model.get_result()


def k_sm_local_search2(
    swm: NDArray[np.float64],
    n_clusters: int,
    min_items: int = 1,
    random_state: Optional[int] = None
) -> LocalSearchResult:
    """Run k-submarket local search with log-likelihood optimization.
    
    Parameters
    ----------
    swm : ndarray of shape (n_items, n_items)
        Switching matrix
    n_clusters : int
        Number of clusters
    min_items : int, default=1
        Minimum items per cluster
    random_state : int, optional
        Random seed
        
    Returns
    -------
    result : LocalSearchResult
        Clustering result
    """
    model = KSMLocalSearch2(
        n_clusters=n_clusters,
        min_items=min_items,
        random_state=random_state
    )
    model.fit(swm)
    return model.get_result()


def k_sm_local_search_constrained(
    swm: NDArray[np.float64],
    n_clusters: int,
    min_items: int,
    fixed_assign: NDArray[np.int64],
    assign_indexes: NDArray[np.int64],
    free_indexes: NDArray[np.int64],
    random_state: Optional[int] = None
) -> LocalSearchResult:
    """Run constrained k-submarket local search clustering.
    
    Parameters
    ----------
    swm : ndarray of shape (n_items, n_items)
        Switching matrix
    n_clusters : int
        Number of clusters
    min_items : int
        Minimum items per cluster
    fixed_assign : ndarray
        Fixed cluster assignments (1-based)
    assign_indexes : ndarray
        Indices of fixed items (1-based)
    free_indexes : ndarray
        Indices of free items (1-based)
    random_state : int, optional
        Random seed
        
    Returns
    -------
    result : LocalSearchResult
        Clustering result
    """
    model = KSMLocalSearchConstrained(
        n_clusters=n_clusters,
        min_items=min_items,
        random_state=random_state
    )
    model.fit_constrained(swm, fixed_assign, assign_indexes, free_indexes)
    return model.get_result()


def k_sm_local_search_constrained2(
    swm: NDArray[np.float64],
    n_clusters: int,
    min_items: int,
    fixed_assign: NDArray[np.int64],
    assign_indexes: NDArray[np.int64],
    free_indexes: NDArray[np.int64],
    random_state: Optional[int] = None
) -> LocalSearchResult:
    """Run constrained k-submarket local search with log-likelihood optimization.
    
    Parameters
    ----------
    swm : ndarray of shape (n_items, n_items)
        Switching matrix
    n_clusters : int
        Number of clusters
    min_items : int
        Minimum items per cluster
    fixed_assign : ndarray
        Fixed cluster assignments (1-based)
    assign_indexes : ndarray
        Indices of fixed items (1-based)
    free_indexes : ndarray
        Indices of free items (1-based)
    random_state : int, optional
        Random seed
        
    Returns
    -------
    result : LocalSearchResult
        Clustering result
    """
    model = KSMLocalSearchConstrained2(
        n_clusters=n_clusters,
        min_items=min_items,
        random_state=random_state
    )
    model.fit_constrained(swm, fixed_assign, assign_indexes, free_indexes)
    return model.get_result()