"""P-value calculation utilities for SUBMARIT validation.

This module provides utilities for calculating p-values from empirical
distributions, supporting both one-tailed and two-tailed tests, as well
as multiple testing corrections.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import warnings
from scipy import stats


@dataclass
class PValueResult:
    """Container for p-value calculation results.
    
    Attributes:
        observed: Observed test statistic value
        p_value: Raw p-value
        p_value_corrected: Corrected p-value (if correction applied)
        percentile: Percentile of observed value in distribution
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        direction: Test direction ('greater', 'less', or 'two-sided')
        n_simulations: Number of simulations in empirical distribution
    """
    observed: float
    p_value: float
    p_value_corrected: Optional[float] = None
    percentile: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    direction: str = 'greater'
    n_simulations: Optional[int] = None
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant.
        
        Parameters
        ----------
        alpha : float, default=0.05
            Significance level
            
        Returns
        -------
        bool
            True if p-value < alpha
        """
        p = self.p_value_corrected if self.p_value_corrected is not None else self.p_value
        return p < alpha
        
    def summary(self) -> str:
        """Generate summary of p-value results."""
        lines = [
            f"P-value Analysis Results",
            f"=======================",
            f"Observed value: {self.observed:.6f}",
            f"P-value ({self.direction}): {self.p_value:.4f}",
        ]
        
        if self.p_value_corrected is not None:
            lines.append(f"Corrected p-value: {self.p_value_corrected:.4f}")
            
        if self.percentile is not None:
            lines.append(f"Percentile: {self.percentile:.1f}%")
            
        if self.ci_lower is not None and self.ci_upper is not None:
            lines.append(f"95% CI: [{self.ci_lower:.6f}, {self.ci_upper:.6f}]")
            
        if self.n_simulations is not None:
            lines.append(f"Based on {self.n_simulations} simulations")
            
        return "\n".join(lines)


def calculate_empirical_p_value(
    observed: float,
    null_distribution: NDArray[np.float64],
    direction: str = 'greater',
    return_percentile: bool = True
) -> PValueResult:
    """Calculate p-value from empirical distribution.
    
    Parameters
    ----------
    observed : float
        Observed test statistic
    null_distribution : ndarray
        Null distribution values (should be sorted)
    direction : {'greater', 'less', 'two-sided'}, default='greater'
        Direction of the test:
        - 'greater': P(X >= observed)
        - 'less': P(X <= observed)
        - 'two-sided': 2 * min(P(X >= observed), P(X <= observed))
    return_percentile : bool, default=True
        Whether to calculate percentile of observed value
        
    Returns
    -------
    result : PValueResult
        P-value calculation results
    """
    n = len(null_distribution)
    
    if n == 0:
        raise ValueError("Null distribution is empty")
        
    # Ensure distribution is sorted
    if not np.all(null_distribution[:-1] <= null_distribution[1:]):
        null_distribution = np.sort(null_distribution)
        
    # Calculate p-value based on direction
    if direction == 'greater':
        # P(X >= observed)
        n_greater_equal = np.sum(null_distribution >= observed)
        p_value = (n_greater_equal + 1) / (n + 1)  # Add 1 for continuity correction
        
    elif direction == 'less':
        # P(X <= observed)
        n_less_equal = np.sum(null_distribution <= observed)
        p_value = (n_less_equal + 1) / (n + 1)
        
    elif direction == 'two-sided':
        # Two-sided test
        n_greater_equal = np.sum(null_distribution >= observed)
        n_less_equal = np.sum(null_distribution <= observed)
        p_greater = (n_greater_equal + 1) / (n + 1)
        p_less = (n_less_equal + 1) / (n + 1)
        p_value = 2 * min(p_greater, p_less)
        p_value = min(p_value, 1.0)  # Ensure p <= 1
        
    else:
        raise ValueError(f"Invalid direction: {direction}")
        
    # Calculate percentile if requested
    percentile = None
    if return_percentile:
        n_below = np.sum(null_distribution < observed)
        percentile = 100 * n_below / n
        
    # Calculate confidence interval (2.5% and 97.5% percentiles)
    ci_lower = np.percentile(null_distribution, 2.5)
    ci_upper = np.percentile(null_distribution, 97.5)
    
    return PValueResult(
        observed=observed,
        p_value=p_value,
        percentile=percentile,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        direction=direction,
        n_simulations=n
    )


def calculate_p_value_range(
    observed: float,
    null_distribution: NDArray[np.float64]
) -> Tuple[float, float]:
    """Calculate upper and lower p-value bounds.
    
    This implements the methodology from MATLAB code where both
    upper and lower p-values are returned as a tuple.
    
    Parameters
    ----------
    observed : float
        Observed test statistic
    null_distribution : ndarray
        Sorted null distribution values
        
    Returns
    -------
    p_range : tuple
        (upper_p, lower_p) where:
        - upper_p: Probability of values >= observed
        - lower_p: Probability of values <= observed
    """
    n = len(null_distribution)
    
    # Find indices with values greater than observed
    gr_indices = np.where(null_distribution > observed)[0]
    
    if len(gr_indices) == 0:
        # Observed is better than all values
        return (0.0, 1.0)
    else:
        # Calculate proportion below observed
        below = (gr_indices[0] - 1) / n
        return (1 - below, below)


def multiple_testing_correction(
    p_values: Union[List[float], NDArray[np.float64]],
    method: str = 'bonferroni',
    alpha: float = 0.05
) -> Dict[str, Union[NDArray[np.float64], float]]:
    """Apply multiple testing correction to p-values.
    
    Parameters
    ----------
    p_values : array-like
        Raw p-values to correct
    method : {'bonferroni', 'holm', 'fdr_bh', 'fdr_by'}, default='bonferroni'
        Correction method:
        - 'bonferroni': Bonferroni correction
        - 'holm': Holm-Bonferroni method
        - 'fdr_bh': Benjamini-Hochberg FDR
        - 'fdr_by': Benjamini-Yekutieli FDR
    alpha : float, default=0.05
        Family-wise error rate or false discovery rate
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'corrected': Corrected p-values
        - 'rejected': Boolean array of rejected hypotheses
        - 'threshold': Adjusted significance threshold
        - 'method': Correction method used
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)
    
    if n_tests == 0:
        raise ValueError("No p-values provided")
        
    if method == 'bonferroni':
        # Bonferroni correction
        corrected = np.minimum(p_values * n_tests, 1.0)
        threshold = alpha / n_tests
        rejected = p_values < threshold
        
    elif method == 'holm':
        # Holm-Bonferroni method
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        corrected = np.zeros_like(p_values)
        rejected = np.zeros(n_tests, dtype=bool)
        
        for i in range(n_tests):
            adjusted_p = sorted_p[i] * (n_tests - i)
            corrected[sorted_idx[i]] = min(adjusted_p, 1.0)
            
            if adjusted_p < alpha:
                rejected[sorted_idx[i]] = True
            else:
                break  # Stop at first non-rejection
                
        threshold = alpha / (n_tests - np.sum(rejected) + 1)
        
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR control
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Find largest i such that P(i) <= (i/m) * alpha
        thresh_idx = np.where(sorted_p <= (np.arange(1, n_tests + 1) / n_tests) * alpha)[0]
        
        if len(thresh_idx) > 0:
            max_idx = thresh_idx[-1]
            threshold = sorted_p[max_idx]
            rejected = p_values <= threshold
        else:
            threshold = 0.0
            rejected = np.zeros(n_tests, dtype=bool)
            
        # Adjusted p-values
        corrected = np.zeros_like(p_values)
        for i in range(n_tests):
            corrected[sorted_idx[i]] = min(
                sorted_p[i] * n_tests / (i + 1), 1.0
            )
            
    elif method == 'fdr_by':
        # Benjamini-Yekutieli FDR control
        c_m = np.sum(1.0 / np.arange(1, n_tests + 1))  # Harmonic sum
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        thresh_idx = np.where(
            sorted_p <= (np.arange(1, n_tests + 1) / (n_tests * c_m)) * alpha
        )[0]
        
        if len(thresh_idx) > 0:
            max_idx = thresh_idx[-1]
            threshold = sorted_p[max_idx]
            rejected = p_values <= threshold
        else:
            threshold = 0.0
            rejected = np.zeros(n_tests, dtype=bool)
            
        corrected = np.minimum(p_values * n_tests * c_m, 1.0)
        
    else:
        raise ValueError(f"Unknown correction method: {method}")
        
    return {
        'corrected': corrected,
        'rejected': rejected,
        'threshold': threshold,
        'method': method,
        'n_tests': n_tests
    }


def combine_p_values(
    p_values: Union[List[float], NDArray[np.float64]],
    method: str = 'fisher',
    weights: Optional[NDArray[np.float64]] = None
) -> Dict[str, float]:
    """Combine multiple p-values into a single test statistic.
    
    Parameters
    ----------
    p_values : array-like
        P-values to combine
    method : {'fisher', 'stouffer', 'min', 'max'}, default='fisher'
        Combination method:
        - 'fisher': Fisher's combined probability test
        - 'stouffer': Stouffer's Z-score method
        - 'min': Minimum p-value
        - 'max': Maximum p-value
    weights : array-like, optional
        Weights for weighted combination (only for Stouffer's method)
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'statistic': Combined test statistic
        - 'p_value': Combined p-value
        - 'method': Combination method used
    """
    p_values = np.asarray(p_values)
    n = len(p_values)
    
    if n == 0:
        raise ValueError("No p-values provided")
        
    # Check for invalid p-values
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("P-values must be between 0 and 1")
        
    if method == 'fisher':
        # Fisher's combined probability test
        # -2 * sum(log(p_i)) ~ chi-squared(2n)
        with np.errstate(divide='ignore'):
            log_p = np.log(p_values)
            log_p[p_values == 0] = -np.inf  # Handle p=0
            
        statistic = -2 * np.sum(log_p)
        p_value = 1 - stats.chi2.cdf(statistic, df=2*n)
        
    elif method == 'stouffer':
        # Stouffer's Z-score method
        # sum(Phi^(-1)(1-p_i)) / sqrt(n) ~ N(0,1)
        z_scores = stats.norm.ppf(1 - p_values)
        
        if weights is not None:
            weights = np.asarray(weights)
            if len(weights) != n:
                raise ValueError("Weights must have same length as p_values")
            weights = weights / np.sqrt(np.sum(weights**2))
            statistic = np.sum(weights * z_scores)
        else:
            statistic = np.sum(z_scores) / np.sqrt(n)
            
        p_value = 1 - stats.norm.cdf(statistic)
        
    elif method == 'min':
        # Minimum p-value (Tippett's method)
        statistic = np.min(p_values)
        p_value = 1 - (1 - statistic)**n
        
    elif method == 'max':
        # Maximum p-value
        statistic = np.max(p_values)
        p_value = statistic**n
        
    else:
        raise ValueError(f"Unknown combination method: {method}")
        
    return {
        'statistic': statistic,
        'p_value': p_value,
        'method': method,
        'n_tests': n
    }


def permutation_test(
    data1: NDArray[np.float64],
    data2: NDArray[np.float64],
    statistic_func: callable,
    n_permutations: int = 10000,
    random_state: Optional[int] = None
) -> PValueResult:
    """Perform permutation test for difference between two groups.
    
    Parameters
    ----------
    data1 : ndarray
        First group data
    data2 : ndarray
        Second group data
    statistic_func : callable
        Function to compute test statistic. Should accept two arrays
        and return a scalar statistic.
    n_permutations : int, default=10000
        Number of permutations
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    result : PValueResult
        Permutation test results
    """
    # Compute observed statistic
    observed = statistic_func(data1, data2)
    
    # Combine data
    combined = np.concatenate([data1, data2])
    n1 = len(data1)
    n_total = len(combined)
    
    # Generate permutation distribution
    rng = np.random.RandomState(random_state)
    perm_stats = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Permute combined data
        perm_idx = rng.permutation(n_total)
        perm_data1 = combined[perm_idx[:n1]]
        perm_data2 = combined[perm_idx[n1:]]
        
        # Compute permutation statistic
        perm_stats[i] = statistic_func(perm_data1, perm_data2)
        
    # Calculate p-value (two-sided by default)
    return calculate_empirical_p_value(
        observed, perm_stats, direction='two-sided'
    )