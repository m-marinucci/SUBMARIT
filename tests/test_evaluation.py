"""Unit tests for evaluation modules."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from submarit.evaluation import (
    ClusterEvaluator,
    ClusterEvaluationResult,
    GAPStatistic,
    EntropyClusterer,
    EvaluationVisualizer,
    StatisticalTests
)


class TestClusterEvaluator:
    """Test cases for ClusterEvaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple 2-cluster switching matrix
        self.swm = np.array([
            [0, 80, 10, 5],
            [80, 0, 5, 10],
            [10, 5, 0, 90],
            [5, 10, 90, 0]
        ], dtype=float)
        
        self.cluster_assign = np.array([1, 1, 2, 2])
        self.evaluator = ClusterEvaluator()
    
    def test_basic_evaluation(self):
        """Test basic evaluation functionality."""
        result = self.evaluator.evaluate(self.swm, 2, self.cluster_assign)
        
        assert isinstance(result, ClusterEvaluationResult)
        assert result.n_clusters == 2
        assert result.n_items == 4
        assert len(result.indexes) == 2
        assert result.counts == [2, 2]
        
    def test_evaluation_metrics(self):
        """Test that evaluation metrics are computed correctly."""
        result = self.evaluator.evaluate(self.swm, 2, self.cluster_assign)
        
        # Check that metrics are finite
        assert np.isfinite(result.diff)
        assert np.isfinite(result.z_value)
        assert np.isfinite(result.log_lh)
        assert np.isfinite(result.scaled_diff)
        
        # Check arrays
        assert len(result.p_hat) == 4
        assert len(result.p) == 4
        assert np.all(result.p_hat >= 0)
        assert np.all(result.p >= 0)
        
    def test_single_cluster(self):
        """Test evaluation with single cluster."""
        single_assign = np.ones(4, dtype=int)
        result = self.evaluator.evaluate(self.swm, 1, single_assign)
        
        assert result.n_clusters == 1
        assert result.counts == [4]
        
    def test_legacy_evaluation(self):
        """Test legacy evaluation function."""
        result = self.evaluator.evaluate_legacy(self.swm, 2, self.cluster_assign)
        
        assert isinstance(result, dict)
        assert 'diff' in result
        assert 'z_value' in result
        assert 'log_lh' in result
        assert 'p' in result
        assert 'p_hat' in result


class TestEntropyClusterer:
    """Test cases for EntropyClusterer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create block-diagonal switching matrix
        n = 20
        self.swm = np.zeros((n, n))
        
        # Cluster 1: items 0-9
        for i in range(10):
            for j in range(10):
                if i != j:
                    self.swm[i, j] = np.random.uniform(50, 100)
                    
        # Cluster 2: items 10-19
        for i in range(10, 20):
            for j in range(10, 20):
                if i != j:
                    self.swm[i, j] = np.random.uniform(50, 100)
    
    def test_entropy_clustering(self):
        """Test basic entropy clustering."""
        clusterer = EntropyClusterer(
            n_clusters=2,
            min_items=2,
            opt_mode=1,
            random_state=42
        )
        
        clusterer.fit(self.swm)
        
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == 20
        assert len(np.unique(clusterer.labels_)) == 2
        
    def test_different_opt_modes(self):
        """Test different optimization modes."""
        for mode in [1, 2, 3]:
            clusterer = EntropyClusterer(
                n_clusters=2,
                min_items=2,
                opt_mode=mode,
                random_state=42
            )
            
            clusterer.fit(self.swm)
            result = clusterer.get_result()
            
            assert result.n_clusters == 2
            assert result.n_items == 20
            assert np.isfinite(result.entropy)
            assert np.isfinite(result.entropy_norm)
            assert np.isfinite(result.entropy_norm2)
            
    def test_min_items_constraint(self):
        """Test minimum items per cluster constraint."""
        clusterer = EntropyClusterer(
            n_clusters=2,
            min_items=5,
            random_state=42
        )
        
        clusterer.fit(self.swm)
        result = clusterer.get_result()
        
        # Check all clusters have at least min_items
        assert all(count >= 5 for count in result.counts)
        
    def test_convergence(self):
        """Test that algorithm converges."""
        clusterer = EntropyClusterer(
            n_clusters=2,
            min_items=2,
            max_iter=100,
            random_state=42
        )
        
        clusterer.fit(self.swm)
        result = clusterer.get_result()
        
        assert result.n_iter <= 100
        assert result.n_iter > 0


class TestGAPStatistic:
    """Test cases for GAPStatistic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Simple switching matrix
        self.swm = np.array([
            [0, 90, 10, 5],
            [90, 0, 5, 10],
            [10, 5, 0, 95],
            [5, 10, 95, 0]
        ], dtype=float)
        
        self.data_matrix = np.random.rand(50, 4) * 100
        
        # Mock clustering function
        def mock_cluster(swm, n_clusters, min_items, n_runs):
            from types import SimpleNamespace
            assign = np.array([1, 1, 2, 2]) if n_clusters == 2 else np.ones(4, dtype=int)
            return SimpleNamespace(assign=assign)
            
        self.cluster_func = mock_cluster
    
    def test_gap_statistic_basic(self):
        """Test basic GAP statistic functionality."""
        gap = GAPStatistic(criterion="diff", n_uniform=3)
        
        result = gap.evaluate(
            swm=self.swm,
            data_matrix=self.data_matrix,
            min_k=1,
            max_k=3,
            min_items=1,
            cluster_func=self.cluster_func,
            n_runs=1
        )
        
        assert len(result.k_values) == 3
        assert len(result.observed_values) == 3
        assert len(result.uniform_values) == 3
        assert len(result.gap_values) == 3
        assert result.best_k in [1, 2, 3]
        
    def test_different_criteria(self):
        """Test different criteria for GAP statistic."""
        for criterion in ["diff", "diff_sq", "log_lh", "z_value"]:
            gap = GAPStatistic(criterion=criterion, n_uniform=2)
            
            result = gap.evaluate(
                swm=self.swm,
                data_matrix=self.data_matrix,
                min_k=2,
                max_k=2,
                min_items=1,
                cluster_func=self.cluster_func,
                n_runs=1
            )
            
            assert result.criterion_name == criterion
            assert len(result.detailed_results) == 4  # All criteria computed


class TestStatisticalTests:
    """Test cases for StatisticalTests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.swm = np.array([
            [0, 80, 10, 5],
            [80, 0, 5, 10],
            [10, 5, 0, 90],
            [5, 10, 90, 0]
        ], dtype=float)
        
        self.assign1 = np.array([1, 1, 2, 2])
        self.assign2 = np.array([1, 2, 1, 2])
    
    def test_permutation_test(self):
        """Test permutation test."""
        result = StatisticalTests.permutation_test(
            swm=self.swm,
            assign1=self.assign1,
            assign2=self.assign2,
            n_permutations=100,
            metric="z_value",
            random_state=42
        )
        
        assert 'p_value' in result
        assert 'original_diff' in result
        assert 0 <= result['p_value'] <= 1
        assert result['n_permutations'] <= 100
        
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence intervals."""
        result = StatisticalTests.bootstrap_confidence_interval(
            swm=self.swm,
            cluster_assign=self.assign1,
            metric="z_value",
            n_bootstrap=100,
            confidence=0.95,
            random_state=42
        )
        
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert result['ci_lower'] <= result['original_value'] <= result['ci_upper']
        assert result['confidence'] == 0.95
        
    def test_cluster_validity_indices(self):
        """Test cluster validity indices."""
        result = StatisticalTests.cluster_validity_indices(
            swm=self.swm,
            cluster_assign=self.assign1
        )
        
        assert 'within_switching_ratio' in result
        assert 'between_switching_ratio' in result
        assert 'silhouette_coefficient' in result
        assert 'dunn_index' in result
        
        # Check ratios sum to 1
        total_ratio = result['within_switching_ratio'] + result['between_switching_ratio']
        assert_array_almost_equal(total_ratio, 1.0, decimal=10)
        
    def test_stability_analysis(self):
        """Test stability analysis."""
        def test_cluster_func(swm, n_clusters, min_items, n_runs):
            from types import SimpleNamespace
            # Add some randomness
            assign = self.assign1.copy()
            if np.random.rand() < 0.2:
                assign[0] = 2  # Swap one assignment
            return SimpleNamespace(assign=assign)
            
        result = StatisticalTests.stability_analysis(
            swm=self.swm,
            cluster_func=test_cluster_func,
            n_clusters=2,
            n_runs=5,
            min_items=1
        )
        
        assert 'z_value_mean' in result
        assert 'z_value_std' in result
        assert 'mean_agreement' in result
        assert result['n_runs'] == 5


class TestEvaluationVisualizer:
    """Test cases for EvaluationVisualizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = EvaluationVisualizer()
        
        # Create mock evaluation result
        self.mock_result = ClusterEvaluationResult(
            assign=np.array([1, 1, 2, 2]),
            indexes=[np.array([0, 1]), np.array([2, 3])],
            counts=[2, 2],
            diff=0.1,
            item_diff=0.025,
            scaled_diff=0.2,
            z_value=2.5,
            max_obj=0.1,
            log_lh=-10.0,
            diff_sq=0.01,
            p_hat=np.array([0.6, 0.7, 0.8, 0.9]),
            p=np.array([0.5, 0.5, 0.5, 0.5]),
            var=[np.array([0.1, 0.1]), np.array([0.1, 0.1])],
            n_clusters=2,
            n_items=4
        )
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        assert hasattr(self.visualizer, 'plt')
        
    def test_plot_methods_no_error(self):
        """Test that plot methods don't raise errors."""
        # These should run without error even if matplotlib is not available
        self.visualizer.plot_cluster_comparison([self.mock_result])
        self.visualizer.plot_cluster_sizes(self.mock_result)
        self.visualizer.plot_probability_comparison(self.mock_result)
        self.visualizer.plot_entropy_evolution([1.0, 0.9, 0.8])


def test_integration():
    """Test integration between modules."""
    # Create test data
    swm = np.eye(10) * 0 + 50
    np.fill_diagonal(swm, 0)
    
    # Run entropy clustering
    clusterer = EntropyClusterer(n_clusters=2, min_items=2, random_state=42)
    clusterer.fit(swm)
    
    # Evaluate clustering
    evaluator = ClusterEvaluator()
    result = evaluator.evaluate(swm, 2, clusterer.labels_)
    
    # Check result consistency
    assert result.n_clusters == 2
    assert result.n_items == 10
    assert len(result.indexes) == 2
    assert sum(result.counts) == 10


if __name__ == "__main__":
    pytest.main([__file__])