"""Integration tests for the SUBMARIT pipeline."""

import numpy as np
import pytest
from numpy.typing import NDArray

from submarit.algorithms.local_search import (
    KSMLocalSearch,
    KSMLocalSearch2,
    KSMLocalSearchConstrained,
    KSMLocalSearchConstrained2,
)
from submarit.core.substitution_matrix import SubstitutionMatrix
from submarit.evaluation.cluster_evaluator import ClusterEvaluator
from submarit.evaluation.entropy_evaluator import EntropyEvaluator
from submarit.evaluation.gap_statistic import GAPStatistic
from submarit.validation.empirical_distributions import EmpiricalDistribution
from submarit.validation.kfold import KFoldValidator
from submarit.validation.multiple_runs import MultipleRunsValidator
from submarit.validation.rand_index import rand_index, adjusted_rand_score
from submarit.validation.topk_analysis import TopKAnalyzer

from .test_fixtures import assert_matrix_properties


class TestIntegrationPipeline:
    """Test the complete SUBMARIT pipeline end-to-end."""
    
    @pytest.mark.integration
    def test_full_pipeline_from_consumer_data(self, consumer_product_data):
        """Test complete pipeline from consumer data to clustering results."""
        # Step 1: Create substitution matrix
        sub_matrix = SubstitutionMatrix()
        product_indexes, product_count = sub_matrix.create_from_consumer_product_data(
            consumer_product_data, normalize=True
        )
        
        assert_matrix_properties(sub_matrix.get_matrix())
        assert len(product_indexes) == product_count
        
        # Step 2: Perform clustering
        n_clusters = 3
        clusterer = KSMLocalSearch(n_clusters=n_clusters, random_state=42)
        clusterer.fit(sub_matrix.get_matrix())
        
        result = clusterer.get_result()
        assert result.NoClusters == n_clusters
        assert len(np.unique(result.Assign)) == n_clusters
        
        # Step 3: Evaluate clustering
        evaluator = ClusterEvaluator()
        metrics = evaluator.evaluate(
            sub_matrix.get_matrix(),
            result.Assign - 1  # Convert to 0-based
        )
        
        assert "silhouette_score" in metrics
        assert "davies_bouldin_score" in metrics
        assert "calinski_harabasz_score" in metrics
        
        # Step 4: Validate with k-fold
        validator = KFoldValidator(n_folds=5, random_state=42)
        validation_results = validator.validate(
            sub_matrix.get_matrix(),
            clusterer,
            n_runs=3
        )
        
        assert "mean_score" in validation_results
        assert "std_score" in validation_results
        assert len(validation_results["fold_scores"]) == 5
    
    @pytest.mark.integration
    def test_full_pipeline_from_sales_data(self, sales_time_series):
        """Test complete pipeline from sales time series data."""
        # Step 1: Create substitution matrix from sales correlation
        sub_matrix = SubstitutionMatrix()
        sub_matrix.create_from_sales_data(sales_time_series, method="correlation")
        
        assert_matrix_properties(sub_matrix.get_matrix())
        
        # Step 2: Multiple clustering algorithms
        n_clusters = 3
        algorithms = [
            KSMLocalSearch(n_clusters=n_clusters, random_state=42),
            KSMLocalSearch2(n_clusters=n_clusters, random_state=42),
        ]
        
        results = []
        for algo in algorithms:
            algo.fit(sub_matrix.get_matrix())
            results.append(algo.get_result())
        
        # Step 3: Compare results
        labels1 = results[0].Assign - 1
        labels2 = results[1].Assign - 1
        
        # Algorithms should produce similar (but not necessarily identical) results
        ari = adjusted_rand_score(labels1, labels2)
        assert ari > 0.5, f"Algorithms produced very different results: ARI={ari}"
        
        # Step 4: GAP statistic
        gap = GAPStatistic(n_refs=10, random_state=42)
        gap_results = gap.compute(
            sub_matrix.get_matrix(),
            max_clusters=5,
            clusterer_class=KSMLocalSearch
        )
        
        assert "gap_values" in gap_results
        assert "optimal_k" in gap_results
        assert 2 <= gap_results["optimal_k"] <= 5
    
    @pytest.mark.integration
    def test_constrained_clustering_pipeline(self, medium_substitution_matrix):
        """Test constrained clustering pipeline."""
        n_clusters = 3
        n_items = medium_substitution_matrix.shape[0]
        
        # Create constraints
        n_fixed = 6
        fixed_indices = np.array([1, 5, 10, 15, 18, 20])  # 1-based
        fixed_assignments = np.array([1, 1, 2, 2, 3, 3])
        free_indices = np.setdiff1d(np.arange(1, n_items + 1), fixed_indices)
        
        # Test both constrained algorithms
        algorithms = [
            KSMLocalSearchConstrained(n_clusters=n_clusters, random_state=42),
            KSMLocalSearchConstrained2(n_clusters=n_clusters, random_state=42),
        ]
        
        for algo in algorithms:
            algo.fit_constrained(
                medium_substitution_matrix,
                fixed_assignments,
                fixed_indices,
                free_indices
            )
            
            result = algo.get_result()
            
            # Check constraints are satisfied
            for idx, assign in zip(fixed_indices, fixed_assignments):
                assert result.Assign[idx - 1] == assign, "Constraint violated"
            
            # Check all clusters have items
            for i in range(1, n_clusters + 1):
                assert result.Count[i] > 0, f"Cluster {i} is empty"
    
    @pytest.mark.integration
    def test_empirical_distribution_pipeline(self, clustered_substitution_matrix):
        """Test empirical distribution and p-value computation."""
        matrix, true_labels = clustered_substitution_matrix
        
        # Cluster the data
        clusterer = KSMLocalSearch(n_clusters=3, random_state=42)
        clusterer.fit(matrix)
        result = clusterer.get_result()
        
        # Generate empirical distribution
        emp_dist = EmpiricalDistribution(n_iterations=100, random_state=42)
        dist_results = emp_dist.generate(
            matrix,
            clusterer,
            statistic="diff"
        )
        
        # Compute p-value
        p_value = emp_dist.compute_p_value(
            result.Diff,
            dist_results["distribution"]
        )
        
        assert 0 <= p_value <= 1
        assert len(dist_results["distribution"]) == 100
        
        # The clustered matrix should have significant structure
        assert p_value < 0.05, "Clustered matrix not detected as significant"
    
    @pytest.mark.integration
    def test_multiple_runs_validation(self, medium_substitution_matrix):
        """Test multiple runs validation for stability."""
        n_clusters = 3
        
        validator = MultipleRunsValidator(n_runs=20)
        clusterer = KSMLocalSearch(n_clusters=n_clusters)
        
        results = validator.validate(medium_substitution_matrix, clusterer)
        
        assert "all_labels" in results
        assert "stability_score" in results
        assert "consensus_matrix" in results
        
        # Check dimensions
        assert len(results["all_labels"]) == 20
        assert results["consensus_matrix"].shape == medium_substitution_matrix.shape
        
        # Stability score should be reasonable
        assert 0 <= results["stability_score"] <= 1
    
    @pytest.mark.integration
    def test_topk_analysis_pipeline(self, large_substitution_matrix):
        """Test top-k analysis pipeline."""
        # Run clustering
        clusterer = KSMLocalSearch(n_clusters=5, random_state=42)
        clusterer.fit(large_substitution_matrix)
        initial_result = clusterer.get_result()
        
        # Perform top-k analysis
        analyzer = TopKAnalyzer()
        topk_results = analyzer.analyze(
            large_substitution_matrix,
            initial_result.Assign - 1,  # Convert to 0-based
            k_values=[5, 10, 20, 30],
            random_state=42
        )
        
        assert "improvements" in topk_results
        assert "final_labels" in topk_results
        assert "history" in topk_results
        
        # Should see some improvement
        assert len(topk_results["improvements"]) > 0
        
        # Final result should be at least as good as initial
        final_score = topk_results.get("final_score", 0)
        initial_score = topk_results.get("initial_score", 0)
        assert final_score >= initial_score
    
    @pytest.mark.integration
    def test_entropy_evaluation_pipeline(self, clustered_substitution_matrix):
        """Test entropy-based evaluation pipeline."""
        matrix, true_labels = clustered_substitution_matrix
        
        # Cluster the data
        clusterer = KSMLocalSearch(n_clusters=3, random_state=42)
        clusterer.fit(matrix)
        predicted_labels = clusterer.get_result().Assign - 1
        
        # Evaluate entropy
        evaluator = EntropyEvaluator()
        
        # Within-cluster entropy (should be low for good clustering)
        within_entropy = evaluator.within_cluster_entropy(matrix, predicted_labels)
        
        # Between-cluster entropy (should be high for good clustering)
        between_entropy = evaluator.between_cluster_entropy(matrix, predicted_labels)
        
        # Total entropy
        total_entropy = evaluator.total_entropy(matrix, predicted_labels)
        
        assert within_entropy >= 0
        assert between_entropy >= 0
        assert total_entropy >= 0
        
        # For a good clustering, within-cluster entropy should be less than between-cluster
        assert within_entropy < between_entropy
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_scale_pipeline(self, large_substitution_matrix):
        """Test pipeline with large-scale data."""
        n_clusters = 10
        
        # Test scalability of different algorithms
        algorithms = [
            ("LocalSearch", KSMLocalSearch(n_clusters=n_clusters, random_state=42)),
            ("LocalSearch2", KSMLocalSearch2(n_clusters=n_clusters, random_state=42)),
        ]
        
        for name, algo in algorithms:
            # Fit algorithm
            algo.fit(large_substitution_matrix)
            result = algo.get_result()
            
            # Basic checks
            assert result.NoClusters == n_clusters
            assert result.NoItems == large_substitution_matrix.shape[0]
            assert len(np.unique(result.Assign)) == n_clusters
            
            # Check convergence
            assert result.Iter < algo.max_iter, f"{name} did not converge"
            
            # Evaluate quality
            evaluator = ClusterEvaluator()
            metrics = evaluator.evaluate(
                large_substitution_matrix,
                result.Assign - 1
            )
            
            # Should produce reasonable clustering
            assert metrics["silhouette_score"] > -0.5
            assert not np.isnan(metrics["davies_bouldin_score"])
    
    @pytest.mark.integration
    def test_matlab_compatibility_pipeline(self, matlab_reference_data):
        """Test MATLAB compatibility in full pipeline."""
        # Use reference data
        swm = matlab_reference_data["swm"]
        n_clusters = matlab_reference_data["n_clusters"]
        
        # Run clustering with same parameters
        clusterer = KSMLocalSearch(
            n_clusters=n_clusters,
            min_items=matlab_reference_data["min_items"],
            random_state=123  # Same as reference
        )
        clusterer.fit(swm)
        result = clusterer.get_result()
        
        # Results should be similar (not exact due to randomness)
        # Check that we get the same number of clusters
        assert len(np.unique(result.Assign)) == n_clusters
        
        # Check objective function is in reasonable range
        ref_diff = matlab_reference_data["diff"]
        assert abs(result.Diff - ref_diff) < abs(ref_diff) * 2  # Within 2x
    
    @pytest.mark.integration
    def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline."""
        # Test with invalid data
        with pytest.raises(ValueError):
            sub_matrix = SubstitutionMatrix()
            sub_matrix.set_data(np.array([1, 2, 3]))  # Not 2D
        
        # Test with too few items for clustering
        small_matrix = np.eye(3)
        clusterer = KSMLocalSearch(n_clusters=5)  # More clusters than items
        
        with pytest.raises(ValueError):
            clusterer.fit(small_matrix)
        
        # Test with invalid constraints
        matrix = np.random.rand(10, 10)
        np.fill_diagonal(matrix, 0)
        
        constrained = KSMLocalSearchConstrained(n_clusters=3)
        
        with pytest.raises(ValueError):
            # Mismatched constraint dimensions
            constrained.fit_constrained(
                matrix,
                fixed_assign=np.array([1, 2]),
                assign_indexes=np.array([1, 2, 3]),  # Mismatch
                free_indexes=np.array([4, 5, 6, 7, 8, 9, 10])
            )