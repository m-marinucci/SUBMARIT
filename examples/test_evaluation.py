"""Test and demonstrate the evaluation functions."""

import numpy as np
from submarit.evaluation import (
    ClusterEvaluator, 
    GAPStatistic,
    EntropyClusterer
)


def create_test_switching_matrix(n_products=20, n_clusters=3):
    """Create a synthetic switching matrix with known cluster structure."""
    # Create block-diagonal structure
    swm = np.zeros((n_products, n_products))
    
    # Assign products to clusters
    cluster_size = n_products // n_clusters
    cluster_assign = np.zeros(n_products, dtype=int)
    
    for i in range(n_clusters):
        start = i * cluster_size
        end = start + cluster_size if i < n_clusters - 1 else n_products
        cluster_assign[start:end] = i + 1
        
        # High switching within cluster
        for j in range(start, end):
            for k in range(start, end):
                if j != k:
                    swm[j, k] = np.random.uniform(50, 100)
    
    # Add some noise (cross-cluster switching)
    noise_mask = np.random.rand(n_products, n_products) < 0.1
    swm[noise_mask] = np.random.uniform(0, 20, size=np.sum(noise_mask))
    
    # Make symmetric
    swm = (swm + swm.T) / 2
    np.fill_diagonal(swm, 0)
    
    return swm, cluster_assign


def test_cluster_evaluator():
    """Test the cluster evaluator."""
    print("=" * 60)
    print("Testing ClusterEvaluator")
    print("=" * 60)
    
    # Create test data
    swm, true_assign = create_test_switching_matrix(n_products=30, n_clusters=3)
    
    # Initialize evaluator
    evaluator = ClusterEvaluator()
    
    # Evaluate true clustering
    result = evaluator.evaluate(swm, n_clusters=3, cluster_assign=true_assign)
    
    print(result.summary())
    print()
    
    # Test with random clustering for comparison
    random_assign = np.random.randint(1, 4, size=30)
    random_result = evaluator.evaluate(swm, n_clusters=3, cluster_assign=random_assign)
    
    print("Comparison with random clustering:")
    print(f"True clustering Z-value: {result.z_value:.4f}")
    print(f"Random clustering Z-value: {random_result.z_value:.4f}")
    print(f"True clustering log-likelihood: {result.log_lh:.4f}")
    print(f"Random clustering log-likelihood: {random_result.log_lh:.4f}")
    print()
    
    # Test legacy evaluation
    legacy_result = evaluator.evaluate_legacy(swm, n_clusters=3, cluster_assign=true_assign)
    print(f"Legacy evaluation - Z-value: {legacy_result['z_value']:.4f}")
    print()


def test_entropy_clusterer():
    """Test the entropy-based clusterer."""
    print("=" * 60)
    print("Testing EntropyClusterer")
    print("=" * 60)
    
    # Create test data
    swm, _ = create_test_switching_matrix(n_products=30, n_clusters=3)
    
    # Test different optimization modes
    for opt_mode in [1, 2, 3]:
        print(f"\nOptimization mode {opt_mode}:")
        clusterer = EntropyClusterer(
            n_clusters=3,
            min_items=2,
            opt_mode=opt_mode,
            random_state=42
        )
        
        # Fit the model
        clusterer.fit(swm)
        result = clusterer.get_result()
        
        print(f"  Converged in {result.n_iter} iterations")
        print(f"  Total entropy: {result.entropy:.6f}")
        print(f"  Normalized entropy: {result.entropy_norm:.6f}")
        print(f"  Max objective: {result.max_obj:.6f}")
        print(f"  Cluster sizes: {result.counts}")


def test_gap_statistic():
    """Test the GAP statistic."""
    print("=" * 60)
    print("Testing GAPStatistic")
    print("=" * 60)
    
    # Create test data  
    swm, _ = create_test_switching_matrix(n_products=30, n_clusters=3)
    data_matrix = np.random.rand(100, 30) * 10  # Synthetic purchase data
    
    # Mock cluster function (would be RunClusters in real usage)
    def mock_cluster_func(swm, n_clusters, min_items, n_runs):
        """Mock clustering function for testing."""
        from types import SimpleNamespace
        
        # Use entropy clusterer as the clustering algorithm
        clusterer = EntropyClusterer(
            n_clusters=n_clusters,
            min_items=min_items,
            random_state=42
        )
        clusterer.fit(swm)
        
        return SimpleNamespace(assign=clusterer.labels_)
    
    # Initialize GAP statistic evaluator
    gap_evaluator = GAPStatistic(
        criterion="diff",
        n_uniform=5,  # Reduced for faster testing
        verbose=True
    )
    
    # Evaluate for k=2 to 5
    result = gap_evaluator.evaluate(
        swm=swm,
        data_matrix=data_matrix,
        min_k=2,
        max_k=5,
        min_items=2,
        cluster_func=mock_cluster_func,
        n_runs=3
    )
    
    print(f"\n{result.summary()}")
    
    # Try to plot (will skip if matplotlib not available)
    gap_evaluator.plot_gap_statistic(result)


def test_integration():
    """Test integration of clustering and evaluation."""
    print("=" * 60)
    print("Testing Integration: Clustering + Evaluation")
    print("=" * 60)
    
    # Create test data
    swm, true_assign = create_test_switching_matrix(n_products=40, n_clusters=4)
    
    # Run entropy clustering
    clusterer = EntropyClusterer(
        n_clusters=4,
        min_items=3,
        opt_mode=2,
        random_state=42
    )
    clusterer.fit(swm)
    
    # Evaluate the clustering
    evaluator = ClusterEvaluator()
    eval_result = evaluator.evaluate(swm, n_clusters=4, cluster_assign=clusterer.labels_)
    
    print("Entropy clustering results:")
    print(clusterer.get_result().summary())
    print("\nEvaluation of entropy clustering:")
    print(eval_result.summary())
    
    # Compare with true clustering
    true_eval = evaluator.evaluate(swm, n_clusters=4, cluster_assign=true_assign)
    print(f"\nTrue clustering Z-value: {true_eval.z_value:.4f}")
    print(f"Entropy clustering Z-value: {eval_result.z_value:.4f}")


def main():
    """Run all tests."""
    print("Testing SUBMARIT Evaluation Functions")
    print("=" * 60)
    print()
    
    test_cluster_evaluator()
    print()
    
    test_entropy_clusterer()
    print()
    
    test_gap_statistic()
    print()
    
    test_integration()


if __name__ == "__main__":
    main()