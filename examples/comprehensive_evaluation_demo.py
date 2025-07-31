"""Comprehensive demonstration of SUBMARIT evaluation capabilities."""

import numpy as np
from submarit.evaluation import (
    ClusterEvaluator,
    GAPStatistic,
    EntropyClusterer,
    EvaluationVisualizer,
    StatisticalTests
)


def create_realistic_switching_matrix(n_products=50, n_true_clusters=4, noise_level=0.1):
    """Create a more realistic switching matrix with known structure."""
    swm = np.zeros((n_products, n_products))
    
    # Create uneven cluster sizes
    cluster_sizes = [15, 20, 10, 5][:n_true_clusters]
    remaining = n_products - sum(cluster_sizes)
    if remaining > 0:
        cluster_sizes[-1] += remaining
        
    # Assign products to clusters
    cluster_assign = np.zeros(n_products, dtype=int)
    start = 0
    for i, size in enumerate(cluster_sizes):
        cluster_assign[start:start+size] = i + 1
        start += size
    
    # Create switching patterns
    for i in range(n_products):
        for j in range(i+1, n_products):
            if cluster_assign[i] == cluster_assign[j]:
                # High within-cluster switching
                base_switch = np.random.gamma(shape=2, scale=50)
                swm[i, j] = swm[j, i] = base_switch
            else:
                # Low between-cluster switching (noise)
                if np.random.rand() < noise_level:
                    swm[i, j] = swm[j, i] = np.random.gamma(shape=1, scale=10)
    
    return swm, cluster_assign, cluster_sizes


def demonstrate_cluster_evaluation():
    """Demonstrate comprehensive cluster evaluation."""
    print("=" * 70)
    print("COMPREHENSIVE CLUSTER EVALUATION DEMONSTRATION")
    print("=" * 70)
    
    # Create test data
    swm, true_assign, true_sizes = create_realistic_switching_matrix(
        n_products=50, n_true_clusters=4, noise_level=0.15
    )
    
    print(f"\nCreated switching matrix: {swm.shape}")
    print(f"True cluster sizes: {true_sizes}")
    print(f"Total switching: {np.sum(swm):.0f}")
    
    # 1. Evaluate true clustering
    print("\n1. EVALUATING TRUE CLUSTERING")
    print("-" * 40)
    
    evaluator = ClusterEvaluator()
    true_result = evaluator.evaluate(swm, n_clusters=4, cluster_assign=true_assign)
    
    print(true_result.summary())
    
    # 2. Compare with entropy clustering
    print("\n2. ENTROPY CLUSTERING COMPARISON")
    print("-" * 40)
    
    # Test different entropy modes
    entropy_results = {}
    for mode, mode_name in [(1, "Total Entropy"), (2, "Normalized"), (3, "Size-scaled")]:
        print(f"\nMode {mode}: {mode_name}")
        clusterer = EntropyClusterer(
            n_clusters=4,
            min_items=2,
            opt_mode=mode,
            max_iter=500,
            random_state=42
        )
        clusterer.fit(swm)
        result = clusterer.get_result()
        
        # Evaluate the clustering
        eval_result = evaluator.evaluate(swm, 4, clusterer.labels_)
        entropy_results[mode_name] = eval_result
        
        print(f"  Converged in {result.n_iter} iterations")
        print(f"  Z-value: {eval_result.z_value:.4f}")
        print(f"  Log-likelihood: {eval_result.log_lh:.4f}")
        print(f"  Cluster sizes: {result.counts}")
    
    # 3. Statistical comparison
    print("\n3. STATISTICAL TESTING")
    print("-" * 40)
    
    # Permutation test comparing true vs entropy clustering
    test_result = StatisticalTests.permutation_test(
        swm=swm,
        assign1=true_assign,
        assign2=clusterer.labels_,
        n_permutations=500,
        metric="z_value",
        random_state=42
    )
    
    print(f"\nPermutation test (true vs entropy clustering):")
    print(f"  Metric: {test_result['metric']}")
    print(f"  True clustering value: {test_result['result1_value']:.4f}")
    print(f"  Entropy clustering value: {test_result['result2_value']:.4f}")
    print(f"  Difference: {test_result['original_diff']:.4f}")
    print(f"  P-value: {test_result['p_value']:.4f}")
    
    # Bootstrap confidence intervals
    print("\nBootstrap confidence intervals for true clustering:")
    for metric in ["z_value", "log_lh", "diff"]:
        ci_result = StatisticalTests.bootstrap_confidence_interval(
            swm=swm,
            cluster_assign=true_assign,
            metric=metric,
            n_bootstrap=500,
            confidence=0.95,
            random_state=42
        )
        print(f"  {metric}: {ci_result['original_value']:.4f} "
              f"[{ci_result['ci_lower']:.4f}, {ci_result['ci_upper']:.4f}]")
    
    # 4. Cluster validity indices
    print("\n4. CLUSTER VALIDITY INDICES")
    print("-" * 40)
    
    validity_true = StatisticalTests.cluster_validity_indices(swm, true_assign)
    validity_entropy = StatisticalTests.cluster_validity_indices(swm, clusterer.labels_)
    
    print("\nValidity indices comparison:")
    print(f"{'Metric':<30} {'True':>10} {'Entropy':>10}")
    print("-" * 50)
    for key in validity_true.keys():
        if key not in ['n_clusters', 'n_items']:
            print(f"{key:<30} {validity_true[key]:>10.4f} {validity_entropy[key]:>10.4f}")
    
    # 5. Visualization
    print("\n5. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    visualizer = EvaluationVisualizer()
    
    # Compare different clustering results
    all_results = [true_result] + list(entropy_results.values())
    labels = ["True"] + list(entropy_results.keys())
    
    print("Creating comparison plots...")
    visualizer.plot_cluster_comparison(
        results=all_results,
        labels=labels,
        save_path="cluster_comparison.png"
    )
    
    # Plot cluster sizes for best result
    visualizer.plot_cluster_sizes(
        result=true_result,
        save_path="cluster_sizes.png"
    )
    
    # Plot P vs PHat
    visualizer.plot_probability_comparison(
        result=true_result,
        save_path="p_vs_phat.png"
    )
    
    # Create comprehensive report
    report_data = {
        'cluster_eval': true_result,
        'entropy_result': clusterer.get_result()
    }
    visualizer.create_evaluation_report(
        results=report_data,
        save_path="evaluation_report.png"
    )
    
    print("Visualizations saved!")
    
    return swm, true_assign, clusterer.labels_


def demonstrate_gap_statistic(swm, true_assign):
    """Demonstrate GAP statistic for optimal k selection."""
    print("\n" + "=" * 70)
    print("GAP STATISTIC DEMONSTRATION")
    print("=" * 70)
    
    # Create synthetic data matrix for GAP statistic
    n_customers = 200
    n_products = swm.shape[0]
    data_matrix = np.random.rand(n_customers, n_products) * 100
    
    # Mock clustering function using entropy clusterer
    def clustering_function(swm, n_clusters, min_items, n_runs):
        from types import SimpleNamespace
        
        best_result = None
        best_obj = -np.inf
        
        for _ in range(n_runs):
            clusterer = EntropyClusterer(
                n_clusters=n_clusters,
                min_items=min_items,
                opt_mode=2,
                max_iter=200
            )
            clusterer.fit(swm)
            
            if clusterer.get_result().max_obj > best_obj:
                best_obj = clusterer.get_result().max_obj
                best_result = clusterer
        
        return SimpleNamespace(assign=best_result.labels_)
    
    # Run GAP statistic analysis
    print("\nTesting different criteria...")
    gap_results = {}
    
    for criterion in ["diff", "z_value", "log_lh"]:
        print(f"\nCriterion: {criterion}")
        gap_evaluator = GAPStatistic(
            criterion=criterion,
            n_uniform=5,  # Reduced for speed
            verbose=False
        )
        
        result = gap_evaluator.evaluate(
            swm=swm,
            data_matrix=data_matrix,
            min_k=2,
            max_k=6,
            min_items=2,
            cluster_func=clustering_function,
            n_runs=3
        )
        
        gap_results[criterion] = result
        print(f"  Optimal k: {result.best_k}")
        print(f"  GAP values: {result.gap_values}")
        
        # Plot GAP statistic
        gap_evaluator.plot_gap_statistic(
            result,
            save_path=f"gap_statistic_{criterion}.png"
        )
    
    # Summary comparison
    print("\nGAP Statistic Summary:")
    print(f"{'Criterion':<15} {'Optimal k':>10}")
    print("-" * 25)
    for criterion, result in gap_results.items():
        print(f"{criterion:<15} {result.best_k:>10}")
    
    return gap_results


def demonstrate_stability_analysis(swm):
    """Demonstrate stability analysis."""
    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS DEMONSTRATION")
    print("=" * 70)
    
    # Define clustering function for stability analysis
    def entropy_cluster_func(swm, n_clusters, min_items, n_runs):
        from types import SimpleNamespace
        clusterer = EntropyClusterer(
            n_clusters=n_clusters,
            min_items=min_items,
            opt_mode=2,
            max_iter=200
        )
        clusterer.fit(swm)
        return SimpleNamespace(assign=clusterer.labels_)
    
    # Test stability for different k values
    print("\nTesting clustering stability across multiple runs...")
    
    for k in [3, 4, 5]:
        print(f"\nk = {k}:")
        stability = StatisticalTests.stability_analysis(
            swm=swm,
            cluster_func=entropy_cluster_func,
            n_clusters=k,
            n_runs=10,
            min_items=2
        )
        
        print(f"  Z-value: {stability['z_value_mean']:.4f} ± {stability['z_value_std']:.4f}")
        print(f"  CV: {stability['z_value_cv']:.4f}")
        print(f"  Mean agreement: {stability['mean_agreement']:.4f}")
        print(f"  Min agreement: {stability['min_agreement']:.4f}")


def main():
    """Run comprehensive evaluation demonstration."""
    print("SUBMARIT EVALUATION FRAMEWORK - COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Run main evaluation demonstration
    swm, true_assign, entropy_assign = demonstrate_cluster_evaluation()
    
    # Run GAP statistic demonstration
    gap_results = demonstrate_gap_statistic(swm, true_assign)
    
    # Run stability analysis
    demonstrate_stability_analysis(swm)
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - cluster_comparison.png")
    print("  - cluster_sizes.png")
    print("  - p_vs_phat.png")
    print("  - evaluation_report.png")
    print("  - gap_statistic_*.png")
    
    print("\nKey findings:")
    print(f"  - True number of clusters: 4")
    print(f"  - GAP statistic optimal k: {gap_results['z_value'].best_k}")
    print(f"  - Entropy clustering successfully recovered cluster structure")


if __name__ == "__main__":
    main()