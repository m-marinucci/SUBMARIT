Evaluation Methods
==================

This section covers all evaluation and validation methods for assessing clustering quality.

Cluster Quality Metrics
-----------------------

Internal Validation Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These metrics evaluate clustering quality without external labels:

**Silhouette Coefficient**

Measures how similar a product is to its own cluster compared to other clusters.

.. math::

    s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}

where :math:`a(i)` is the average distance to products in the same cluster and :math:`b(i)` is the average distance to the nearest cluster.

.. code-block:: python

    from submarit.evaluation import silhouette_score
    
    score = silhouette_score(S, clusters)
    print(f"Silhouette score: {score:.3f}")  # Range: [-1, 1], higher is better
    
    # Per-sample scores
    sample_scores = silhouette_samples(S, clusters)

**Calinski-Harabasz Index**

Ratio of between-cluster to within-cluster variance:

.. math::

    CH = \frac{SS_B / (k-1)}{SS_W / (n-k)}

.. code-block:: python

    from submarit.evaluation import calinski_harabasz_score
    
    ch_score = calinski_harabasz_score(S, clusters)
    print(f"CH index: {ch_score:.2f}")  # Higher is better

**Davies-Bouldin Index**

Average similarity between each cluster and its most similar cluster:

.. code-block:: python

    from submarit.evaluation import davies_bouldin_score
    
    db_score = davies_bouldin_score(S, clusters)
    print(f"DB index: {db_score:.3f}")  # Lower is better

**Dunn Index**

Ratio of minimum inter-cluster to maximum intra-cluster distance:

.. code-block:: python

    from submarit.evaluation import dunn_index
    
    dunn = dunn_index(S, clusters)
    print(f"Dunn index: {dunn:.3f}")  # Higher is better

Comprehensive Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.evaluation import ClusterEvaluator
    
    evaluator = ClusterEvaluator()
    
    # Evaluate all metrics at once
    metrics = evaluator.evaluate(S, clusters)
    
    # Pretty print results
    evaluator.print_report(metrics)
    
    # Get LaTeX table
    latex_table = evaluator.to_latex(metrics)
    
    # Compare multiple clusterings
    results = []
    for k in range(2, 11):
        ls = LocalSearch(n_clusters=k)
        clusters = ls.fit_predict(S)
        metrics = evaluator.evaluate(S, clusters)
        metrics['k'] = k
        results.append(metrics)
    
    # Find best k by metric
    best_k_silhouette = max(results, key=lambda x: x['silhouette'])['k']

Statistical Validation
----------------------

Gap Statistic
~~~~~~~~~~~~~

Compares within-cluster dispersion to that expected under null hypothesis:

.. code-block:: python

    from submarit.evaluation import gap_statistic
    
    # Single k value
    gap, std = gap_statistic(S, n_clusters=5, n_bootstrap=50)
    
    # Find optimal k
    gaps, stds = [], []
    k_values = range(2, 11)
    
    for k in k_values:
        gap, std = gap_statistic(S, k, n_bootstrap=50)
        gaps.append(gap)
        stds.append(std)
    
    # Apply 1-std rule
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - stds[i + 1]:
            optimal_k = k_values[i]
            break

Stability Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.evaluation import stability_analysis
    
    # Bootstrap stability
    stability_scores = stability_analysis(
        S, 
        n_clusters=5,
        method='bootstrap',
        n_iterations=100
    )
    
    print(f"Stability: {np.mean(stability_scores):.3f} ± {np.std(stability_scores):.3f}")
    
    # Noise injection stability
    noise_stability = stability_analysis(
        S,
        n_clusters=5,
        method='noise',
        noise_level=0.1,
        n_iterations=50
    )

Visualization Tools
-------------------

Substitution Matrix Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.evaluation.visualization import plot_substitution_matrix
    import matplotlib.pyplot as plt
    
    # Basic plot
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_substitution_matrix(S, clusters, ax=ax)
    plt.show()
    
    # With product names
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_substitution_matrix(
        S, 
        clusters,
        labels=product_names,
        ax=ax,
        cmap='RdBu_r',
        show_dendogram=True
    )

Cluster Quality Plots
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.evaluation.visualization import (
        plot_silhouette_analysis,
        plot_cluster_comparison
    )
    
    # Silhouette plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_silhouette_analysis(S, clusters, ax=ax)
    
    # Compare different k values
    fig = plot_cluster_comparison(S, k_range=range(2, 11))

Elbow Method Plot
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.evaluation.visualization import plot_elbow_method
    
    # Calculate within-cluster sum of squares
    wcss = []
    k_values = range(2, 11)
    
    for k in k_values:
        ls = LocalSearch(n_clusters=k)
        ls.fit(S)
        wcss.append(ls.objective_)
    
    # Plot elbow
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_elbow_method(k_values, wcss, ax=ax)

3D Visualization
~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.evaluation.visualization import plot_3d_clusters
    from sklearn.decomposition import PCA
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(S)
    
    # 3D scatter plot
    fig = plot_3d_clusters(X_3d, clusters, product_names)

Entropy-Based Evaluation
------------------------

.. code-block:: python

    from submarit.evaluation import EntropyEvaluator
    
    # Initialize with product attributes
    evaluator = EntropyEvaluator()
    
    # Calculate entropy metrics
    entropy_metrics = evaluator.evaluate(
        clusters,
        product_attributes,  # DataFrame with categorical attributes
        attribute_columns=['brand', 'category', 'price_range']
    )
    
    # Normalized mutual information
    nmi = evaluator.normalized_mutual_info(clusters, true_labels)

Comparative Analysis
--------------------

.. code-block:: python

    from submarit.evaluation import ComparativeAnalyzer
    
    # Compare multiple algorithms
    algorithms = {
        'Local Search': LocalSearch(n_clusters=5),
        'K-Means': KMeansAdapter(n_clusters=5),
        'Hierarchical': HierarchicalAdapter(n_clusters=5)
    }
    
    analyzer = ComparativeAnalyzer()
    comparison = analyzer.compare(S, algorithms)
    
    # Generate report
    report = analyzer.generate_report(comparison)
    print(report)
    
    # Plot comparison
    fig = analyzer.plot_comparison(comparison)

Cluster Profiling
-----------------

.. code-block:: python

    from submarit.evaluation import ClusterProfiler
    
    profiler = ClusterProfiler()
    
    # Generate cluster profiles
    profiles = profiler.create_profiles(
        clusters,
        product_features,
        product_names,
        feature_names
    )
    
    # Print cluster summaries
    for cluster_id, profile in profiles.items():
        print(f"\nCluster {cluster_id}:")
        print(f"Size: {profile['size']}")
        print(f"Top features: {profile['top_features']}")
        print(f"Representative products: {profile['representatives']}")

Export Results
--------------

.. code-block:: python

    from submarit.evaluation import ResultsExporter
    
    exporter = ResultsExporter()
    
    # Export to various formats
    exporter.to_excel('results.xlsx', {
        'clusters': clusters,
        'metrics': metrics,
        'profiles': profiles
    })
    
    exporter.to_latex('results.tex', metrics)
    exporter.to_html('results.html', full_report)

Best Practices
--------------

1. **Always use multiple metrics** - No single metric captures all aspects
2. **Validate stability** - Ensure clusters are robust to data perturbations  
3. **Visualize results** - Visual inspection often reveals insights metrics miss
4. **Compare with baselines** - Random clustering provides lower bound
5. **Consider domain knowledge** - Metrics should align with business objectives

Example: Complete Evaluation Pipeline
-------------------------------------

.. code-block:: python

    from submarit import SubmarketAnalyzer
    from submarit.evaluation import create_evaluation_report
    
    # Load data
    X, product_names = load_data('products.csv')
    S = create_substitution_matrix(X)
    
    # Find optimal k
    analyzer = SubmarketAnalyzer()
    k_results = analyzer.find_optimal_k(S, k_range=range(2, 11))
    optimal_k = k_results['optimal_k']
    
    # Perform clustering
    clusters = analyzer.cluster(S, n_clusters=optimal_k)
    
    # Comprehensive evaluation
    report = create_evaluation_report(
        S, 
        clusters,
        product_names=product_names,
        product_features=X,
        include_visualization=True,
        output_dir='evaluation_results'
    )
    
    print(report['summary'])