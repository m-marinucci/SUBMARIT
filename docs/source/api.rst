API Reference
=============

This is the complete API reference for SUBMARIT.

.. contents:: Table of Contents
   :local:
   :depth: 2

Core Module
-----------

.. currentmodule:: submarit.core

Substitution Matrix Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: submarit.core.substitution_matrix
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: create_substitution_matrix

Example usage:

.. code-block:: python

    from submarit.core import create_substitution_matrix
    import numpy as np
    
    # Create product feature matrix
    X = np.random.rand(100, 20)  # 100 products, 20 features
    
    # Default Euclidean distance
    S = create_substitution_matrix(X)
    
    # Using cosine similarity
    S_cosine = create_substitution_matrix(X, metric='cosine')
    
    # With custom parameters
    S_custom = create_substitution_matrix(
        X,
        metric='minkowski',
        metric_params={'p': 3},
        normalize=True
    )

Base Classes
~~~~~~~~~~~~

.. automodule:: submarit.core.base
   :members:
   :undoc-members:
   :show-inheritance:

Algorithms Module
-----------------

.. currentmodule:: submarit.algorithms

Local Search Algorithm
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: submarit.algorithms.LocalSearch
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

    from submarit.algorithms import LocalSearch
    
    # Initialize with parameters
    ls = LocalSearch(
        n_clusters=5,
        max_iter=100,
        n_restarts=10,
        tol=1e-4,
        random_state=42
    )
    
    # Fit and predict
    clusters = ls.fit_predict(S)
    
    # Access results
    print(f"Final objective value: {ls.objective_}")
    print(f"Number of iterations: {ls.n_iter_}")
    print(f"Cluster centers: {ls.cluster_centers_}")

Algorithm Parameters
~~~~~~~~~~~~~~~~~~~~

.. list-table:: LocalSearch Parameters
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - n_clusters
     - Required
     - Number of clusters (submarkets) to find
   * - max_iter
     - 100
     - Maximum number of iterations per restart
   * - n_restarts
     - 10
     - Number of random restarts to avoid local optima
   * - tol
     - 1e-4
     - Convergence tolerance
   * - init
     - 'random'
     - Initialization method: 'random', 'k-means++', or array
   * - random_state
     - None
     - Random seed for reproducibility

Evaluation Module
-----------------

.. currentmodule:: submarit.evaluation

Cluster Evaluator
~~~~~~~~~~~~~~~~~

.. autoclass:: submarit.evaluation.ClusterEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Available metrics:

- **silhouette**: Silhouette coefficient (-1 to 1, higher is better)
- **calinski_harabasz**: Calinski-Harabasz index (higher is better)
- **davies_bouldin**: Davies-Bouldin index (lower is better)
- **dunn**: Dunn index (higher is better)
- **within_cluster_sum**: Within-cluster sum of squares (lower is better)

Example:

.. code-block:: python

    from submarit.evaluation import ClusterEvaluator
    
    evaluator = ClusterEvaluator()
    
    # Evaluate all metrics
    metrics = evaluator.evaluate(S, clusters)
    
    # Evaluate specific metrics
    metrics = evaluator.evaluate(S, clusters, 
                                metrics=['silhouette', 'dunn'])
    
    # Get detailed cluster statistics
    stats = evaluator.cluster_statistics(S, clusters)

Gap Statistic
~~~~~~~~~~~~~

.. autofunction:: submarit.evaluation.gap_statistic

Example:

.. code-block:: python

    from submarit.evaluation import gap_statistic
    
    # Calculate gap statistic
    gap, std = gap_statistic(S, n_clusters=5, n_bootstrap=50)
    
    # Find optimal number of clusters
    gaps = []
    for k in range(2, 11):
        gap, _ = gap_statistic(S, k, n_bootstrap=20)
        gaps.append(gap)
    
    optimal_k = np.argmax(gaps) + 2

Entropy Evaluator
~~~~~~~~~~~~~~~~~

.. autoclass:: submarit.evaluation.EntropyEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Statistical Tests
~~~~~~~~~~~~~~~~~

.. automodule:: submarit.evaluation.statistical_tests
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
~~~~~~~~~~~~~

.. automodule:: submarit.evaluation.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

    from submarit.evaluation.visualization import (
        plot_substitution_matrix,
        plot_cluster_heatmap,
        plot_evaluation_metrics
    )
    
    # Plot substitution matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_substitution_matrix(S, clusters, ax=ax)
    
    # Plot cluster heatmap
    plot_cluster_heatmap(S, clusters)
    
    # Plot evaluation metrics
    plot_evaluation_metrics(metrics_dict)

Validation Module
-----------------

.. currentmodule:: submarit.validation

K-Fold Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: submarit.validation.KFoldValidator
   :members:
   :undoc-members:
   :show-inheritance:

Multiple Runs Validation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: submarit.validation.MultipleRunsValidator
   :members:
   :undoc-members:
   :show-inheritance:

Rand Index Calculation
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: submarit.validation.rand_index
   :members:
   :undoc-members:
   :show-inheritance:

P-Value Analysis
~~~~~~~~~~~~~~~~

.. automodule:: submarit.validation.p_values
   :members:
   :undoc-members:
   :show-inheritance:

Top-K Analysis
~~~~~~~~~~~~~~

.. automodule:: submarit.validation.topk_analysis
   :members:
   :undoc-members:
   :show-inheritance:

IO Module
---------

.. currentmodule:: submarit.io

Data Input/Output
~~~~~~~~~~~~~~~~~

.. automodule:: submarit.io.data_io
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

    from submarit.io import load_data, save_results
    
    # Load data from various formats
    X, names = load_data('products.csv')
    X, names = load_data('products.xlsx', sheet_name='Sheet1')
    X, names = load_data('products.json')
    
    # Save results
    save_results('results.json', {
        'clusters': clusters,
        'metrics': metrics,
        'parameters': params
    })

MATLAB Compatibility
~~~~~~~~~~~~~~~~~~~~

.. automodule:: submarit.io.matlab_io
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

    from submarit.io import load_matlab_data, save_matlab_data
    
    # Load MATLAB .mat file
    data = load_matlab_data('submarkets.mat')
    X = data['features']
    S = data['substitution_matrix']
    
    # Save to MATLAB format
    save_matlab_data('results.mat', {
        'clusters': clusters,
        'substitution_matrix': S,
        'metrics': metrics
    })

Utils Module
------------

.. currentmodule:: submarit.utils

MATLAB Compatibility Utils
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: submarit.utils.matlab_compat
   :members:
   :undoc-members:
   :show-inheritance:

Full API Example
----------------

Here's a complete example using the full API:

.. code-block:: python

    import numpy as np
    from submarit.core import create_substitution_matrix
    from submarit.algorithms import LocalSearch
    from submarit.evaluation import ClusterEvaluator, gap_statistic
    from submarit.validation import KFoldValidator
    from submarit.io import load_data, save_results
    
    # Load data
    X, product_names = load_data('products.csv')
    
    # Create substitution matrix
    S = create_substitution_matrix(X, metric='cosine')
    
    # Find optimal number of clusters
    gaps = []
    for k in range(2, 11):
        gap, _ = gap_statistic(S, k, n_bootstrap=20)
        gaps.append(gap)
    optimal_k = np.argmax(gaps) + 2
    
    # Perform clustering
    clusterer = LocalSearch(n_clusters=optimal_k, n_restarts=20)
    clusters = clusterer.fit_predict(S)
    
    # Evaluate results
    evaluator = ClusterEvaluator()
    metrics = evaluator.evaluate(S, clusters)
    
    # Validate stability
    validator = KFoldValidator(n_splits=5)
    stability_scores = validator.validate(X, n_clusters=optimal_k)
    
    # Save results
    save_results('analysis_results.json', {
        'product_names': product_names,
        'clusters': clusters.tolist(),
        'metrics': metrics,
        'stability': {
            'scores': stability_scores,
            'mean': np.mean(stability_scores),
            'std': np.std(stability_scores)
        },
        'parameters': {
            'n_clusters': optimal_k,
            'metric': 'cosine',
            'algorithm': 'local_search'
        }
    })