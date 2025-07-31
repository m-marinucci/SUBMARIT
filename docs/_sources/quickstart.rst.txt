Quick Start Tutorial
====================

This tutorial will get you started with SUBMARIT in 5 minutes.

Basic Usage
-----------

1. Import and Load Data
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    from submarit.core import create_substitution_matrix
    from submarit.algorithms import LocalSearch
    from submarit.evaluation import ClusterEvaluator
    
    # Load your data (rows: products, columns: features)
    # Example with synthetic data
    n_products = 100
    n_features = 20
    X = np.random.rand(n_products, n_features)
    
    # Or load from file
    # X = pd.read_csv('your_data.csv').values

2. Create Substitution Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create substitution matrix from product features
    S = create_substitution_matrix(X, metric='euclidean')
    
    # S[i,j] represents substitutability between products i and j
    print(f"Substitution matrix shape: {S.shape}")
    print(f"Substitution values range: [{S.min():.3f}, {S.max():.3f}]")

3. Identify Submarkets
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize clustering algorithm
    local_search = LocalSearch(
        n_clusters=5,           # Number of submarkets
        max_iter=100,          # Maximum iterations
        n_restarts=10,         # Random restarts for robustness
        random_state=42
    )
    
    # Fit the model
    clusters = local_search.fit_predict(S)
    
    print(f"Found {len(np.unique(clusters))} submarkets")
    print(f"Cluster sizes: {np.bincount(clusters)}")

4. Evaluate Results
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create evaluator
    evaluator = ClusterEvaluator()
    
    # Calculate quality metrics
    metrics = evaluator.evaluate(S, clusters)
    
    print("\nClustering Quality Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

5. Visualize Results
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.evaluation.visualization import plot_substitution_matrix
    import matplotlib.pyplot as plt
    
    # Plot substitution matrix with cluster boundaries
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_substitution_matrix(S, clusters, ax=ax)
    plt.title("Product Substitution Matrix with Submarkets")
    plt.show()

Complete Example
----------------

Here's a complete example analyzing product submarkets:

.. code-block:: python

    import numpy as np
    from submarit import SubmarketAnalyzer
    
    # Initialize analyzer with all components
    analyzer = SubmarketAnalyzer(
        n_clusters=5,
        algorithm='local_search',
        validation_method='kfold',
        random_state=42
    )
    
    # Generate example data (replace with your data)
    n_products = 200
    n_features = 30
    X = np.random.rand(n_products, n_features)
    
    # Add product names (optional)
    product_names = [f"Product_{i}" for i in range(n_products)]
    
    # Run complete analysis
    results = analyzer.analyze(X, product_names=product_names)
    
    # Access results
    print(f"Optimal number of clusters: {results['optimal_k']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Cluster assignments: {results['clusters']}")
    
    # Get detailed report
    report = analyzer.generate_report(results)
    print(report)

Working with Real Data
----------------------

CSV Files
~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from submarit.io import load_data
    
    # Load from CSV
    X, product_names = load_data('products.csv', 
                                 name_column='product_id',
                                 feature_columns=None)  # Use all numeric columns

Excel Files
~~~~~~~~~~~

.. code-block:: python

    # Load from Excel
    X, product_names = load_data('products.xlsx', 
                                 sheet_name='ProductFeatures',
                                 name_column='SKU')

MATLAB Files
~~~~~~~~~~~~

.. code-block:: python

    from submarit.io import load_matlab_data
    
    # Load from MATLAB .mat file
    data = load_matlab_data('submarkets.mat')
    X = data['features']
    S = data.get('substitution_matrix', None)

Command Line Usage
------------------

SUBMARIT can also be used from the command line:

.. code-block:: bash

    # Basic clustering
    submarit cluster data.csv --n-clusters 5 --output results.json
    
    # With visualization
    submarit cluster data.csv --n-clusters 5 --plot --output results.json
    
    # Automatic cluster number selection
    submarit cluster data.csv --auto-k --k-range 2 10 --output results.json

Next Steps
----------

- Learn about `advanced algorithms <api/algorithms.html>`_
- Explore `evaluation methods <api/evaluation.html>`_
- Read the `MATLAB migration guide <migration_guide.html>`_
- Check out `example notebooks <https://github.com/yourusername/submarit/tree/main/examples>`_

Common Patterns
---------------

Optimal Number of Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.evaluation import gap_statistic
    
    # Test different numbers of clusters
    k_values = range(2, 11)
    gaps, stds = [], []
    
    for k in k_values:
        gap, std = gap_statistic(S, k, n_bootstrap=50)
        gaps.append(gap)
        stds.append(std)
    
    # Find optimal k using elbow method
    optimal_k = k_values[np.argmax(gaps)]
    print(f"Optimal number of clusters: {optimal_k}")

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.validation import KFoldValidator
    
    # Set up k-fold validation
    validator = KFoldValidator(n_splits=5)
    
    # Validate clustering stability
    scores = validator.validate(X, n_clusters=5)
    print(f"Validation scores: {scores}")
    print(f"Mean score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

Handling Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.core import create_sparse_substitution_matrix
    
    # For large datasets, use sparse representation
    S_sparse = create_sparse_substitution_matrix(X, 
                                                threshold=0.1,  # Keep only top 10%
                                                metric='cosine')
    
    # Use mini-batch processing
    from submarit.algorithms import MiniBatchLocalSearch
    
    clusterer = MiniBatchLocalSearch(n_clusters=5, batch_size=1000)
    clusters = clusterer.fit_predict(S_sparse)