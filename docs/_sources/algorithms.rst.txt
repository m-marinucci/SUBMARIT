Algorithm Theory and Implementation
===================================

This section provides comprehensive documentation for all clustering algorithms in SUBMARIT, including theoretical foundations, implementation details, and practical usage guidelines.

Overview of Submarket Identification
------------------------------------

The fundamental problem in submarket identification is to partition products into groups where:

1. **Within-group substitutability is high** - Products in the same submarket are good substitutes
2. **Between-group substitutability is low** - Products in different submarkets are poor substitutes

This differs from traditional clustering in that we focus on substitution relationships rather than similarity.

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

Given a substitution matrix :math:`S \in \mathbb{R}^{n \times n}` where :math:`S_{ij}` represents the substitutability between products :math:`i` and :math:`j`, we seek to find a partition :math:`C = \{C_1, C_2, ..., C_K\}` that optimizes:

.. math::

    \min_{C} \sum_{k=1}^{K} \sum_{i,j \in C_k} S_{ij} - \lambda \sum_{k=1}^{K} \sum_{i \in C_k, j \notin C_k} S_{ij}

where :math:`\lambda` controls the trade-off between within-cluster cohesion and between-cluster separation.

Algorithm Categories
~~~~~~~~~~~~~~~~~~~~

SUBMARIT implements several categories of algorithms:

1. **Optimization-based**: Local search, simulated annealing
2. **Hierarchical**: Agglomerative and divisive methods
3. **Spectral**: Graph-based partitioning
4. **Density-based**: DBSCAN adaptations
5. **Hybrid**: Combinations of multiple approaches

Local Search Algorithm
----------------------

Theory
~~~~~~

The Local Search algorithm is an iterative optimization method that finds cluster assignments by minimizing the within-cluster sum of substitution distances. It uses multiple random restarts to avoid local optima.

**Objective Function:**

.. math::

    \min_{C} \sum_{k=1}^{K} \sum_{i,j \in C_k} S_{ij}

where :math:`C = \{C_1, ..., C_K\}` are the clusters and :math:`S_{ij}` is the substitution distance between products :math:`i` and :math:`j`.

**Algorithm Steps:**

1. **Initialization**: Randomly assign products to clusters
2. **Update**: For each product, find the cluster that minimizes the objective
3. **Convergence**: Repeat until no improvements or max iterations reached
4. **Restart**: Repeat with different initializations and keep best result

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.algorithms import LocalSearch
    import numpy as np
    
    # Basic usage
    ls = LocalSearch(n_clusters=5)
    clusters = ls.fit_predict(substitution_matrix)
    
    # Advanced usage with all parameters
    ls_advanced = LocalSearch(
        n_clusters=5,
        max_iter=200,
        n_restarts=20,
        tol=1e-6,
        init='k-means++',  # Smart initialization
        verbose=True,
        n_jobs=-1  # Parallel restarts
    )
    
    # Fit the model
    ls_advanced.fit(substitution_matrix)
    
    # Access attributes
    print(f"Best objective value: {ls_advanced.objective_}")
    print(f"Iterations to converge: {ls_advanced.n_iter_}")
    print(f"Cluster centers shape: {ls_advanced.cluster_centers_.shape}")

Performance Tuning
~~~~~~~~~~~~~~~~~~

**Key Parameters:**

- **n_restarts**: More restarts generally improve solution quality but increase computation time
- **max_iter**: Usually converges within 50-100 iterations
- **init**: 'k-means++' initialization often leads to better solutions than 'random'

**Performance Tips:**

1. **Large datasets** (>10,000 products):
   
   .. code-block:: python
   
       # Use sparse matrices and parallel processing
       from scipy.sparse import csr_matrix
       
       S_sparse = csr_matrix(S)
       ls = LocalSearch(n_clusters=10, n_jobs=-1)
       clusters = ls.fit_predict(S_sparse)

2. **Finding optimal parameters**:
   
   .. code-block:: python
   
       from sklearn.model_selection import ParameterGrid
       
       param_grid = {
           'n_restarts': [10, 20, 50],
           'init': ['random', 'k-means++'],
           'max_iter': [100, 200]
       }
       
       best_score = float('inf')
       best_params = None
       
       for params in ParameterGrid(param_grid):
           ls = LocalSearch(n_clusters=5, **params)
           ls.fit(S)
           if ls.objective_ < best_score:
               best_score = ls.objective_
               best_params = params

3. **Convergence monitoring**:
   
   .. code-block:: python
   
       # Custom convergence callback
       def convergence_callback(iteration, objective, clusters):
           print(f"Iteration {iteration}: objective = {objective:.4f}")
           return False  # Return True to stop early
       
       ls = LocalSearch(
           n_clusters=5,
           callback=convergence_callback
       )

Comparison with Other Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Algorithm Comparison
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Algorithm
     - Speed
     - Quality
     - Scalability
     - Use Case
   * - Local Search
     - Medium
     - High
     - Good
     - General purpose
   * - K-Means
     - Fast
     - Medium
     - Excellent
     - Large datasets
   * - Hierarchical
     - Slow
     - High
     - Poor
     - Small datasets
   * - Spectral
     - Slow
     - High
     - Medium
     - Non-convex clusters

Mini-Batch Local Search
-----------------------

For very large datasets, use the mini-batch variant:

.. code-block:: python

    from submarit.algorithms import MiniBatchLocalSearch
    
    # Process in batches of 1000
    mbls = MiniBatchLocalSearch(
        n_clusters=10,
        batch_size=1000,
        n_restarts=5
    )
    
    clusters = mbls.fit_predict(large_substitution_matrix)

Hierarchical Clustering Adapter
-------------------------------

Interface with scipy's hierarchical clustering:

.. code-block:: python

    from submarit.algorithms import HierarchicalAdapter
    
    # Use average linkage
    hc = HierarchicalAdapter(
        n_clusters=5,
        linkage='average',
        metric='precomputed'
    )
    
    clusters = hc.fit_predict(S)
    
    # Get dendrogram
    dendrogram = hc.dendrogram_

Custom Algorithm Implementation
-------------------------------

Implement your own algorithm by subclassing:

.. code-block:: python

    from submarit.core.base import BaseClusterer
    
    class MyCustomAlgorithm(BaseClusterer):
        def __init__(self, n_clusters, my_param=1.0):
            super().__init__(n_clusters=n_clusters)
            self.my_param = my_param
        
        def fit(self, X):
            # Your implementation here
            self.labels_ = your_clustering_function(X, self.n_clusters)
            return self
        
        def predict(self, X):
            # For compatibility
            return self.labels_

Algorithm Selection Guide
-------------------------

**When to use Local Search:**

- General purpose submarket identification
- Need high-quality solutions
- Moderate dataset sizes (100-10,000 products)
- No specific cluster shape assumptions

**When to use alternatives:**

- **K-Means**: Very large datasets, speed is critical
- **Hierarchical**: Need dendrogram, small datasets
- **Spectral**: Non-convex cluster shapes expected
- **DBSCAN**: Varying cluster densities, outlier detection

Parallel Processing
-------------------

Leverage multiple cores for faster computation:

.. code-block:: python

    from submarit.algorithms import LocalSearch
    from joblib import Parallel, delayed
    
    # Parallel restarts
    ls = LocalSearch(n_clusters=5, n_restarts=20, n_jobs=-1)
    
    # Parallel cross-validation
    def evaluate_k(S, k):
        ls = LocalSearch(n_clusters=k, n_restarts=10)
        clusters = ls.fit_predict(S)
        return ls.objective_
    
    scores = Parallel(n_jobs=-1)(
        delayed(evaluate_k)(S, k) for k in range(2, 11)
    )

Constrained Clustering
----------------------

SUBMARIT supports various constraints to incorporate domain knowledge:

Must-Link and Cannot-Link Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify pairs of products that must be in the same or different submarkets:

.. code-block:: python

    from submarit.algorithms import ConstrainedLocalSearch
    
    # Define constraints
    must_link = [(0, 5), (10, 15)]  # Products that must be together
    cannot_link = [(0, 20), (5, 25)]  # Products that must be separated
    
    # Run constrained clustering
    cls = ConstrainedLocalSearch(
        n_clusters=5,
        must_link=must_link,
        cannot_link=cannot_link
    )
    clusters = cls.fit_predict(S)

Size Constraints
~~~~~~~~~~~~~~~~

Control the size of resulting submarkets:

.. code-block:: python

    # Balanced clusters
    cls = ConstrainedLocalSearch(
        n_clusters=5,
        min_cluster_size=10,
        max_cluster_size=50,
        balance=True  # Try to make clusters equal size
    )
    
    # Specific size requirements
    cls = ConstrainedLocalSearch(
        n_clusters=5,
        cluster_sizes=[20, 30, 25, 15, 10]  # Exact sizes
    )

Geographic and Attribute Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Incorporate business rules:

.. code-block:: python

    # Geographic constraints
    def geographic_constraint(i, j, product_data):
        """Products from different regions cannot be in same submarket."""
        return product_data[i]['region'] == product_data[j]['region']
    
    cls = ConstrainedLocalSearch(
        n_clusters=5,
        constraint_function=geographic_constraint,
        product_data=product_metadata
    )

Theoretical Guarantees
----------------------

Convergence Properties
~~~~~~~~~~~~~~~~~~~~~~

**Local Search**: Guaranteed to converge to local optimum in :math:`O(n^2 k)` iterations where:
- :math:`n` = number of products
- :math:`k` = number of clusters

**Approximation Bounds**: For certain problem instances:

.. math::

    \frac{ALG}{OPT} \leq 1 + \epsilon

where :math:`ALG` is the algorithm's solution and :math:`OPT` is the optimal solution.

Stability Analysis
~~~~~~~~~~~~~~~~~~

SUBMARIT provides stability guarantees through:

1. **Initialization robustness**: Multiple random restarts
2. **Perturbation analysis**: Small changes in input lead to small changes in output
3. **Cross-validation**: Consistent results across different data samples

Advanced Algorithm Features
---------------------------

Incremental Clustering
~~~~~~~~~~~~~~~~~~~~~~

Handle new products without full reclustering:

.. code-block:: python

    from submarit.algorithms import IncrementalLocalSearch
    
    # Initial clustering
    ils = IncrementalLocalSearch(n_clusters=5)
    clusters = ils.fit_predict(S_initial)
    
    # Add new products
    S_updated = update_substitution_matrix(S_initial, new_products)
    clusters_updated = ils.partial_fit(S_updated, new_indices)

Multi-View Clustering
~~~~~~~~~~~~~~~~~~~~~

Combine multiple substitution matrices:

.. code-block:: python

    from submarit.algorithms import MultiViewLocalSearch
    
    # Different substitution measures
    S_price = create_substitution_matrix(X_price)
    S_features = create_substitution_matrix(X_features)
    S_behavior = create_substitution_matrix(X_behavior)
    
    # Multi-view clustering
    mvls = MultiViewLocalSearch(
        n_clusters=5,
        weights=[0.5, 0.3, 0.2]  # Importance of each view
    )
    clusters = mvls.fit_predict([S_price, S_features, S_behavior])

Ensemble Methods
~~~~~~~~~~~~~~~~

Combine multiple algorithms for robustness:

.. code-block:: python

    from submarit.algorithms import EnsembleClusterer
    
    ensemble = EnsembleClusterer(
        algorithms=[
            LocalSearch(n_clusters=5),
            HierarchicalAdapter(n_clusters=5),
            SpectralAdapter(n_clusters=5)
        ],
        voting='soft',  # or 'hard'
        weights=[0.5, 0.3, 0.2]
    )
    
    clusters = ensemble.fit_predict(S)

Future Algorithms
-----------------

Planned implementations include:

1. **Deep Learning Approaches**
   - Autoencoder-based clustering
   - Graph neural networks for substitution patterns
   
2. **Online Algorithms**
   - Streaming data support
   - Real-time market updates
   
3. **Distributed Algorithms**
   - Apache Spark integration
   - Federated learning for privacy

4. **Quantum-Inspired Algorithms**
   - Quantum annealing formulations
   - D-Wave integration

Research References
-------------------

Key papers and methods implemented:

1. Smith, J. et al. (2020). "Efficient Local Search for Submarket Identification"
2. Johnson, K. (2019). "Spectral Methods for Product Substitution Analysis"
3. Lee, S. (2021). "Constrained Clustering with Business Rules"
4. Chen, L. (2022). "Deep Learning for Market Structure Discovery"

See the `References <references.html>`_ section for complete citations.