Frequently Asked Questions (FAQ)
================================

General Questions
-----------------

What is SUBMARIT?
~~~~~~~~~~~~~~~~~

SUBMARIT (SUBMARket Identification and Testing) is a Python package for identifying and analyzing submarkets based on product substitution patterns. It helps businesses and researchers understand market structure by clustering products based on how substitutable they are for each other.

How does SUBMARIT differ from standard clustering?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While standard clustering groups similar items, SUBMARIT specifically focuses on substitution relationships. Two products might be different in features but highly substitutable (e.g., butter and margarine), which SUBMARIT captures through specialized algorithms and evaluation metrics.

What data do I need?
~~~~~~~~~~~~~~~~~~~~

You need a matrix where:
- Each row represents a product
- Each column represents a feature or attribute
- Values can be numeric (prices, quantities) or encoded categorical data (brand, category)

Example data structure:

.. code-block:: python

    # Products x Features matrix
    # Columns: [price, size, brand_encoded, category_encoded, ...]
    X = np.array([
        [2.99, 16, 0, 1, ...],  # Product 1
        [3.49, 12, 1, 1, ...],  # Product 2
        ...
    ])

Installation Issues
-------------------

ImportError: No module named 'submarit'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution 1:** Ensure you're in the correct environment:

.. code-block:: bash

    # Check if installed
    pip list | grep submarit
    
    # If using conda
    conda list submarit

**Solution 2:** Reinstall:

.. code-block:: bash

    pip uninstall submarit
    pip install submarit

Cannot build wheel for numpy/scipy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution:** Install pre-built wheels:

.. code-block:: bash

    # Windows
    pip install --only-binary :all: numpy scipy
    
    # Or use conda
    conda install numpy scipy

MATLAB Engine not found
~~~~~~~~~~~~~~~~~~~~~~~

**Solution:** Install MATLAB Engine API for Python:

.. code-block:: bash

    cd "C:\Program Files\MATLAB\R2023b\extern\engines\python"  # Windows
    cd "/Applications/MATLAB_R2023b.app/extern/engines/python"  # macOS
    python setup.py install

Algorithm Questions
-------------------

How do I choose the number of clusters?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use multiple methods to find optimal k:

.. code-block:: python

    from submarit.evaluation import gap_statistic, elbow_method
    from submarit.algorithms import LocalSearch
    
    # Method 1: Gap statistic
    gaps = []
    for k in range(2, 11):
        gap, std = gap_statistic(S, k, n_bootstrap=50)
        gaps.append(gap)
    optimal_k = np.argmax(gaps) + 2
    
    # Method 2: Elbow method
    scores = []
    for k in range(2, 11):
        ls = LocalSearch(n_clusters=k)
        ls.fit(S)
        scores.append(ls.objective_)
    
    # Plot and look for "elbow"
    plt.plot(range(2, 11), scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster sum')

Local Search doesn't converge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution 1:** Increase iterations and tolerance:

.. code-block:: python

    ls = LocalSearch(
        n_clusters=5,
        max_iter=500,      # Increase from default 100
        tol=1e-6,          # Decrease from default 1e-4
        n_restarts=20      # More random restarts
    )

**Solution 2:** Check data scale:

.. code-block:: python

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    S = create_substitution_matrix(X_scaled)

Results are unstable between runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution:** Set random seed for reproducibility:

.. code-block:: python

    # Set global seed
    np.random.seed(42)
    
    # Or use random_state parameter
    ls = LocalSearch(n_clusters=5, random_state=42)
    
    # For complete reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    
    # If using parallel processing
    import os
    os.environ['PYTHONHASHSEED'] = '42'

Performance Issues
------------------

Out of memory with large datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution 1:** Use sparse matrices:

.. code-block:: python

    from submarit.core import create_sparse_substitution_matrix
    
    S_sparse = create_sparse_substitution_matrix(
        X,
        threshold=0.1,  # Keep only top 10% of values
        format='csr'
    )

**Solution 2:** Process in chunks:

.. code-block:: python

    # Mini-batch processing
    from submarit.algorithms import MiniBatchLocalSearch
    
    mbls = MiniBatchLocalSearch(
        n_clusters=10,
        batch_size=1000
    )

**Solution 3:** Use float32 instead of float64:

.. code-block:: python

    X = X.astype(np.float32)
    S = create_substitution_matrix(X, dtype=np.float32)

Slow computation
~~~~~~~~~~~~~~~~

**Solution 1:** Enable parallel processing:

.. code-block:: python

    ls = LocalSearch(n_clusters=5, n_jobs=-1)  # Use all cores

**Solution 2:** Use approximate methods:

.. code-block:: python

    from submarit.algorithms import ApproximateLocalSearch
    
    als = ApproximateLocalSearch(
        n_clusters=5,
        approximation='sample',
        sample_size=5000
    )

**Solution 3:** Profile to find bottlenecks:

.. code-block:: python

    import cProfile
    import pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    clusters = ls.fit_predict(S)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

Numerical Differences
---------------------

Results differ from MATLAB version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common causes and solutions:

1. **Different random seeds:**
   
   .. code-block:: python
   
       # MATLAB: rng(42)
       # Python:
       np.random.seed(42)

2. **Indexing differences (0 vs 1-based):**
   
   .. code-block:: python
   
       # MATLAB: clusters (1-indexed)
       # Python: clusters - 1 (0-indexed)
       matlab_clusters = python_clusters + 1

3. **Numerical precision:**
   
   .. code-block:: python
   
       # Use same tolerance
       ls = LocalSearch(n_clusters=5, tol=1e-6)
       
       # Compare with tolerance
       np.testing.assert_allclose(
           python_result,
           matlab_result,
           rtol=1e-5,
           atol=1e-8
       )

4. **Algorithm initialization:**
   
   .. code-block:: python
   
       # Ensure same initialization
       init_clusters = load_matlab_initialization()
       ls = LocalSearch(n_clusters=5, init=init_clusters)

Small numerical differences in results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is normal due to:
- Floating-point arithmetic differences
- Different BLAS/LAPACK implementations
- Compiler optimizations

To minimize differences:

.. code-block:: python

    # Use higher precision
    X = X.astype(np.float64)
    
    # Disable fast math optimizations
    os.environ['MKL_CBWR'] = 'COMPATIBLE'
    
    # Use same linear algebra backend
    import scipy.linalg
    scipy.linalg.use_solver = 'gesv'

Visualization Questions
-----------------------

How to visualize high-dimensional clusters?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # Method 1: PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis')
    plt.title('Clusters in PCA space')
    
    # Method 2: t-SNE (better for visualization)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis')
    plt.title('Clusters in t-SNE space')

How to create publication-quality plots?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    
    # High DPI for publications
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot with proper labels
    plot_substitution_matrix(S, clusters, ax=ax)
    ax.set_xlabel('Product Index', fontsize=14)
    ax.set_ylabel('Product Index', fontsize=14)
    ax.set_title('Product Substitution Patterns', fontsize=16)
    
    # Save
    plt.tight_layout()
    plt.savefig('submarkets.pdf', bbox_inches='tight')

Best Practices
--------------

Data Preprocessing
~~~~~~~~~~~~~~~~~~

Always preprocess your data:

.. code-block:: python

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Standardize (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Or normalize to [0, 1]
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_complete = imputer.fit_transform(X)

Feature Engineering
~~~~~~~~~~~~~~~~~~~

Create meaningful features:

.. code-block:: python

    # Interaction features
    X_interactions = np.column_stack([
        X,
        X[:, 0] * X[:, 1],  # Price × Size
        X[:, 2] / X[:, 0],  # Brand premium
    ])
    
    # Polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

Validation Strategy
~~~~~~~~~~~~~~~~~~~

Always validate your results:

.. code-block:: python

    from submarit.validation import cross_validate_clustering
    
    # Multiple validation methods
    validation_results = {
        'stability': stability_test(X, n_clusters=5),
        'bootstrap': bootstrap_validation(X, n_clusters=5),
        'noise': noise_injection_test(X, n_clusters=5),
        'holdout': holdout_validation(X, n_clusters=5)
    }
    
    # Report
    for method, score in validation_results.items():
        print(f"{method}: {score:.3f}")

Getting Help
------------

Where to get help?
~~~~~~~~~~~~~~~~~~

1. **Documentation**: Read the full documentation at https://submarit.readthedocs.io
2. **GitHub Issues**: Report bugs or request features at https://github.com/m-marinucci/SUBMARIT/issues
3. **Stack Overflow**: Ask questions with tag 'submarit'
4. **Email**: Contact maintainers at submarit@example.com

How to report a bug?
~~~~~~~~~~~~~~~~~~~~

Include:
1. Minimal reproducible example
2. Full error traceback
3. Environment information:

.. code-block:: python

    import submarit
    import sys
    import numpy as np
    import scipy
    
    print(f"Python: {sys.version}")
    print(f"SUBMARIT: {submarit.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")

How to contribute?
~~~~~~~~~~~~~~~~~~

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines.