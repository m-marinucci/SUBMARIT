MATLAB to Python Migration Guide
=================================

This guide helps MATLAB users transition from the original MATLAB SUBMARIT implementation to the Python version.

.. note::
   The original MATLAB implementation was developed by Stephen France (Mississippi State University) 
   and other contributors. This Python implementation maintains compatibility while offering 
   modern improvements and performance optimizations.

Function Mapping Table
----------------------

Core Functions
~~~~~~~~~~~~~~

.. list-table:: Core Function Mappings
   :widths: 40 40 20
   :header-rows: 1

   * - MATLAB Function
     - Python Equivalent
     - Notes
   * - ``create_substitution_matrix(X)``
     - ``submarit.core.create_substitution_matrix(X)``
     - Identical interface
   * - ``local_search(S, k, options)``
     - ``LocalSearch(n_clusters=k, **options).fit_predict(S)``
     - Object-oriented API
   * - ``evaluate_clusters(S, clusters)``
     - ``ClusterEvaluator().evaluate(S, clusters)``
     - Returns dict instead of struct
   * - ``gap_statistic(S, k, B)``
     - ``gap_statistic(S, k, n_bootstrap=B)``
     - Parameter name change
   * - ``kfold_validation(X, k, nfolds)``
     - ``KFoldValidator(n_splits=nfolds).validate(X, k)``
     - Class-based approach

Data Structure Conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Data Structure Mappings
   :widths: 30 30 40
   :header-rows: 1

   * - MATLAB
     - Python
     - Conversion Example
   * - ``matrix``
     - ``numpy.ndarray``
     - ``X = np.array(matlab_matrix)``
   * - ``cell array``
     - ``list`` or ``numpy.object_``
     - ``names = list(cell_array)``
   * - ``struct``
     - ``dict`` or ``namedtuple``
     - ``params = dict(matlab_struct)``
   * - ``table``
     - ``pandas.DataFrame``
     - ``df = pd.DataFrame(table_data)``
   * - ``sparse matrix``
     - ``scipy.sparse``
     - ``S = scipy.sparse.csr_matrix(S_matlab)``

Common Operations
~~~~~~~~~~~~~~~~~

.. list-table:: Operation Mappings
   :widths: 35 35 30
   :header-rows: 1

   * - MATLAB
     - Python
     - Notes
   * - ``size(X)``
     - ``X.shape``
     - Returns tuple
   * - ``length(x)``
     - ``len(x)`` or ``x.size``
     - Use ``len`` for 1D
   * - ``X'``
     - ``X.T`` or ``X.transpose()``
     - 
   * - ``X(:)``
     - ``X.flatten()`` or ``X.ravel()``
     - 
   * - ``X(1:10, :)``
     - ``X[0:10, :]`` or ``X[:10, :]``
     - 0-based indexing
   * - ``find(X > 0)``
     - ``np.where(X > 0)`` or ``np.argwhere(X > 0)``
     - 
   * - ``mean(X)``
     - ``np.mean(X, axis=0)``
     - Specify axis
   * - ``std(X)``
     - ``np.std(X, axis=0, ddof=1)``
     - MATLAB uses ddof=1

Code Conversion Examples
------------------------

Example 1: Basic Substitution Matrix Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**MATLAB:**

.. code-block:: matlab

    % Load data
    data = readtable('products.csv');
    X = table2array(data(:, 2:end));
    
    % Create substitution matrix
    S = create_substitution_matrix(X, 'metric', 'euclidean');
    
    % Cluster
    options.maxiter = 100;
    options.nrestarts = 10;
    [clusters, obj] = local_search(S, 5, options);

**Python:**

.. code-block:: python

    # Load data
    import pandas as pd
    import numpy as np
    from submarit.core import create_substitution_matrix
    from submarit.algorithms import LocalSearch
    
    data = pd.read_csv('products.csv')
    X = data.iloc[:, 1:].values  # All columns except first
    
    # Create substitution matrix
    S = create_substitution_matrix(X, metric='euclidean')
    
    # Cluster
    ls = LocalSearch(n_clusters=5, max_iter=100, n_restarts=10)
    clusters = ls.fit_predict(S)
    obj = ls.objective_

Example 2: Evaluation and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**MATLAB:**

.. code-block:: matlab

    % Evaluate clustering
    metrics = evaluate_clusters(S, clusters);
    fprintf('Silhouette: %.3f\n', metrics.silhouette);
    
    % Visualize
    figure;
    imagesc(S);
    colorbar;
    title('Substitution Matrix');
    
    % Plot sorted matrix
    [sorted_S, idx] = sort_matrix_by_clusters(S, clusters);
    figure;
    imagesc(sorted_S);

**Python:**

.. code-block:: python

    # Evaluate clustering
    from submarit.evaluation import ClusterEvaluator
    
    evaluator = ClusterEvaluator()
    metrics = evaluator.evaluate(S, clusters)
    print(f"Silhouette: {metrics['silhouette']:.3f}")
    
    # Visualize
    import matplotlib.pyplot as plt
    from submarit.evaluation.visualization import plot_substitution_matrix
    
    plt.figure(figsize=(10, 8))
    plt.imshow(S, cmap='viridis')
    plt.colorbar()
    plt.title('Substitution Matrix')
    plt.show()
    
    # Plot sorted matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_substitution_matrix(S, clusters, ax=ax)
    plt.show()

Example 3: Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**MATLAB:**

.. code-block:: matlab

    % K-fold cross-validation
    nfolds = 5;
    scores = zeros(nfolds, 1);
    
    for i = 1:nfolds
        [train_idx, test_idx] = get_fold_indices(size(X, 1), nfolds, i);
        X_train = X(train_idx, :);
        X_test = X(test_idx, :);
        
        % Train and evaluate
        S_train = create_substitution_matrix(X_train);
        clusters_train = local_search(S_train, 5);
        
        score = evaluate_fold(X_test, clusters_train);
        scores(i) = score;
    end
    
    fprintf('CV Score: %.3f ± %.3f\n', mean(scores), std(scores));

**Python:**

.. code-block:: python

    # K-fold cross-validation
    from sklearn.model_selection import KFold
    from submarit.validation import KFoldValidator
    
    # Method 1: Using SUBMARIT's validator
    validator = KFoldValidator(n_splits=5)
    scores = validator.validate(X, n_clusters=5)
    print(f"CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # Method 2: Manual implementation (similar to MATLAB)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        
        # Train and evaluate
        S_train = create_substitution_matrix(X_train)
        ls = LocalSearch(n_clusters=5)
        clusters_train = ls.fit_predict(S_train)
        
        score = evaluate_fold(X_test, clusters_train)
        scores.append(score)
    
    print(f"CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

Common Pitfalls and Solutions
-----------------------------

1. Indexing Differences
~~~~~~~~~~~~~~~~~~~~~~~

**MATLAB (1-based):**

.. code-block:: matlab

    X(1, 1)      % First element
    X(end, :)    % Last row
    X(2:5, :)    % Rows 2-5

**Python (0-based):**

.. code-block:: python

    X[0, 0]      # First element
    X[-1, :]     # Last row
    X[1:5, :]    # Rows 2-5 (exclusive end)

2. Broadcasting Behavior
~~~~~~~~~~~~~~~~~~~~~~~~

**MATLAB:**

.. code-block:: matlab

    A = [1; 2; 3];  % Column vector
    B = [4, 5, 6];  % Row vector
    C = A + B;      % Error in MATLAB

**Python:**

.. code-block:: python

    A = np.array([[1], [2], [3]])  # Column vector
    B = np.array([4, 5, 6])         # Row vector
    C = A + B                       # Broadcasting works!

3. Function Return Values
~~~~~~~~~~~~~~~~~~~~~~~~~

**MATLAB:**

.. code-block:: matlab

    [U, S, V] = svd(X);  % Multiple outputs
    [~, idx] = max(x);   % Ignore first output

**Python:**

.. code-block:: python

    U, S, V = np.linalg.svd(X)  # Multiple outputs
    idx = np.argmax(x)           # Direct function for index

4. Default Random State
~~~~~~~~~~~~~~~~~~~~~~~

**MATLAB:**

.. code-block:: matlab

    rng(42);  % Set random seed
    x = rand(100, 1);

**Python:**

.. code-block:: python

    np.random.seed(42)  # Set random seed
    x = np.random.rand(100, 1)
    
    # Better: use RandomState
    rng = np.random.RandomState(42)
    x = rng.rand(100, 1)

Numerical Differences
---------------------

Precision and Tolerance
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # MATLAB and Python may have different default tolerances
    # Be explicit about tolerances
    
    # MATLAB: eps
    # Python equivalent:
    eps = np.finfo(float).eps
    
    # For algorithms
    ls = LocalSearch(n_clusters=5, tol=1e-6)  # Specify tolerance

Linear Algebra Differences
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # MATLAB uses LAPACK/BLAS, Python uses NumPy's version
    # Results may differ slightly
    
    # For exact reproducibility
    import scipy.linalg
    
    # Use same backend as MATLAB
    eigenvalues = scipy.linalg.eigh(S, driver='ev')

MATLAB Integration
------------------

Using MATLAB Engine
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matlab.engine
    
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    
    # Call MATLAB functions from Python
    matlab_result = eng.your_matlab_function(data)
    
    # Convert to Python
    python_result = np.array(matlab_result)
    
    # Stop engine
    eng.quit()

Loading MATLAB Files
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from scipy.io import loadmat, savemat
    
    # Load .mat file
    mat_data = loadmat('data.mat')
    X = mat_data['X']
    clusters = mat_data['clusters'].squeeze()  # Remove singleton dimensions
    
    # Save to .mat file
    savemat('results.mat', {
        'clusters': clusters,
        'metrics': metrics,
        'S': S
    })

Performance Comparison
----------------------

.. list-table:: Performance Characteristics
   :widths: 30 35 35
   :header-rows: 1

   * - Operation
     - MATLAB
     - Python (NumPy)
   * - Matrix multiplication
     - Very fast (MKL)
     - Fast (OpenBLAS/MKL)
   * - For loops
     - Slow
     - Very slow (use vectorization)
   * - Memory usage
     - Copy-on-write
     - Views when possible
   * - Parallel computing
     - Parallel Computing Toolbox
     - multiprocessing/joblib
   * - GPU support
     - GPU Computing Toolbox
     - CuPy/PyTorch/TensorFlow

Best Practices for Migration
----------------------------

1. **Start with small examples** - Verify numerical equivalence
2. **Use MATLAB compatibility layer** during transition:
   
   .. code-block:: python
   
       from submarit.utils.matlab_compat import matlab_style_api
       
       # Use MATLAB-like interface
       S = matlab_style_api.create_substitution_matrix(X)

3. **Validate results** against MATLAB output:
   
   .. code-block:: python
   
       # Load MATLAB results
       matlab_results = loadmat('matlab_results.mat')
       
       # Compare
       np.testing.assert_allclose(
           python_clusters, 
           matlab_results['clusters'].squeeze(),
           rtol=1e-5
       )

4. **Profile both implementations** to ensure performance parity
5. **Document any numerical differences** for your team

Additional Resources
--------------------

- `NumPy for MATLAB users <https://numpy.org/doc/stable/user/numpy-for-matlab-users.html>`_
- `SciPy Tutorial <https://docs.scipy.org/doc/scipy/tutorial/index.html>`_
- `Python Data Science Handbook <https://jakevdp.github.io/PythonDataScienceHandbook/>`_