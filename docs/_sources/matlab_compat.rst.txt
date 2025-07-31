MATLAB Compatibility Reference
==============================

This reference provides detailed compatibility information between MATLAB and Python implementations.

MATLAB Compatibility Layer
--------------------------

SUBMARIT provides a MATLAB-style API for easier transition:

.. code-block:: python

    from submarit.utils.matlab_compat import MatlabAPI
    
    # Initialize MATLAB-style interface
    m = MatlabAPI()
    
    # Use MATLAB-like function calls
    S = m.create_substitution_matrix(X, 'euclidean')
    clusters = m.local_search(S, 5)
    metrics = m.evaluate_clusters(S, clusters)

Function Equivalence Table
--------------------------

Core Functions
~~~~~~~~~~~~~~

.. list-table:: Detailed Function Mappings
   :widths: 25 25 50
   :header-rows: 1

   * - MATLAB
     - Python
     - Notes & Differences
   * - ``create_substitution_matrix``
     - ``submarit.core.create_substitution_matrix``
     - Python version supports sparse output
   * - ``local_search``
     - ``LocalSearch.fit_predict``
     - Python uses object-oriented design
   * - ``evaluate_clusters``
     - ``ClusterEvaluator.evaluate``
     - Python returns dict, MATLAB returns struct
   * - ``gap_statistic``
     - ``gap_statistic``
     - Identical algorithm, minor numerical differences
   * - ``plot_clusters``
     - ``plot_substitution_matrix``
     - Python uses matplotlib instead of MATLAB plots
   * - ``save_results``
     - ``io.save_results``
     - Python supports JSON, MATLAB uses .mat

Parameter Mappings
~~~~~~~~~~~~~~~~~~

.. list-table:: Parameter Name Conversions
   :widths: 30 30 40
   :header-rows: 1

   * - MATLAB Parameter
     - Python Parameter
     - Example
   * - ``'Distance'``
     - ``metric``
     - ``metric='euclidean'``
   * - ``'NumClusters'``
     - ``n_clusters``
     - ``n_clusters=5``
   * - ``'MaxIter'``
     - ``max_iter``
     - ``max_iter=100``
   * - ``'Replicates'``
     - ``n_restarts``
     - ``n_restarts=10``
   * - ``'Display'``
     - ``verbose``
     - ``verbose=True``
   * - ``'Options'``
     - ``**kwargs``
     - ``LocalSearch(**options_dict)``

Data Type Conversions
---------------------

Automatic Conversions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.utils.matlab_compat import matlab_to_python, python_to_matlab
    
    # Convert MATLAB data to Python
    python_data = matlab_to_python(matlab_data)
    
    # Convert back
    matlab_data = python_to_matlab(python_data)

Manual Conversions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from scipy.io import loadmat, savemat
    
    # Cell arrays to lists
    matlab_cells = mat_data['cell_array'][0]
    python_list = [str(cell[0]) for cell in matlab_cells]
    
    # Struct to dict
    matlab_struct = mat_data['params']
    python_dict = {
        field: matlab_struct[field][0, 0]
        for field in matlab_struct.dtype.names
    }
    
    # Logical to boolean
    matlab_logical = mat_data['is_valid']
    python_bool = matlab_logical.astype(bool)

Numerical Compatibility
-----------------------

Ensuring Numerical Agreement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Set compatible random state
    def set_compatible_random_state(seed=42):
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # For MATLAB compatibility
        if 'matlab.engine' in sys.modules:
            eng = matlab.engine.start_matlab()
            eng.rng(seed)
            return eng
        return None
    
    # Use same numerical tolerance
    MATLAB_EPS = 2.220446049250313e-16  # MATLAB's eps
    
    # Configure algorithms for compatibility
    ls = LocalSearch(
        n_clusters=5,
        tol=MATLAB_EPS * 100,  # Similar to MATLAB's default
        random_state=42
    )

Handling Numerical Differences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def compare_with_matlab(python_result, matlab_file, tolerance=1e-10):
        """Compare Python results with MATLAB output."""
        matlab_data = loadmat(matlab_file)
        matlab_result = matlab_data['result'].squeeze()
        
        # Check shape
        assert python_result.shape == matlab_result.shape, \
            f"Shape mismatch: {python_result.shape} vs {matlab_result.shape}"
        
        # Check values
        max_diff = np.max(np.abs(python_result - matlab_result))
        mean_diff = np.mean(np.abs(python_result - matlab_result))
        
        print(f"Maximum difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        
        if max_diff > tolerance:
            # Find where differences occur
            diff_mask = np.abs(python_result - matlab_result) > tolerance
            diff_indices = np.where(diff_mask)
            print(f"Differences at indices: {diff_indices}")
        
        return max_diff <= tolerance

File I/O Compatibility
----------------------

Reading MATLAB Files
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.io.matlab_io import read_matlab_submarkets
    
    # Read complete MATLAB workspace
    data = read_matlab_submarkets('matlab_results.mat')
    
    # Access components
    X = data['features']
    S = data['substitution_matrix']
    clusters = data['clusters']
    params = data['parameters']
    
    # Handle MATLAB quirks
    if clusters.ndim > 1:
        clusters = clusters.squeeze()  # Remove singleton dimensions
    if clusters.min() == 1:
        clusters = clusters - 1  # Convert to 0-based indexing

Writing MATLAB Files
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.io.matlab_io import write_matlab_submarkets
    
    # Prepare data for MATLAB
    matlab_data = {
        'features': X,
        'substitution_matrix': S,
        'clusters': clusters + 1,  # Convert to 1-based indexing
        'metrics': struct_from_dict(metrics),
        'parameters': struct_from_dict(params),
        'product_names': np.array(product_names, dtype=object)
    }
    
    write_matlab_submarkets('python_results.mat', matlab_data)

Algorithm Compatibility
-----------------------

Local Search Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Python implementation matches MATLAB's algorithm:

.. code-block:: python

    # MATLAB algorithm pseudo-code:
    # for iter = 1:maxiter
    #     for i = 1:n
    #         best_cluster = find_best_cluster(i, clusters, S)
    #         clusters(i) = best_cluster
    #     end
    #     if no_change, break
    # end
    
    # Exact Python equivalent:
    class MatlabCompatibleLocalSearch(LocalSearch):
        def _update_step(self, S, clusters):
            """MATLAB-compatible update step."""
            n = len(clusters)
            changed = False
            
            # MATLAB-style iteration order
            for i in range(n):
                old_cluster = clusters[i]
                
                # Compute costs for all clusters
                costs = np.zeros(self.n_clusters)
                for k in range(self.n_clusters):
                    mask = clusters == k
                    if k == old_cluster:
                        mask[i] = False
                    else:
                        mask[i] = True
                    
                    costs[k] = S[i, mask].sum()
                
                # Find best cluster
                best_cluster = np.argmin(costs)
                
                if best_cluster != old_cluster:
                    clusters[i] = best_cluster
                    changed = True
            
            return clusters, changed

Evaluation Metrics
~~~~~~~~~~~~~~~~~~

Ensure same metric definitions:

.. code-block:: python

    # MATLAB silhouette calculation
    def matlab_compatible_silhouette(S, clusters):
        """Calculate silhouette score using MATLAB's method."""
        n = len(clusters)
        silhouettes = np.zeros(n)
        
        for i in range(n):
            # Same cluster distances
            same_cluster = clusters == clusters[i]
            same_cluster[i] = False
            
            if np.sum(same_cluster) > 0:
                a_i = np.mean(S[i, same_cluster])
            else:
                a_i = 0
            
            # Other cluster distances
            b_i = np.inf
            for k in range(int(clusters.max()) + 1):
                if k != clusters[i]:
                    other_cluster = clusters == k
                    if np.sum(other_cluster) > 0:
                        mean_dist = np.mean(S[i, other_cluster])
                        b_i = min(b_i, mean_dist)
            
            # Silhouette
            if max(a_i, b_i) > 0:
                silhouettes[i] = (b_i - a_i) / max(a_i, b_i)
            else:
                silhouettes[i] = 0
        
        return np.mean(silhouettes)

Visualization Compatibility
---------------------------

Reproducing MATLAB Plots
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    
    def matlab_style_plot(S, clusters):
        """Create MATLAB-style visualization."""
        # Set MATLAB-like style
        plt.style.use('classic')
        
        # MATLAB's default figure size
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        
        # Sort by clusters (MATLAB style)
        idx = np.argsort(clusters)
        S_sorted = S[idx][:, idx]
        
        # MATLAB's default colormap
        im = ax.imshow(S_sorted, cmap='jet', aspect='equal')
        
        # MATLAB-style colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Substitution Distance')
        
        # MATLAB-style labels
        ax.set_xlabel('Product Index')
        ax.set_ylabel('Product Index')
        ax.set_title('Substitution Matrix (Sorted by Clusters)')
        
        # MATLAB grid
        ax.grid(True, alpha=0.3)
        
        return fig, ax

Color Mapping
~~~~~~~~~~~~~

.. code-block:: python

    # MATLAB color order
    matlab_colors = [
        [0, 0.4470, 0.7410],      # Blue
        [0.8500, 0.3250, 0.0980], # Orange
        [0.9290, 0.6940, 0.1250], # Yellow
        [0.4940, 0.1840, 0.5560], # Purple
        [0.4660, 0.6740, 0.1880], # Green
        [0.3010, 0.7450, 0.9330], # Cyan
        [0.6350, 0.0780, 0.1840], # Red
    ]
    
    def get_matlab_colors(n):
        """Get MATLAB-style colors for n clusters."""
        colors = matlab_colors * (n // 7 + 1)
        return colors[:n]

Testing Compatibility
---------------------

Compatibility Test Suite
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.tests.matlab_compat import run_compatibility_tests
    
    # Run all compatibility tests
    results = run_compatibility_tests(
        matlab_data_dir='tests/matlab_data/',
        tolerance=1e-10
    )
    
    # Individual tests
    def test_substitution_matrix_compat():
        X = load_test_data()
        
        # Python version
        S_python = create_substitution_matrix(X)
        
        # Load MATLAB version
        S_matlab = loadmat('test_S.mat')['S']
        
        # Compare
        assert np.allclose(S_python, S_matlab, rtol=1e-10)

Debugging Differences
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def debug_matlab_differences(python_result, matlab_result):
        """Detailed debugging of differences."""
        
        # Basic statistics
        print("Python - Mean:", np.mean(python_result))
        print("MATLAB - Mean:", np.mean(matlab_result))
        print("Python - Std:", np.std(python_result))
        print("MATLAB - Std:", np.std(matlab_result))
        
        # Difference analysis
        diff = python_result - matlab_result
        print("\nDifference statistics:")
        print("Max absolute:", np.max(np.abs(diff)))
        print("Mean absolute:", np.mean(np.abs(diff)))
        print("Relative error:", np.mean(np.abs(diff) / (np.abs(matlab_result) + 1e-10)))
        
        # Find largest differences
        largest_diff_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(f"\nLargest difference at {largest_diff_idx}:")
        print(f"Python: {python_result[largest_diff_idx]}")
        print(f"MATLAB: {matlab_result[largest_diff_idx]}")
        
        # Plot differences
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.hist(diff.flatten(), bins=50)
        plt.title('Difference Distribution')
        
        plt.subplot(132)
        plt.scatter(matlab_result.flatten(), python_result.flatten(), alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('MATLAB')
        plt.ylabel('Python')
        plt.title('Value Comparison')
        
        plt.subplot(133)
        plt.imshow(np.abs(diff), cmap='hot')
        plt.colorbar()
        plt.title('Absolute Differences')
        
        plt.tight_layout()
        plt.show()

Migration Checklist
-------------------

Before Migration
~~~~~~~~~~~~~~~~

1. ✓ Export all MATLAB data to .mat files
2. ✓ Document all custom MATLAB functions
3. ✓ Note any toolbox dependencies
4. ✓ Save random seeds and parameters

During Migration
~~~~~~~~~~~~~~~~

1. ✓ Use compatibility layer initially
2. ✓ Validate each step against MATLAB
3. ✓ Document any differences found
4. ✓ Update tests incrementally

After Migration
~~~~~~~~~~~~~~~

1. ✓ Run full compatibility test suite
2. ✓ Compare performance metrics
3. ✓ Update documentation
4. ✓ Train team on Python version