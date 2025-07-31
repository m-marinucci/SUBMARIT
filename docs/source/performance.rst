Performance Tuning Guide
========================

This guide helps you optimize SUBMARIT for speed and memory efficiency with large datasets.

Performance Considerations
--------------------------

Dataset Size Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Recommended Approaches by Dataset Size
   :widths: 20 20 30 30
   :header-rows: 1

   * - Products
     - Memory Required
     - Recommended Approach
     - Expected Time
   * - < 1,000
     - < 1 GB
     - Standard dense matrix
     - < 1 minute
   * - 1,000 - 10,000
     - 1-8 GB
     - Dense matrix with optimization
     - 1-10 minutes
   * - 10,000 - 100,000
     - 8-80 GB
     - Sparse matrix, mini-batch
     - 10-60 minutes
   * - > 100,000
     - > 80 GB
     - Distributed, approximate methods
     - > 1 hour

Memory Optimization
-------------------

Sparse Matrices
~~~~~~~~~~~~~~~

For datasets with many zero or near-zero substitution values:

.. code-block:: python

    from scipy.sparse import csr_matrix
    from submarit.core import create_sparse_substitution_matrix
    
    # Create sparse substitution matrix
    S_sparse = create_sparse_substitution_matrix(
        X,
        threshold=0.1,  # Keep only top 10% of connections
        metric='cosine',
        format='csr'  # Compressed sparse row format
    )
    
    print(f"Memory usage: {S_sparse.data.nbytes / 1e6:.2f} MB")
    print(f"Sparsity: {1 - S_sparse.nnz / (S_sparse.shape[0]**2):.2%}")

Memory-Mapped Arrays
~~~~~~~~~~~~~~~~~~~~

For datasets too large for memory:

.. code-block:: python

    import numpy as np
    from submarit.core import create_mmap_substitution_matrix
    
    # Create memory-mapped substitution matrix
    S_mmap = create_mmap_substitution_matrix(
        X,
        output_file='substitution_matrix.dat',
        dtype=np.float32,  # Use float32 to save space
        chunks=1000  # Process in chunks
    )

Chunked Processing
~~~~~~~~~~~~~~~~~~

Process large matrices in chunks:

.. code-block:: python

    from submarit.utils import chunked_substitution_matrix
    
    def process_in_chunks(X, chunk_size=5000):
        n = len(X)
        S = np.zeros((n, n), dtype=np.float32)
        
        for i in range(0, n, chunk_size):
            for j in range(0, n, chunk_size):
                chunk_i = X[i:i+chunk_size]
                chunk_j = X[j:j+chunk_size]
                S[i:i+chunk_size, j:j+chunk_size] = compute_chunk(chunk_i, chunk_j)
        
        return S

Speed Optimization
------------------

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Leverage multiple CPU cores:

.. code-block:: python

    from submarit.algorithms import LocalSearch
    from joblib import Parallel, delayed
    import multiprocessing
    
    # Use all available cores
    n_cores = multiprocessing.cpu_count()
    
    # Parallel clustering with different random seeds
    ls = LocalSearch(
        n_clusters=10,
        n_restarts=20,
        n_jobs=n_cores  # Parallel restarts
    )
    
    # Parallel substitution matrix computation
    from submarit.core import parallel_substitution_matrix
    
    S = parallel_substitution_matrix(X, n_jobs=n_cores, batch_size=100)

Vectorization
~~~~~~~~~~~~~

Use NumPy's vectorized operations:

.. code-block:: python

    # Slow: Python loops
    def slow_distance(X):
        n = len(X)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = np.linalg.norm(X[i] - X[j])
        return D
    
    # Fast: Vectorized
    def fast_distance(X):
        # Use broadcasting
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=2)
    
    # Faster: Use scipy
    from scipy.spatial.distance import cdist
    D = cdist(X, X, metric='euclidean')

Numba JIT Compilation
~~~~~~~~~~~~~~~~~~~~~

Speed up custom functions:

.. code-block:: python

    from numba import jit, prange
    
    @jit(nopython=True, parallel=True)
    def fast_local_search_update(S, clusters, n_clusters):
        n = len(clusters)
        changed = True
        
        while changed:
            changed = False
            for i in prange(n):  # Parallel loop
                best_cluster = clusters[i]
                best_cost = compute_cost(S, i, clusters, best_cluster)
                
                for k in range(n_clusters):
                    if k != clusters[i]:
                        cost = compute_cost(S, i, clusters, k)
                        if cost < best_cost:
                            best_cost = cost
                            best_cluster = k
                
                if best_cluster != clusters[i]:
                    clusters[i] = best_cluster
                    changed = True
        
        return clusters

Algorithm-Specific Optimizations
--------------------------------

Local Search Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.algorithms import OptimizedLocalSearch
    
    # Use optimized implementation
    ols = OptimizedLocalSearch(
        n_clusters=10,
        max_iter=100,
        tol=1e-4,
        early_stopping=True,  # Stop when improvement is minimal
        cache_distances=True,  # Cache frequently accessed distances
        use_triangle_inequality=True  # Skip unnecessary distance calculations
    )
    
    # Mini-batch version for large datasets
    from submarit.algorithms import MiniBatchLocalSearch
    
    mbls = MiniBatchLocalSearch(
        n_clusters=10,
        batch_size=1000,
        n_init=3,
        max_no_improvement=10
    )

Approximate Methods
~~~~~~~~~~~~~~~~~~~

For very large datasets, use approximations:

.. code-block:: python

    from submarit.algorithms import ApproximateLocalSearch
    
    # Use locality-sensitive hashing
    als = ApproximateLocalSearch(
        n_clusters=10,
        approximation='lsh',
        n_hash_functions=10,
        accuracy=0.9  # 90% accuracy vs exact method
    )
    
    # Use random sampling
    als_sample = ApproximateLocalSearch(
        n_clusters=10,
        approximation='sample',
        sample_size=10000,  # Work with subset
        n_iterations=5  # Refine with full data
    )

Profiling and Benchmarking
--------------------------

Profile Your Code
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import cProfile
    import pstats
    from submarit.utils import Timer
    
    # Simple timing
    with Timer() as t:
        S = create_substitution_matrix(X)
    print(f"Matrix creation took {t.elapsed:.2f} seconds")
    
    # Detailed profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    clusters = LocalSearch(n_clusters=5).fit_predict(S)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 time-consuming functions

Memory Profiling
~~~~~~~~~~~~~~~~

.. code-block:: python

    from memory_profiler import profile
    
    @profile
    def memory_intensive_function(X):
        S = create_substitution_matrix(X)
        ls = LocalSearch(n_clusters=10)
        clusters = ls.fit_predict(S)
        return clusters
    
    # Run with: python -m memory_profiler your_script.py

Benchmarking Suite
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.benchmarks import run_benchmark
    
    # Benchmark different configurations
    results = run_benchmark(
        dataset_sizes=[100, 1000, 10000],
        n_clusters_list=[5, 10, 20],
        algorithms=['local_search', 'kmeans', 'hierarchical'],
        metrics=['time', 'memory', 'quality']
    )
    
    # Plot results
    from submarit.benchmarks import plot_benchmark_results
    plot_benchmark_results(results)

Best Practices
--------------

1. **Data Preprocessing**
   
   .. code-block:: python
   
       # Normalize features for faster convergence
       from sklearn.preprocessing import StandardScaler
       X_scaled = StandardScaler().fit_transform(X)
       
       # Remove redundant features
       from sklearn.feature_selection import VarianceThreshold
       selector = VarianceThreshold(threshold=0.01)
       X_reduced = selector.fit_transform(X_scaled)

2. **Caching Results**
   
   .. code-block:: python
   
       import joblib
       
       # Cache substitution matrix
       try:
           S = joblib.load('substitution_matrix.pkl')
       except FileNotFoundError:
           S = create_substitution_matrix(X)
           joblib.dump(S, 'substitution_matrix.pkl')

3. **Progressive Refinement**
   
   .. code-block:: python
   
       # Start with coarse clustering, then refine
       def progressive_clustering(X, final_k=50):
           # Stage 1: Coarse clustering
           coarse_k = 10
           coarse_clusters = LocalSearch(n_clusters=coarse_k).fit_predict(X)
           
           # Stage 2: Refine each coarse cluster
           final_clusters = np.zeros(len(X), dtype=int)
           offset = 0
           
           for i in range(coarse_k):
               mask = coarse_clusters == i
               X_subset = X[mask]
               
               if len(X_subset) > final_k // coarse_k:
                   sub_k = final_k // coarse_k
                   sub_clusters = LocalSearch(n_clusters=sub_k).fit_predict(X_subset)
                   final_clusters[mask] = sub_clusters + offset
                   offset += sub_k
               else:
                   final_clusters[mask] = offset
                   offset += 1
           
           return final_clusters

Hardware Considerations
-----------------------

CPU Optimization
~~~~~~~~~~~~~~~~

- Use Intel MKL for optimized linear algebra: ``conda install mkl``
- Set thread affinity: ``export OMP_NUM_THREADS=8``
- Disable hyperthreading for compute-intensive tasks

GPU Acceleration
~~~~~~~~~~~~~~~~

For extremely large datasets:

.. code-block:: python

    # Using CuPy for GPU arrays
    import cupy as cp
    
    def gpu_distance_matrix(X):
        X_gpu = cp.asarray(X)
        # Compute pairwise distances on GPU
        diff = X_gpu[:, cp.newaxis, :] - X_gpu[cp.newaxis, :, :]
        distances = cp.linalg.norm(diff, axis=2)
        return cp.asnumpy(distances)  # Transfer back to CPU

Distributed Computing
~~~~~~~~~~~~~~~~~~~~~

For cluster computing:

.. code-block:: python

    from dask.distributed import Client
    import dask.array as da
    
    # Setup Dask client
    client = Client('scheduler-address:8786')
    
    # Convert to Dask array
    X_dask = da.from_array(X, chunks=(1000, X.shape[1]))
    
    # Compute in parallel across cluster
    S_dask = compute_distributed_substitution_matrix(X_dask)
    S = S_dask.compute()  # Trigger computation

Cloud Deployment
----------------

AWS Configuration
~~~~~~~~~~~~~~~~~

Deploy SUBMARIT on AWS for large-scale processing:

.. code-block:: python

    # Using AWS Batch
    import boto3
    from submarit.cloud import AWSBatchRunner
    
    runner = AWSBatchRunner(
        job_definition='submarit-clustering',
        job_queue='high-memory-queue',
        vcpus=16,
        memory=64000  # 64GB
    )
    
    # Submit job
    job_id = runner.submit(
        data_s3_path='s3://bucket/data.csv',
        n_clusters=20,
        algorithm='local_search'
    )
    
    # Monitor progress
    status = runner.get_status(job_id)

Google Cloud Platform
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.cloud import GCPDataprocRunner
    
    runner = GCPDataprocRunner(
        cluster_name='submarit-cluster',
        num_workers=10,
        worker_machine_type='n1-highmem-8'
    )
    
    # Run distributed clustering
    results = runner.run_clustering(
        gcs_path='gs://bucket/data.csv',
        n_clusters=50,
        max_iter=1000
    )

Azure ML Pipeline
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from azureml.core import Workspace, Experiment
    from submarit.cloud import AzureMLRunner
    
    # Configure compute
    runner = AzureMLRunner(
        workspace=ws,
        compute_target='gpu-cluster',
        environment='submarit-env'
    )
    
    # Run experiment
    run = runner.submit_experiment(
        data_path='datastore://products/data.csv',
        config={
            'n_clusters': 30,
            'algorithm': 'gpu_local_search',
            'batch_size': 10000
        }
    )

Edge Computing
--------------

For real-time submarket analysis at retail locations:

Lightweight Models
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.edge import EdgeClusterer
    
    # Create lightweight model
    edge_model = EdgeClusterer(
        n_clusters=5,
        max_products=1000,
        memory_limit='512MB',
        cpu_limit=2
    )
    
    # Export for edge deployment
    edge_model.export_onnx('edge_model.onnx')
    edge_model.export_tflite('edge_model.tflite')

Incremental Updates
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Edge device code
    from submarit.edge import IncrementalEdgeClusterer
    
    clusterer = IncrementalEdgeClusterer.load('edge_model.pkl')
    
    # Process new products in real-time
    while True:
        new_products = get_new_products()
        if new_products:
            clusterer.partial_update(new_products)
            
        # Periodic sync with cloud
        if time_to_sync():
            clusterer.sync_with_cloud()

Performance Monitoring
----------------------

Real-time Metrics
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.monitoring import PerformanceMonitor
    
    monitor = PerformanceMonitor(
        metrics=['cpu', 'memory', 'disk', 'network'],
        interval=1.0  # seconds
    )
    
    with monitor:
        clusters = LocalSearch(n_clusters=10).fit_predict(S)
    
    # Get performance report
    report = monitor.get_report()
    print(f"Peak memory: {report['memory_peak_mb']:.2f} MB")
    print(f"CPU time: {report['cpu_time']:.2f} seconds")
    print(f"Wall time: {report['wall_time']:.2f} seconds")

Bottleneck Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.profiling import bottleneck_analysis
    
    # Automatic bottleneck detection
    analysis = bottleneck_analysis(
        function=lambda: LocalSearch(5).fit_predict(S),
        data_size=len(S),
        iterations=10
    )
    
    print("Bottlenecks found:")
    for bottleneck in analysis['bottlenecks']:
        print(f"- {bottleneck['function']}: {bottleneck['percent']:.1f}% of time")
        print(f"  Suggestion: {bottleneck['optimization_hint']}")

Advanced Optimization Techniques
--------------------------------

JIT Compilation Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from numba import jit, cuda
    from submarit.optimization import auto_optimize
    
    # Automatic optimization selection
    @auto_optimize(target=['cpu', 'gpu'])
    def optimized_distance_computation(X):
        # Framework automatically selects best implementation
        pass
    
    # GPU-specific optimization
    @cuda.jit
    def gpu_local_search_kernel(S, clusters, n_clusters):
        # CUDA kernel for GPU execution
        idx = cuda.grid(1)
        if idx < len(clusters):
            # Parallel cluster assignment update
            pass

Memory Mapping Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from submarit.optimization import SmartMemoryManager
    
    # Intelligent memory management
    manager = SmartMemoryManager(
        available_memory='16GB',
        swap_path='/fast_ssd/swap',
        compression='lz4'
    )
    
    # Automatically handles large matrices
    with manager:
        S = create_substitution_matrix(very_large_X)
        clusters = LocalSearch(20).fit_predict(S)

Optimization Decision Tree
--------------------------

Use this decision tree to choose optimization strategies:

.. code-block:: text

    Dataset Size?
    ├── < 1,000 products
    │   └── Use default settings
    ├── 1,000 - 10,000 products
    │   ├── Memory < 8GB?
    │   │   ├── Yes → Use sparse matrices
    │   │   └── No → Use dense matrices with parallel processing
    │   └── Time critical?
    │       ├── Yes → Use approximate methods
    │       └── No → Use exact methods with multiple restarts
    └── > 10,000 products
        ├── Memory < 32GB?
        │   ├── Yes → Use mini-batch or distributed computing
        │   └── No → Use GPU acceleration if available
        └── Real-time requirements?
            ├── Yes → Use edge computing with incremental updates
            └── No → Use cloud computing with batch processing

Performance Benchmarks
----------------------

Latest benchmark results (v2.0):

.. list-table:: Performance Benchmarks
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Dataset Size
     - Algorithm
     - Time (seconds)
     - Memory (GB)
     - Hardware
   * - 1,000
     - Local Search
     - 0.5
     - 0.1
     - 4-core CPU
   * - 10,000
     - Local Search
     - 45
     - 8
     - 8-core CPU
   * - 10,000
     - GPU Local Search
     - 5
     - 4
     - NVIDIA V100
   * - 100,000
     - Mini-batch LS
     - 600
     - 16
     - 16-core CPU
   * - 100,000
     - Distributed LS
     - 120
     - 8/node
     - 10-node cluster
   * - 1,000,000
     - Approximate LS
     - 1800
     - 32
     - 32-core CPU
   * - 1,000,000
     - Cloud GPU
     - 300
     - 16
     - 4x NVIDIA A100