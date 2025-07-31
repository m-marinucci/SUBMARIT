"""Performance benchmarks for SUBMARIT algorithms."""

import time
from typing import Dict, List, Tuple

import numpy as np
import pytest
from memory_profiler import memory_usage
from numpy.typing import NDArray

from submarit.algorithms.local_search import (
    KSMLocalSearch,
    KSMLocalSearch2,
    KSMLocalSearchConstrained,
    KSMLocalSearchConstrained2,
)
from submarit.core.substitution_matrix import SubstitutionMatrix
from submarit.evaluation.cluster_evaluator import ClusterEvaluator
from submarit.evaluation.gap_statistic import GAPStatistic
from submarit.validation.empirical_distributions import EmpiricalDistribution
from submarit.validation.kfold import KFoldValidator
from submarit.validation.multiple_runs import MultipleRunsValidator


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self):
        self.timing_results = {}
        self.memory_results = {}
        self.scalability_results = {}
        self.algorithm_comparison = {}
    
    def add_timing(self, name: str, size: int, time_seconds: float):
        """Add timing result."""
        if name not in self.timing_results:
            self.timing_results[name] = {}
        self.timing_results[name][size] = time_seconds
    
    def add_memory(self, name: str, size: int, memory_mb: float):
        """Add memory usage result."""
        if name not in self.memory_results:
            self.memory_results[name] = {}
        self.memory_results[name][size] = memory_mb
    
    def add_scalability(self, name: str, metric: str, value: float):
        """Add scalability metric."""
        if name not in self.scalability_results:
            self.scalability_results[name] = {}
        self.scalability_results[name][metric] = value
    
    def summary(self) -> str:
        """Generate summary report."""
        lines = ["Performance Benchmark Summary", "=" * 40]
        
        # Timing results
        if self.timing_results:
            lines.append("\nTiming Results (seconds):")
            for name, results in self.timing_results.items():
                lines.append(f"\n{name}:")
                for size, time in sorted(results.items()):
                    lines.append(f"  Size {size}: {time:.3f}s")
        
        # Memory results
        if self.memory_results:
            lines.append("\nMemory Usage (MB):")
            for name, results in self.memory_results.items():
                lines.append(f"\n{name}:")
                for size, memory in sorted(results.items()):
                    lines.append(f"  Size {size}: {memory:.1f} MB")
        
        # Scalability
        if self.scalability_results:
            lines.append("\nScalability Metrics:")
            for name, metrics in self.scalability_results.items():
                lines.append(f"\n{name}:")
                for metric, value in metrics.items():
                    lines.append(f"  {metric}: {value:.3f}")
        
        return "\n".join(lines)


@pytest.fixture(scope="module")
def benchmark_results():
    """Shared benchmark results container."""
    return BenchmarkResults()


class TestAlgorithmPerformance:
    """Benchmark algorithm performance."""
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("size", [10, 50, 100, 200, 500])
    def test_local_search_timing(self, size, benchmark_results):
        """Benchmark LocalSearch algorithm timing."""
        # Generate test matrix
        np.random.seed(42)
        matrix = np.random.rand(size, size)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        # Time the algorithm
        n_clusters = min(5, size // 10)
        clusterer = KSMLocalSearch(n_clusters=n_clusters, random_state=42)
        
        start_time = time.time()
        clusterer.fit(matrix)
        end_time = time.time()
        
        elapsed = end_time - start_time
        benchmark_results.add_timing("LocalSearch", size, elapsed)
        
        # Basic performance check
        if size <= 100:
            assert elapsed < 1.0, f"LocalSearch too slow for size {size}: {elapsed}s"
        elif size <= 500:
            assert elapsed < 10.0, f"LocalSearch too slow for size {size}: {elapsed}s"
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("size", [10, 50, 100, 200])
    def test_local_search2_timing(self, size, benchmark_results):
        """Benchmark LocalSearch2 algorithm timing."""
        # Generate test matrix
        np.random.seed(42)
        matrix = np.random.rand(size, size)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        # Time the algorithm
        n_clusters = min(5, size // 10)
        clusterer = KSMLocalSearch2(n_clusters=n_clusters, random_state=42)
        
        start_time = time.time()
        clusterer.fit(matrix)
        end_time = time.time()
        
        elapsed = end_time - start_time
        benchmark_results.add_timing("LocalSearch2", size, elapsed)
        
        # LocalSearch2 might be slower due to log-likelihood computation
        if size <= 100:
            assert elapsed < 2.0, f"LocalSearch2 too slow for size {size}: {elapsed}s"
    
    @pytest.mark.benchmark
    def test_algorithm_comparison(self, benchmark_results):
        """Compare performance of different algorithms."""
        sizes = [20, 50, 100]
        n_runs = 5
        
        for size in sizes:
            # Generate matrix
            np.random.seed(42)
            matrix = np.random.rand(size, size)
            matrix = (matrix + matrix.T) / 2
            np.fill_diagonal(matrix, 0)
            row_sums = matrix.sum(axis=1)
            matrix = matrix / row_sums[:, np.newaxis]
            
            n_clusters = 3
            
            # Test each algorithm
            algorithms = [
                ("LocalSearch", KSMLocalSearch),
                ("LocalSearch2", KSMLocalSearch2),
            ]
            
            for name, algo_class in algorithms:
                times = []
                for _ in range(n_runs):
                    clusterer = algo_class(n_clusters=n_clusters, random_state=42)
                    start = time.time()
                    clusterer.fit(matrix)
                    end = time.time()
                    times.append(end - start)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                benchmark_results.add_timing(f"{name}_avg", size, avg_time)
                benchmark_results.add_timing(f"{name}_std", size, std_time)
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_memory_usage(self, benchmark_results):
        """Benchmark memory usage of algorithms."""
        sizes = [50, 100, 200]
        
        for size in sizes:
            # Generate matrix
            np.random.seed(42)
            matrix = np.random.rand(size, size)
            matrix = (matrix + matrix.T) / 2
            np.fill_diagonal(matrix, 0)
            row_sums = matrix.sum(axis=1)
            matrix = matrix / row_sums[:, np.newaxis]
            
            n_clusters = 3
            
            # Measure memory for LocalSearch
            def run_local_search():
                clusterer = KSMLocalSearch(n_clusters=n_clusters, random_state=42)
                clusterer.fit(matrix)
                return clusterer
            
            mem_usage = memory_usage(run_local_search, interval=0.1, timeout=30)
            max_memory = max(mem_usage) - min(mem_usage)  # Delta
            benchmark_results.add_memory("LocalSearch", size, max_memory)
            
            # Memory should scale roughly with matrix size
            expected_memory = (size * size * 8) / (1024 * 1024)  # Rough estimate in MB
            assert max_memory < expected_memory * 10, f"Excessive memory usage: {max_memory} MB"
    
    @pytest.mark.benchmark
    def test_convergence_speed(self, benchmark_results):
        """Test convergence speed of algorithms."""
        size = 100
        n_clusters = 5
        
        # Generate matrix
        np.random.seed(42)
        matrix = np.random.rand(size, size)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        # Test different max_iter settings
        max_iters = [10, 20, 50, 100]
        
        for max_iter in max_iters:
            clusterer = KSMLocalSearch(
                n_clusters=n_clusters,
                max_iter=max_iter,
                random_state=42
            )
            
            start = time.time()
            clusterer.fit(matrix)
            end = time.time()
            
            result = clusterer.get_result()
            converged = result.Iter < max_iter
            
            benchmark_results.add_scalability(
                f"LocalSearch_maxiter_{max_iter}",
                "converged",
                1.0 if converged else 0.0
            )
            benchmark_results.add_scalability(
                f"LocalSearch_maxiter_{max_iter}",
                "iterations",
                result.Iter
            )
            benchmark_results.add_scalability(
                f"LocalSearch_maxiter_{max_iter}",
                "time",
                end - start
            )


class TestComponentPerformance:
    """Benchmark individual component performance."""
    
    @pytest.mark.benchmark
    def test_substitution_matrix_creation_performance(self, benchmark_results):
        """Benchmark substitution matrix creation."""
        sizes = [(100, 20), (500, 50), (1000, 100)]  # (n_consumers, n_products)
        
        for n_consumers, n_products in sizes:
            # Generate consumer data
            np.random.seed(42)
            data = np.random.randint(0, 2, size=(n_consumers, n_products))
            
            # Time matrix creation
            start = time.time()
            sub_matrix = SubstitutionMatrix()
            sub_matrix.create_from_consumer_product_data(data, normalize=True)
            end = time.time()
            
            elapsed = end - start
            benchmark_results.add_timing(
                "SubstitutionMatrix_creation",
                n_consumers * n_products,
                elapsed
            )
            
            # Should be fast even for large data
            assert elapsed < 1.0, f"Matrix creation too slow: {elapsed}s"
    
    @pytest.mark.benchmark
    def test_evaluation_performance(self, benchmark_results):
        """Benchmark clustering evaluation performance."""
        sizes = [50, 100, 200]
        
        for size in sizes:
            # Generate matrix and clustering
            np.random.seed(42)
            matrix = np.random.rand(size, size)
            matrix = (matrix + matrix.T) / 2
            np.fill_diagonal(matrix, 0)
            labels = np.random.randint(0, 5, size)
            
            # Time evaluation
            evaluator = ClusterEvaluator()
            start = time.time()
            metrics = evaluator.evaluate(matrix, labels)
            end = time.time()
            
            elapsed = end - start
            benchmark_results.add_timing("ClusterEvaluator", size, elapsed)
            
            # Evaluation should be fast
            assert elapsed < 0.5, f"Evaluation too slow for size {size}: {elapsed}s"
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_gap_statistic_performance(self, benchmark_results):
        """Benchmark GAP statistic performance."""
        size = 50
        
        # Generate matrix
        np.random.seed(42)
        matrix = np.random.rand(size, size)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        # Time GAP statistic
        gap = GAPStatistic(n_refs=10, random_state=42)
        start = time.time()
        results = gap.compute(
            matrix,
            max_clusters=5,
            clusterer_class=KSMLocalSearch
        )
        end = time.time()
        
        elapsed = end - start
        benchmark_results.add_timing("GAPStatistic", size, elapsed)
        
        # GAP statistic is expensive but should complete in reasonable time
        assert elapsed < 30.0, f"GAP statistic too slow: {elapsed}s"
    
    @pytest.mark.benchmark
    def test_empirical_distribution_performance(self, benchmark_results):
        """Benchmark empirical distribution generation."""
        size = 50
        n_iterations = 100
        
        # Generate matrix
        np.random.seed(42)
        matrix = np.random.rand(size, size)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        # Create clusterer
        clusterer = KSMLocalSearch(n_clusters=3, random_state=42)
        
        # Time empirical distribution
        emp_dist = EmpiricalDistribution(n_iterations=n_iterations, random_state=42)
        start = time.time()
        results = emp_dist.generate(matrix, clusterer, statistic="diff")
        end = time.time()
        
        elapsed = end - start
        benchmark_results.add_timing("EmpiricalDistribution", n_iterations, elapsed)
        
        # Should complete in reasonable time
        assert elapsed < 20.0, f"Empirical distribution too slow: {elapsed}s"


class TestScalabilityBenchmarks:
    """Test scalability characteristics."""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_linear_scalability(self, benchmark_results):
        """Test if algorithm scales linearly with problem size."""
        sizes = [20, 40, 80, 160]
        times = []
        
        for size in sizes:
            # Generate matrix
            np.random.seed(42)
            matrix = np.random.rand(size, size)
            matrix = (matrix + matrix.T) / 2
            np.fill_diagonal(matrix, 0)
            row_sums = matrix.sum(axis=1)
            matrix = matrix / row_sums[:, np.newaxis]
            
            # Time algorithm
            clusterer = KSMLocalSearch(n_clusters=3, random_state=42)
            start = time.time()
            clusterer.fit(matrix)
            end = time.time()
            
            times.append(end - start)
        
        # Check if scaling is roughly linear or better
        # Time complexity should be O(n²) or better
        time_ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        
        # Each doubling of size should increase time by at most 4x (quadratic)
        for i, ratio in enumerate(time_ratios):
            benchmark_results.add_scalability(
                "LocalSearch_scaling",
                f"ratio_{sizes[i]}_{sizes[i+1]}",
                ratio
            )
            assert ratio < 5.0, f"Poor scaling: {ratio}x increase for 2x size increase"
    
    @pytest.mark.benchmark
    def test_cluster_number_impact(self, benchmark_results):
        """Test impact of number of clusters on performance."""
        size = 100
        cluster_numbers = [2, 5, 10, 20]
        
        # Generate matrix
        np.random.seed(42)
        matrix = np.random.rand(size, size)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        for n_clusters in cluster_numbers:
            clusterer = KSMLocalSearch(n_clusters=n_clusters, random_state=42)
            start = time.time()
            clusterer.fit(matrix)
            end = time.time()
            
            elapsed = end - start
            benchmark_results.add_timing("LocalSearch_by_k", n_clusters, elapsed)
            
            # More clusters might take more time, but should be reasonable
            assert elapsed < 5.0, f"Too slow with {n_clusters} clusters: {elapsed}s"
    
    @pytest.mark.benchmark
    def test_sparsity_impact(self, benchmark_results):
        """Test impact of matrix sparsity on performance."""
        size = 100
        sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # Fraction of non-zero entries
        
        for sparsity in sparsity_levels:
            # Generate sparse matrix
            np.random.seed(42)
            matrix = np.random.rand(size, size)
            matrix = (matrix + matrix.T) / 2
            
            # Apply sparsity
            mask = np.random.rand(size, size) < sparsity
            matrix = matrix * mask
            
            np.fill_diagonal(matrix, 0)
            row_sums = matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            matrix = matrix / row_sums[:, np.newaxis]
            
            # Time algorithm
            clusterer = KSMLocalSearch(n_clusters=3, random_state=42)
            start = time.time()
            clusterer.fit(matrix)
            end = time.time()
            
            elapsed = end - start
            benchmark_results.add_timing("LocalSearch_sparsity", sparsity, elapsed)


@pytest.fixture(scope="module", autouse=True)
def print_benchmark_summary(benchmark_results):
    """Print benchmark summary at the end of the module."""
    yield
    print("\n" + benchmark_results.summary())