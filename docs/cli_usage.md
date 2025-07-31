# SUBMARIT CLI Usage Guide

The SUBMARIT command-line interface provides comprehensive functionality for submarket identification and clustering analysis.

## Installation

```bash
pip install submarit
```

## Basic Usage

### Clustering

Run basic clustering on a substitution matrix:

```bash
submarit cluster input_data.npz output_results.json -k 3 -n 100 --verbose
```

Options:
- `-k, --n-clusters`: Number of clusters (default: 3)
- `-m, --method`: Algorithm version - 'v1' (PHat-P) or 'v2' (log-likelihood) (default: v1)
- `-n, --n-runs`: Number of random initializations (default: 10)
- `-s, --seed`: Random seed for reproducibility
- `--min-items`: Minimum items per cluster (default: 1)
- `--max-iter`: Maximum iterations per run (default: 100)
- `-f, --format`: Output format: json, yaml, npz, csv, mat (default: json)
- `--save-all`: Save all runs, not just the best
- `-v, --verbose`: Verbose output
- `--progress`: Show progress bar

### Top-K Analysis

Analyze stability of top-k clustering solutions:

```bash
submarit topk input_data.npz output_topk.json -k 3 -n 100 --top-k 10 --n-random 10000 -j 4
```

Options:
- `-k, --n-clusters`: Number of clusters
- `-n, --n-runs`: Number of clustering runs
- `--top-k`: Number of top solutions to analyze (default: 10)
- `--n-random`: Number of random samples for empirical distribution (default: 10000)
- `-m, --method`: Algorithm version (v1 or v2)
- `--min-items`: Minimum items per cluster
- `-j, --n-jobs`: Number of parallel jobs (-1 for all cores)
- `--progress`: Show progress bars

### Selecting Optimal K

Find the optimal number of clusters:

```bash
submarit select-k input_data.npz -k 2-10 -m all --progress
```

Options:
- `-k, --k-range`: Range of k values to test (e.g., "2-10")
- `-m, --method`: Method - 'gap', 'stability', or 'all' (default: gap)
- `--n-runs`: Runs per k value (default: 10)
- `--n-refs`: Number of reference datasets for GAP statistic (default: 10)
- `--algorithm`: Clustering algorithm (v1 or v2)
- `--progress`: Show progress bar

### Stability Analysis

Evaluate clustering stability:

```bash
submarit stability input_data.npz -k 3 -n 100 --verbose
```

Options:
- `-k, --n-clusters`: Number of clusters (required)
- `-n, --n-runs`: Number of runs for stability analysis (default: 100)
- `-m, --method`: Algorithm version (v1 or v2)
- `--min-items`: Minimum items per cluster
- `-s, --seed`: Random seed

### Configuration File Execution

Run multiple experiments using a configuration file:

```bash
submarit run config.yaml -o results/ --verbose
```

Options:
- `-o, --output-dir`: Output directory for results (default: ./results)
- `-v, --verbose`: Verbose output
- `--dry-run`: Show what would be run without executing

Example configuration file (YAML):

```yaml
input_file: data/switching_matrix.npz
experiments:
  - name: baseline_k3
    n_clusters: 3
    method: v1
    n_runs: 100
    min_items: 2
    seed: 42
    
  - name: baseline_k4
    n_clusters: 4
    method: v1
    n_runs: 100
    min_items: 2
    seed: 42
    
  - name: topk_analysis_k3
    n_clusters: 3
    method: v1
    n_runs: 200
    top_k: 10
    n_random: 10000
    n_jobs: 4
    
  - name: likelihood_k3
    n_clusters: 3
    method: v2
    n_runs: 100
    min_items: 2
```

Example configuration file (JSON):

```json
{
  "input_file": "data/switching_matrix.npz",
  "experiments": [
    {
      "name": "baseline_k3",
      "n_clusters": 3,
      "method": "v1",
      "n_runs": 100,
      "min_items": 2,
      "seed": 42
    },
    {
      "name": "topk_analysis_k3",
      "n_clusters": 3,
      "method": "v1",
      "n_runs": 200,
      "top_k": 10,
      "n_random": 10000,
      "n_jobs": 4
    }
  ]
}
```

## Advanced Examples

### Parallel Processing

Use multiple cores for faster computation:

```bash
# Use 4 cores
submarit topk data.npz results.json -k 3 -n 1000 --top-k 20 -j 4

# Use all available cores
submarit topk data.npz results.json -k 3 -n 1000 --top-k 20 -j -1
```

### Batch Processing

Process multiple datasets:

```bash
for file in data/*.npz; do
    output="results/$(basename $file .npz)_k3.json"
    submarit cluster "$file" "$output" -k 3 -n 100 --progress
done
```

### Pipeline Example

Complete analysis pipeline:

```bash
# 1. Find optimal number of clusters
submarit select-k data.npz -k 2-10 -m all --progress

# 2. Run clustering with optimal k (e.g., k=4)
submarit cluster data.npz results/clustering_k4.json -k 4 -n 100 --save-all

# 3. Analyze top solutions
submarit topk data.npz results/topk_k4.json -k 4 -n 500 --top-k 20 -j 4

# 4. Check stability
submarit stability data.npz -k 4 -n 100 --verbose
```

### Output Formats

Save results in different formats:

```bash
# JSON (default)
submarit cluster data.npz results.json -k 3

# YAML
submarit cluster data.npz results.yaml -k 3 -f yaml

# NumPy archive
submarit cluster data.npz results.npz -k 3 -f npz

# CSV (cluster assignments only)
submarit cluster data.npz results.csv -k 3 -f csv

# MATLAB format
submarit cluster data.npz results.mat -k 3 -f mat
```

## Understanding Output

### Clustering Output

The clustering command produces output with:
- `assignments`: Cluster assignments for each item (1-based)
- `n_clusters`: Number of clusters
- `n_items`: Number of items clustered
- `cluster_counts`: Number of items in each cluster
- `log_likelihood`: Log-likelihood of the solution
- `diff`: PHat-P difference value
- `z_value`: Z-statistic for the clustering
- `iterations`: Number of iterations until convergence
- `parameters`: Input parameters used

### Top-K Analysis Output

The top-k command produces:
- `avg_rand`: Average Rand index among top-k solutions
- `avg_adj_rand`: Average adjusted Rand index
- `rand_p_value`: P-value for Rand index
- `adj_rand_p_value`: P-value for adjusted Rand index
- `stability`: Overall and configuration stability metrics
- `best_solutions`: Details of the top-k solutions
- `confidence_intervals`: CI values at various levels

### Stability Analysis Output

The stability command shows:
- `overall_stability`: Overall clustering stability (0-1, higher is better)
- `objective_mean`: Mean objective value across runs
- `objective_std`: Standard deviation of objective values
- `item_stability`: Stability score for each item (with -v flag)

## Performance Tips

1. **Use parallel processing**: Add `-j -1` to use all cores
2. **Start with fewer runs**: Test with `-n 10` before scaling up
3. **Enable progress bars**: Use `--progress` for long-running tasks
4. **Save intermediate results**: Use `--save-all` to keep all runs
5. **Set random seeds**: Use `-s` for reproducible results

## Troubleshooting

### Common Issues

1. **"Not enough items for clustering"**
   - Ensure your data has enough items for the specified number of clusters and minimum items constraint
   - Try reducing `--min-items` or `-k`

2. **"All clustering runs failed"**
   - Check data format and validity
   - Try different algorithm (`-m v2` instead of `v1`)
   - Reduce number of clusters

3. **Memory errors with large datasets**
   - Reduce `--n-random` for top-k analysis
   - Use fewer parallel jobs (`-j 2` instead of `-j -1`)
   - Process in batches using configuration files

4. **Slow performance**
   - Enable parallel processing with `-j`
   - Reduce number of runs `-n`
   - Use `--progress` to monitor progress

### Debug Mode

For detailed debugging information:

```bash
# Maximum verbosity
submarit cluster data.npz output.json -k 3 -n 10 --verbose --progress

# Dry run to check configuration
submarit run config.yaml --dry-run --verbose
```

## Citation

If you use SUBMARIT in your research, please cite:

```
@software{submarit,
  title = {SUBMARIT: SUBMARket Identification and Testing},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/m-marinucci/SUBMARIT}
}
```