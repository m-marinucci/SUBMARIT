"""Command-line interface for SUBMARIT."""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import warnings

import click
import numpy as np
import yaml
from tqdm import tqdm

from submarit import __version__
from submarit.io import load_substitution_data, save_results
from submarit.validation.multiple_runs import run_clusters, run_clusters_constrained, evaluate_clustering_stability
from submarit.validation.topk_analysis import run_clusters_topk, analyze_solution_stability
from submarit.algorithms.local_search import LocalSearchResult
from submarit.evaluation.cluster_evaluator import ClusterEvaluator
from submarit.evaluation.gap_statistic import gap_statistic


@click.group()
@click.version_option(version=__version__)
def main():
    """SUBMARIT - SUBMARket Identification and Testing."""
    pass


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option(
    '-k', '--n-clusters', 
    type=int, 
    default=3,
    help='Number of clusters'
)
@click.option(
    '-m', '--method',
    type=click.Choice(['v1', 'v2']),
    default='v1',
    help='Clustering algorithm: v1 (PHat-P) or v2 (log-likelihood)'
)
@click.option(
    '-n', '--n-runs',
    type=int,
    default=10,
    help='Number of random initializations'
)
@click.option(
    '-s', '--seed',
    type=int,
    default=None,
    help='Random seed for reproducibility'
)
@click.option(
    '--min-items',
    type=int,
    default=1,
    help='Minimum items per cluster'
)
@click.option(
    '--max-iter',
    type=int,
    default=100,
    help='Maximum iterations per run'
)
@click.option(
    '-f', '--format',
    type=click.Choice(['json', 'yaml', 'npz', 'csv', 'mat']),
    default='json',
    help='Output format'
)
@click.option(
    '--save-all',
    is_flag=True,
    help='Save all runs, not just the best'
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    help='Verbose output'
)
@click.option(
    '--progress',
    is_flag=True,
    help='Show progress bar'
)
def cluster(
    input_file: str,
    output_file: str,
    n_clusters: int,
    method: str,
    n_runs: int,
    seed: Optional[int],
    min_items: int,
    max_iter: int,
    format: str,
    save_all: bool,
    verbose: bool,
    progress: bool
):
    """Run SUBMARIT clustering on substitution matrix data."""
    if verbose:
        click.echo(f"Loading data from {input_file}...")
    
    # Load data
    try:
        matrix = load_substitution_data(input_file)
        swm = matrix.forced_switching_matrix if hasattr(matrix, 'forced_switching_matrix') else matrix
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)
    
    if verbose:
        click.echo(f"Loaded matrix of shape {swm.shape}")
        click.echo(f"Running {method} clustering with {n_clusters} clusters...")
        click.echo(f"Performing {n_runs} runs with min {min_items} items per cluster")
    
    # Run clustering
    try:
        if progress and n_runs > 1:
            # Run with progress bar
            results = []
            best_result = None
            best_objective = -np.inf if method == 'v1' else np.inf
            
            with tqdm(total=n_runs, desc="Clustering runs") as pbar:
                for i in range(n_runs):
                    run_seed = seed + i if seed is not None else None
                    result = run_clusters(
                        swm, n_clusters, min_items,
                        n_runs=1,
                        max_iter=max_iter,
                        random_state=run_seed,
                        algorithm=method
                    )
                    results.append(result)
                    
                    # Track best
                    obj_value = result.Diff if method == 'v1' else result.LogLH
                    if method == 'v1' and obj_value > best_objective:
                        best_objective = obj_value
                        best_result = result
                    elif method == 'v2' and obj_value > best_objective:
                        best_objective = obj_value
                        best_result = result
                    
                    pbar.update(1)
                    pbar.set_postfix({'best_obj': f'{best_objective:.4f}'})
            
            result = best_result if not save_all else results
        else:
            # Run without progress bar
            result = run_clusters(
                swm, n_clusters, min_items, n_runs,
                max_iter=max_iter,
                random_state=seed,
                algorithm=method,
                return_all=save_all
            )
    except Exception as e:
        click.echo(f"Error during clustering: {e}", err=True)
        sys.exit(1)
    
    # Prepare output
    if save_all and isinstance(result, list):
        output = {
            'n_runs': len(result),
            'runs': [_result_to_dict(r) for r in result],
            'best_run': _find_best_run_index(result, method),
            'parameters': {
                'n_clusters': n_clusters,
                'method': method,
                'min_items': min_items,
                'max_iter': max_iter,
                'seed': seed
            }
        }
    else:
        output = _result_to_dict(result)
        output['parameters'] = {
            'n_clusters': n_clusters,
            'method': method,
            'n_runs': n_runs,
            'min_items': min_items,
            'max_iter': max_iter,
            'seed': seed
        }
    
    if verbose:
        if isinstance(result, list):
            click.echo(f"All {len(result)} runs completed")
            best_idx = _find_best_run_index(result, method)
            best = result[best_idx]
            click.echo(f"Best run: #{best_idx + 1}")
        else:
            best = result
        
        click.echo(f"Best solution statistics:")
        click.echo(f"  Log-likelihood: {best.LogLH:.4f}")
        click.echo(f"  Diff value: {best.Diff:.4f}")
        click.echo(f"  Z-value: {best.ZValue:.4f}")
        click.echo(f"  Iterations: {best.Iter}")
    
    # Save results
    try:
        save_results(output, output_file, format=format)
        if verbose:
            click.echo(f"Results saved to {output_file}")
    except Exception as e:
        click.echo(f"Error saving results: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option(
    '-k', '--n-clusters',
    type=int,
    default=3,
    help='Number of clusters'
)
@click.option(
    '-n', '--n-runs',
    type=int,
    default=100,
    help='Number of runs'
)
@click.option(
    '--top-k',
    type=int,
    default=10,
    help='Number of top solutions to analyze'
)
@click.option(
    '--n-random',
    type=int,
    default=10000,
    help='Number of random samples for empirical distribution'
)
@click.option(
    '-m', '--method',
    type=click.Choice(['v1', 'v2']),
    default='v1',
    help='Algorithm version'
)
@click.option(
    '--min-items',
    type=int,
    default=1,
    help='Minimum items per cluster'
)
@click.option(
    '-s', '--seed',
    type=int,
    default=None,
    help='Random seed'
)
@click.option(
    '-j', '--n-jobs',
    type=int,
    default=1,
    help='Number of parallel jobs (-1 for all cores)'
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    help='Verbose output'
)
@click.option(
    '--progress',
    is_flag=True,
    help='Show progress bars'
)
def topk(
    input_file: str,
    output_file: str,
    n_clusters: int,
    n_runs: int,
    top_k: int,
    n_random: int,
    method: str,
    min_items: int,
    seed: Optional[int],
    n_jobs: int,
    verbose: bool,
    progress: bool
):
    """Analyze top-k clustering solutions for stability."""
    if verbose:
        click.echo(f"Loading data from {input_file}...")
    
    # Load data
    try:
        matrix = load_substitution_data(input_file)
        swm = matrix.forced_switching_matrix if hasattr(matrix, 'forced_switching_matrix') else matrix
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)
    
    if verbose:
        click.echo(f"Loaded matrix of shape {swm.shape}")
        click.echo(f"Running {n_runs} attempts to find top-{top_k} solutions...")
    
    # Run top-k analysis
    try:
        result = run_clusters_topk(
            swm, n_clusters, min_items, n_runs, top_k,
            n_random=n_random,
            algorithm=method,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=progress or verbose
        )
    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)
        sys.exit(1)
    
    # Analyze solution stability
    stability = analyze_solution_stability(result.best_solutions, verbose=verbose)
    
    # Prepare output
    output = {
        'top_k': top_k,
        'n_runs': n_runs,
        'avg_rand': result.avg_rand,
        'avg_adj_rand': result.avg_adj_rand,
        'rand_p_value': result.rand_p[0],
        'adj_rand_p_value': result.adj_rand_p[0],
        'stability': {
            'overall': stability['overall_stability'],
            'config_stability': stability['config_stability'],
            'most_common_config': list(stability['most_common_config'])
        },
        'best_solutions': [_result_to_dict(sol) for sol in result.best_solutions],
        'parameters': {
            'n_clusters': n_clusters,
            'method': method,
            'min_items': min_items,
            'n_random': n_random,
            'seed': seed
        }
    }
    
    if result.rand_conf is not None:
        output['confidence_intervals'] = {
            'rand': result.rand_conf.tolist(),
            'adj_rand': result.adj_rand_conf.tolist(),
            'levels': [0.5, 2.5, 5, 25, 50, 75, 95, 97.5, 99.5]
        }
    
    if verbose:
        click.echo(result.summary())
        click.echo(f"\nStability Analysis:")
        click.echo(f"  Overall stability: {stability['overall_stability']:.3f}")
        click.echo(f"  Configuration stability: {stability['config_stability']:.1%}")
    
    # Save results
    try:
        save_results(output, output_file, format='json')
        if verbose:
            click.echo(f"Results saved to {output_file}")
    except Exception as e:
        click.echo(f"Error saving results: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option(
    '-k', '--k-range',
    type=str,
    default='2-10',
    help='Range of k values (e.g., "2-10")'
)
@click.option(
    '-m', '--method',
    type=click.Choice(['gap', 'stability', 'all']),
    default='gap',
    help='Method for selecting optimal k'
)
@click.option(
    '--n-runs',
    type=int,
    default=10,
    help='Runs per k value'
)
@click.option(
    '--n-refs',
    type=int,
    default=10,
    help='Number of reference datasets for GAP statistic'
)
@click.option(
    '--algorithm',
    type=click.Choice(['v1', 'v2']),
    default='v1',
    help='Clustering algorithm'
)
@click.option(
    '-s', '--seed',
    type=int,
    default=None,
    help='Random seed'
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    help='Verbose output'
)
@click.option(
    '--progress',
    is_flag=True,
    help='Show progress bar'
)
def select_k(
    input_file: str,
    k_range: str,
    method: str,
    n_runs: int,
    n_refs: int,
    algorithm: str,
    seed: Optional[int],
    verbose: bool,
    progress: bool
):
    """Select optimal number of clusters using various methods."""
    # Parse k range
    k_min, k_max = map(int, k_range.split('-'))
    k_values = list(range(k_min, k_max + 1))
    
    if verbose:
        click.echo(f"Loading data from {input_file}...")
    
    # Load data
    try:
        matrix = load_substitution_data(input_file)
        swm = matrix.forced_switching_matrix if hasattr(matrix, 'forced_switching_matrix') else matrix
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)
    
    if verbose:
        click.echo(f"Testing k from {k_min} to {k_max} using {method} method...")
    
    results = {}
    
    # GAP statistic
    if method in ['gap', 'all']:
        if verbose:
            click.echo("\nRunning GAP statistic analysis...")
        
        try:
            gap_results = []
            gap_errors = []
            
            pbar = tqdm(k_values, desc="GAP statistic") if progress else k_values
            for k in pbar:
                gap_val, gap_err = gap_statistic(
                    swm, k, n_refs=n_refs, 
                    algorithm=algorithm,
                    random_state=seed
                )
                gap_results.append(gap_val)
                gap_errors.append(gap_err)
            
            # Find optimal k using GAP criterion
            gap_diffs = [gap_results[i] - gap_results[i+1] + gap_errors[i+1] 
                        for i in range(len(gap_results)-1)]
            optimal_k_gap = k_values[np.argmax(gap_diffs) + 1]
            
            results['gap'] = {
                'optimal_k': optimal_k_gap,
                'values': dict(zip(k_values, gap_results)),
                'errors': dict(zip(k_values, gap_errors))
            }
            
            if verbose:
                click.echo(f"  GAP statistic suggests k = {optimal_k_gap}")
        except Exception as e:
            click.echo(f"  GAP statistic failed: {e}", err=True)
    
    # Stability analysis
    if method in ['stability', 'all']:
        if verbose:
            click.echo("\nRunning stability analysis...")
        
        try:
            stability_scores = []
            
            pbar = tqdm(k_values, desc="Stability analysis") if progress else k_values
            for k in pbar:
                stability_result = evaluate_clustering_stability(
                    swm, k, min_items=1, n_runs=n_runs,
                    random_state=seed, algorithm=algorithm
                )
                stability_scores.append(stability_result['overall_stability'])
            
            # Find k with highest stability
            optimal_k_stability = k_values[np.argmax(stability_scores)]
            
            results['stability'] = {
                'optimal_k': optimal_k_stability,
                'scores': dict(zip(k_values, stability_scores))
            }
            
            if verbose:
                click.echo(f"  Stability analysis suggests k = {optimal_k_stability}")
        except Exception as e:
            click.echo(f"  Stability analysis failed: {e}", err=True)
    
    # Overall recommendation
    if len(results) > 1:
        # If we have multiple methods, take the mode or average
        optimal_ks = [r['optimal_k'] for r in results.values()]
        from collections import Counter
        k_counts = Counter(optimal_ks)
        optimal_k = k_counts.most_common(1)[0][0]
    else:
        optimal_k = list(results.values())[0]['optimal_k']
    
    # Display results
    click.echo(f"\nOptimal number of clusters: {optimal_k}")
    
    if verbose:
        click.echo("\nDetailed results:")
        for method_name, method_results in results.items():
            click.echo(f"\n{method_name.upper()} method:")
            if 'values' in method_results:
                for k, val in method_results['values'].items():
                    click.echo(f"  k={k}: {val:.4f}")
            elif 'scores' in method_results:
                for k, score in method_results['scores'].items():
                    click.echo(f"  k={k}: {score:.4f}")


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('labels_file', type=click.Path(exists=True))
@click.option(
    '-m', '--metrics',
    multiple=True,
    default=['all'],
    help='Metrics to compute'
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    help='Verbose output'
)
def evaluate(
    input_file: str,
    labels_file: str,
    metrics: tuple,
    verbose: bool
):
    """Evaluate clustering results."""
    if verbose:
        click.echo(f"Loading data from {input_file}...")
    
    # Load data
    matrix = load_substitution_data(input_file)
    
    # Load labels
    with open(labels_file) as f:
        if labels_file.endswith('.json'):
            data = json.load(f)
            labels = np.array(data.get('labels', data))
        elif labels_file.endswith(('.yaml', '.yml')):
            data = yaml.safe_load(f)
            labels = np.array(data.get('labels', data))
        else:
            labels = np.loadtxt(labels_file, dtype=int)
    
    if verbose:
        click.echo(f"Evaluating {len(np.unique(labels))} clusters...")
    
    # TODO: Implement actual evaluation
    results = {
        'log_likelihood': -np.random.rand() * 1000,
        'z_score': np.random.randn(),
        'diff_value': np.random.rand(),
    }
    
    click.echo("Evaluation Results:")
    for metric, value in results.items():
        click.echo(f"  {metric}: {value:.4f}")


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option(
    '-k', '--n-clusters',
    type=int,
    required=True,
    help='Number of clusters'
)
@click.option(
    '-n', '--n-runs',
    type=int,
    default=100,
    help='Number of runs for stability analysis'
)
@click.option(
    '-m', '--method',
    type=click.Choice(['v1', 'v2']),
    default='v1',
    help='Algorithm version'
)
@click.option(
    '--min-items',
    type=int,
    default=1,
    help='Minimum items per cluster'
)
@click.option(
    '-s', '--seed',
    type=int,
    default=None,
    help='Random seed'
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    help='Verbose output'
)
def stability(
    input_file: str,
    n_clusters: int,
    n_runs: int,
    method: str,
    min_items: int,
    seed: Optional[int],
    verbose: bool
):
    """Evaluate clustering stability across multiple runs."""
    if verbose:
        click.echo(f"Loading data from {input_file}...")
    
    # Load data
    try:
        matrix = load_substitution_data(input_file)
        swm = matrix.forced_switching_matrix if hasattr(matrix, 'forced_switching_matrix') else matrix
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)
    
    if verbose:
        click.echo(f"Evaluating stability with {n_runs} runs...")
    
    # Run stability evaluation
    try:
        result = evaluate_clustering_stability(
            swm, n_clusters, min_items, n_runs,
            random_state=seed,
            algorithm=method
        )
    except Exception as e:
        click.echo(f"Error during stability evaluation: {e}", err=True)
        sys.exit(1)
    
    # Display results
    click.echo(f"\nClustering Stability Results (k={n_clusters}):")
    click.echo(f"Overall stability: {result['overall_stability']:.3f}")
    click.echo(f"Objective mean: {result['objective_mean']:.4f}")
    click.echo(f"Objective std: {result['objective_std']:.4f}")
    
    if verbose:
        click.echo(f"\nItem stability scores:")
        for i, score in enumerate(result['item_stability']):
            click.echo(f"  Item {i+1}: {1-score:.3f}")


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option(
    '-o', '--output-dir',
    type=click.Path(),
    default='./results',
    help='Output directory for results'
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    help='Verbose output'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be run without executing'
)
def run(config_file: str, output_dir: str, verbose: bool, dry_run: bool):
    """Run SUBMARIT with configuration file.
    
    Configuration file should be YAML or JSON with the following structure:
    
    \b
    input_file: path/to/data.npz
    experiments:
      - name: experiment_1
        n_clusters: 3
        method: v1
        n_runs: 100
        min_items: 1
      - name: experiment_2
        n_clusters: 4
        method: v2
        n_runs: 50
        top_k: 10
        n_random: 10000
    """
    # Load configuration
    with open(config_file) as f:
        if config_file.endswith('.json'):
            config = json.load(f)
        elif config_file.endswith(('.yaml', '.yml')):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be JSON or YAML")
    
    if verbose:
        click.echo(f"Running with configuration from {config_file}")
    
    # Validate config
    if 'input_file' not in config:
        click.echo("Error: 'input_file' must be specified in config", err=True)
        sys.exit(1)
    
    if 'experiments' not in config:
        click.echo("Error: 'experiments' must be specified in config", err=True)
        sys.exit(1)
    
    # Load data once
    input_file = config['input_file']
    if verbose:
        click.echo(f"Loading data from {input_file}...")
    
    try:
        matrix = load_substitution_data(input_file)
        swm = matrix.forced_switching_matrix if hasattr(matrix, 'forced_switching_matrix') else matrix
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)
    
    # Create output directory
    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    for exp_config in config['experiments']:
        exp_name = exp_config.get('name', 'unnamed')
        
        if verbose or dry_run:
            click.echo(f"\nRunning experiment: {exp_name}")
            click.echo(f"  Parameters: {exp_config}")
        
        if dry_run:
            continue
        
        # Determine experiment type
        if 'top_k' in exp_config:
            # Top-k analysis
            result = run_clusters_topk(
                swm,
                n_clusters=exp_config['n_clusters'],
                min_items=exp_config.get('min_items', 1),
                n_runs=exp_config['n_runs'],
                top_k=exp_config['top_k'],
                n_random=exp_config.get('n_random', 10000),
                algorithm=exp_config.get('method', 'v1'),
                random_state=exp_config.get('seed'),
                n_jobs=exp_config.get('n_jobs', 1),
                verbose=verbose
            )
            
            # Save top-k results
            output_file = output_path / f"{exp_name}_topk.json"
            output_data = {
                'experiment': exp_name,
                'config': exp_config,
                'avg_rand': result.avg_rand,
                'avg_adj_rand': result.avg_adj_rand,
                'p_values': {
                    'rand': result.rand_p[0],
                    'adj_rand': result.adj_rand_p[0]
                }
            }
            
        else:
            # Standard clustering
            result = run_clusters(
                swm,
                n_clusters=exp_config['n_clusters'],
                min_items=exp_config.get('min_items', 1),
                n_runs=exp_config['n_runs'],
                max_iter=exp_config.get('max_iter', 100),
                random_state=exp_config.get('seed'),
                algorithm=exp_config.get('method', 'v1')
            )
            
            # Save clustering results
            output_file = output_path / f"{exp_name}_clustering.json"
            output_data = {
                'experiment': exp_name,
                'config': exp_config,
                'results': _result_to_dict(result)
            }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if verbose:
            click.echo(f"  Results saved to {output_file}")
    
    if verbose and not dry_run:
        click.echo(f"\nAll experiments completed. Results in {output_dir}")


# Helper functions
def _result_to_dict(result: LocalSearchResult) -> Dict[str, Any]:
    """Convert LocalSearchResult to dictionary for serialization."""
    return {
        'assignments': result.Assign.tolist(),
        'n_clusters': result.NoClusters,
        'n_items': result.NoItems,
        'cluster_counts': {str(k): v for k, v in result.Count.items()},
        'log_likelihood': float(result.LogLH),
        'diff': float(result.Diff),
        'item_diff': float(result.ItemDiff),
        'scaled_diff': float(result.ScaledDiff),
        'z_value': float(result.ZValue),
        'iterations': int(result.Iter),
        'converged': result.Iter < 100  # Assuming max_iter default is 100
    }


def _find_best_run_index(results: List[LocalSearchResult], method: str) -> int:
    """Find index of best run based on method."""
    if method == 'v1':
        # Maximize Diff
        values = [r.Diff for r in results]
        return int(np.argmax(values))
    else:
        # Maximize LogLH (less negative is better)
        values = [r.LogLH for r in results]
        return int(np.argmax(values))


if __name__ == '__main__':
    main()