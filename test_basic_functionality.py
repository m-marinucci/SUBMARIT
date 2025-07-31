#!/usr/bin/env python3
"""Basic functionality test for SUBMARIT."""

import numpy as np
from submarit.core import SubstitutionMatrix, create_substitution_matrix
from submarit.algorithms.local_search import KSMLocalSearch, KSMLocalSearch2
from submarit.evaluation.cluster_evaluator import ClusterEvaluator
from submarit.evaluation.gap_statistic import GAPStatistic
from submarit.validation.rand_index import RandIndex
from submarit.io.data_io import save_results

def test_basic_pipeline():
    """Test the basic SUBMARIT pipeline."""
    print("=== SUBMARIT Basic Functionality Test ===\n")
    
    # 1. Create test data
    print("1. Creating test consumer-product data...")
    np.random.seed(42)
    n_consumers, n_products = 200, 30
    
    # Create data with 3 natural clusters
    consumer_product = np.zeros((n_consumers, n_products))
    for i in range(n_consumers):
        if i < 70:  # Cluster 1 consumers prefer products 0-9
            consumer_product[i, :10] = np.random.poisson(3, 10)
            consumer_product[i, 10:] = np.random.poisson(0.1, 20)
        elif i < 130:  # Cluster 2 consumers prefer products 10-19
            consumer_product[i, :10] = np.random.poisson(0.1, 10)
            consumer_product[i, 10:20] = np.random.poisson(3, 10)
            consumer_product[i, 20:] = np.random.poisson(0.1, 10)
        else:  # Cluster 3 consumers prefer products 20-29
            consumer_product[i, :20] = np.random.poisson(0.1, 20)
            consumer_product[i, 20:] = np.random.poisson(3, 10)
    
    # 2. Create substitution matrix
    print("2. Creating substitution matrix...")
    fswm, indexes, count = create_substitution_matrix(consumer_product)
    print(f"   - Created {count}x{count} substitution matrix")
    print(f"   - Products retained: {count}/{n_products}")
    
    # 3. Run clustering
    print("\n3. Running clustering algorithms...")
    results = {}
    
    for name, cls in [('KSMLocalSearch', KSMLocalSearch), 
                      ('KSMLocalSearch2', KSMLocalSearch2)]:
        print(f"\n   {name}:")
        clusterer = cls(n_clusters=3, random_state=42)
        labels = clusterer.fit_predict(fswm)
        
        print(f"   - Converged in {clusterer.n_iter_} iterations")
        print(f"   - Cluster sizes: {np.bincount(labels)}")
        
        # Evaluate
        evaluator = ClusterEvaluator()
        eval_result = evaluator.evaluate(fswm, labels)
        print(f"   - Log-likelihood: {eval_result.log_likelihood:.4f}")
        print(f"   - Z-value: {eval_result.z_value:.4f}")
        
        results[name] = {
            'labels': labels.tolist(),
            'n_iter': int(clusterer.n_iter_),
            'log_likelihood': float(eval_result.log_likelihood),
            'z_value': float(eval_result.z_value)
        }
    
    # 4. Compare results
    print("\n4. Comparing clustering results...")
    rand_index = RandIndex()
    labels1 = np.array(results['KSMLocalSearch']['labels'])
    labels2 = np.array(results['KSMLocalSearch2']['labels'])
    
    ri, ari = rand_index.calculate(labels1, labels2)
    print(f"   - Rand Index: {ri:.4f}")
    print(f"   - Adjusted Rand Index: {ari:.4f}")
    
    # 5. Save results
    print("\n5. Saving results...")
    save_results(results, 'test_results.json')
    print("   - Results saved to test_results.json")
    
    print("\n✅ All tests passed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_basic_pipeline()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)