#!/usr/bin/env python3
"""Test script for k-submarket local search algorithms."""

import numpy as np
from src.submarit.algorithms import (
    KSMLocalSearch,
    KSMLocalSearch2,
    KSMLocalSearchConstrained,
    KSMLocalSearchConstrained2,
    k_sm_local_search,
    k_sm_local_search2,
    k_sm_local_search_constrained,
    k_sm_local_search_constrained2
)


def create_test_switching_matrix(n_items=10, n_clusters=3, noise_level=0.1):
    """Create a synthetic switching matrix with cluster structure."""
    np.random.seed(42)
    
    # Create base switching patterns within clusters
    swm = np.zeros((n_items, n_items))
    items_per_cluster = n_items // n_clusters
    
    for k in range(n_clusters):
        start = k * items_per_cluster
        end = min((k + 1) * items_per_cluster, n_items)
        
        # High switching within cluster
        for i in range(start, end):
            for j in range(start, end):
                if i != j:
                    swm[i, j] = np.random.poisson(50)
                    
    # Add noise - low switching between clusters
    for i in range(n_items):
        for j in range(n_items):
            if swm[i, j] == 0 and i != j:
                swm[i, j] = np.random.poisson(noise_level * 10)
                
    return swm


def test_basic_algorithms():
    """Test basic local search algorithms."""
    print("Testing basic k-submarket local search algorithms...")
    print("=" * 60)
    
    # Create test data
    swm = create_test_switching_matrix(n_items=15, n_clusters=3)
    
    # Test KSMLocalSearch
    print("\n1. Testing KSMLocalSearch (PHat-P optimization):")
    model1 = KSMLocalSearch(n_clusters=3, min_items=2, random_state=42)
    model1.fit(swm)
    result1 = model1.get_result()
    
    print(f"   - Converged in {result1.Iter} iterations")
    print(f"   - Cluster assignments: {result1.Assign}")
    print(f"   - Objective (Diff): {result1.Diff:.4f}")
    print(f"   - Log-likelihood: {result1.LogLH:.4f}")
    print(f"   - Z-value: {result1.ZValue:.4f}")
    
    # Test KSMLocalSearch2
    print("\n2. Testing KSMLocalSearch2 (Log-likelihood optimization):")
    model2 = KSMLocalSearch2(n_clusters=3, min_items=2, random_state=42)
    model2.fit(swm)
    result2 = model2.get_result()
    
    print(f"   - Converged in {result2.Iter} iterations")
    print(f"   - Cluster assignments: {result2.Assign}")
    print(f"   - Objective (-LogLH): {result2.MaxObj:.4f}")
    print(f"   - Log-likelihood: {result2.LogLH:.4f}")
    print(f"   - Z-value: {result2.ZValue:.4f}")
    
    # Test functional interface
    print("\n3. Testing functional interface:")
    result3 = k_sm_local_search(swm, n_clusters=3, min_items=2, random_state=42)
    print(f"   - k_sm_local_search: Diff = {result3.Diff:.4f}")
    
    result4 = k_sm_local_search2(swm, n_clusters=3, min_items=2, random_state=42)
    print(f"   - k_sm_local_search2: LogLH = {result4.LogLH:.4f}")


def test_constrained_algorithms():
    """Test constrained local search algorithms."""
    print("\n\nTesting constrained k-submarket local search algorithms...")
    print("=" * 60)
    
    # Create test data
    swm = create_test_switching_matrix(n_items=15, n_clusters=3)
    
    # Define constraints: fix first 3 items to clusters 1, 2, 3
    fixed_assign = np.array([1, 2, 3])  # 1-based cluster assignments
    assign_indexes = np.array([1, 2, 3])  # 1-based item indices
    free_indexes = np.arange(4, 16)  # Items 4-15 are free (1-based)
    
    # Test KSMLocalSearchConstrained
    print("\n1. Testing KSMLocalSearchConstrained:")
    model1 = KSMLocalSearchConstrained(n_clusters=3, min_items=2, random_state=42)
    model1.fit_constrained(swm, fixed_assign, assign_indexes, free_indexes)
    result1 = model1.get_result()
    
    print(f"   - Converged in {result1.Iter} iterations")
    print(f"   - Fixed items kept assignments: {result1.Assign[0:3]}")
    print(f"   - Objective (Diff): {result1.Diff:.4f}")
    print(f"   - Log-likelihood: {result1.LogLH:.4f}")
    
    # Test KSMLocalSearchConstrained2
    print("\n2. Testing KSMLocalSearchConstrained2:")
    model2 = KSMLocalSearchConstrained2(n_clusters=3, min_items=2, random_state=42)
    model2.fit_constrained(swm, fixed_assign, assign_indexes, free_indexes)
    result2 = model2.get_result()
    
    print(f"   - Converged in {result2.Iter} iterations")
    print(f"   - Fixed items kept assignments: {result2.Assign[0:3]}")
    print(f"   - Objective (Diff): {result2.MaxObj:.4f}")
    print(f"   - Log-likelihood: {result2.LogLH:.4f}")
    
    # Test functional interface
    print("\n3. Testing constrained functional interface:")
    result3 = k_sm_local_search_constrained(
        swm, 3, 2, fixed_assign, assign_indexes, free_indexes, random_state=42
    )
    print(f"   - k_sm_local_search_constrained: Diff = {result3.Diff:.4f}")
    
    result4 = k_sm_local_search_constrained2(
        swm, 3, 2, fixed_assign, assign_indexes, free_indexes, random_state=42
    )
    print(f"   - k_sm_local_search_constrained2: LogLH = {result4.LogLH:.4f}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n\nTesting edge cases...")
    print("=" * 60)
    
    # Test minimum items constraint
    print("\n1. Testing minimum items constraint:")
    swm = create_test_switching_matrix(n_items=6, n_clusters=3)
    
    try:
        model = KSMLocalSearch(n_clusters=3, min_items=3)
        model.fit(swm)
        print("   - Error: Should have raised ValueError for insufficient items")
    except ValueError as e:
        print(f"   - Correctly raised ValueError: {e}")
    
    # Test with zero sales
    print("\n2. Testing with zero sales row:")
    swm[0, :] = 0  # First item has no sales
    model = KSMLocalSearch(n_clusters=2, min_items=1)
    model.fit(swm)
    result = model.get_result()
    print(f"   - Successfully handled zero sales, LogLH = {result.LogLH:.4f}")
    
    # Test cluster counts
    print("\n3. Testing cluster membership counts:")
    swm = create_test_switching_matrix(n_items=12, n_clusters=3)
    model = KSMLocalSearch(n_clusters=3, min_items=2)
    model.fit(swm)
    result = model.get_result()
    
    for k in range(1, 4):
        print(f"   - Cluster {k}: {result.Count[k]} items")


def test_result_structure():
    """Test that result structure matches MATLAB output."""
    print("\n\nTesting result structure...")
    print("=" * 60)
    
    swm = create_test_switching_matrix(n_items=10, n_clusters=2)
    result = k_sm_local_search(swm, n_clusters=2, min_items=2)
    
    # Check all expected fields
    expected_fields = [
        'SWM', 'NoClusters', 'NoItems', 'Assign', 'Indexes', 'Count',
        'Diff', 'DiffSq', 'ItemDiff', 'ScaledDiff', 'ZValue', 'MaxObj',
        'LogLH', 'LogLH2', 'Iter', 'Var', 'SDComp', 'SDiff'
    ]
    
    print("Checking result fields:")
    for field in expected_fields:
        if hasattr(result, field):
            value = getattr(result, field)
            if isinstance(value, (int, float, np.number)):
                print(f"   - {field}: {value:.4f}" if isinstance(value, float) else f"   - {field}: {value}")
            elif isinstance(value, dict):
                print(f"   - {field}: Dict with {len(value)} entries")
            elif isinstance(value, np.ndarray):
                print(f"   - {field}: Array with shape {value.shape}")
            else:
                print(f"   - {field}: Present")
        else:
            print(f"   - {field}: MISSING!")
            
    # Test dictionary conversion
    print("\nTesting dictionary conversion:")
    result_dict = result.to_dict()
    print(f"   - Dictionary has {len(result_dict)} keys")


if __name__ == "__main__":
    print("K-SUBMARKET LOCAL SEARCH TEST SUITE")
    print("===================================\n")
    
    test_basic_algorithms()
    test_constrained_algorithms()
    test_edge_cases()
    test_result_structure()
    
    print("\n\nAll tests completed!")
    print("=" * 60)