#!/usr/bin/env python3
"""
Test Divergence V3 Invariant

Extracts candidate weight vectors from nullspace and tests invariance
over many Rule 30 evolution steps.
"""

import argparse
import sys
import random
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.divergence_v3 import enumerate_patterns, divergence_v3, format_weights
from experiments.rule30.invariant_solver_v3 import build_invariance_system, find_nullspace, analyze_nullspace
from experiments.rule30.rule30_algebra import rule30_step, rule30_evolve

Pattern = Tuple[int, int, int]


def extract_weight_vector(nullspace: np.ndarray, vector_index: int = 0) -> Dict[Pattern, float]:
    """
    Turn a nullspace basis vector into a pattern -> weight mapping.
    
    Args:
        nullspace: Nullspace basis matrix
        vector_index: Which basis vector to use (default: 0)
        
    Returns:
        Dict mapping patterns to weights
    """
    patterns = enumerate_patterns()
    
    if nullspace.shape[1] == 0:
        raise RuntimeError("No non-trivial nullspace found (no invariant of this form).")
    
    if vector_index >= nullspace.shape[1]:
        vector_index = 0
    
    w_vec = nullspace[:, vector_index]  # Select basis vector
    
    # Normalize for readability (scale so max absolute value is 1)
    max_abs = np.max(np.abs(w_vec))
    if max_abs > 1e-6:
        w_vec = w_vec / max_abs
    
    weights: Dict[Pattern, float] = {}
    for p, w in zip(patterns, w_vec):
        if abs(w) > 1e-6:
            weights[p] = float(w)
    
    return weights


def run_invariance_test(
    weights: Dict[Pattern, float],
    initial_row: List[int],
    steps: int = 100,
    cyclic: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Test whether divergence_v3 is invariant under Rule 30 evolution.
    
    Args:
        weights: Pattern weights
        initial_row: Starting row
        steps: Number of evolution steps
        cyclic: Use cyclic boundary conditions
        verbose: Print progress
        
    Returns:
        Dict with test results
    """
    row = initial_row.copy()
    d0 = divergence_v3(row, weights, cyclic=cyclic)
    
    if verbose:
        print(f"Initial divergence: {d0:.9f}")
    
    values = [d0]
    deviations = [0.0]
    
    for t in range(1, steps + 1):
        row = rule30_step(row, cyclic=cyclic)
        dt = divergence_v3(row, weights, cyclic=cyclic)
        delta = dt - d0
        
        values.append(dt)
        deviations.append(abs(delta))
        
        if verbose and (t % 10 == 0 or t <= 5):
            print(f"[t={t:3d}] divergence = {dt:.9f}, Δ = {delta:+.3e}")
    
    max_deviation = max(deviations)
    mean_deviation = np.mean(deviations)
    std_deviation = np.std(deviations)
    
    # Check if invariant (small deviations)
    tolerance = 1e-6
    is_invariant = max_deviation < tolerance
    
    return {
        'initial_value': d0,
        'values': values,
        'deviations': deviations,
        'max_deviation': float(max_deviation),
        'mean_deviation': float(mean_deviation),
        'std_deviation': float(std_deviation),
        'is_invariant': is_invariant,
        'tolerance': tolerance
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test Divergence V3 invariants using linear algebra",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--num-rows',
        type=int,
        default=300,
        help='Number of random rows for building system (default: 300)'
    )
    
    parser.add_argument(
        '--row-length',
        type=int,
        default=200,
        help='Length of rows (default: 200)'
    )
    
    parser.add_argument(
        '--test-steps',
        type=int,
        default=50,
        help='Number of evolution steps for testing (default: 50)'
    )
    
    parser.add_argument(
        '--vector-index',
        type=int,
        default=0,
        help='Which nullspace basis vector to use (default: 0)'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-8,
        help='Tolerance for nullspace computation (default: 1e-8)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("DIVERGENCE V3 INVARIANT TEST")
    print("="*70)
    print()
    
    # Step 1: Build linear system
    print(f"Step 1: Building invariance system...")
    print(f"  Sampling {args.num_rows} random rows of length {args.row_length}")
    A = build_invariance_system(
        num_rows=args.num_rows,
        row_length=args.row_length,
        cyclic=True
    )
    print(f"  System matrix shape: {A.shape}")
    print()
    
    # Step 2: Find nullspace
    print(f"Step 2: Finding nullspace (tolerance={args.tolerance})...")
    nullspace = find_nullspace(A, tol=args.tolerance)
    
    analysis = analyze_nullspace(nullspace)
    print(f"  {analysis['message']}")
    print(f"  Nullspace shape: {nullspace.shape}")
    print()
    
    if not analysis['has_invariants']:
        print("="*70)
        print("RESULT: No invariant found")
        print("="*70)
        print("\nThis means:")
        print("  - No 3-bit weighted sum is exactly preserved by Rule 30")
        print("  - Try extending to 5-bit windows or different functionals")
        print("  - Or check if numerical issues (try different tolerance)")
        return
    
    # Step 3: Extract weight vector
    print(f"Step 3: Extracting weight vector (index={args.vector_index})...")
    try:
        weights = extract_weight_vector(nullspace, vector_index=args.vector_index)
        print(f"  Extracted {len(weights)} non-zero weights")
        print()
        print("Candidate invariant formula:")
        print(f"  D3(s) = {format_weights(weights)}")
        print()
    except RuntimeError as e:
        print(f"  Error: {e}")
        return
    
    # Step 4: Test invariance
    print(f"Step 4: Testing invariance over {args.test_steps} steps...")
    test_row = [random.randint(0, 1) for _ in range(args.row_length)]
    
    result = run_invariance_test(
        weights,
        test_row,
        steps=args.test_steps,
        cyclic=True,
        verbose=True
    )
    
    print()
    print("="*70)
    print("INVARIANCE TEST RESULTS")
    print("="*70)
    print(f"Initial value:     {result['initial_value']:.9f}")
    print(f"Max deviation:     {result['max_deviation']:.9e}")
    print(f"Mean deviation:    {result['mean_deviation']:.9e}")
    print(f"Std deviation:     {result['std_deviation']:.9e}")
    print()
    
    if result['is_invariant']:
        print("✓✓✓ INVARIANT CONFIRMED!")
        print(f"Divergence stays within {result['tolerance']:.1e} over {args.test_steps} steps")
        print()
        print("This is a TRUE algebraic invariant of Rule 30!")
        print(f"Formula: D3(s) = {format_weights(weights)}")
    else:
        print("⚠ Not exactly invariant (deviations exceed tolerance)")
        print("Possible reasons:")
        print("  - Numerical precision issues")
        print("  - Need more constraints in system")
        print("  - Approximate invariant (not exact)")
        print()
        if result['max_deviation'] < 1e-3:
            print("However, deviations are small (< 1e-3)")
            print("This may be an approximate invariant worth investigating")
    
    print()


if __name__ == "__main__":
    main()

