#!/usr/bin/env python3
"""
Analyze Divergence Structure

Analyzes how divergence relates to pattern frequencies to find the exact formula.
Goal: Discover D(s) = Σ α_p · freq_p(s)
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.rule30_optimized import generate_center_column_direct
from experiments.rule30.diagnostics import create_sequence_vectors, _compute_field_divergence

def compute_direct_divergence(sequence: list, window_size: int = 5) -> float:
    """Compute divergence directly from sequence."""
    vectors = create_sequence_vectors(sequence)
    if len(vectors) < window_size:
        return 0.0
    divergence_values = []
    for i in range(len(vectors) - window_size + 1):
        window_vecs = vectors[i:i + window_size]
        divergence = _compute_field_divergence(window_vecs, window_vecs)
        divergence_values.append(divergence)
    return float(np.mean(divergence_values)) if divergence_values else 0.0
from experiments.rule30.pattern_analysis import (
    compute_pattern_frequencies,
    analyze_divergence_vs_patterns,
    find_invariant_pattern_combination
)
from experiments.rule30.rational_divergence import find_exact_invariant_value


def generate_random_sequences(n: int, count: int = 100) -> list:
    """Generate random binary sequences for comparison."""
    import random
    sequences = []
    for _ in range(count):
        seq = [random.randint(0, 1) for _ in range(n)]
        sequences.append(seq)
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Analyze divergence structure to find exact formula",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--rule30-steps',
        type=int,
        nargs='+',
        default=[1000, 5000, 10000],
        help='Rule 30 sequence lengths to test (default: 1000 5000 10000)'
    )
    
    parser.add_argument(
        '--random-count',
        type=int,
        default=50,
        help='Number of random sequences to generate (default: 50)'
    )
    
    parser.add_argument(
        '--pattern-length',
        type=int,
        default=3,
        help='Pattern length to analyze (default: 3)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("DIVERGENCE STRUCTURE ANALYSIS")
    print("="*70)
    print()
    
    # Generate Rule 30 sequences
    print("Step 1: Generating Rule 30 sequences...")
    rule30_sequences = []
    rule30_divergences = []
    
    for n in args.rule30_steps:
        seq = generate_center_column_direct(n, show_progress=False)
        div = compute_direct_divergence(seq)
        rule30_sequences.append(seq)
        rule30_divergences.append(div)
        print(f"  n={n:6,}: divergence={div:.9f}, length={len(seq):,}")
    
    print()
    
    # Generate random sequences
    print(f"Step 2: Generating {args.random_count} random sequences...")
    random_sequences = []
    random_divergences = []
    
    for n in args.rule30_steps:
        rand_seqs = generate_random_sequences(n, count=args.random_count // len(args.rule30_steps))
        for seq in rand_seqs:
            div = compute_direct_divergence(seq)
            random_sequences.append(seq)
            random_divergences.append(div)
    
    print(f"  Generated {len(random_sequences)} random sequences")
    print(f"  Divergence range: [{min(random_divergences):.6f}, {max(random_divergences):.6f}]")
    print()
    
    # Analyze pattern frequencies
    print(f"Step 3: Analyzing {args.pattern_length}-bit pattern frequencies...")
    
    # Analyze Rule 30
    print("\nRule 30 pattern frequencies:")
    for i, seq in enumerate(rule30_sequences):
        freqs = compute_pattern_frequencies(seq, pattern_length=args.pattern_length)
        print(f"\n  n={args.rule30_steps[i]:,}:")
        for pattern in sorted(freqs.keys()):
            print(f"    '{pattern}': {freqs[pattern]:.4f}")
    
    print()
    
    # Try to fit divergence as linear combination
    print("Step 4: Fitting divergence as linear combination of pattern frequencies...")
    
    # Combine Rule 30 and random for regression
    all_sequences = rule30_sequences + random_sequences
    all_divergences = rule30_divergences + random_divergences
    
    result = analyze_divergence_vs_patterns(
        all_sequences,
        all_divergences,
        pattern_length=args.pattern_length
    )
    
    if 'error' in result:
        print(f"  Error: {result['error']}")
    else:
        print(f"\n  Formula: D(s) = {result['formula']}")
        print(f"  R² = {result['r_squared']:.6f}")
        print(f"  Max residual: {result['max_residual']:.9e}")
        print(f"  Is exact: {result['is_exact']}")
        
        if result['is_exact']:
            print("\n  ✓✓✓ EXACT FORMULA FOUND!")
            print(f"  D(s) = {result['formula']}")
        else:
            print("\n  ⚠ Formula is approximate, not exact")
    
    print()
    
    # Test Rule 30 specifically
    print("Step 5: Testing invariant for Rule 30 sequences...")
    rule30_result = find_invariant_pattern_combination(rule30_sequences)
    
    if 'formula' in rule30_result:
        print(f"\n  Rule 30 formula: D(s) = {rule30_result['formula']}")
        print(f"  Invariant value: {rule30_result.get('invariant_value', 'N/A'):.9f}")
        print(f"  Is invariant: {rule30_result.get('is_invariant', False)}")
        
        if rule30_result.get('is_invariant'):
            print("\n  ✓✓✓ INVARIANT CONFIRMED!")
            print(f"  The formula D(s) = {rule30_result['formula']}")
            print(f"  gives constant value {rule30_result['invariant_value']:.9f} for Rule 30")
    
    print()
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. If exact formula found, verify with rational arithmetic")
    print("2. Derive how Rule 30 updates pattern frequencies")
    print("3. Prove the linear combination is conserved algebraically")


if __name__ == '__main__':
    main()

