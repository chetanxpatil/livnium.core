#!/usr/bin/env python3
"""
Invariant Hunter - Proof Search Lab

Searches for exact algebraic invariants in Rule 30 center column.
Enumerates candidate invariants and tests them for exact conservation.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from fractions import Fraction
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.rule30_optimized import generate_center_column_direct
from experiments.rule30.pattern_analysis import (
    compute_pattern_frequencies,
    analyze_divergence_vs_patterns,
    find_invariant_pattern_combination
)
from experiments.rule30.rational_divergence import find_exact_invariant_value


def enumerate_candidate_invariants(
    pattern_length: int = 3,
    max_coefficient: int = 10
) -> List[Dict]:
    """
    Enumerate candidate invariants as linear combinations of pattern frequencies.
    
    Args:
        pattern_length: Length of patterns (default: 3, gives 8 patterns)
        max_coefficient: Maximum absolute value for coefficients
        
    Returns:
        List of candidate invariants with coefficients
    """
    # Generate all patterns
    patterns = []
    for i in range(2 ** pattern_length):
        pattern = ''.join(str((i >> j) & 1) for j in range(pattern_length - 1, -1, -1))
        patterns.append(pattern)
    
    candidates = []
    
    # Enumerate simple combinations
    # Start with small coefficients
    for a0 in range(-max_coefficient, max_coefficient + 1):
        for a1 in range(-max_coefficient, max_coefficient + 1):
            for a2 in range(-max_coefficient, max_coefficient + 1):
                # Skip trivial (all zeros)
                if a0 == 0 and a1 == 0 and a2 == 0:
                    continue
                
                # For 3-bit patterns, we have 8 patterns
                # Use first 3 as basis, rest determined by constraint
                coefficients = [a0, a1, a2] + [0] * (len(patterns) - 3)
                
                candidates.append({
                    'patterns': patterns,
                    'coefficients': coefficients,
                    'as_formula': _format_candidate(patterns[:3], coefficients[:3])
                })
    
    return candidates


def _format_candidate(patterns: List[str], coeffs: List[int]) -> str:
    """Format candidate invariant as formula."""
    terms = []
    for p, c in zip(patterns, coeffs):
        if c != 0:
            if c == 1:
                terms.append(f"freq('{p}')")
            elif c == -1:
                terms.append(f"-freq('{p}')")
            else:
                terms.append(f"{c}*freq('{p}')")
    
    if not terms:
        return "0"
    
    return " + ".join(terms)


def test_candidate_invariant(
    candidate: Dict,
    sequences: List[List[int]],
    target_value: float = -0.572222233,
    tolerance: float = 1e-6
) -> Dict:
    """
    Test if a candidate invariant is conserved across sequences.
    
    Args:
        candidate: Candidate invariant with patterns and coefficients
        sequences: List of sequences to test
        target_value: Expected invariant value
        tolerance: Tolerance for matching target
        
    Returns:
        Dict with test results
    """
    values = []
    
    for seq in sequences:
        freqs = compute_pattern_frequencies(seq, pattern_length=len(candidate['patterns'][0]))
        
        # Compute invariant value
        value = 0.0
        for pattern, coeff in zip(candidate['patterns'], candidate['coefficients']):
            value += coeff * freqs.get(pattern, 0.0)
        
        values.append(value)
    
    values = np.array(values)
    mean_val = np.mean(values)
    std_val = np.std(values)
    max_dev = np.max(np.abs(values - mean_val))
    
    # Check if conserved (low std) and matches target
    is_conserved = std_val < tolerance
    matches_target = abs(mean_val - target_value) < tolerance
    
    return {
        'candidate': candidate,
        'mean_value': float(mean_val),
        'std_value': float(std_val),
        'max_deviation': float(max_dev),
        'is_conserved': is_conserved,
        'matches_target': matches_target,
        'values': values.tolist()
    }


def hunt_invariants(
    n_steps_list: List[int],
    target_value: float = -0.572222233,
    max_coefficient: int = 5
) -> List[Dict]:
    """
    Hunt for exact invariants by testing candidates.
    
    Args:
        n_steps_list: List of sequence lengths to test
        target_value: Expected invariant value
        max_coefficient: Maximum coefficient to try
        
    Returns:
        List of candidate invariants that pass tests
    """
    print("="*70)
    print("INVARIANT HUNTER")
    print("="*70)
    print(f"Target value: {target_value:.9f}")
    print(f"Testing sequences: {n_steps_list}")
    print(f"Max coefficient: {max_coefficient}")
    print()
    
    # Generate Rule 30 sequences
    print("Generating Rule 30 sequences...")
    sequences = []
    for n in n_steps_list:
        seq = generate_center_column_direct(n, show_progress=False)
        sequences.append(seq)
        print(f"  Generated {n:,} steps: {len(seq):,} bits")
    
    print()
    
    # Enumerate candidates
    print(f"Enumerating candidate invariants (max_coeff={max_coefficient})...")
    candidates = enumerate_candidate_invariants(max_coefficient=max_coefficient)
    print(f"  Generated {len(candidates):,} candidates")
    print()
    
    # Test candidates
    print("Testing candidates...")
    passing_candidates = []
    
    for i, candidate in enumerate(candidates):
        if (i + 1) % 1000 == 0:
            print(f"  Tested {i+1:,}/{len(candidates):,} candidates...")
        
        result = test_candidate_invariant(candidate, sequences, target_value)
        
        if result['is_conserved'] and result['matches_target']:
            passing_candidates.append(result)
            print(f"  ✓ Found candidate: {result['candidate']['as_formula']}")
            print(f"    Mean: {result['mean_value']:.9f}, Std: {result['std_value']:.9e}")
    
    print()
    print(f"Found {len(passing_candidates)} passing candidates")
    
    return passing_candidates


def main():
    parser = argparse.ArgumentParser(
        description="Hunt for exact algebraic invariants in Rule 30",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[1000, 10000],
        help='Sequence lengths to test (default: 1000 10000)'
    )
    
    parser.add_argument(
        '--target',
        type=float,
        default=-0.572222233,
        help='Target invariant value (default: -0.572222233)'
    )
    
    parser.add_argument(
        '--max-coeff',
        type=int,
        default=5,
        help='Maximum coefficient to try (default: 5, warning: grows exponentially)'
    )
    
    args = parser.parse_args()
    
    # Hunt for invariants
    passing = hunt_invariants(
        args.steps,
        target_value=args.target,
        max_coefficient=args.max_coeff
    )
    
    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    if passing:
        print(f"\nFound {len(passing)} candidate invariants:")
        for i, result in enumerate(passing[:10], 1):  # Show first 10
            print(f"\n{i}. {result['candidate']['as_formula']}")
            print(f"   Mean: {result['mean_value']:.9f}")
            print(f"   Std:  {result['std_value']:.9e}")
        
        if len(passing) > 10:
            print(f"\n... and {len(passing) - 10} more")
    else:
        print("\nNo exact invariants found with given parameters.")
        print("Try increasing --max-coeff or checking different pattern lengths.")
    
    print()


if __name__ == '__main__':
    main()

