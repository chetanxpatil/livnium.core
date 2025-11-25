#!/usr/bin/env python3
"""
Extract Symbolic Divergence Formula

Extracts the exact symbolic form of the divergence function D(s) by:
1. Testing all possible small neighborhoods (5-bit, 7-bit windows)
2. Building a lookup table: pattern → divergence contribution
3. Expressing the exact functional form
4. Verifying it matches the numerical implementation

Goal: Turn numerical divergence into exact symbolic expression.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from fractions import Fraction
from itertools import product
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.diagnostics import create_sequence_vectors, _compute_field_divergence


def test_all_patterns(pattern_length: int = 5) -> Dict[Tuple[int, ...], float]:
    """
    Test divergence on all possible bit patterns of given length.
    
    Args:
        pattern_length: Length of patterns to test
        
    Returns:
        Dict mapping pattern tuple to divergence value
    """
    print(f"Testing all {2**pattern_length} patterns of length {pattern_length}...")
    
    results = {}
    
    # Generate all possible patterns
    for pattern_bits in product([0, 1], repeat=pattern_length):
        # Convert to sequence (repeat pattern to get enough length)
        sequence = list(pattern_bits) * 10  # Repeat to get enough vectors
        
        # Compute divergence
        vectors = create_sequence_vectors(sequence)
        
        if len(vectors) < 5:
            continue
        
        # Compute divergence for this pattern
        window_size = min(5, len(vectors))
        divergence_values = []
        
        for i in range(len(vectors) - window_size + 1):
            window_vecs = vectors[i:i + window_size]
            divergence = _compute_field_divergence(window_vecs, window_vecs)
            divergence_values.append(divergence)
        
        if divergence_values:
            mean_div = float(np.mean(divergence_values))
            results[pattern_bits] = mean_div
    
    print(f"  Tested {len(results)} patterns")
    return results


def analyze_divergence_structure(pattern_results: Dict[Tuple[int, ...], float]) -> Dict:
    """
    Analyze the structure of divergence across patterns.
    
    Looks for:
    - Symmetries
    - Dependencies on specific bit positions
    - Functional relationships
    """
    print("\nAnalyzing divergence structure...")
    
    analysis = {
        'pattern_length': len(next(iter(pattern_results.keys()))) if pattern_results else 0,
        'unique_values': len(set(pattern_results.values())),
        'min_value': min(pattern_results.values()) if pattern_results else 0,
        'max_value': max(pattern_results.values()) if pattern_results else 0,
        'mean_value': float(np.mean(list(pattern_results.values()))) if pattern_results else 0,
        'patterns_by_value': {}
    }
    
    # Group patterns by divergence value
    value_to_patterns = {}
    for pattern, value in pattern_results.items():
        if value not in value_to_patterns:
            value_to_patterns[value] = []
        value_to_patterns[value].append(pattern)
    
    analysis['patterns_by_value'] = {
        value: [''.join(str(b) for b in p) for p in patterns]
        for value, patterns in value_to_patterns.items()
    }
    
    # Check for symmetries
    print(f"  Unique divergence values: {analysis['unique_values']}")
    print(f"  Value range: [{analysis['min_value']:.9f}, {analysis['max_value']:.9f}]")
    print(f"  Mean value: {analysis['mean_value']:.9f}")
    
    return analysis


def extract_functional_form(pattern_results: Dict[Tuple[int, ...], float]) -> Dict:
    """
    Extract the functional form of divergence.
    
    Tries to express divergence as a function of:
    - Bit positions
    - Pattern properties (density, transitions, etc.)
    - Geometric features
    """
    print("\nExtracting functional form...")
    
    if not pattern_results:
        return {'error': 'No pattern results'}
    
    pattern_length = len(next(iter(pattern_results.keys())))
    
    # Analyze dependencies
    dependencies = {}
    
    # Check if divergence depends on specific bit positions
    for pos in range(pattern_length):
        # Group by bit value at this position
        bit0_values = []
        bit1_values = []
        
        for pattern, div in pattern_results.items():
            if pattern[pos] == 0:
                bit0_values.append(div)
            else:
                bit1_values.append(div)
        
        if bit0_values and bit1_values:
            mean0 = np.mean(bit0_values)
            mean1 = np.mean(bit1_values)
            diff = abs(mean0 - mean1)
            
            dependencies[f'bit_{pos}'] = {
                'mean_if_0': float(mean0),
                'mean_if_1': float(mean1),
                'difference': float(diff),
                'significant': diff > 1e-6
            }
    
    # Check if divergence depends on pattern density
    density_values = {}
    for pattern, div in pattern_results.items():
        density = sum(pattern) / len(pattern)
        if density not in density_values:
            density_values[density] = []
        density_values[density].append(div)
    
    density_analysis = {
        density: {
            'mean_div': float(np.mean(divs)),
            'std_div': float(np.std(divs)),
            'count': len(divs)
        }
        for density, divs in density_values.items()
    }
    
    # Check if divergence depends on transitions
    transition_analysis = {}
    for pattern, div in pattern_results.items():
        transitions = sum(1 for i in range(len(pattern)-1) if pattern[i] != pattern[i+1])
        if transitions not in transition_analysis:
            transition_analysis[transitions] = []
        transition_analysis[transitions].append(div)
    
    transition_stats = {
        trans: {
            'mean_div': float(np.mean(divs)),
            'std_div': float(np.std(divs)),
            'count': len(divs)
        }
        for trans, divs in transition_analysis.items()
    }
    
    return {
        'pattern_length': pattern_length,
        'bit_dependencies': dependencies,
        'density_dependence': density_analysis,
        'transition_dependence': transition_stats,
        'lookup_table': {
            ''.join(str(b) for b in pattern): float(div)
            for pattern, div in pattern_results.items()
        }
    }


def verify_on_rule30(functional_form: Dict, n_steps: int = 1000) -> Dict:
    """
    Verify the extracted functional form on actual Rule 30 sequence.
    
    Args:
        functional_form: Extracted functional form
        n_steps: Sequence length to test
        
    Returns:
        Verification results
    """
    print(f"\nVerifying functional form on Rule 30 sequence (n={n_steps})...")
    
    from experiments.rule30.rule30_optimized import generate_center_column_direct
    from experiments.rule30.diagnostics import create_sequence_vectors, _compute_field_divergence
    
    def compute_direct_divergence(seq: List[int], window_size: int = 5) -> float:
        """Compute divergence directly from sequence."""
        vectors = create_sequence_vectors(seq)
        if len(vectors) < window_size:
            return 0.0
        divergence_values = []
        for i in range(len(vectors) - window_size + 1):
            window_vecs = vectors[i:i + window_size]
            divergence = _compute_field_divergence(window_vecs, window_vecs)
            divergence_values.append(divergence)
        return float(np.mean(divergence_values)) if divergence_values else 0.0
    
    # Generate Rule 30 sequence
    sequence = generate_center_column_direct(n_steps, show_progress=False)
    
    # Compute actual divergence
    actual_div = compute_direct_divergence(sequence)
    
    # Try to predict using functional form
    # For now, use lookup table if available
    predicted_div = None
    
    if 'lookup_table' in functional_form:
        # Use pattern frequencies and lookup table
        from experiments.rule30.pattern_analysis import compute_pattern_frequencies
        
        pattern_length = functional_form['pattern_length']
        freqs = compute_pattern_frequencies(sequence, pattern_length)
        
        # Predict as weighted sum
        predicted_div = 0.0
        total_weight = 0.0
        
        for pattern, freq in freqs.items():
            if pattern in functional_form['lookup_table']:
                pattern_div = functional_form['lookup_table'][pattern]
                predicted_div += pattern_div * freq
                total_weight += freq
        
        if total_weight > 0:
            predicted_div = predicted_div / total_weight
    
    return {
        'actual_divergence': actual_div,
        'predicted_divergence': predicted_div,
        'error': abs(actual_div - predicted_div) if predicted_div is not None else None,
        'matches': abs(actual_div - predicted_div) < 1e-6 if predicted_div is not None else False
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract symbolic form of divergence function",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--pattern-length',
        type=int,
        default=5,
        help='Pattern length to analyze (default: 5, warning: 2^n patterns)'
    )
    
    parser.add_argument(
        '--verify',
        type=int,
        default=1000,
        help='Verify on Rule 30 sequence of this length (default: 1000)'
    )
    
    args = parser.parse_args()
    
    if args.pattern_length > 7:
        print("Warning: pattern_length > 7 will generate 128+ patterns (slow)")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("="*70)
    print("SYMBOLIC DIVERGENCE EXTRACTOR")
    print("="*70)
    print(f"Pattern length: {args.pattern_length}")
    print(f"Total patterns to test: {2**args.pattern_length}")
    print()
    
    # Step 1: Test all patterns
    pattern_results = test_all_patterns(args.pattern_length)
    
    if not pattern_results:
        print("Error: No patterns tested successfully")
        return
    
    # Step 2: Analyze structure
    structure = analyze_divergence_structure(pattern_results)
    
    # Step 3: Extract functional form
    functional_form = extract_functional_form(pattern_results)
    
    # Step 4: Verify on Rule 30
    verification = verify_on_rule30(functional_form, args.verify)
    
    # Print summary
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    
    print(f"\nPattern Analysis:")
    print(f"  Unique divergence values: {structure['unique_values']}")
    print(f"  Value range: [{structure['min_value']:.9f}, {structure['max_value']:.9f}]")
    
    print(f"\nFunctional Dependencies:")
    significant_bits = [
        name for name, dep in functional_form.get('bit_dependencies', {}).items()
        if dep.get('significant', False)
    ]
    if significant_bits:
        print(f"  Significant bit positions: {significant_bits}")
    else:
        print(f"  No strong bit-position dependencies found")
    
    print(f"\nVerification on Rule 30:")
    print(f"  Actual divergence: {verification['actual_divergence']:.9f}")
    if verification['predicted_divergence'] is not None:
        print(f"  Predicted divergence: {verification['predicted_divergence']:.9f}")
        print(f"  Error: {verification['error']:.9e}")
        if verification['matches']:
            print(f"  ✓✓✓ MATCHES!")
        else:
            print(f"  ⚠ Does not match exactly")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review lookup table to understand pattern → divergence mapping")
    print("2. Express divergence as explicit function of bit patterns")
    print("3. Derive how Rule 30 update preserves this function")
    print("4. Write formal proof: D(T(s)) = D(s)")
    print()


if __name__ == '__main__':
    main()

