#!/usr/bin/env python3
"""
Test Divergence Invariant Across Different CA Rules

Tests if other cellular automata rules exhibit similar divergence invariants.
This helps determine if the invariant is unique to Rule 30 or a general property.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.diagnostics import create_sequence_vectors, _compute_field_divergence


def generate_ca_center_column(rule_number: int, n_steps: int) -> list:
    """
    Generate center column for a given CA rule.
    
    Args:
        rule_number: CA rule number (0-255)
        n_steps: Number of steps to generate
        
    Returns:
        Center column as list of 0s and 1s
    """
    # Rule lookup table: 8-bit pattern -> new cell value
    rule_bits = [(rule_number >> i) & 1 for i in range(8)]
    
    # Initialize: single black cell in center
    max_width = n_steps * 2 + 10
    center_idx = max_width // 2
    current_row = [0] * max_width
    current_row[center_idx] = 1
    
    center_column = [1]  # First row center is 1
    
    for step in range(1, n_steps):
        new_row = [0] * max_width
        
        # Apply CA rule
        for i in range(1, max_width - 1):
            # Get 3-cell neighborhood
            left = current_row[i - 1]
            center = current_row[i]
            right = current_row[i + 1]
            
            # Convert to rule index (0-7)
            pattern = (left << 2) | (center << 1) | right
            new_cell = rule_bits[pattern]
            new_row[i] = new_cell
        
        # Extract center cell
        center_cell = new_row[center_idx]
        center_column.append(center_cell)
        
        # Trim row (simplified - just keep reasonable window)
        first_nonzero = next((i for i, x in enumerate(new_row) if x != 0), 0)
        last_nonzero = next((i for i in range(len(new_row) - 1, -1, -1) if new_row[i] != 0), len(new_row) - 1)
        padding = max(1, (n_steps - step) // 2)
        start_idx = max(0, first_nonzero - padding)
        end_idx = min(len(new_row), last_nonzero + padding + 1)
        
        # Ensure valid indices
        if start_idx < end_idx and end_idx <= len(new_row):
            trimmed_row = new_row[start_idx:end_idx]
            # Adjust center_idx for trimmed row
            center_idx = center_idx - start_idx
            current_row = trimmed_row
        else:
            # Fallback: keep current row if trimming fails
            current_row = new_row
    
    return center_column


def compute_divergence(sequence: list, window_size: int = 5) -> float:
    """Compute divergence directly from sequence."""
    vectors = create_sequence_vectors(sequence)
    
    if len(vectors) < window_size:
        return 0.0
    
    divergence_values = []
    for i in range(len(vectors) - window_size + 1):
        window_vecs = vectors[i:i + window_size]
        divergence = _compute_field_divergence(window_vecs, window_vecs)
        divergence_values.append(divergence)
    
    if not divergence_values:
        return 0.0
    
    return float(np.mean(divergence_values))


def test_rule(rule_number: int, n_steps: int = 10000) -> dict:
    """
    Test a CA rule for divergence invariant.
    
    Returns:
        Dict with rule number, divergence, and properties
    """
    print(f"Testing Rule {rule_number}...")
    
    try:
        # Generate center column
        center_column = generate_ca_center_column(rule_number, n_steps)
        
        if len(center_column) < 100:
            print(f"  ⚠ Sequence too short: {len(center_column)} bits")
            return None
        
        # Compute divergence
        divergence = compute_divergence(center_column)
        
        # Analyze sequence
        ones_count = sum(center_column)
        ones_ratio = ones_count / len(center_column)
        
        result = {
            'rule': rule_number,
            'divergence': divergence,
            'sequence_length': len(center_column),
            'ones_ratio': ones_ratio,
            'is_chaotic': abs(ones_ratio - 0.5) < 0.1  # Rough heuristic
        }
        
        print(f"  Divergence: {divergence:.9f}")
        print(f"  Ones ratio: {ones_ratio:.4f}")
        print(f"  Sequence length: {len(center_column):,}")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test divergence invariant across different CA rules",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--rules',
        type=int,
        nargs='+',
        default=[30, 90, 110, 150, 184],
        help='CA rules to test (default: 30 90 110 150 184)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=10000,
        help='Number of steps (default: 10000)'
    )
    
    parser.add_argument(
        '--all-rules',
        action='store_true',
        help='Test all 256 rules (slow!)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("CA RULE DIVERGENCE INVARIANT TEST")
    print("="*70)
    print(f"Testing rules: {args.rules if not args.all_rules else 'all 256'}")
    print(f"Steps: {args.steps:,}")
    print("="*70)
    print()
    
    if args.all_rules:
        rules_to_test = list(range(256))
    else:
        rules_to_test = args.rules
    
    results = []
    
    for rule_num in rules_to_test:
        result = test_rule(rule_num, args.steps)
        if result:
            results.append(result)
        print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Rule':<6} {'Divergence':<15} {'Ones Ratio':<12} {'Length':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['rule']:<6} {r['divergence']:>14.9f} {r['ones_ratio']:>11.4f} {r['sequence_length']:>9,}")
    
    print()
    
    # Check for similar invariants
    rule30_result = next((r for r in results if r['rule'] == 30), None)
    if rule30_result:
        rule30_div = rule30_result['divergence']
        print(f"Rule 30 divergence: {rule30_div:.9f}")
        print()
        print("Rules with similar divergence (±0.01):")
        similar = [r for r in results if abs(r['divergence'] - rule30_div) < 0.01 and r['rule'] != 30]
        if similar:
            for r in similar:
                print(f"  Rule {r['rule']}: {r['divergence']:.9f}")
        else:
            print("  None found - Rule 30's invariant appears unique")
    
    print()
    print("="*70)


if __name__ == '__main__':
    main()

