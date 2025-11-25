#!/usr/bin/env python3
"""
Test Divergence V2 Invariants

Tests if Divergence V2 (structure-aware) produces invariants for Rule 30.
Tests across multiple sequence lengths to check for stability.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.rule30_optimized import generate_center_column_direct
from experiments.rule30.divergence_v2 import (
    divergence_v2_transition_based,
    divergence_v2_neighborhood,
    divergence_v2_pattern_weighted
)


def test_divergence_v2_invariant(
    divergence_func,
    n_steps_list: list,
    show_progress: bool = True
) -> dict:
    """
    Test if divergence V2 produces invariant values across sequence lengths.
    
    Args:
        divergence_func: Divergence function to test
        n_steps_list: List of sequence lengths to test
        show_progress: Show progress indicator
        
    Returns:
        Dict with results: values, mean, std, is_invariant
    """
    print("="*70)
    print("DIVERGENCE V2 INVARIANT TEST")
    print("="*70)
    print(f"Testing {divergence_func.__name__}")
    print(f"Sequence lengths: {n_steps_list}")
    print()
    
    values = []
    
    for n_steps in n_steps_list:
        if show_progress:
            print(f"Generating Rule 30 ({n_steps:,} steps)...", end=" ", flush=True)
        
        # Generate Rule 30 center column
        sequence = generate_center_column_direct(n_steps, show_progress=False)
        
        # Compute divergence V2
        div_value = divergence_func(sequence)
        values.append(div_value)
        
        if show_progress:
            print(f"divergence = {div_value:.9f}")
    
    values = np.array(values)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    range_val = max_val - min_val
    
    # Check if invariant (low standard deviation)
    tolerance = 1e-3  # Allow 0.1% variation
    is_invariant = std_val < tolerance
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Mean divergence:  {mean_val:.9f}")
    print(f"Std deviation:    {std_val:.9e}")
    print(f"Range:            [{min_val:.9f}, {max_val:.9f}]")
    print(f"Spread:           {range_val:.9e}")
    print()
    
    if is_invariant:
        print("✓✓✓ INVARIANT DETECTED!")
        print(f"Divergence is stable across sequence lengths (std < {tolerance})")
    else:
        print("⚠ Divergence varies across sequence lengths")
        print("This may indicate:")
        print("  - Divergence depends on sequence length")
        print("  - No invariant exists for this divergence function")
        print("  - Need longer sequences to converge")
    
    return {
        'values': values.tolist(),
        'n_steps': n_steps_list,
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'is_invariant': is_invariant
    }


def plot_divergence_v2_stability(results: dict, divergence_name: str, output_path: str = None):
    """
    Plot divergence V2 values across sequence lengths.
    
    Args:
        results: Results dict from test_divergence_v2_invariant
        divergence_name: Name of divergence function
        output_path: Optional path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    n_steps = results['n_steps']
    values = results['values']
    mean_val = results['mean']
    
    # Plot 1: Divergence vs sequence length
    ax1.plot(n_steps, values, 'bo-', linewidth=2, markersize=8, label='Measured')
    ax1.axhline(y=mean_val, color='r', linestyle='--', 
                linewidth=2, label=f'Mean ({mean_val:.9f})')
    ax1.fill_between(n_steps, 
                     [mean_val - results['std']] * len(n_steps),
                     [mean_val + results['std']] * len(n_steps),
                     alpha=0.2, color='gray', label=f'±1 std ({results["std"]:.9e})')
    ax1.set_xlabel('Sequence Length (n)', fontsize=12)
    ax1.set_ylabel('Divergence V2', fontsize=12)
    ax1.set_title(f'{divergence_name} - Stability Across Sequence Lengths', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Deviation from mean (log scale)
    deviations = np.abs(np.array(values) - mean_val)
    ax2.semilogy(n_steps, deviations, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=results['std'], color='orange', linestyle=':', 
                label=f'Std dev ({results["std"]:.9e})')
    ax2.set_xlabel('Sequence Length (n)', fontsize=12)
    ax2.set_ylabel('Deviation from Mean (log scale)', fontsize=12)
    ax2.set_title('Convergence Precision', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Test Divergence V2 invariants for Rule 30",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--divergence',
        type=str,
        choices=['transition', 'neighborhood', 'pattern-weighted', 'all'],
        default='neighborhood',
        help='Divergence function to test (default: neighborhood)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[1000, 5000, 10000, 20000, 50000],
        help='Sequence lengths to test (default: 1000 5000 10000 20000 50000)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate stability plot'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/rule30/results',
        help='Output directory for plots (default: experiments/rule30/results)'
    )
    
    args = parser.parse_args()
    
    # Map divergence names to functions
    divergence_funcs = {
        'transition': divergence_v2_transition_based,
        'neighborhood': divergence_v2_neighborhood,
        'pattern-weighted': divergence_v2_pattern_weighted
    }
    
    # Test selected divergence function(s)
    if args.divergence == 'all':
        funcs_to_test = divergence_funcs.items()
    else:
        funcs_to_test = [(args.divergence, divergence_funcs[args.divergence])]
    
    all_results = {}
    
    for div_name, div_func in funcs_to_test:
        print()
        results = test_divergence_v2_invariant(div_func, args.steps)
        all_results[div_name] = results
        
        # Plot if requested
        if args.plot:
            output_path = f"{args.output_dir}/divergence_v2_{div_name}_stability.png"
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            plot_divergence_v2_stability(results, div_name.replace('-', ' ').title(), output_path)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for div_name, results in all_results.items():
        status = "✓ INVARIANT" if results['is_invariant'] else "✗ VARIES"
        print(f"\n{div_name.replace('-', ' ').title()}:")
        print(f"  Status: {status}")
        print(f"  Mean:   {results['mean']:.9f}")
        print(f"  Std:    {results['std']:.9e}")
    
    print()


if __name__ == '__main__':
    main()

