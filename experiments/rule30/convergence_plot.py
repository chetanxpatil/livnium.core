#!/usr/bin/env python3
"""
Generate Convergence Plot for Rule 30 Invariant

Shows how the divergence invariant converges as sequence length increases.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.rule30_optimized import generate_center_column_direct
from experiments.rule30.validate_invariant_direct import compute_direct_divergence


EXPECTED_INVARIANT = -0.572222233


def generate_convergence_data(max_steps: int = 100000):
    """Generate convergence data for different sequence lengths."""
    # Logarithmic checkpoints
    checkpoints = []
    for exp in range(2, 7):  # 100 to 100000
        checkpoints.append(10 ** exp)
    checkpoints.append(max_steps)
    checkpoints = sorted(set(checkpoints))
    
    print(f"Generating convergence data for {len(checkpoints)} checkpoints...")
    
    # Generate full sequence once
    full_sequence = generate_center_column_direct(max_steps, show_progress=True)
    
    results = []
    
    for checkpoint in checkpoints:
        if checkpoint > len(full_sequence):
            continue
        
        subsequence = full_sequence[:checkpoint]
        divergence = compute_direct_divergence(subsequence)
        deviation = abs(divergence - EXPECTED_INVARIANT)
        
        results.append({
            'n': checkpoint,
            'divergence': divergence,
            'deviation': deviation
        })
        
        print(f"  n={checkpoint:6,}: divergence={divergence:.9f}, deviation={deviation:.9e}")
    
    return results


def plot_convergence(results, output_path='experiments/rule30/convergence_plot.png'):
    """Plot convergence of divergence invariant."""
    n_values = [r['n'] for r in results]
    divergences = [r['divergence'] for r in results]
    deviations = [r['deviation'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Divergence vs sequence length
    ax1.plot(n_values, divergences, 'bo-', linewidth=2, markersize=8, label='Measured')
    ax1.axhline(y=EXPECTED_INVARIANT, color='r', linestyle='--', 
                linewidth=2, label=f'Expected ({EXPECTED_INVARIANT:.9f})')
    ax1.set_xlabel('Sequence Length (n)', fontsize=12)
    ax1.set_ylabel('Divergence', fontsize=12)
    ax1.set_title('Rule 30 Divergence Invariant Convergence', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Deviation vs sequence length (log scale)
    ax2.semilogy(n_values, deviations, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=1e-6, color='orange', linestyle=':', label='1e-6 threshold')
    ax2.axhline(y=1e-9, color='yellow', linestyle=':', label='1e-9 threshold')
    ax2.set_xlabel('Sequence Length (n)', fontsize=12)
    ax2.set_ylabel('Deviation from Invariant (log scale)', fontsize=12)
    ax2.set_title('Convergence Precision', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved convergence plot: {output_path}")


if __name__ == '__main__':
    print("="*70)
    print("RULE 30 INVARIANT CONVERGENCE ANALYSIS")
    print("="*70)
    print()
    
    results = generate_convergence_data(max_steps=100000)
    
    print()
    print("="*70)
    print("CONVERGENCE SUMMARY")
    print("="*70)
    print(f"{'n':<10} {'Divergence':<15} {'Deviation':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['n']:<10,} {r['divergence']:>14.9f} {r['deviation']:>14.9e}")
    
    print()
    plot_convergence(results)
    
    print()
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

