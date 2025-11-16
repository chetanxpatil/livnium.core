#!/usr/bin/env python3
"""
Dual-Cube Visualization Demo

Visual demonstration of the dual-cube semantic physics system:
- Positive cube (+cube): stable meanings, attractors
- Negative cube (-cube): contradictions, conflicts, decohered states
- Cross-cube dynamics: contradiction flow, decoherence drift
- Semantic decoherence as displacement
- Confusion diagnosis

This is a visual representation of a geometric semantic physics engine.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum.hierarchical.geometry.dual_cube_system import DualCubeSystem


def visualize_dual_cubes_3d(system: DualCubeSystem, title: str = "Dual-Cube System"):
    """
    Visualize positive and negative cubes in 3D space.
    
    Shows:
    - Positive cube states (blue) = stable meanings
    - Negative cube states (red) = contradictions
    - Dual states (purple) = exist in both
    """
    fig = plt.figure(figsize=(16, 8))
    
    # Positive cube subplot
    ax1 = fig.add_subplot(121, projection='3d')
    
    pos_states = system.positive_cube.states
    if pos_states:
        pos_coords = np.array([s.coordinates for s in pos_states])
        pos_amplitudes = [abs(s.amplitude) for s in pos_states]
        
        # Color by amplitude (darker = stronger)
        ax1.scatter(pos_coords[:, 0], pos_coords[:, 1], pos_coords[:, 2],
                   c=pos_amplitudes, cmap='Blues', s=50, alpha=0.6, label='Positive States')
    
    ax1.set_title('Positive Cube (+cube)\nStable Meanings, Attractors', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    
    # Negative cube subplot
    ax2 = fig.add_subplot(122, projection='3d')
    
    neg_states = system.negative_cube.states
    if neg_states:
        neg_coords = np.array([s.coordinates for s in neg_states])
        neg_amplitudes = [abs(s.amplitude) for s in neg_states]
        
        ax2.scatter(neg_coords[:, 0], neg_coords[:, 1], neg_coords[:, 2],
                   c=neg_amplitudes, cmap='Reds', s=50, alpha=0.6, label='Negative States')
    
    ax2.set_title('Negative Cube (-cube)\nContradictions, Conflicts, Decohered', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_decoherence_evolution(system: DualCubeSystem, steps: int = 10):
    """
    Visualize decoherence evolution over time.
    
    Shows how mass/energy moves from +cube to -cube.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Track evolution
    iterations = []
    positive_energies = []
    negative_energies = []
    decoherence_fractions = []
    total_energies = []
    
    # Initial state
    decoherence = system.get_decoherence_measure()
    iterations.append(0)
    positive_energies.append(decoherence['positive_energy'])
    negative_energies.append(decoherence['negative_energy'])
    decoherence_fractions.append(decoherence['decoherence_fraction'])
    total_energies.append(decoherence['total_energy'])
    
    # Apply decoherence steps
    for step in range(1, steps + 1):
        system.apply_decoherence_drift(rate=0.1)
        decoherence = system.get_decoherence_measure()
        iterations.append(step)
        positive_energies.append(decoherence['positive_energy'])
        negative_energies.append(decoherence['negative_energy'])
        decoherence_fractions.append(decoherence['decoherence_fraction'])
        total_energies.append(decoherence['total_energy'])
    
    # Plot 1: Energy evolution
    ax1 = axes[0, 0]
    ax1.plot(iterations, positive_energies, 'b-', label='Positive Energy', linewidth=2)
    ax1.plot(iterations, negative_energies, 'r-', label='Negative Energy', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Evolution: +cube ↔ -cube', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Decoherence fraction
    ax2 = axes[0, 1]
    ax2.plot(iterations, decoherence_fractions, 'g-', linewidth=2, marker='o')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Decoherence Fraction')
    ax2.set_title('Decoherence as Displacement\n(% mass in -cube)', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Total energy (conservation)
    ax3 = axes[1, 0]
    ax3.plot(iterations, total_energies, 'k-', linewidth=2, marker='s')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Total Energy')
    ax3.set_title('Mass/Energy Conservation\n(should stay constant)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy ratio
    ax4 = axes[1, 1]
    ratios = [neg / pos if pos > 0 else 0 for neg, pos in zip(negative_energies, positive_energies)]
    ax4.plot(iterations, ratios, 'm-', linewidth=2, marker='^')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Negative / Positive Ratio')
    ax4.set_title('Semantic Balance\n(ratio of contradiction to meaning)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Semantic Decoherence Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_attractors(system: DualCubeSystem):
    """
    Visualize attractors in positive and negative cubes.
    
    Shows stable patterns (attractors) in each cube.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Positive cube attractors
    ax1 = axes[0]
    pos_states = system.positive_cube.states
    if pos_states:
        pos_coords = np.array([s.coordinates for s in pos_states])
        pos_amplitudes = [abs(s.amplitude) for s in pos_states]
        
        # Quantize to find clusters (attractors)
        quantized = []
        for coords in pos_coords:
            quantized.append(tuple(int(round(c * 10)) for c in coords))
        
        from collections import Counter
        clusters = Counter(quantized)
        
        # Plot with size proportional to cluster size
        sizes = [clusters[q] * 20 for q in quantized]
        ax1.scatter(pos_coords[:, 0], pos_coords[:, 1], s=sizes, c=pos_amplitudes,
                   cmap='Blues', alpha=0.6)
        
        # Highlight top attractors
        top_attractors = clusters.most_common(5)
        for sig, count in top_attractors:
            x = sig[0] / 10.0
            y = sig[1] / 10.0
            ax1.scatter([x], [y], s=500, c='yellow', marker='*', 
                       edgecolors='black', linewidths=2, label=f'Attractor ({count} states)')
    
    ax1.set_title('Positive Cube Attractors\n(Stable Semantic Patterns)', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Negative cube patterns
    ax2 = axes[1]
    neg_states = system.negative_cube.states
    if neg_states:
        neg_coords = np.array([s.coordinates for s in neg_states])
        neg_amplitudes = [abs(s.amplitude) for s in neg_states]
        
        # Quantize
        quantized = []
        for coords in neg_coords:
            quantized.append(tuple(int(round(c * 10)) for c in coords))
        
        from collections import Counter
        clusters = Counter(quantized)
        
        sizes = [clusters[q] * 20 for q in quantized]
        ax2.scatter(neg_coords[:, 0], neg_coords[:, 1], s=sizes, c=neg_amplitudes,
                   cmap='Reds', alpha=0.6)
        
        # Highlight top anti-attractors
        top_anti = clusters.most_common(5)
        for sig, count in top_anti:
            x = sig[0] / 10.0
            y = sig[1] / 10.0
            ax2.scatter([x], [y], s=500, c='cyan', marker='*',
                       edgecolors='black', linewidths=2, label=f'Anti-Attractor ({count} states)')
    
    ax2.set_title('Negative Cube Patterns\n(Contradiction Zones, Chaos)', fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Attractors: Stable Patterns vs Contradiction Zones', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_confusion_diagnosis(system: DualCubeSystem):
    """
    Visualize confusion diagnosis across the space.
    
    Shows which regions are "clear" (mostly +cube) vs "confused" (mostly -cube).
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Create grid of test points
    x_range = np.linspace(-1, 1, 20)
    y_range = np.linspace(-1, 1, 20)
    z_fixed = 0.0
    
    confusion_map = np.zeros((len(x_range), len(y_range)))
    
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            diagnosis = system.diagnose_confusion((x, y, z_fixed))
            confusion_map[i, j] = diagnosis['confusion_score']
    
    # Plot confusion map
    ax = fig.add_subplot(121)
    im = ax.imshow(confusion_map, cmap='RdYlGn', aspect='auto', origin='lower',
                   extent=[-1, 1, -1, 1], vmin=0, vmax=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Confusion Map\n(Red = Confused, Green = Clear)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Confusion Score')
    
    # Plot 3D view
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Sample points and color by confusion
    sample_points = []
    confusion_scores = []
    for i in range(0, len(x_range), 2):
        for j in range(0, len(y_range), 2):
            x = x_range[i]
            y = y_range[j]
            diagnosis = system.diagnose_confusion((x, y, z_fixed))
            sample_points.append([x, y, z_fixed])
            confusion_scores.append(diagnosis['confusion_score'])
    
    sample_points = np.array(sample_points)
    confusion_scores = np.array(confusion_scores)
    
    scatter = ax2.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
                         c=confusion_scores, cmap='RdYlGn', s=100, alpha=0.7)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Confusion Diagnosis', fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Confusion Score')
    
    plt.suptitle('Confusion Diagnosis: Center of Gravity Shift', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def demo_dual_cube_visualization():
    """
    Main visualization demo showing all aspects of dual-cube semantic physics.
    """
    print("=" * 70)
    print("DUAL-CUBE SEMANTIC PHYSICS VISUALIZATION")
    print("=" * 70)
    print("\nThis demonstrates:")
    print("  - Positive cube: stable meanings, attractors")
    print("  - Negative cube: contradictions, conflicts")
    print("  - Cross-cube dynamics: decoherence drift")
    print("  - Semantic decoherence as displacement")
    print("  - Confusion diagnosis")
    print("=" * 70)
    
    # Create system
    print("\n1. Creating dual-cube system...")
    system = DualCubeSystem(base_dimension=3, num_levels=3)
    
    # Add states to positive cube (stable meanings)
    print("2. Adding stable states to positive cube...")
    for i in range(200):
        # Mix of random and structured (structured will form attractors)
        if i % 3 == 0:
            x = np.random.uniform(-0.8, 0.8)
            y = np.random.uniform(-0.8, 0.8)
            z = np.random.uniform(-0.8, 0.8)
        else:
            # Structured pattern (attractor)
            t = i * 0.05
            x = np.sin(t) * 0.5
            y = np.cos(t) * 0.5
            z = t * 0.1
        system.add_positive_state((x, y, z), amplitude=1.0+0j)
    
    # Add some contradictory states
    print("3. Adding contradictory states...")
    for i in range(50):
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)
        z = np.random.uniform(-0.5, 0.5)
        state = system.add_positive_state((x, y, z), amplitude=2.0+0j)
        
        # Detect and move contradictions
        contradiction = system.detect_contradiction(state, system.positive_cube.states[:10])
        if contradiction > system.contradiction_threshold:
            system.move_to_negative_cube(state, contradiction)
    
    # Add some states directly to negative cube
    print("4. Adding states to negative cube...")
    for i in range(100):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        system.add_negative_state((x, y, z), amplitude=0.5+0j)
    
    # Visualization 1: Dual cubes in 3D
    print("\n5. Generating 3D visualization of dual cubes...")
    fig1 = visualize_dual_cubes_3d(system, "Dual-Cube Semantic Universe")
    plt.savefig('dual_cubes_3d.png', dpi=150, bbox_inches='tight')
    print("   Saved: dual_cubes_3d.png")
    
    # Visualization 2: Decoherence evolution
    print("6. Generating decoherence evolution visualization...")
    # Create fresh system for evolution
    system_evo = DualCubeSystem(base_dimension=3, num_levels=3)
    for i in range(500):
        x = np.random.normal(0, 0.5)
        y = np.random.normal(0, 0.5)
        z = np.random.normal(0, 0.5)
        system_evo.add_positive_state((x, y, z), amplitude=1.0+0j)
    
    fig2 = visualize_decoherence_evolution(system_evo, steps=15)
    plt.savefig('decoherence_evolution.png', dpi=150, bbox_inches='tight')
    print("   Saved: decoherence_evolution.png")
    
    # Visualization 3: Attractors
    print("7. Generating attractor visualization...")
    fig3 = visualize_attractors(system)
    plt.savefig('attractors.png', dpi=150, bbox_inches='tight')
    print("   Saved: attractors.png")
    
    # Visualization 4: Confusion diagnosis
    print("8. Generating confusion diagnosis visualization...")
    fig4 = visualize_confusion_diagnosis(system)
    plt.savefig('confusion_diagnosis.png', dpi=150, bbox_inches='tight')
    print("   Saved: confusion_diagnosis.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    
    summary = system.get_system_summary()
    print(f"\nSystem Summary:")
    print(f"  Positive cube: {summary['positive_cube']['states']} states, "
          f"energy={summary['positive_cube']['energy']:.3f}")
    print(f"  Negative cube: {summary['negative_cube']['states']} states, "
          f"energy={summary['negative_cube']['energy']:.3f}")
    print(f"  Decoherence fraction: {summary['decoherence_fraction']:.3f}")
    
    decoherence = system.get_decoherence_measure()
    print(f"\nDecoherence Metrics:")
    print(f"  {decoherence['decoherence_fraction']*100:.1f}% of mass in negative cube")
    print(f"  This is semantic decoherence as displacement")
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION DEMO COMPLETE")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  - dual_cubes_3d.png: 3D view of dual cubes")
    print("  - decoherence_evolution.png: Evolution of decoherence over time")
    print("  - attractors.png: Attractors in positive and negative cubes")
    print("  - confusion_diagnosis.png: Confusion map across space")
    print("\nThis demonstrates a geometric semantic physics engine.")
    print("=" * 70)
    
    # Show plots
    plt.show()


if __name__ == '__main__':
    demo_dual_cube_visualization()

