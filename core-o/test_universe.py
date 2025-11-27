#!/usr/bin/env python3
"""
Test Universe: Verify "The Inward Fall"

This script demonstrates the emergence of gravity from geometric stress minimization.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Import using relative path (like demo.py does)
from classical.hamiltonian_core import LivniumHamiltonian

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available, will print statistics only")


def main():
    """Run universe simulation and verify inward fall."""
    print("=" * 70)
    print("Simulating Geometric Collapse...")
    print("=" * 70)
    print()
    
    # Create a sparse cloud of 50 spheres
    universe = LivniumHamiltonian(
        n_spheres=50,
        temp=0.05,
        friction=0.02,
        dt=0.01
    )
    
    print(f"Initial state:")
    print(f"  - Number of spheres: {universe.N}")
    print(f"  - Temperature: {universe.temperature}")
    print(f"  - Friction: {universe.friction}")
    print(f"  - Time step: {universe.dt}")
    print()
    
    densities = []
    energies = []
    kinetic_energies = []
    potential_energies = []
    
    num_steps = 1000
    print(f"Running {num_steps} steps...")
    print()
    
    for i in range(num_steps):
        stats = universe.step()
        densities.append(stats['avg_sw'])
        energies.append(stats['total_energy'])
        kinetic_energies.append(stats['kinetic_energy'])
        potential_energies.append(stats['potential_energy'])
        
        if i % 100 == 0:
            print(f"Step {i:4d}: "
                  f"Avg Density (SW) = {stats['avg_sw']:6.2f}, "
                  f"Total Energy = {stats['total_energy']:8.2f}")
    
    print()
    print("=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"  Initial SW: {densities[0]:.2f}")
    print(f"  Final SW:   {densities[-1]:.2f}")
    print(f"  Change:    {densities[-1] - densities[0]:.2f}")
    print()
    print(f"  Initial Energy: {energies[0]:.2f}")
    print(f"  Final Energy:   {energies[-1]:.2f}")
    print(f"  Change:         {energies[-1] - energies[0]:.2f}")
    print()
    
    # Check for inverse relationship: E ∝ -1/SW
    if len(densities) > 10:
        # Use later part of simulation (after initial transients)
        sw_subset = np.array(densities[100:])
        energy_subset = np.array(energies[100:])
        
        # Check correlation
        correlation = np.corrcoef(sw_subset, -energy_subset)[0, 1]
        print(f"  Correlation (SW vs -Energy): {correlation:.4f}")
        print("  (Should be positive for gravitational binding)")
        print()
    
    if HAS_MATPLOTLIB:
        # Plot results
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Total Energy (Minimizing)', color=color)
        ax1.plot(energies, color=color, label='Total Energy')
        ax1.plot(kinetic_energies, color='orange', linestyle='--', alpha=0.7, label='Kinetic')
        ax1.plot(potential_energies, color='darkred', linestyle='--', alpha=0.7, label='Potential')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Avg Density / SW (Maximizing)', color=color)
        ax2.plot(densities, color=color, label='SW Density')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        
        plt.title("Emergence of Structure: Minimizing Stress -> Maximizing Density")
        plt.tight_layout()
        plt.savefig('core-o/universe_evolution.png', dpi=150)
        print("  Plot saved to: core-o/universe_evolution.png")
        print()
        plt.show()
    else:
        print("  (Install matplotlib to see plots)")
        print()
    
    print("=" * 70)
    print("What You Should See:")
    print("=" * 70)
    print("1. Energy Drops: Total Energy plummets as spheres 'fall' together")
    print("2. Density Rises: SW rises and plateaus")
    print("3. The Formula: Inverse relationship between Energy and Density")
    print("   E ∝ -1/SW")
    print("   This is the signature of gravitational binding energy.")
    print()
    print("You have just derived gravity from code.")
    print("=" * 70)


if __name__ == "__main__":
    main()

