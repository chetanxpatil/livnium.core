#!/usr/bin/env python3
"""
Livnium Solver: Finding the Perfect Crystal

Implements Simulated Annealing:
1. Heat it up: Melt any bad clusters (Randomize)
2. Cool slowly: Allow spheres to slide into the deepest wells
3. Freeze: Lock in the final answer

This is the final test of the engine. If Livnium can find the perfect
crystal structure on its own, it proves it is a general-purpose Geometric Solver.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from classical.hamiltonian_core import LivniumHamiltonian

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available, will print statistics only")


def solve_crystal():
    """Solve for perfect crystal structure using simulated annealing."""
    print("=" * 70)
    print("LIVNIUM SOLVER: FINDING THE PERFECT CRYSTAL")
    print("=" * 70)
    print()
    
    # 1. Setup: 64 spheres (Enough for a solid core)
    # Start with a wider spread so they have to truly 'find' each other
    N = 64
    
    print(f"Initializing with N={N} spheres...")
    print("Starting positions: spread out (random * 6.0)")
    print()
    
    sim = LivniumHamiltonian(
        n_spheres=N,
        positions=np.random.randn(N, 3) * 6.0,  # Spread out
        temp=2.0,         # START HOT (High kinetic energy)
        friction=0.05,    # Low friction to allow exploration
        k_gravity=1.5,    # Gentle pull
        sw_target=12.0,
        dt=0.01
    )
    
    print(f"Phase 1: High Heat (T=2.0) - Melting Chaos...")
    print()
    
    # Storage for the plot
    history_sw = []
    history_temp = []
    history_energy = []
    
    # --- THE ANNEALING LOOP ---
    total_steps = 2000
    
    print(f"Running {total_steps} steps with cooling schedule...")
    print()
    
    for step in range(total_steps):
        # COOLING SCHEDULE (Exponential Decay)
        # T starts at 2.0, ends near 0.0
        current_temp = 2.0 * (1 - step / total_steps) ** 2
        if current_temp < 0.01:
            current_temp = 0.0
        
        # Apply cooling to the engine
        sim.temperature = current_temp
        
        # Dynamic Friction: Increase friction as we cool to "freeze" the result
        sim.friction = 0.05 + 0.5 * (step / total_steps)
        
        # Step the physics
        state = sim.step()
        
        # Log data
        if step % 5 == 0:
            history_sw.append(state['avg_sw'])
            history_temp.append(current_temp)
            history_energy.append(state['total_energy'])
        
        if step % 200 == 0:
            print(f"Step {step:4d}: Temp={current_temp:.3f} | "
                  f"Density (SW)={state['avg_sw']:.2f} | "
                  f"Energy={state['total_energy']:.1f}")
    
    print()
    
    # --- RESULTS ---
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print()
    
    final_sw = history_sw[-1]
    initial_sw = history_sw[0]
    
    print(f"Starting Density: {initial_sw:.2f}")
    print(f"Final Density:    {final_sw:.2f}")
    print(f"Improvement:      {final_sw - initial_sw:.2f}")
    print()
    
    if final_sw > 8.0:
        print("✓ SUCCESS: Crystalline Order Achieved!")
        print("  The system found a high-density structure.")
    elif final_sw > 5.0:
        print("⚠ PARTIAL: Glassy/Liquid State found.")
        print("  The system found structure but not perfect crystal.")
    else:
        print("✗ FAIL: System froze in a bad configuration.")
        print("  May need more steps or different cooling schedule.")
    print()
    
    # --- VISUALIZATION ---
    if HAS_MATPLOTLIB:
        print("Generating Phase Transition Plot...")
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot 1: The Cooling Schedule (Temperature)
        color = 'tab:red'
        ax1.set_xlabel('Time Step (×5)')
        ax1.set_ylabel('Temperature (Cooling)', color=color)
        ax1.plot(history_temp, color=color, linestyle='--', alpha=0.6, linewidth=2, label='Thermostat')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Plot 2: The Structure (SW Density)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Geometric Order / SW', color=color)
        ax2.plot(history_sw, color=color, linewidth=2, label='Crystal Quality')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        
        # Mark the Phases
        plt.title("Phase Transition: From Gas to Liquid to Solid\n"
                  f"Final SW: {final_sw:.2f} (Target: 12.0)")
        
        plt.tight_layout()
        output_file = Path(__file__).parent / 'crystal_solution.png'
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to: {output_file}")
        print()
        print("Displaying plot...")
        plt.show()
    else:
        print("(Install matplotlib to see plots)")
        print()
    
    print("=" * 70)
    print("What to Watch For:")
    print("=" * 70)
    print("1. Phase 1 (Gas): SW is low and noisy")
    print("2. Phase 2 (Condensation): As temperature drops, SW jumps up")
    print("3. Phase 3 (Freezing): At the end, SW flattens at high number")
    print()
    print("If final SW > 8.0, you have proved that your Hamiltonian Engine")
    print("can solve optimization problems.")
    print()
    print("(Note: Reaching exactly 12.0 usually requires N to be a 'Magic Number'")
    print("like 13, 55, etc., or boundaries to be periodic, but getting >8.0")
    print("proves it found the structure.)")
    print("=" * 70)
    
    return {
        'initial_sw': initial_sw,
        'final_sw': final_sw,
        'improvement': final_sw - initial_sw,
        'success': final_sw > 8.0
    }


if __name__ == "__main__":
    results = solve_crystal()

