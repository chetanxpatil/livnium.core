#!/usr/bin/env python3
"""
Radial Density Law Extraction: Discover Quantized Shells

This script implements COM-corrected Radial Density Scan to discover
discrete layering (shells) in the Livnium soft-matter core.

Hypothesis: Spheres should organize into discrete layers (shells) to
maximize packing efficiency, not just pile up randomly.

Predicted Signature:
- Monotonic decay from high density (center) to low density (surface)
- Oscillations (wiggles) corresponding to atomic shells
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


def discover_radial_structure():
    """Discover radial density structure and quantized shells."""
    print("=" * 70)
    print("LIVNIUM: RADIAL DENSITY LAW EXTRACTION")
    print("=" * 70)
    print()
    
    # 1. Configuration: N=200 for multi-shell visibility
    N = 200
    print(f"Annealing {N} spheres to form a deep core...")
    print()
    
    # Spawn widely to allow meaningful collapse
    initial_positions = np.random.randn(N, 3) * 6.0
    
    sim = LivniumHamiltonian(
        n_spheres=N,
        positions=initial_positions,
        # Standard Path A parameters (Soft Matter / Optimization Mode)
        temp=2.0,
        friction=0.05,
        k_repulsion=800.0,
        k_gravity=2.5,
        sw_target=12.0,
        dt=0.01
    )
    
    # 2. Annealing Process
    # We use a standard exponential cooling schedule
    steps = 2500
    print(f"Running {steps} steps of annealing...")
    print("(Exponential cooling: High Temp -> Freezing)")
    print()
    
    for i in range(steps):
        prog = i / steps
        
        # Cool down: High Temp -> Freezing
        sim.temperature = 2.0 * (1 - prog)**2
        
        # Friction ramp: Liquid -> Solid lock
        sim.friction = 0.05 + 0.2 * prog
        
        state = sim.step()
        
        if i % 500 == 0:
            print(f"  Step {i}: Temp={sim.temperature:.3f}, Friction={sim.friction:.3f}, "
                  f"Avg SW={state['avg_sw']:.2f}")
    
    # 3. Data Extraction (Drift Correction)
    print()
    print("Extracting final state...")
    state = sim.step()
    pos = state['positions']
    sw = state['sw']
    
    # Calculate Center of Mass (COM)
    center_of_mass = np.mean(pos, axis=0)
    print(f"Cluster COM detected at: {np.round(center_of_mass, 2)}")
    print()
    
    # Compute Corrected Radii
    # r = || pos - COM ||
    radial_dist = np.linalg.norm(pos - center_of_mass, axis=1)
    
    # Statistics
    print("Radial Structure Statistics:")
    print(f"  Min radius: {np.min(radial_dist):.2f}")
    print(f"  Max radius: {np.max(radial_dist):.2f}")
    print(f"  Mean radius: {np.mean(radial_dist):.2f}")
    print()
    print(f"  Min SW (crust): {np.min(sw):.2f}")
    print(f"  Max SW (core): {np.max(sw):.2f}")
    print(f"  Differentiation factor: {np.max(sw) / max(np.min(sw), 0.01):.1f}x")
    print()
    
    # 4. Law Discovery: Density vs Radius
    print("Plotting Radial Density Profile...")
    
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # A. Raw Data Points
        sc = ax.scatter(radial_dist, sw, c=sw, cmap='plasma', alpha=0.5, s=30, label='Raw Sphere Data')
        
        # B. Trend Line (The Equation of State)
        # Sort by radius to create a line
        sort_idx = np.argsort(radial_dist)
        r_sorted = radial_dist[sort_idx]
        sw_sorted = sw[sort_idx]
        
        # Apply Moving Average Smoothing to reveal shells
        # Window size determines how "smooth" the law looks vs how much detail is preserved
        window = 15
        if len(sw_sorted) > window:
            sw_smooth = np.convolve(sw_sorted, np.ones(window) / window, mode='valid')
            r_smooth = r_sorted[window - 1:]
            
            ax.plot(r_smooth, sw_smooth, 'k-', linewidth=2.5, label='Mean Density Profile')
            
            # Heuristic Shell Detection: Look for local peaks in the smoothed line
            # Find local maxima (potential shell boundaries)
            from scipy.signal import find_peaks
            try:
                peaks, properties = find_peaks(sw_smooth, height=np.mean(sw_smooth), distance=window//2)
                if len(peaks) > 0:
                    shell_radii = r_smooth[peaks]
                    shell_densities = sw_smooth[peaks]
                    ax.scatter(shell_radii, shell_densities, c='red', s=100, marker='x', 
                             label=f'Detected Shells ({len(peaks)})', zorder=5)
                    print(f"Detected {len(peaks)} potential shell boundaries")
                    for i, (r, d) in enumerate(zip(shell_radii, shell_densities)):
                        print(f"  Shell {i+1}: r={r:.2f}, SW={d:.2f}")
            except ImportError:
                # scipy not available, skip peak detection
                pass
        
        # C. Formatting
        ax.set_xlabel('Radial Distance (r) from Center of Mass')
        ax.set_ylabel('Geometric Density (SW)')
        ax.set_title(f'Livnium Equation of State (N={N})\n'
                     f'Differentiation: {np.max(sw) / max(np.min(sw), 0.01):.1f}x')
        
        # Reference lines for expected shell locations (approximate)
        # Shell 1 is usually around r=1.0 - 1.2 (tight core)
        # Shell 2 is usually around r=2.0 - 2.2
        ax.axvline(x=1.1, color='gray', linestyle=':', alpha=0.5, label='~Shell 1')
        ax.axvline(x=2.1, color='gray', linestyle=':', alpha=0.5, label='~Shell 2')
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(sc, label='SW Intensity')
        
        plt.tight_layout()
        output_file = Path(__file__).parent / 'radial_density_law.png'
        plt.savefig(output_file, dpi=150)
        print(f"Analysis complete. Saved to: {output_file}")
        print()
        print("Displaying plot...")
        plt.show()
    else:
        print("(Install matplotlib to see plots)")
        print()
    
    print("=" * 70)
    print("Interpretation:")
    print("=" * 70)
    print("• Raw Data Points: Individual spheres colored by density")
    print("• Black Line: Mean density profile (smoothed)")
    print("• Red X's: Detected shell boundaries (if scipy available)")
    print("• Gray Dotted Lines: Expected shell locations")
    print()
    print("If the black line wiggles or bumps at specific intervals,")
    print("you have discovered discrete layering of soft matter.")
    print("=" * 70)
    
    return {
        'radial_dist': radial_dist,
        'sw': sw,
        'center_of_mass': center_of_mass,
        'differentiation_factor': np.max(sw) / max(np.min(sw), 0.01)
    }


if __name__ == "__main__":
    results = discover_radial_structure()

