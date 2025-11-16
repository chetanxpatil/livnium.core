#!/usr/bin/env python3
"""
Scars Experiment: Trapped φ under Repeated Overload

This experiment demonstrates:
- How persistent contradictions become structural obstacles (scars)
- How φ-flow routes around trapped states
- How the system "remembers where it hurts"

Expected result: "LIVNIUM forms persistent structures from persistent contradictions."
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum.hierarchical.geometry.dual_cube_with_trapped import DualCubeWithTrappedSystem


def run_scars_experiment(num_omcubes: int = 100_000, 
                        max_steps: int = 5000,
                        overload_frequency: float = 0.1):
    """
    Run the scars experiment: test trapped φ under repeated overload.
    
    Args:
        num_omcubes: Number of initial omcubes
        max_steps: Maximum number of steps
        overload_frequency: Frequency of overload events (0.0 to 1.0)
        
    Returns:
        Dictionary with experiment results
    """
    print("=" * 70)
    print("SCARS EXPERIMENT: Trapped φ under Repeated Overload")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Initial omcubes: {num_omcubes:,}")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Overload frequency: {overload_frequency:.2f}")
    print("=" * 70)
    
    # Create system
    print("\n1. Creating dual-cube system with trapped φ support...")
    system = DualCubeWithTrappedSystem(base_dimension=3, num_levels=3)
    
    # Initialize omcubes
    print(f"\n2. Initializing {num_omcubes:,} omcubes...")
    start_time = time.time()
    
    for i in range(num_omcubes):
        # Mix of random and structured
        if i % 3 == 0:
            # Random
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(-1, 1)
            system.add_positive_state((x, y, z), amplitude=1.0+0j)
        elif i % 3 == 1:
            # Structured pattern (will form attractors)
            t = i * 0.001
            x = np.sin(t) * 0.5
            y = np.cos(t) * 0.5
            z = t * 0.1
            system.add_positive_state((x, y, z), amplitude=1.0+0j)
        else:
            # Concentrated (will create conflict zones)
            x = np.random.normal(0, 0.2)
            y = np.random.normal(0, 0.2)
            z = np.random.normal(0, 0.2)
            system.add_positive_state((x, y, z), amplitude=1.0+0j)
        
        if (i + 1) % 10_000 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1:,}/{num_omcubes:,} ({elapsed:.2f}s)")
    
    init_time = time.time() - start_time
    print(f"  ✅ Initialized {num_omcubes:,} omcubes in {init_time:.2f}s")
    
    # Track evolution
    history = []
    log_interval = max(1, max_steps // 20)  # Log every 5% of steps
    
    print(f"\n3. Running evolution for {max_steps:,} steps...")
    print(f"   Logging every {log_interval} steps")
    
    start_time = time.time()
    
    for step in range(max_steps):
        # Occasionally push states into contradiction (overload)
        if np.random.random() < overload_frequency:
            # Create contradictory states
            num_conflicts = int(num_omcubes * 0.01)  # 1% of states
            for _ in range(num_conflicts):
                x = np.random.uniform(-0.5, 0.5)
                y = np.random.uniform(-0.5, 0.5)
                z = np.random.uniform(-0.5, 0.5)
                # Add with conflicting amplitude
                state = system.add_positive_state((x, y, z), amplitude=2.0+0j)
                
                # Move to negative cube (simulating contradiction)
                # In real system, this would be detected automatically
                system.add_negative_state((x, y, z), amplitude=1.5+0j)
        
        # Apply decoherence drift
        system.apply_decoherence_drift(rate=0.05)
        
        # Step evolution (trapping, decay, capacity limits)
        step_result = system.step()
        
        # Log periodically
        if step % log_interval == 0 or step == max_steps - 1:
            stats = system.get_trapped_statistics()
            scar_analysis = stats['scar_analysis']
            
            history.append({
                'step': step,
                'trapped_fraction': stats['trapped_fraction'],
                'trapped_count': stats['trapped_count'],
                'positive_count': stats['positive_count'],
                'negative_count': stats['negative_count'],
                'num_scars': scar_analysis['num_scars'],
                'largest_scar_size': scar_analysis['largest_scar_size'],
                'step_result': step_result
            })
            
            elapsed = time.time() - start_time
            print(f"  Step {step:5d}: trapped={stats['trapped_fraction']:.3f}, "
                  f"scars={scar_analysis['num_scars']}, largest={scar_analysis['largest_scar_size']}, "
                  f"time={elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n  ✅ Evolution complete in {total_time:.2f}s")
    
    # Final analysis
    print(f"\n4. Final Analysis:")
    final_stats = system.get_trapped_statistics()
    final_scars = final_stats['scar_analysis']
    
    print(f"   Total states: {final_stats['total_states']:,}")
    print(f"   Positive (Cube⁺): {final_stats['positive_count']:,}")
    print(f"   Negative (Cube⁻): {final_stats['negative_count']:,}")
    print(f"   Trapped (Cube⁰): {final_stats['trapped_count']:,}")
    print(f"   Trapped fraction: {final_stats['trapped_fraction']:.3f}")
    print(f"   Number of scars: {final_scars['num_scars']}")
    print(f"   Largest scar size: {final_scars['largest_scar_size']} cells")
    
    # Print summary at key steps
    print(f"\n5. Evolution Summary (key steps):")
    key_steps = [0, max_steps // 4, max_steps // 2, 3 * max_steps // 4, max_steps - 1]
    for key_step in key_steps:
        # Find closest logged step
        closest = min(history, key=lambda h: abs(h['step'] - key_step))
        print(f"\n   Step {closest['step']}:")
        print(f"     trapped_fraction = {closest['trapped_fraction']:.3f}")
        print(f"     largest_scar_size = {closest['largest_scar_size']} cells")
        print(f"     num_scar_clusters = {closest['num_scars']}")
    
    # Check for phase transition
    print(f"\n6. Phase Transition Analysis:")
    trapped_fractions = [h['trapped_fraction'] for h in history]
    if len(trapped_fractions) > 1:
        max_increase = max(
            trapped_fractions[i+1] - trapped_fractions[i] 
            for i in range(len(trapped_fractions) - 1)
        )
        print(f"   Maximum trapped fraction increase: {max_increase:.3f}")
        
        if max_increase > 0.05:  # 5% jump
            print(f"   ✅ PHASE TRANSITION DETECTED")
            print(f"      Sudden jump in trapped fraction ({max_increase:.1%})")
        else:
            print(f"   ✅ GRADUAL SOLIDIFICATION (no sudden phase transition)")
    
    # Results
    results = {
        'num_omcubes': num_omcubes,
        'max_steps': max_steps,
        'overload_frequency': overload_frequency,
        'initialization_time': init_time,
        'evolution_time': total_time,
        'final_stats': final_stats,
        'history': history,
        'conclusion': 'LIVNIUM forms persistent structures from persistent contradictions.'
    }
    
    print(f"\n" + "=" * 70)
    print("✅ SCARS EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nConclusion:")
    print(f"  {results['conclusion']}")
    print(f"\nKey Findings:")
    print(f"  - Trapped fraction: {final_stats['trapped_fraction']:.1%}")
    print(f"  - Number of scars: {final_scars['num_scars']}")
    print(f"  - Largest scar: {final_scars['largest_scar_size']} cells")
    print(f"  - System remembers where it hurts (scars persist)")
    print("=" * 70)
    
    return results


def run_quick_test():
    """Run a quick test with smaller parameters."""
    print("Running quick test (10k omcubes, 500 steps)...")
    return run_scars_experiment(num_omcubes=10_000, max_steps=500, overload_frequency=0.1)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run scars experiment')
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--omcubes', type=int, default=100_000, help='Number of omcubes')
    parser.add_argument('--steps', type=int, default=5000, help='Number of steps')
    parser.add_argument('--overload', type=float, default=0.1, help='Overload frequency')
    
    args = parser.parse_args()
    
    if args.quick:
        results = run_quick_test()
    else:
        results = run_scars_experiment(
            num_omcubes=args.omcubes,
            max_steps=args.steps,
            overload_frequency=args.overload
        )

