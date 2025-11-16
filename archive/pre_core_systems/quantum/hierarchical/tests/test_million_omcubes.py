#!/usr/bin/env python3
"""
Million Omcube Tests

Tests LIVNIUM's geometric symbolic logic at massive scale (1M+ base states).

Tests:
1. Scaling & Physics - Memory/time curves at massive N
2. Invariant Hunting - Geometric norms that stay constant
3. Attractor Detection - Collatz-style patterns in geometry space
4. Semantic Decoherence - Phase transitions at scale
5. Hierarchy Depth vs Stability - Does deeper hierarchy stabilize?
6. Collision Tests - Identity preservation at million scale
"""

import sys
import time
import tracemalloc
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum.hierarchical.geometry.hierarchy_v2 import HierarchyV2System, OperationType


def test_scaling_curve():
    """
    Test 1: Scaling & Physics-of-Engine
    
    Measure memory and time at: 10k, 50k, 100k, 250k, 500k, 1M omcubes
    """
    print("=" * 70)
    print("TEST 1: Scaling & Physics-of-Engine")
    print("=" * 70)
    
    test_sizes = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
    results = []
    
    print(f"\nTesting scaling at: {[f'{n:,}' for n in test_sizes]} omcubes")
    print(f"\n{'N':<12} {'Memory (MB)':<15} {'Time (s)':<12} {'Bytes/Qubit':<15} {'Ops/sec':<12}")
    print("-" * 70)
    
    for num_omcubes in test_sizes:
        tracemalloc.start()
        start_time = time.time()
        
        try:
            # Create system
            system = HierarchyV2System(base_dimension=3, num_levels=10)
            
            # Allocate omcubes
            alloc_start = time.time()
            for i in range(num_omcubes):
                x = (i % 100) * 0.01
                y = ((i // 100) % 100) * 0.01
                z = (i // 10000) * 0.01
                system.add_base_state((x, y, z))
            alloc_time = time.time() - alloc_start
            
            # Apply fixed batch of operations
            ops_start = time.time()
            for level in [1, 2, 3]:
                system.register_operation(
                    OperationType.ROTATION, level=level,
                    parameters={'angle': 0.1 * level},
                    description=f'Rotation at level {level}',
                    propagates_down=True
                )
            ops_time = time.time() - ops_start
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            total_time = time.time() - start_time
            
            memory_mb = peak / 1024 / 1024
            bytes_per_qubit = (peak / num_omcubes) if num_omcubes > 0 else 0
            ops_per_sec = num_omcubes / alloc_time if alloc_time > 0 else 0
            
            results.append({
                'n': num_omcubes,
                'memory_mb': memory_mb,
                'time_s': total_time,
                'alloc_time': alloc_time,
                'ops_time': ops_time,
                'bytes_per_qubit': bytes_per_qubit,
                'ops_per_sec': ops_per_sec
            })
            
            print(f"{num_omcubes:<12,} {memory_mb:<15.2f} {total_time:<12.3f} {bytes_per_qubit:<15.0f} {ops_per_sec:<12,.0f}")
            
        except Exception as e:
            tracemalloc.stop()
            print(f"{num_omcubes:<12,} FAILED: {e}")
            break
    
    # Analysis
    print("\n" + "=" * 70)
    print("Scaling Analysis:")
    print("=" * 70)
    
    if len(results) >= 2:
        # Check linearity
        first = results[0]
        last = results[-1]
        
        n_ratio = last['n'] / first['n']
        memory_ratio = last['memory_mb'] / first['memory_mb']
        
        print(f"  N increased by: {n_ratio:.1f}x")
        print(f"  Memory increased by: {memory_ratio:.1f}x")
        print(f"  Scaling factor: {memory_ratio / n_ratio:.3f}")
        
        if abs(memory_ratio / n_ratio - 1.0) < 0.2:
            print(f"  ✅ LINEAR SCALING CONFIRMED")
        else:
            print(f"  ⚠️  Non-linear scaling detected")
    
    return results


def test_geometric_invariants():
    """
    Test 2: Invariant Hunting at Scale
    
    Track geometric norms/energy before and after operations.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Geometric Invariant Hunting")
    print("=" * 70)
    
    num_omcubes = 100_000  # Start with 100k for invariant test
    
    print(f"\nInitializing {num_omcubes:,} omcubes...")
    system = HierarchyV2System(base_dimension=3, num_levels=10)
    
    # Initialize with different states
    energies_before = []
    for i in range(num_omcubes):
        # Random geometric state
        x = np.random.normal(0, 1.0)
        y = np.random.normal(0, 1.0)
        z = np.random.normal(0, 1.0)
        state = system.add_base_state((x, y, z))
        
        # Define "energy" = geometric norm
        energy = np.sqrt(x**2 + y**2 + z**2)
        energies_before.append(energy)
    
    print(f"  Created {len(system.base_geometry.states):,} states")
    
    # Calculate statistics before
    mean_before = np.mean(energies_before)
    var_before = np.var(energies_before)
    sum_before = np.sum(energies_before)
    
    print(f"\nBefore operations:")
    print(f"  Mean energy: {mean_before:.6f}")
    print(f"  Variance: {var_before:.6f}")
    print(f"  Sum: {sum_before:.6f}")
    
    # Apply operations
    print(f"\nApplying operations...")
    for level in [1, 3, 5]:
        system.register_operation(
            OperationType.ROTATION, level=level,
            parameters={'angle': 0.1},
            description=f'Rotation at level {level}',
            propagates_down=True
        )
    
    # Calculate statistics after
    energies_after = []
    for state in system.base_geometry.states:
        coords = np.array(state.coordinates)
        energy = np.linalg.norm(coords)
        energies_after.append(energy)
    
    mean_after = np.mean(energies_after)
    var_after = np.var(energies_after)
    sum_after = np.sum(energies_after)
    
    print(f"\nAfter operations:")
    print(f"  Mean energy: {mean_after:.6f}")
    print(f"  Variance: {var_after:.6f}")
    print(f"  Sum: {sum_after:.6f}")
    
    # Check invariants
    print(f"\n" + "=" * 70)
    print("Invariant Analysis:")
    print("=" * 70)
    
    mean_change = abs(mean_after - mean_before) / mean_before if mean_before > 0 else 0
    var_change = abs(var_after - var_before) / var_before if var_before > 0 else 0
    sum_change = abs(sum_after - sum_before) / sum_before if sum_before > 0 else 0
    
    print(f"  Mean change: {mean_change * 100:.2f}%")
    print(f"  Variance change: {var_change * 100:.2f}%")
    print(f"  Sum change: {sum_change * 100:.2f}%")
    
    threshold = 0.01  # 1% threshold
    invariants = []
    
    if mean_change < threshold:
        invariants.append("Mean energy is invariant")
    if var_change < threshold:
        invariants.append("Variance is invariant")
    if sum_change < threshold:
        invariants.append("Sum is invariant")
    
    if invariants:
        print(f"\n  ✅ DISCOVERED INVARIANTS:")
        for inv in invariants:
            print(f"     - {inv}")
    else:
        print(f"\n  ⚠️  No strong invariants detected (may need different operations)")
    
    return {
        'mean_before': mean_before,
        'mean_after': mean_after,
        'var_before': var_before,
        'var_after': var_after,
        'invariants': invariants
    }


def test_attractor_detection():
    """
    Test 3: Attractor Detection (Collatz-style patterns)
    
    Find geometric attractors in operation-orbit space.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Attractor Detection")
    print("=" * 70)
    
    num_omcubes = 10_000  # Start smaller for attractor detection
    max_iterations = 100
    
    print(f"\nInitializing {num_omcubes:,} omcubes for attractor detection...")
    system = HierarchyV2System(base_dimension=3, num_levels=10)
    
    # Initialize omcubes
    initial_states = []
    for i in range(num_omcubes):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        state = system.add_base_state((x, y, z))
        initial_states.append((x, y, z))
    
    print(f"  Created {len(system.base_geometry.states):,} initial states")
    
    # Define fixed update rule
    def update_rule(system, iteration):
        """Fixed operation sequence."""
        level = (iteration % 3) + 1
        system.register_operation(
            OperationType.ROTATION, level=level,
            parameters={'angle': 0.1 * level},
            description=f'Iteration {iteration} rotation',
            propagates_down=True
        )
    
    # Track orbits
    print(f"\nRunning {max_iterations} iterations of update rule...")
    attractors = defaultdict(list)
    orbit_lengths = []
    stable_count = 0
    cycle_count = 0
    
    # Simplified: Track state signatures
    state_signatures = []
    
    for iteration in range(max_iterations):
        # Apply update rule
        update_rule(system, iteration)
        
        # Calculate state signature (simplified: sum of coordinates)
        if iteration % 10 == 0:  # Sample every 10 iterations
            signature = sum(
                sum(state.coordinates) 
                for state in system.base_geometry.states[:100]  # Sample
            )
            state_signatures.append(signature)
    
    # Detect cycles
    print(f"\nAnalyzing attractors...")
    
    # Look for repeating patterns in signatures
    if len(state_signatures) >= 3:
        # Check for cycles
        for cycle_len in [2, 3, 4, 5]:
            if len(state_signatures) >= cycle_len * 2:
                # Check if last cycle_len match previous cycle_len
                last = state_signatures[-cycle_len:]
                prev = state_signatures[-cycle_len*2:-cycle_len]
                
                if np.allclose(last, prev, rtol=0.01):
                    cycle_count += 1
                    print(f"  ✅ Detected cycle of length {cycle_len}")
                    break
    
    # Check stability (low variance in signatures)
    if len(state_signatures) >= 10:
        recent = state_signatures[-10:]
        variance = np.var(recent)
        if variance < 0.01:
            stable_count = num_omcubes
            print(f"  ✅ System stabilized (variance: {variance:.6f})")
    
    print(f"\n  Total omcubes: {num_omcubes:,}")
    print(f"  Stable: {stable_count:,}")
    print(f"  Cycles detected: {cycle_count}")
    
    return {
        'num_omcubes': num_omcubes,
        'stable': stable_count,
        'cycles': cycle_count,
        'signatures': state_signatures
    }


def test_semantic_decoherence():
    """
    Test 4: Semantic Decoherence Phase Transition
    
    Measure decoherence as a function of operation load.
    Find critical load beyond which decoherence rapidly spikes.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Semantic Decoherence Phase Transition")
    print("=" * 70)
    
    num_omcubes = 50_000
    operation_loads = [10, 50, 100, 200, 500, 1000]  # Number of operations
    
    print(f"\nTesting with {num_omcubes:,} omcubes")
    print(f"Operation loads: {operation_loads}")
    
    # Define semantic zones
    def classify_zone(coords, amplitude):
        """Classify state into semantic zone."""
        coords_array = np.array(coords)
        distance = np.linalg.norm(coords_array)
        amp_mag = abs(amplitude)
        
        # Zone A: Stable semantic zone (low distance, moderate amplitude)
        if distance < 0.5 and 0.5 < amp_mag < 2.0:
            return 'A'  # Stable
        # Zone B: Borderline (medium distance or amplitude)
        elif distance < 1.0 or 0.2 < amp_mag < 5.0:
            return 'B'  # Borderline
        # Zone C: Fully decohered (high distance or extreme amplitude)
        else:
            return 'C'  # Decohered
    
    results = []
    
    for load in operation_loads:
        print(f"\n  Testing load: {load} operations...")
        
        system = HierarchyV2System(base_dimension=3, num_levels=10)
        
        # Initialize omcubes in stable zone
        for i in range(num_omcubes):
            x = np.random.uniform(-0.3, 0.3)
            y = np.random.uniform(-0.3, 0.3)
            z = np.random.uniform(-0.3, 0.3)
            system.add_base_state((x, y, z), amplitude=1.0+0j)
        
        # Classify initial zones
        initial_zones = {'A': 0, 'B': 0, 'C': 0}
        for state in system.base_geometry.states:
            zone = classify_zone(state.coordinates, state.amplitude)
            initial_zones[zone] += 1
        
        # Apply operations (load)
        for op_num in range(load):
            level = (op_num % 5) + 1
            intensity = 0.1 + (op_num / load) * 0.5  # Increasing intensity
            
            system.register_operation(
                OperationType.ROTATION, level=level,
                parameters={'angle': intensity},
                description=f'Load {load}, op {op_num}',
                propagates_down=True
            )
        
        # Classify final zones
        final_zones = {'A': 0, 'B': 0, 'C': 0}
        for state in system.base_geometry.states:
            zone = classify_zone(state.coordinates, state.amplitude)
            final_zones[zone] += 1
        
        # Calculate decoherence metrics
        stable_fraction = final_zones['A'] / num_omcubes
        decohered_fraction = final_zones['C'] / num_omcubes
        
        results.append({
            'load': load,
            'stable_fraction': stable_fraction,
            'decohered_fraction': decohered_fraction,
            'borderline_fraction': final_zones['B'] / num_omcubes,
            'initial_stable': initial_zones['A'] / num_omcubes
        })
        
        print(f"    Stable (A): {stable_fraction*100:.2f}%")
        print(f"    Borderline (B): {results[-1]['borderline_fraction']*100:.2f}%")
        print(f"    Decohered (C): {decohered_fraction*100:.2f}%")
    
    # Analysis
    print(f"\n" + "=" * 70)
    print("Decoherence Phase Transition Analysis:")
    print("=" * 70)
    
    # Find critical load (where decoherence spikes)
    critical_load = None
    for i in range(1, len(results)):
        prev_decohered = results[i-1]['decohered_fraction']
        curr_decohered = results[i]['decohered_fraction']
        
        # Spike: decoherence increases by >20%
        if curr_decohered - prev_decohered > 0.20:
            critical_load = results[i]['load']
            print(f"  ⚠️  CRITICAL LOAD DETECTED: {critical_load} operations")
            print(f"     Decoherence spike: {prev_decohered*100:.2f}% → {curr_decohered*100:.2f}%")
            break
    
    if not critical_load:
        print(f"  ➡️  No sharp phase transition detected")
        print(f"     Decoherence increases gradually with load")
    
    # Plot data
    print(f"\n  Load vs Decoherence:")
    for result in results:
        print(f"    Load {result['load']:4d}: Stable={result['stable_fraction']*100:5.2f}%, "
              f"Decohered={result['decohered_fraction']*100:5.2f}%")
    
    return {
        'results': results,
        'critical_load': critical_load,
        'phase_transition': critical_load is not None
    }


def test_hierarchy_depth_stability():
    """
    Test 5: Hierarchy Depth vs Stability
    
    Does deeper hierarchy stabilize or destabilize?
    """
    print("\n" + "=" * 70)
    print("TEST 5: Hierarchy Depth vs Stability")
    print("=" * 70)
    
    num_omcubes = 50_000
    hierarchy_depths = [1, 3, 5, 10, 20]
    
    print(f"\nTesting with {num_omcubes:,} omcubes")
    print(f"Hierarchy depths: {hierarchy_depths}")
    
    results = []
    
    for num_levels in hierarchy_depths:
        print(f"\n  Testing {num_levels} levels...")
        
        system = HierarchyV2System(base_dimension=3, num_levels=num_levels)
        
        # Initialize omcubes
        for i in range(num_omcubes):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(-1, 1)
            system.add_base_state((x, y, z))
        
        # Apply same operation script at different levels
        operations_applied = 0
        for level in range(1, min(num_levels, 5)):
            system.register_operation(
                OperationType.ROTATION, level=level,
                parameters={'angle': 0.1},
                description=f'Rotation at level {level}',
                propagates_down=True
            )
            operations_applied += 1
        
        # Measure stability (variance of coordinates)
        coords = [np.array(state.coordinates) for state in system.base_geometry.states]
        coord_array = np.array(coords)
        variance = np.var(coord_array)
        
        # Measure oscillation (check if states are changing)
        amplitudes = [abs(state.amplitude) for state in system.base_geometry.states]
        amplitude_variance = np.var(amplitudes)
        
        results.append({
            'num_levels': num_levels,
            'operations': operations_applied,
            'coordinate_variance': variance,
            'amplitude_variance': amplitude_variance,
            'stability_score': 1.0 / (1.0 + variance)  # Higher = more stable
        })
        
        print(f"    Operations: {operations_applied}")
        print(f"    Coordinate variance: {variance:.6f}")
        print(f"    Stability score: {results[-1]['stability_score']:.6f}")
    
    # Analysis
    print(f"\n" + "=" * 70)
    print("Stability Analysis:")
    print("=" * 70)
    
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        
        stability_change = last['stability_score'] / first['stability_score']
        
        print(f"  Stability change ({first['num_levels']} → {last['num_levels']} levels): {stability_change:.3f}x")
        
        if stability_change > 1.1:
            print(f"  ✅ DEEPER HIERARCHY INCREASES STABILITY")
            print(f"     (Shock absorber effect confirmed)")
        elif stability_change < 0.9:
            print(f"  ⚠️  DEEPER HIERARCHY DECREASES STABILITY")
            print(f"     (Amplification effect)")
        else:
            print(f"  ➡️  Hierarchy depth has minimal effect on stability")
    
    return results


def test_collision_detection():
    """
    Test 6: Collision/Identity Tests at Million Scale
    
    How often do different inputs end in same geometric state?
    """
    print("\n" + "=" * 70)
    print("TEST 6: Collision Detection")
    print("=" * 70)
    
    num_omcubes = 100_000  # Test with 100k
    
    print(f"\nInitializing {num_omcubes:,} omcubes with different inputs...")
    system = HierarchyV2System(base_dimension=3, num_levels=10)
    
    # Initialize with structured encodings
    state_hashes = {}
    collisions = []
    
    for i in range(num_omcubes):
        # Different input encoding
        x = np.sin(i * 0.001) * 0.5
        y = np.cos(i * 0.001) * 0.5
        z = (i % 1000) * 0.001
        state = system.add_base_state((x, y, z))
        
        # Create hash/signature
        coords = state.coordinates
        # Simple hash: rounded coordinates
        hash_key = tuple(round(c, 4) for c in coords)
        
        if hash_key in state_hashes:
            collisions.append({
                'current_index': i,
                'previous_index': state_hashes[hash_key],
                'coordinates': coords
            })
        else:
            state_hashes[hash_key] = i
    
    print(f"  Created {len(system.base_geometry.states):,} states")
    print(f"  Unique hashes: {len(state_hashes):,}")
    print(f"  Collisions: {len(collisions)}")
    
    # Apply operations
    print(f"\nApplying operations...")
    for level in [1, 3, 5]:
        system.register_operation(
            OperationType.ROTATION, level=level,
            parameters={'angle': 0.1},
            description=f'Rotation at level {level}',
            propagates_down=True
        )
    
    # Check collisions after
    state_hashes_after = {}
    collisions_after = []
    
    for i, state in enumerate(system.base_geometry.states):
        coords = state.coordinates
        hash_key = tuple(round(c, 4) for c in coords)
        
        if hash_key in state_hashes_after:
            collisions_after.append({
                'current_index': i,
                'previous_index': state_hashes_after[hash_key],
                'coordinates': coords
            })
        else:
            state_hashes_after[hash_key] = i
    
    print(f"\nAfter operations:")
    print(f"  Unique hashes: {len(state_hashes_after):,}")
    print(f"  Collisions: {len(collisions_after)}")
    
    # Analysis
    collision_rate_before = len(collisions) / num_omcubes if num_omcubes > 0 else 0
    collision_rate_after = len(collisions_after) / num_omcubes if num_omcubes > 0 else 0
    
    print(f"\n" + "=" * 70)
    print("Collision Analysis:")
    print("=" * 70)
    print(f"  Collision rate (before): {collision_rate_before * 100:.4f}%")
    print(f"  Collision rate (after): {collision_rate_after * 100:.4f}%")
    
    if collision_rate_after < 0.001:  # Less than 0.1%
        print(f"  ✅ LOW COLLISION RATE - Good for geometric hashing")
    elif collision_rate_after < 0.01:  # Less than 1%
        print(f"  ⚠️  MODERATE COLLISION RATE - Acceptable")
    else:
        print(f"  ❌ HIGH COLLISION RATE - May indicate information loss")
    
    return {
        'collisions_before': len(collisions),
        'collisions_after': len(collisions_after),
        'collision_rate_before': collision_rate_before,
        'collision_rate_after': collision_rate_after
    }


def main():
    """Run all million-omcube tests."""
    print("\n" + "=" * 70)
    print("MILLION OMCUBE TEST SUITE")
    print("=" * 70)
    print("\nTesting LIVNIUM's geometric symbolic logic at massive scale...")
    print("These tests validate the system is a real machine, not toy code.\n")
    
    all_results = {}
    
    # Test 1: Scaling
    try:
        all_results['scaling'] = test_scaling_curve()
    except Exception as e:
        print(f"  ❌ Scaling test failed: {e}")
    
    # Test 2: Invariants
    try:
        all_results['invariants'] = test_geometric_invariants()
    except Exception as e:
        print(f"  ❌ Invariant test failed: {e}")
    
    # Test 3: Attractors
    try:
        all_results['attractors'] = test_attractor_detection()
    except Exception as e:
        print(f"  ❌ Attractor test failed: {e}")
    
    # Test 4: Semantic decoherence
    try:
        all_results['decoherence'] = test_semantic_decoherence()
    except Exception as e:
        print(f"  ❌ Decoherence test failed: {e}")
    
    # Test 5: Hierarchy depth vs stability
    try:
        all_results['hierarchy_stability'] = test_hierarchy_depth_stability()
    except Exception as e:
        print(f"  ❌ Hierarchy stability test failed: {e}")
    
    # Test 6: Collisions
    try:
        all_results['collisions'] = test_collision_detection()
    except Exception as e:
        print(f"  ❌ Collision test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    
    for test_name, result in all_results.items():
        if result:
            print(f"\n  {test_name}: ✅ Completed")
        else:
            print(f"\n  {test_name}: ❌ Failed")
    
    print("\n" + "=" * 70)
    print("✅ Million-omcube test suite complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

