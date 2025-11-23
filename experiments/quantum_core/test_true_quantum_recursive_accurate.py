"""
Accurate TrueQuantumRegister capacity test in recursive geometry.

This test:
1. Tracks exactly which nodes are visited
2. Calculates theoretical max based on visited nodes (not full tree)
3. Shows accurate utilization percentages
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.classical import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.recursive import RecursiveGeometryEngine
from core.quantum.true_quantum_layer import TrueQuantumRegister
from core.quantum.quantum_gates import QuantumGates


def test_level_accurate(level, depth=0, max_qubits_per_register=10, 
                       visited_stats=None, max_children_per_level=None):
    """
    Test TrueQuantumRegister at a level and track accurate stats.
    
    Returns:
        (qubits_created, registers_created, visited_stats)
    """
    if visited_stats is None:
        visited_stats = {}
    
    geometry = level.geometry
    cells = list(geometry.lattice.keys())
    num_cells = len(cells)
    
    # Track visited
    if depth not in visited_stats:
        visited_stats[depth] = {'geometries': 0, 'total_cells': 0}
    visited_stats[depth]['geometries'] += 1
    visited_stats[depth]['total_cells'] += num_cells
    
    # Create TrueQuantumRegisters
    qubits_created = 0
    registers_created = 0
    
    for i in range(0, num_cells, max_qubits_per_register):
        size = min(max_qubits_per_register, num_cells - i)
        if size < 2:
            continue
        
        try:
            reg = TrueQuantumRegister(list(range(size)))
            H = QuantumGates.hadamard()
            reg.apply_gate(H, 0)
            if size >= 2:
                reg.apply_cnot(0, 1)
            reg.measure_qubit(0)
            
            qubits_created += size
            registers_created += 1
        except Exception as e:
            # Stop if we hit memory limit
            break
    
    # Recursively process children
    if depth < 2:  # Limit depth for testing
        children = list(level.children.values())
        
        # Apply sampling limit if specified
        if max_children_per_level and len(children) > max_children_per_level:
            import random
            children = random.sample(children, max_children_per_level)
        
        for child in children:
            cq, cr, visited_stats = test_level_accurate(
                child, depth + 1, max_qubits_per_register, 
                visited_stats, max_children_per_level
            )
            qubits_created += cq
            registers_created += cr
    
    return qubits_created, registers_created, visited_stats


def run_accurate_test():
    """Run accurate capacity test."""
    print("="*70)
    print("ACCURATE TrueQuantumRegister Capacity Test")
    print("="*70)
    print()
    
    config = LivniumCoreConfig(lattice_size=5, enable_quantum=True)
    base = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base, max_depth=2)
    print("✓ Built recursive geometry (5×5×5, depth 2)\n")
    
    # Test 1: Full tree (no sampling)
    print("TEST 1: Full Tree (No Sampling)")
    print("-"*70)
    total_qubits, total_registers, visited_stats = test_level_accurate(
        engine.levels[0],
        depth=0,
        max_qubits_per_register=10,
        max_children_per_level=None  # No limit
    )
    
    print(f"\nResults:")
    print(f"  Total qubits: {total_qubits:,}")
    print(f"  Total registers: {total_registers:,}")
    
    print(f"\nVisited nodes:")
    actual_theoretical = 0
    for depth in sorted(visited_stats.keys()):
        stats = visited_stats[depth]
        cells_per_geo = stats['total_cells'] // stats['geometries'] if stats['geometries'] > 0 else 0
        max_qubits_possible = stats['total_cells']  # If we used all cells
        actual_theoretical += max_qubits_possible
        print(f"  Depth {depth}: {stats['geometries']:,} geometries, "
              f"{stats['total_cells']:,} cells, "
              f"max {max_qubits_possible:,} qubits possible")
    
    print(f"\nActual theoretical max (from visited): {actual_theoretical:,} qubits")
    print(f"Actual achieved: {total_qubits:,} qubits")
    print(f"Utilization: {100*total_qubits/actual_theoretical:.1f}%" if actual_theoretical > 0 else "N/A")
    
    # Test 2: Sampled (for comparison)
    print("\n" + "="*70)
    print("TEST 2: Sampled Tree (5 children per level)")
    print("-"*70)
    total_qubits_sampled, total_registers_sampled, visited_stats_sampled = test_level_accurate(
        engine.levels[0],
        depth=0,
        max_qubits_per_register=10,
        max_children_per_level=5  # Sample limit
    )
    
    print(f"\nResults:")
    print(f"  Total qubits: {total_qubits_sampled:,}")
    print(f"  Total registers: {total_registers_sampled:,}")
    
    print(f"\nVisited nodes:")
    sampled_theoretical = 0
    for depth in sorted(visited_stats_sampled.keys()):
        stats = visited_stats_sampled[depth]
        max_qubits_possible = stats['total_cells']
        sampled_theoretical += max_qubits_possible
        print(f"  Depth {depth}: {stats['geometries']:,} geometries, "
              f"{stats['total_cells']:,} cells, "
              f"max {max_qubits_possible:,} qubits possible")
    
    print(f"\nSampled theoretical max: {sampled_theoretical:,} qubits")
    print(f"Sampled achieved: {total_qubits_sampled:,} qubits")
    print(f"Sampled utilization: {100*total_qubits_sampled/sampled_theoretical:.1f}%" if sampled_theoretical > 0 else "N/A")
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Full tree:  {total_qubits:,} qubits from {actual_theoretical:,} possible ({100*total_qubits/actual_theoretical:.1f}%)")
    print(f"Sampled:    {total_qubits_sampled:,} qubits from {sampled_theoretical:,} possible ({100*total_qubits_sampled/sampled_theoretical:.1f}%)")
    print(f"\nFull tree has {total_qubits/total_qubits_sampled:.1f}x more qubits than sampled" if total_qubits_sampled > 0 else "")


if __name__ == "__main__":
    run_accurate_test()

