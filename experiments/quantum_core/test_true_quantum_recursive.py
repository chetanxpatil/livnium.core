"""
Test TrueQuantumRegister capacity in recursive geometry.

This tests how many qubits can be run using TrueQuantumRegister
(real tensor product quantum mechanics) at each recursive level.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple

# Make repo root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.recursive import RecursiveGeometryEngine
from core.quantum.true_quantum_layer import TrueQuantumRegister
from core.quantum.quantum_gates import QuantumGates

# Optional psutil for memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_memory_usage():
    """Get current memory usage in MB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    else:
        return 0.0  # Return 0 if psutil not available


def test_true_quantum_at_level(level, level_name="Level 0", max_qubits_per_register=10, verbose=True):
    """
    Test TrueQuantumRegister at a single level.
    
    Args:
        level: GeometryLevel instance
        level_name: Name for logging
        max_qubits_per_register: Maximum qubits per TrueQuantumRegister
        verbose: Print progress
        
    Returns:
        (total_qubits, successful_registers, failed_registers)
    """
    geometry = level.geometry
    cells = list(geometry.lattice.keys())
    num_cells = len(cells)
    
    if verbose:
        print(f"  [{level_name}] {num_cells} cells available")
    
    # We can create multiple TrueQuantumRegisters
    # Each register can handle up to max_qubits_per_register qubits
    # Strategy: Group cells into registers of size max_qubits_per_register
    
    total_qubits = 0
    successful_registers = 0
    failed_registers = 0
    
    # Create registers from available cells
    registers_created = 0
    for i in range(0, num_cells, max_qubits_per_register):
        qubit_group = cells[i:i+max_qubits_per_register]
        num_qubits = len(qubit_group)
        
        if num_qubits < 2:  # Need at least 2 qubits for meaningful test
            continue
        
        try:
            # Create TrueQuantumRegister with qubit IDs
            qubit_ids = list(range(num_qubits))
            register = TrueQuantumRegister(qubit_ids)
            
            # Test operations
            H = QuantumGates.hadamard()
            register.apply_gate(H, target_id=0)
            
            if num_qubits >= 2:
                register.apply_cnot(control_id=0, target_id=1)
            
            # Measure
            result = register.measure_qubit(0)
            
            total_qubits += num_qubits
            successful_registers += 1
            registers_created += 1
            
        except MemoryError as e:
            if verbose:
                print(f"    ❌ MemoryError at {num_qubits} qubits: {e}")
            failed_registers += 1
            break
        except Exception as e:
            if verbose:
                print(f"    ❌ Error at {num_qubits} qubits: {e}")
            failed_registers += 1
            # Continue with next register
    
    if verbose and registers_created > 0:
        print(f"    ✅ Created {successful_registers} TrueQuantumRegisters with {total_qubits} total qubits")
    
    return total_qubits, successful_registers, failed_registers


def test_recursive_true_quantum(level, current_depth=0, max_depth=3, 
                                max_qubits_per_register=10,
                                max_registers_per_level=None,
                                visited_nodes=None):
    """
    Recursively test TrueQuantumRegister across all levels.
    
    Args:
        level: GeometryLevel instance
        current_depth: Current recursion depth
        max_depth: Maximum depth to test
        max_qubits_per_register: Max qubits per TrueQuantumRegister
        max_registers_per_level: Limit registers per level (None = no limit)
        visited_nodes: Dict to track visited nodes for accurate theoretical calc
        
    Returns:
        (total_qubits, total_registers, total_failed, visited_nodes)
    """
    if visited_nodes is None:
        visited_nodes = {}
    
    level_name = f"Depth {current_depth}"
    
    # Only print at depth 0 and 1 to avoid spam
    verbose = (current_depth <= 1)
    
    # Test at this level
    level_qubits, level_successful, level_failed = test_true_quantum_at_level(
        level, level_name, max_qubits_per_register, verbose=verbose
    )
    
    # Track visited nodes
    if current_depth not in visited_nodes:
        visited_nodes[current_depth] = {'geometries': 0, 'cells': 0}
    visited_nodes[current_depth]['geometries'] += 1
    visited_nodes[current_depth]['cells'] += len(level.geometry.lattice)
    
    total_qubits = level_qubits
    total_registers = level_successful
    total_failed = level_failed
    
    # Recursively test children
    if current_depth < max_depth:
        children = list(level.children.values())
        
        # Only sample if a limit is explicitly set
        if max_registers_per_level is not None and len(children) > max_registers_per_level:
            import random
            children = random.sample(children, max_registers_per_level)
        
        for child in children:
            child_qubits, child_registers, child_failed, visited_nodes = test_recursive_true_quantum(
                child, current_depth + 1, max_depth,
                max_qubits_per_register, max_registers_per_level, visited_nodes
            )
            
            total_qubits += child_qubits
            total_registers += child_registers
            total_failed += child_failed
    
    return total_qubits, total_registers, total_failed, visited_nodes


def test_true_quantum_recursive_capacity(base_lattice_size=5, max_depth=2, 
                                         max_qubits_per_register=10):
    """
    Test TrueQuantumRegister capacity in recursive geometry.
    
    Args:
        base_lattice_size: Base lattice size (must be odd, >= 3)
        max_depth: Maximum recursion depth
        max_qubits_per_register: Maximum qubits per TrueQuantumRegister
    """
    print("="*70)
    print("TRUE QUANTUM REGISTER - RECURSIVE CAPACITY TEST")
    print("="*70)
    print(f"Base: {base_lattice_size}×{base_lattice_size}×{base_lattice_size}")
    print(f"Max depth: {max_depth}")
    print(f"Max qubits per register: {max_qubits_per_register}")
    print()
    
    start_memory = get_memory_usage()
    
    config = LivniumCoreConfig(
        lattice_size=base_lattice_size,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
    )
    
    base = LivniumCoreSystem(config)
    print("Building recursive geometry...")
    engine = RecursiveGeometryEngine(base, max_depth=max_depth)
    print("✓ Built\n")
    
    print("="*70)
    print("Testing TrueQuantumRegister at each level")
    print("="*70)
    
    level_0 = engine.levels[0]
    
    total_qubits, total_registers, total_failed, visited_nodes = test_recursive_true_quantum(
        level_0,
        current_depth=0,
        max_depth=max_depth,
        max_qubits_per_register=max_qubits_per_register,
        max_registers_per_level=None  # No limit - test full capacity
    )
    
    end_memory = get_memory_usage()
    memory_used = end_memory - start_memory
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total TrueQuantumRegister qubits: {total_qubits:,}")
    print(f"Total registers created: {total_registers}")
    print(f"Failed registers: {total_failed}")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Average qubits per register: {total_qubits / total_registers:.1f}" if total_registers > 0 else "")
    print()
    
    # Calculate ACTUAL theoretical max based on visited nodes
    print("Visited nodes breakdown:")
    actual_theoretical = 0
    for depth in sorted(visited_nodes.keys()):
        stats = visited_nodes[depth]
        cells_per_geometry = stats['cells'] // stats['geometries'] if stats['geometries'] > 0 else 0
        # Calculate how many registers we could create from visited cells
        max_registers_from_visited = (stats['cells'] // max_qubits_per_register) if max_qubits_per_register > 0 else 0
        max_qubits_from_visited = stats['cells']  # If we used all cells
        actual_theoretical += max_qubits_from_visited
        print(f"  Depth {depth}: {stats['geometries']:,} geometries, {stats['cells']:,} cells, "
              f"max {max_qubits_from_visited:,} qubits possible")
    
    print()
    print(f"Actual theoretical max (from visited nodes): {actual_theoretical:,} qubits")
    print(f"Actual achieved: {total_qubits:,} qubits")
    print(f"Utilization: {100*total_qubits/actual_theoretical:.1f}%" if actual_theoretical > 0 else "N/A")
    print()
    
    # Also show what FULL tree would be (for reference)
    base_cells = base_lattice_size ** 3
    full_tree_max = base_cells * max_qubits_per_register
    for d in range(1, max_depth + 1):
        # At depth d, we'd have base_cells^d geometries, each with base_cells cells
        # But child geometries are typically 3×3×3 = 27 cells
        child_cells = 27  # Typical child size
        num_geometries = base_cells ** d
        full_tree_max += num_geometries * child_cells
    
    print(f"Full tree theoretical max (all branches): {full_tree_max:,} qubits")
    print(f"Visited vs Full: {100*actual_theoretical/full_tree_max:.1f}% of tree visited" if full_tree_max > 0 else "")
    
    return {
        'total_qubits': total_qubits,
        'total_registers': total_registers,
        'failed': total_failed,
        'memory_mb': memory_used,
        'base_lattice_size': base_lattice_size,
        'max_depth': max_depth,
        'max_qubits_per_register': max_qubits_per_register
    }


def test_different_register_sizes():
    """Test with different register sizes."""
    print("\n" + "="*70)
    print("TESTING DIFFERENT REGISTER SIZES")
    print("="*70)
    
    register_sizes = [3, 5, 8, 10, 15, 20]
    results = []
    
    for size in register_sizes:
        print(f"\n--- Testing with {size} qubits per register ---")
        try:
            result = test_true_quantum_recursive_capacity(
                base_lattice_size=5,
                max_depth=2,  # Reduced depth for testing
                max_qubits_per_register=size
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Failed at {size} qubits/register: {e}")
            break
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Register Size vs Capacity")
    print("="*70)
    for r in results:
        print(f"  {r['max_qubits_per_register']:2d} qubits/register: "
              f"{r['total_qubits']:6,} total qubits, "
              f"{r['total_registers']:3d} registers, "
              f"{r['memory_mb']:6.2f} MB")


if __name__ == "__main__":
    print("Testing TrueQuantumRegister capacity in recursive geometry\n")
    
    # Test 1: Basic recursive test
    print("TEST 1: Basic Recursive Test")
    result1 = test_true_quantum_recursive_capacity(
        base_lattice_size=5,
        max_depth=2,  # Start with depth 2 to avoid memory issues
        max_qubits_per_register=10
    )
    
    # Test 2: Different register sizes
    test_different_register_sizes()

