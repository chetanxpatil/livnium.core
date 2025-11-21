"""
Test Recursive Geometry Omcube Capacity

Tests how many omcubes can be represented using recursive geometry.
The recursive engine compresses entanglement into lower-scale geometry,
allowing simulation of many omcubes with limited resources.
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
from core.quantum import QuantumLattice, QuantumCell


def count_omcubes_at_level(level, quantum_lattices: Dict[int, QuantumLattice]) -> int:
    """Count omcubes at a specific recursive level."""
    if level not in quantum_lattices:
        return 0
    
    quantum_lattice = quantum_lattices[level]
    return len(quantum_lattice.quantum_cells)


def count_total_omcubes_recursive(recursive_engine: RecursiveGeometryEngine,
                                  quantum_lattices: Dict[int, QuantumLattice]) -> int:
    """Count total omcubes across all recursive levels."""
    total = 0
    
    for level_id, level in recursive_engine.levels.items():
        if level_id in quantum_lattices:
            total += len(quantum_lattices[level_id].quantum_cells)
    
    return total


def count_omcubes_recursive(level) -> int:
    """Recursively count all omcubes in a level and its children."""
    from core.recursive.recursive_geometry_engine import GeometryLevel
    
    total = len(level.geometry.lattice)  # Cells at this level
    
    # Count children recursively
    for child_level in level.children.values():
        total += count_omcubes_recursive(child_level)
    
    return total


def test_recursive_omcube_capacity(base_lattice_size: int = 5,
                                   max_depth: int = 3,
                                   target_omcubes: int = 3000) -> Dict:
    """
    Test how many omcubes can be represented recursively.
    
    Args:
        base_lattice_size: Size of base lattice (must be odd, ≥3)
        max_depth: Maximum recursion depth
        target_omcubes: Target number of omcubes to test
    
    Returns:
        Dictionary with capacity metrics
    """
    print("=" * 70)
    print("Recursive Geometry Omcube Capacity Test")
    print("=" * 70)
    print(f"Base lattice: {base_lattice_size}×{base_lattice_size}×{base_lattice_size}")
    print(f"Max depth: {max_depth}")
    print(f"Target omcubes: {target_omcubes:,}")
    print()
    
    # Initialize base system
    config = LivniumCoreConfig(
        lattice_size=base_lattice_size,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    base_system = LivniumCoreSystem(config)
    
    # Initialize recursive geometry engine
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=base_system,
        max_depth=max_depth
    )
    
    # Count omcubes at each level
    level_0 = recursive_engine.levels[0]
    base_omcubes = len(level_0.geometry.lattice)
    print(f"Level 0: {base_omcubes:,} cells → {base_omcubes:,} omcubes")
    
    level_breakdown = {0: base_omcubes}
    total_omcubes = base_omcubes
    
    # Count omcubes at each recursive level
    for level_id in range(1, max_depth + 1):
        level_qubits = 0
        num_subgeometries = 0
        
        # Count all child geometries at this level
        for parent_coords, child_level in level_0.children.items():
            if child_level.level_id == level_id:
                num_subgeometries += 1
                level_qubits += len(child_level.geometry.lattice)
            else:
                # Recursively count deeper levels
                def count_deeper(child, target_level):
                    if child.level_id == target_level:
                        return len(child.geometry.lattice)
                    total = 0
                    for grandchild in child.children.values():
                        total += count_deeper(grandchild, target_level)
                    return total
                
                level_omcubes += count_deeper(child_level, level_id)
        
        if level_omcubes > 0:
            level_breakdown[level_id] = level_omcubes
            total_omcubes += level_omcubes
            print(f"Level {level_id}: {level_omcubes:,} omcubes (from sub-geometries)")
    
    # Also count recursively to get exact total
    total_omcubes_exact = count_omcubes_recursive(level_0)
    
    print()
    print("=" * 70)
    print("Capacity Summary")
    print("=" * 70)
    print(f"Total omcubes: {total_omcubes_exact:,}")
    print(f"Target omcubes: {target_omcubes:,}")
    if target_omcubes > 0:
        print(f"Capacity: {total_omcubes_exact / target_omcubes * 100:.1f}% of target")
    print()
    print("Breakdown by level:")
    for level_id, count in sorted(level_breakdown.items()):
        print(f"  Level {level_id}: {count:,} omcubes")
    
    # Calculate theoretical maximum
    base_cells = base_lattice_size ** 3
    theoretical_max = base_cells
    
    # For each depth, calculate how many cells we'd have if fully subdivided
    for depth in range(1, max_depth + 1):
        # At depth 1: each of base_cells contains a (base_size-2)^3 geometry
        # At depth 2: each of those contains a 3×3×3 geometry
        if depth == 1:
            child_size = max(3, base_lattice_size - 2)
            cells_per_child = child_size ** 3
            cells_at_depth = base_cells * cells_per_child
        else:
            # Depth 2+: 3×3×3 = 27 cells each
            cells_per_child = 27
            # Number of parent cells that can be subdivided
            parent_cells = base_cells * (max(3, base_lattice_size - 2) ** 3) if depth == 2 else theoretical_max - base_cells
            cells_at_depth = parent_cells * cells_per_child
        
        theoretical_max += cells_at_depth
    
    print()
    print(f"Theoretical maximum (if fully subdivided): {theoretical_max:,} omcubes")
    print(f"Actual capacity: {total_omcubes_exact:,} omcubes")
    if theoretical_max > 0:
        print(f"Utilization: {total_omcubes_exact / theoretical_max * 100:.2f}%")
    
    # Test if we can reach target
    can_reach_target = total_omcubes_exact >= target_omcubes
    
    print()
    print("=" * 70)
    if can_reach_target:
        print(f"✅ SUCCESS: Can represent {total_omcubes_exact:,} omcubes (exceeds target of {target_omcubes:,})")
    else:
        print(f"⚠️  PARTIAL: Can represent {total_omcubes_exact:,} omcubes (below target of {target_omcubes:,})")
        print(f"   Need {target_omcubes - total_omcubes_exact:,} more omcubes")
        print(f"   Suggestions:")
        print(f"   - Increase base_lattice_size (currently {base_lattice_size})")
        print(f"   - Increase max_depth (currently {max_depth})")
    print("=" * 70)
    
    return {
        'total_omcubes': total_omcubes_exact,
        'target_omcubes': target_omcubes,
        'can_reach_target': can_reach_target,
        'level_breakdown': level_breakdown,
        'theoretical_max': theoretical_max,
        'utilization': total_omcubes_exact / theoretical_max if theoretical_max > 0 else 0
    }


def test_scaling():
    """Test how capacity scales with different parameters."""
    print("\n" + "=" * 70)
    print("Scaling Analysis")
    print("=" * 70)
    
    test_cases = [
        (3, 2, 100),
        (3, 3, 500),
        (5, 2, 1000),
        (5, 3, 3000),
        (7, 2, 2000),
        (7, 3, 5000),
    ]
    
    results = []
    
    for lattice_size, max_depth, target in test_cases:
        print(f"\n--- Testing: {lattice_size}×{lattice_size}×{lattice_size}, depth={max_depth}, target={target} ---")
        try:
            result = test_recursive_omcube_capacity(
                base_lattice_size=lattice_size,
                max_depth=max_depth,
                target_omcubes=target
            )
            results.append({
                'lattice_size': lattice_size,
                'max_depth': max_depth,
                'target': target,
                'actual': result['total_omcubes'],
                'success': result['can_reach_target']
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'lattice_size': lattice_size,
                'max_depth': max_depth,
                'target': target,
                'actual': 0,
                'success': False,
                'error': str(e)
            })
    
    print("\n" + "=" * 70)
    print("Scaling Summary")
    print("=" * 70)
    print(f"{'Lattice':<10} {'Depth':<8} {'Target':<10} {'Actual':<12} {'Status':<10}")
    print("-" * 70)
    for r in results:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        print(f"{r['lattice_size']}×{r['lattice_size']}×{r['lattice_size']:<4} "
              f"{r['max_depth']:<8} {r['target']:<10,} {r['actual']:<12,} {status:<10}")


if __name__ == "__main__":
    # Test with target of 2,953-4,000 omcubes
    print("Testing capacity for 2,953-4,000 omcubes...")
    print()
    
    # Test 1: Conservative (target 2,953)
    print("TEST 1: Target 2,953 omcubes")
    result1 = test_recursive_omcube_capacity(
        base_lattice_size=5,
        max_depth=3,
        target_omcubes=2953
    )
    
    # Test 2: Ambitious (target 4,000)
    print("\n\n")
    print("TEST 2: Target 4,000 omcubes")
    result2 = test_recursive_omcube_capacity(
        base_lattice_size=7,
        max_depth=3,
        target_omcubes=4000
    )
    
    # Scaling analysis
    test_scaling()

