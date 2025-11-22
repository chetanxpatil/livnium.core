"""
Test Recursive Geometry Stability

Tests that recursive geometry (universe inside universe) remains stable:
- Recursive projection correctness
- Stability across 3-5 iterations
- No drift in SW
- No explosion/degeneration of cell classes
- No ghost inversions in child blocks
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from core.classical.livnium_core_system import LivniumCoreSystem, CellClass
from core.recursive.recursive_geometry_engine import RecursiveGeometryEngine
from core.config import LivniumCoreConfig


def test_recursive_stability_base_5x5x5():
    """Test recursive stability with base 5×5×5 system."""
    print("=" * 60)
    print("Test 1: Recursive Stability (Base 5×5×5)")
    print("=" * 60)
    
    # Create base system
    config = LivniumCoreConfig(lattice_size=5)
    base_system = LivniumCoreSystem(config)
    
    # Create recursive engine
    recursive_engine = RecursiveGeometryEngine(base_system, max_depth=3)
    
    # Build hierarchy
    recursive_engine.build_hierarchy()
    
    print(f"Base system: {len(base_system.lattice)} cells")
    print(f"Recursive levels: {len(recursive_engine.levels)}")
    
    # Track SW and class counts at each level
    level_stats = []
    
    for level_id, level in recursive_engine.levels.items():
        geometry = level.geometry
        total_sw = geometry.get_total_symbolic_weight()
        expected_sw = geometry.get_expected_total_sw()
        class_counts = geometry.get_class_counts()
        expected_counts = geometry.get_expected_class_counts()
        
        level_stats.append({
            'level': level_id,
            'cells': len(geometry.lattice),
            'total_sw': total_sw,
            'expected_sw': expected_sw,
            'sw_error': abs(total_sw - expected_sw),
            'class_counts': class_counts,
            'expected_counts': expected_counts
        })
        
        print(f"\nLevel {level_id}:")
        print(f"  Cells: {len(geometry.lattice)}")
        print(f"  SW: {total_sw:.6f} (expected: {expected_sw:.6f}, error: {abs(total_sw - expected_sw):.6f})")
        print(f"  Class counts:")
        for cls in CellClass:
            actual = class_counts.get(cls, 0)
            expected = expected_counts.get(cls, 0)
            match = "✅" if actual == expected else "❌"
            print(f"    {cls.name}: {actual} (expected: {expected}) {match}")
    
    # Check stability: SW should match expected at all levels
    sw_stable = all(stat['sw_error'] < 1e-6 for stat in level_stats)
    print(f"\nSW Stability: {'✅ PASS' if sw_stable else '❌ FAIL'}")
    assert sw_stable, "SW must match expected at all recursive levels"
    
    # Check class counts stability
    class_stable = True
    for stat in level_stats:
        for cls in CellClass:
            if stat['class_counts'].get(cls, 0) != stat['expected_counts'].get(cls, 0):
                class_stable = False
                break
    
    print(f"Class Count Stability: {'✅ PASS' if class_stable else '❌ FAIL'}")
    assert class_stable, "Class counts must match expected at all levels"
    
    print("\n✅ Recursive stability test passed!")


def test_recursive_stability_iterations():
    """Test stability across multiple iterations."""
    print("\n" + "=" * 60)
    print("Test 2: Stability Across Iterations")
    print("=" * 60)
    
    config = LivniumCoreConfig(lattice_size=5)
    base_system = LivniumCoreSystem(config)
    recursive_engine = RecursiveGeometryEngine(base_system, max_depth=2)
    
    # Run 5 iterations
    initial_sw = base_system.get_total_symbolic_weight()
    sw_history = [initial_sw]
    
    for iteration in range(5):
        # Rebuild hierarchy
        recursive_engine.build_hierarchy()
        
        # Check SW at each level
        for level_id, level in recursive_engine.levels.items():
            geometry = level.geometry
            current_sw = geometry.get_total_symbolic_weight()
            expected_sw = geometry.get_expected_total_sw()
            
            error = abs(current_sw - expected_sw)
            if error > 1e-6:
                print(f"Iteration {iteration+1}, Level {level_id}: SW error = {error:.6f}")
                assert False, f"SW drift detected at iteration {iteration+1}, level {level_id}"
        
        # Record base SW
        base_sw = base_system.get_total_symbolic_weight()
        sw_history.append(base_sw)
    
    # Check SW doesn't drift
    sw_drift = max(sw_history) - min(sw_history)
    print(f"SW drift across 5 iterations: {sw_drift:.6f}")
    print(f"SW Stability: {'✅ PASS' if sw_drift < 1e-6 else '❌ FAIL'}")
    assert sw_drift < 1e-6, "SW must not drift across iterations"
    
    print("\n✅ Iteration stability test passed!")


def test_recursive_no_ghost_inversions():
    """Test that child blocks don't have ghost inversions."""
    print("\n" + "=" * 60)
    print("Test 3: No Ghost Inversions in Child Blocks")
    print("=" * 60)
    
    config = LivniumCoreConfig(lattice_size=5)
    base_system = LivniumCoreSystem(config)
    recursive_engine = RecursiveGeometryEngine(base_system, max_depth=2)
    recursive_engine.build_hierarchy()
    
    # Check each child level
    for level_id, level in recursive_engine.levels.items():
        if level_id == 0:
            continue  # Skip base level
        
        geometry = level.geometry
        
        # Check all cells have valid face_exposure
        invalid_cells = []
        for coords, cell in geometry.lattice.items():
            if cell.face_exposure is None or cell.face_exposure < 0 or cell.face_exposure > 3:
                invalid_cells.append((coords, cell.face_exposure))
            if cell.symbolic_weight < 0:
                invalid_cells.append((coords, f"negative SW: {cell.symbolic_weight}"))
        
        if invalid_cells:
            print(f"Level {level_id}: Found {len(invalid_cells)} invalid cells")
            for coords, issue in invalid_cells[:5]:  # Show first 5
                print(f"  {coords}: {issue}")
            assert False, f"Ghost inversions detected at level {level_id}"
        
        print(f"Level {level_id}: ✅ All cells valid")
    
    print("\n✅ No ghost inversions test passed!")


def test_recursive_class_distribution():
    """Test that class distributions remain valid at all depths."""
    print("\n" + "=" * 60)
    print("Test 4: Class Distribution Stability")
    print("=" * 60)
    
    config = LivniumCoreConfig(lattice_size=5)
    base_system = LivniumCoreSystem(config)
    recursive_engine = RecursiveGeometryEngine(base_system, max_depth=3)
    recursive_engine.build_hierarchy()
    
    # Check class distributions at each level
    for level_id, level in recursive_engine.levels.items():
        geometry = level.geometry
        class_counts = geometry.get_class_counts()
        expected_counts = geometry.get_expected_class_counts()
        
        # Check no class has negative count
        for cls in CellClass:
            count = class_counts.get(cls, 0)
            assert count >= 0, f"Level {level_id}: {cls.name} has negative count"
            
            # Check count matches expected
            expected = expected_counts.get(cls, 0)
            if count != expected:
                print(f"Level {level_id}: {cls.name} count mismatch ({count} vs {expected})")
                assert False, f"Level {level_id}: Class count mismatch for {cls.name}"
        
        print(f"Level {level_id}: ✅ Class distribution valid")
    
    print("\n✅ Class distribution test passed!")


if __name__ == "__main__":
    test_recursive_stability_base_5x5x5()
    test_recursive_stability_iterations()
    test_recursive_no_ghost_inversions()
    test_recursive_class_distribution()
    print("\n" + "=" * 60)
    print("All recursive geometry stability tests passed! ✅")
    print("=" * 60)

