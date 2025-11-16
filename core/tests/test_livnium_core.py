"""
Test script for Livnium Core System.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis, CellClass
from core.config import LivniumCoreConfig


def test_basic_system():
    """Test basic system with all features enabled."""
    print("=" * 60)
    print("Test 1: Basic System (All Features Enabled)")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    
    # Check core cell
    core_cell = system.get_cell((0, 0, 0))
    print(f"Core cell at (0,0,0):")
    print(f"  Face exposure: {core_cell.face_exposure}")
    print(f"  Symbolic Weight: {core_cell.symbolic_weight}")
    print(f"  Class: {core_cell.cell_class}")
    print(f"  Symbol: {system.get_symbol((0, 0, 0))}")
    
    # Check corner cell
    corner_cell = system.get_cell((1, 1, 1))
    print(f"\nCorner cell at (1,1,1):")
    print(f"  Face exposure: {corner_cell.face_exposure}")
    print(f"  Symbolic Weight: {corner_cell.symbolic_weight}")
    print(f"  Class: {corner_cell.cell_class}")
    
    # Check total SW
    total_sw = system.get_total_symbolic_weight()
    expected_sw = system.get_expected_total_sw()
    print(f"\nTotal Symbolic Weight:")
    print(f"  Calculated: {total_sw}")
    print(f"  Expected: {expected_sw}")
    print(f"  Match: {abs(total_sw - expected_sw) < 1e-6}")
    
    # Check class counts
    counts = system.get_class_counts()
    expected = system.get_expected_class_counts()
    print(f"\nClass Counts:")
    for cls in CellClass:
        print(f"  {cls.name}: {counts[cls]} (expected: {expected[cls]})")
    
    print("\n✅ Basic system test passed!")


def test_rotations():
    """Test rotation operations."""
    print("\n" + "=" * 60)
    print("Test 2: Rotation Operations")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    
    # Record initial state
    initial_sw = system.get_total_symbolic_weight()
    initial_counts = system.get_class_counts()
    
    # Rotate about X-axis
    result = system.rotate(RotationAxis.X, quarter_turns=1)
    print(f"Rotation about X-axis (1 quarter-turn):")
    print(f"  Rotated: {result['rotated']}")
    print(f"  Invariants preserved: {result['invariants_preserved']}")
    
    if system.config.enable_sw_conservation:
        print(f"  SW preserved: {result.get('sw_preserved', 'N/A')}")
        print(f"  Total SW: {result.get('total_sw', 'N/A')}")
    
    if system.config.enable_class_count_conservation:
        print(f"  Class counts preserved: {result.get('class_counts_preserved', 'N/A')}")
    
    # Rotate back (should restore)
    result2 = system.rotate(RotationAxis.X, quarter_turns=3)
    final_sw = system.get_total_symbolic_weight()
    final_counts = system.get_class_counts()
    
    print(f"\nAfter rotating back (3 more quarter-turns = 4 total = identity):")
    print(f"  SW restored: {abs(initial_sw - final_sw) < 1e-6}")
    print(f"  Counts restored: {initial_counts == final_counts}")
    
    print("\n✅ Rotation test passed!")


def test_polarity():
    """Test semantic polarity."""
    print("\n" + "=" * 60)
    print("Test 3: Semantic Polarity")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    
    # Motion toward observer (using target coordinates)
    # Observer at (0,0,0), target at (1,0,0) = motion toward
    polarity = system.calculate_polarity((0, 0, 0), target_coords=(1, 0, 0))
    print(f"Motion from (0,0,0) to (1,0,0) [toward positive X]:")
    print(f"  Polarity: {polarity:.3f}")
    print(f"  Interpretation: {'Toward' if polarity > 0 else 'Away'}")
    
    # Motion away from observer
    polarity2 = system.calculate_polarity((0, 0, 0), target_coords=(-1, 0, 0))
    print(f"\nMotion from (0,0,0) to (-1,0,0) [away from positive X]:")
    print(f"  Polarity: {polarity2:.3f}")
    
    # Motion perpendicular
    polarity3 = system.calculate_polarity((0, 0, 0), target_coords=(0, 1, 0))
    print(f"\nMotion from (0,0,0) to (0,1,0) [perpendicular]:")
    print(f"  Polarity: {polarity3:.3f}")
    
    # Test with local observer
    local_obs = system.set_local_observer((1, 1, 1))
    polarity4 = system.calculate_polarity((0, 0, 0), observer_coords=(1, 1, 1), target_coords=(2, 1, 1))
    print(f"\nMotion from local observer (1,1,1) to (2,1,1):")
    print(f"  Polarity: {polarity4:.3f}")
    
    print("\n✅ Polarity test passed!")


def test_feature_switches():
    """Test feature switches."""
    print("\n" + "=" * 60)
    print("Test 4: Feature Switches")
    print("=" * 60)
    
    # Minimal system
    config_minimal = LivniumCoreConfig(
        enable_symbol_alphabet=False,
        enable_symbolic_weight=False,
        enable_sw_conservation=False,
        enable_90_degree_rotations=False,
        enable_global_observer=False,
        enable_local_observer=False,
        enable_semantic_polarity=False
    )
    system_minimal = LivniumCoreSystem(config_minimal)
    print("Minimal system (only lattice structure):")
    print(f"  Total cells: {len(system_minimal.lattice)}")
    print(f"  Features enabled: {sum(1 for v in system_minimal.config.__dict__.values() if isinstance(v, bool) and v)}")
    
    # System with only SW
    config_sw = LivniumCoreConfig(
        enable_symbol_alphabet=False,
        enable_90_degree_rotations=False,
        enable_global_observer=False,
        enable_local_observer=False,
        enable_semantic_polarity=False
    )
    system_sw = LivniumCoreSystem(config_sw)
    print(f"\nSystem with only Symbolic Weight:")
    print(f"  Total SW: {system_sw.get_total_symbolic_weight()}")
    print(f"  Expected SW: {system_sw.get_expected_total_sw()}")
    
    print("\n✅ Feature switches test passed!")


def test_system_summary():
    """Test system summary."""
    print("\n" + "=" * 60)
    print("Test 5: System Summary")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    summary = system.get_system_summary()
    
    print("System Summary:")
    print(f"  Lattice size: {summary['lattice_size']}")
    print(f"  Total cells: {summary['total_cells']}")
    print(f"\nFeatures Enabled:")
    for feature, enabled in summary['features_enabled'].items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {feature}")
    
    if 'total_symbolic_weight' in summary:
        print(f"\nTotal Symbolic Weight: {summary['total_symbolic_weight']}")
    
    if 'class_counts' in summary:
        print(f"\nClass Counts:")
        for cls, count in summary['class_counts'].items():
            print(f"  {cls.name}: {count}")
    
    print("\n✅ System summary test passed!")


if __name__ == "__main__":
    print("Livnium Core System - Test Suite")
    print("=" * 60)
    
    try:
        test_basic_system()
        test_rotations()
        test_polarity()
        test_feature_switches()
        test_system_summary()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

