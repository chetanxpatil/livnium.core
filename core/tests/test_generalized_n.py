"""
Test script for Generalized N×N×N Livnium Core System.

Tests the system with N=3, N=5, N=7 to verify scaling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis, CellClass
from core.config import LivniumCoreConfig


def test_n3():
    """Test N=3 (canonical 3×3×3)."""
    print("=" * 60)
    print("Test: N=3 (3×3×3 Lattice)")
    print("=" * 60)
    
    config = LivniumCoreConfig(lattice_size=3)
    system = LivniumCoreSystem(config)
    
    # Check total cells
    total_cells = len(system.lattice)
    expected_cells = 3 ** 3
    print(f"Total cells: {total_cells} (expected: {expected_cells})")
    assert total_cells == expected_cells, f"Cell count mismatch: {total_cells} != {expected_cells}"
    
    # Check alphabet size
    if system.config.enable_symbol_alphabet:
        alphabet_size = len(set(system.symbol_map.values()))
        print(f"Alphabet size: {alphabet_size} (expected: {expected_cells})")
        assert alphabet_size == expected_cells, f"Alphabet size mismatch: {alphabet_size} != {expected_cells}"
    
    # Check total SW
    total_sw = system.get_total_symbolic_weight()
    expected_sw = system.get_expected_total_sw()
    print(f"Total SW: {total_sw} (expected: {expected_sw})")
    assert abs(total_sw - expected_sw) < 1e-6, f"SW mismatch: {total_sw} != {expected_sw}"
    assert expected_sw == 486, f"N=3 should have SW=486, got {expected_sw}"
    
    # Check class counts
    counts = system.get_class_counts()
    expected = system.get_expected_class_counts()
    print(f"Class counts:")
    for cls in CellClass:
        print(f"  {cls.name}: {counts[cls]} (expected: {expected[cls]})")
        assert counts[cls] == expected[cls], f"{cls.name} count mismatch"
    
    print("✅ N=3 test passed!\n")


def test_n5():
    """Test N=5 (5×5×5)."""
    print("=" * 60)
    print("Test: N=5 (5×5×5 Lattice)")
    print("=" * 60)
    
    config = LivniumCoreConfig(lattice_size=5)
    system = LivniumCoreSystem(config)
    
    # Check total cells
    total_cells = len(system.lattice)
    expected_cells = 5 ** 3
    print(f"Total cells: {total_cells} (expected: {expected_cells})")
    assert total_cells == expected_cells, f"Cell count mismatch: {total_cells} != {expected_cells}"
    
    # Check alphabet size
    if system.config.enable_symbol_alphabet:
        alphabet_size = len(set(system.symbol_map.values()))
        print(f"Alphabet size: {alphabet_size} (expected: {expected_cells})")
        assert alphabet_size == expected_cells, f"Alphabet size mismatch: {alphabet_size} != {expected_cells}"
    
    # Check total SW
    total_sw = system.get_total_symbolic_weight()
    expected_sw = system.get_expected_total_sw()
    print(f"Total SW: {total_sw} (expected: {expected_sw})")
    assert abs(total_sw - expected_sw) < 1e-6, f"SW mismatch: {total_sw} != {expected_sw}"
    assert expected_sw == 1350, f"N=5 should have SW=1350, got {expected_sw}"
    
    # Check class counts
    counts = system.get_class_counts()
    expected = system.get_expected_class_counts()
    print(f"Class counts:")
    for cls in CellClass:
        print(f"  {cls.name}: {counts[cls]} (expected: {expected[cls]})")
        assert counts[cls] == expected[cls], f"{cls.name} count mismatch"
    
    # Test rotation preserves invariants
    result = system.rotate(RotationAxis.X, quarter_turns=1)
    assert result['invariants_preserved'], "Rotation should preserve invariants"
    
    # Rotate back
    system.rotate(RotationAxis.X, quarter_turns=3)
    final_sw = system.get_total_symbolic_weight()
    assert abs(final_sw - expected_sw) < 1e-6, "Rotation back should restore SW"
    
    print("✅ N=5 test passed!\n")


def test_n7():
    """Test N=7 (7×7×7)."""
    print("=" * 60)
    print("Test: N=7 (7×7×7 Lattice)")
    print("=" * 60)
    
    config = LivniumCoreConfig(lattice_size=7)
    system = LivniumCoreSystem(config)
    
    # Check total cells
    total_cells = len(system.lattice)
    expected_cells = 7 ** 3
    print(f"Total cells: {total_cells} (expected: {expected_cells})")
    assert total_cells == expected_cells, f"Cell count mismatch: {total_cells} != {expected_cells}"
    
    # Check alphabet size
    if system.config.enable_symbol_alphabet:
        alphabet_size = len(set(system.symbol_map.values()))
        print(f"Alphabet size: {alphabet_size} (expected: {expected_cells})")
        assert alphabet_size == expected_cells, f"Alphabet size mismatch: {alphabet_size} != {expected_cells}"
    
    # Check total SW
    total_sw = system.get_total_symbolic_weight()
    expected_sw = system.get_expected_total_sw()
    print(f"Total SW: {total_sw} (expected: {expected_sw})")
    assert abs(total_sw - expected_sw) < 1e-6, f"SW mismatch: {total_sw} != {expected_sw}"
    # N=7: 54(5)² + 216(5) + 216 = 54×25 + 1080 + 216 = 1350 + 1080 + 216 = 2646
    assert expected_sw == 2646, f"N=7 should have SW=2646, got {expected_sw}"
    
    # Check class counts
    counts = system.get_class_counts()
    expected = system.get_expected_class_counts()
    print(f"Class counts:")
    for cls in CellClass:
        print(f"  {cls.name}: {counts[cls]} (expected: {expected[cls]})")
        assert counts[cls] == expected[cls], f"{cls.name} count mismatch"
    
    print("✅ N=7 test passed!\n")


def test_alphabet_scaling():
    """Test that alphabet scales correctly with N."""
    print("=" * 60)
    print("Test: Alphabet Scaling Σ(N)")
    print("=" * 60)
    
    for n in [3, 5, 7]:
        config = LivniumCoreConfig(lattice_size=n)
        system = LivniumCoreSystem(config)
        
        if system.config.enable_symbol_alphabet:
            alphabet = system.generate_alphabet(n)
            expected_size = n ** 3
            actual_size = len(set(system.symbol_map.values()))
            
            print(f"N={n}: Alphabet size = {actual_size} (expected: {expected_size})")
            assert actual_size == expected_size, f"N={n}: Alphabet size mismatch"
            assert len(alphabet) == expected_size, f"N={n}: Generated alphabet size mismatch"
            
            # Check bijection: each coordinate has unique symbol
            symbols = list(system.symbol_map.values())
            assert len(symbols) == len(set(symbols)), f"N={n}: Symbols are not unique (not bijective)"
    
    print("✅ Alphabet scaling test passed!\n")


def test_formula_verification():
    """Verify formulas for multiple N values."""
    print("=" * 60)
    print("Test: Formula Verification (General N)")
    print("=" * 60)
    
    test_cases = [
        (3, 486),
        (5, 1350),
        (7, 2646),  # 54(5)² + 216(5) + 216 = 1350 + 1080 + 216 = 2646
    ]
    
    for n, expected_sw in test_cases:
        config = LivniumCoreConfig(lattice_size=n)
        system = LivniumCoreSystem(config)
        
        calculated_sw = system.get_expected_total_sw()
        actual_sw = system.get_total_symbolic_weight()
        
        print(f"N={n}:")
        print(f"  Formula: ΣSW = {calculated_sw}")
        print(f"  Actual:  ΣSW = {actual_sw}")
        print(f"  Expected: {expected_sw}")
        
        assert abs(calculated_sw - expected_sw) < 1e-6, f"N={n}: Formula mismatch"
        assert abs(actual_sw - expected_sw) < 1e-6, f"N={n}: Actual SW mismatch"
        
        # Verify class counts formula
        counts = system.get_class_counts()
        expected_counts = system.get_expected_class_counts()
        for cls in CellClass:
            assert counts[cls] == expected_counts[cls], f"N={n}: {cls.name} count mismatch"
    
    print("✅ Formula verification test passed!\n")


if __name__ == "__main__":
    print("Livnium Core System - Generalized N×N×N Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_n3()
        test_n5()
        test_n7()
        test_alphabet_scaling()
        test_formula_verification()
        
        print("=" * 60)
        print("✅ ALL GENERALIZED N×N×N TESTS PASSED!")
        print("=" * 60)
        print()
        print("System is fully generalized for any odd N ≥ 3:")
        print("  ✅ N=3: 27 cells, SW=486")
        print("  ✅ N=5: 125 cells, SW=1350")
        print("  ✅ N=7: 343 cells, SW=2646")
        print("  ✅ Alphabet Σ(N) scales with N³")
        print("  ✅ All formulas work for general N")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

