"""
Test assertions for LivniumCoreSystem.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from core.classical.livnium_core_system import (
    LivniumCoreSystem,
    RotationAxis,
    CellClass,
    LatticeCell,
    Observer,
    RotationGroup
)
from core.config import LivniumCoreConfig


def test_basic_initialization():
    """Test basic system initialization."""
    system = LivniumCoreSystem()
    
    assert system.lattice_size == 3, "Default lattice size should be 3"
    assert len(system.lattice) == 27, "3×3×3 lattice should have 27 cells"
    assert system.global_observer is not None, "Global observer should be initialized"
    assert system.global_observer.coordinates == (0, 0, 0), "Global observer should be at (0,0,0)"
    assert system.global_observer.is_global, "Global observer should be marked as global"


def test_cell_classification():
    """Test cell classification (core, center, edge, corner)."""
    system = LivniumCoreSystem()
    
    # Core cell (0,0,0)
    core_cell = system.get_cell((0, 0, 0))
    assert core_cell.face_exposure == 0, "Core cell should have f=0"
    assert core_cell.symbolic_weight == 0.0, "Core cell should have SW=0"
    assert core_cell.cell_class == CellClass.CORE, "Cell at (0,0,0) should be CORE"
    
    # Center cells (face centers)
    center_cell = system.get_cell((0, 0, 1))
    assert center_cell.face_exposure == 1, "Center cell should have f=1"
    assert center_cell.symbolic_weight == 9.0, "Center cell should have SW=9"
    assert center_cell.cell_class == CellClass.CENTER, "Face center should be CENTER"
    
    # Edge cells
    edge_cell = system.get_cell((0, 1, 1))
    assert edge_cell.face_exposure == 2, "Edge cell should have f=2"
    assert edge_cell.symbolic_weight == 18.0, "Edge cell should have SW=18"
    assert edge_cell.cell_class == CellClass.EDGE, "Edge cell should be EDGE"
    
    # Corner cells
    corner_cell = system.get_cell((1, 1, 1))
    assert corner_cell.face_exposure == 3, "Corner cell should have f=3"
    assert corner_cell.symbolic_weight == 27.0, "Corner cell should have SW=27"
    assert corner_cell.cell_class == CellClass.CORNER, "Corner cell should be CORNER"


def test_symbolic_weight_formula():
    """Test SW = 9·f formula."""
    system = LivniumCoreSystem()
    
    for coords, cell in system.lattice.items():
        expected_sw = 9.0 * cell.face_exposure
        assert abs(cell.symbolic_weight - expected_sw) < 1e-6, \
            f"Cell at {coords}: SW should be 9·f, got {cell.symbolic_weight}, expected {expected_sw}"


def test_total_symbolic_weight():
    """Test total symbolic weight calculation."""
    system = LivniumCoreSystem()
    
    total_sw = system.get_total_symbolic_weight()
    expected_sw = system.get_expected_total_sw()
    
    assert abs(total_sw - expected_sw) < 1e-6, \
        f"Total SW should be {expected_sw}, got {total_sw}"
    assert expected_sw == 486.0, "For N=3, expected total SW should be 486"


def test_class_counts():
    """Test class count invariants."""
    system = LivniumCoreSystem()
    
    counts = system.get_class_counts()
    expected = system.get_expected_class_counts()
    
    assert counts == expected, f"Class counts should match expected: {expected}, got {counts}"
    assert counts[CellClass.CORE] == 1, "Should have 1 core cell"
    assert counts[CellClass.CENTER] == 6, "Should have 6 center cells"
    assert counts[CellClass.EDGE] == 12, "Should have 12 edge cells"
    assert counts[CellClass.CORNER] == 8, "Should have 8 corner cells"


def test_symbol_alphabet():
    """Test symbol alphabet initialization."""
    system = LivniumCoreSystem()
    
    assert len(system.symbol_map) == 27, "Should have 27 symbols for 3×3×3"
    
    # Check that symbols are assigned (coordinates are sorted, so first coord gets '0')
    sorted_coords = sorted(system.lattice.keys())
    first_coord = sorted_coords[0]
    first_symbol = system.get_symbol(first_coord)
    assert first_symbol == '0', f"First coordinate {first_coord} should have symbol '0'"
    
    # Check core cell has a symbol (but not necessarily '0' since coords are sorted)
    core_symbol = system.get_symbol((0, 0, 0))
    assert core_symbol is not None, "Core cell should have a symbol"
    assert core_symbol in system.symbol_map.values(), "Core symbol should be in symbol map"
    
    # Check all symbols are unique
    symbols = list(system.symbol_map.values())
    assert len(symbols) == len(set(symbols)), "All symbols should be unique"


def test_rotation_preserves_invariants():
    """Test that rotations preserve symbolic weight and class counts."""
    system = LivniumCoreSystem()
    
    initial_sw = system.get_total_symbolic_weight()
    initial_counts = system.get_class_counts()
    
    # Rotate around X axis
    result = system.rotate(RotationAxis.X, quarter_turns=1)
    
    assert result['rotated'], "Rotation should succeed"
    assert result['invariants_preserved'], "Invariants should be preserved"
    
    final_sw = system.get_total_symbolic_weight()
    final_counts = system.get_class_counts()
    
    assert abs(final_sw - initial_sw) < 1e-6, "Total SW should be preserved"
    assert final_counts == initial_counts, "Class counts should be preserved"


def test_rotation_around_all_axes():
    """Test rotations around X, Y, Z axes."""
    system = LivniumCoreSystem()
    
    for axis in [RotationAxis.X, RotationAxis.Y, RotationAxis.Z]:
        result = system.rotate(axis, quarter_turns=1)
        assert result['rotated'], f"Rotation around {axis.name} should succeed"
        assert result['invariants_preserved'], f"Invariants should be preserved for {axis.name}"


def test_full_rotation_identity():
    """Test that 4 quarter-turns = identity."""
    system = LivniumCoreSystem()
    
    # Get initial state
    initial_core_symbol = system.get_symbol((0, 0, 0))
    
    # Rotate 4 times (should return to original)
    result = system.rotate(RotationAxis.X, quarter_turns=4)
    assert not result['rotated'], "Full rotation should be identity (no rotation)"
    
    # Rotate 4 times explicitly
    system.rotate(RotationAxis.X, quarter_turns=1)
    system.rotate(RotationAxis.X, quarter_turns=1)
    system.rotate(RotationAxis.X, quarter_turns=1)
    system.rotate(RotationAxis.X, quarter_turns=1)
    
    final_core_symbol = system.get_symbol((0, 0, 0))
    assert final_core_symbol == initial_core_symbol, "4 rotations should return to original"


def test_rotation_group():
    """Test rotation group operations."""
    # Test rotation matrix generation
    for axis in [RotationAxis.X, RotationAxis.Y, RotationAxis.Z]:
        matrix = RotationGroup.get_rotation_matrix(axis, quarter_turns=1)
        assert matrix.shape == (3, 3), f"Rotation matrix for {axis.name} should be 3×3"
        
        # Test that 4 rotations = identity
        matrix_4 = RotationGroup.get_rotation_matrix(axis, quarter_turns=4)
        identity = np.eye(3)
        assert np.allclose(matrix_4, identity), f"4 rotations around {axis.name} should be identity"
    
    # Test coordinate rotation
    coords = (1, 0, 0)
    rotated = RotationGroup.rotate_coordinates(coords, RotationAxis.Z, quarter_turns=1)
    assert rotated == (0, 1, 0), "Rotation (1,0,0) around Z should give (0,1,0)"


def test_semantic_polarity():
    """Test semantic polarity calculation."""
    system = LivniumCoreSystem()
    
    # Test polarity with motion vector
    polarity = system.calculate_polarity((1, 0, 0))
    assert -1.0 <= polarity <= 1.0, "Polarity should be in [-1, 1]"
    
    # Test polarity with target coordinates
    polarity_target = system.calculate_polarity((0, 0, 0), target_coords=(1, 0, 0))
    assert -1.0 <= polarity_target <= 1.0, "Polarity with target should be in [-1, 1]"


def test_local_observer():
    """Test local observer creation."""
    system = LivniumCoreSystem()
    
    observer = system.set_local_observer((1, 0, 0))
    assert observer.coordinates == (1, 0, 0), "Local observer should be at specified coordinates"
    assert observer.is_local, "Observer should be marked as local"
    assert not observer.is_global, "Local observer should not be global"
    assert len(system.local_observers) == 1, "Should have 1 local observer"


def test_generalized_n():
    """Test system with different N values."""
    for n in [3, 5, 7]:
        config = LivniumCoreConfig(lattice_size=n)
        system = LivniumCoreSystem(config)
        
        assert system.lattice_size == n, f"Lattice size should be {n}"
        assert len(system.lattice) == n**3, f"Should have {n**3} cells for N={n}"
        
        total_sw = system.get_total_symbolic_weight()
        expected_sw = system.get_expected_total_sw()
        assert abs(total_sw - expected_sw) < 1e-6, \
            f"For N={n}, total SW should be {expected_sw}, got {total_sw}"


def test_config_feature_switches():
    """Test feature switches in configuration."""
    # Test with some features disabled (respecting all dependencies)
    config = LivniumCoreConfig(
        enable_symbol_alphabet=False,
        enable_symbolic_weight=False,
        enable_face_exposure=False,
        enable_class_structure=False,
        enable_global_observer=False,
        enable_local_observer=False,  # Must disable if global is disabled
        enable_semantic_polarity=False,  # Must disable if global observer is disabled
        enable_90_degree_rotations=False,
        enable_sw_conservation=False,  # Must disable if symbolic_weight is disabled
        enable_class_count_conservation=False  # Must disable if class_structure is disabled
    )
    system = LivniumCoreSystem(config)
    
    assert system.global_observer is None, "Global observer should be None when disabled"
    assert len(system.symbol_map) == 0, "Symbol map should be empty when disabled"
    
    # Test rotation with feature disabled
    try:
        system.rotate(RotationAxis.X)
        assert False, "Rotation should fail when feature is disabled"
    except ValueError:
        pass  # Expected


def test_system_summary():
    """Test system summary generation."""
    system = LivniumCoreSystem()
    summary = system.get_system_summary()
    
    assert 'lattice_size' in summary, "Summary should include lattice_size"
    assert 'total_cells' in summary, "Summary should include total_cells"
    assert 'features_enabled' in summary, "Summary should include features_enabled"
    assert summary['total_cells'] == 27, "Should have 27 cells for N=3"


def test_invalid_lattice_size():
    """Test that invalid lattice sizes raise errors."""
    # Test even N
    try:
        config = LivniumCoreConfig(lattice_size=4)
        system = LivniumCoreSystem(config)
        assert False, "Even lattice size should raise ValueError"
    except ValueError:
        pass  # Expected
    
    # Test N < 3
    try:
        config = LivniumCoreConfig(lattice_size=1)
        system = LivniumCoreSystem(config)
        assert False, "Lattice size < 3 should raise ValueError"
    except ValueError:
        pass  # Expected


if __name__ == "__main__":
    print("Running LivniumCoreSystem tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_cell_classification()
    print("✓ Cell classification")
    
    test_symbolic_weight_formula()
    print("✓ Symbolic weight formula")
    
    test_total_symbolic_weight()
    print("✓ Total symbolic weight")
    
    test_class_counts()
    print("✓ Class counts")
    
    test_symbol_alphabet()
    print("✓ Symbol alphabet")
    
    test_rotation_preserves_invariants()
    print("✓ Rotation preserves invariants")
    
    test_rotation_around_all_axes()
    print("✓ Rotation around all axes")
    
    test_full_rotation_identity()
    print("✓ Full rotation identity")
    
    test_rotation_group()
    print("✓ Rotation group")
    
    test_semantic_polarity()
    print("✓ Semantic polarity")
    
    test_local_observer()
    print("✓ Local observer")
    
    test_generalized_n()
    print("✓ Generalized N")
    
    test_config_feature_switches()
    print("✓ Config feature switches")
    
    test_system_summary()
    print("✓ System summary")
    
    test_invalid_lattice_size()
    print("✓ Invalid lattice size")
    
    print("\nAll tests passed! ✓")

