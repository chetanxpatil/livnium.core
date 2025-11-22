"""
Simple Test for Constraint Encoder

Tests that constraints are correctly encoded as tension fields.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
import importlib

# Import with space in module name
encoder_module = importlib.import_module('core.encoder.constraint_encoder')
ConstraintEncoder = encoder_module.ConstraintEncoder
TensionField = encoder_module.TensionField


def test_equality_constraint():
    """Test equality constraint encoding."""
    print("="*60)
    print("Test 1: Equality Constraint → Tension Field")
    print("="*60)
    
    # Create system
    config = LivniumCoreConfig(lattice_size=3)
    system = LivniumCoreSystem(config)
    
    # Create encoder
    encoder = ConstraintEncoder(system)
    
    # Define variables
    var1_coords = [(0, 0, 0), (1, 0, 0)]
    var2_coords = [(0, 1, 0), (1, 1, 0)]
    
    # Encode constraint: var1 = var2
    field = encoder.encode_equality_constraint(
        constraint_id="eq1",
        var1_coords=var1_coords,
        var2_coords=var2_coords,
        target_value=0.0
    )
    
    print(f"  Constraint: {field.description}")
    print(f"  Involved coords: {len(field.involved_coords)}")
    
    # Set initial values (violating constraint)
    system.get_cell((0, 0, 0)).symbolic_weight = 20.0
    system.get_cell((1, 0, 0)).symbolic_weight = 20.0
    system.get_cell((0, 1, 0)).symbolic_weight = 10.0
    system.get_cell((1, 1, 0)).symbolic_weight = 10.0
    
    # Check tension (should be high - constraint violated)
    tension = field.get_tension(system)
    curvature = field.get_curvature(system)
    
    print(f"  Tension (violated): {tension:.4f}")
    print(f"  Curvature (violated): {curvature:.4f}")
    
    # Fix constraint (satisfy it)
    system.get_cell((0, 1, 0)).symbolic_weight = 20.0
    system.get_cell((1, 1, 0)).symbolic_weight = 20.0
    
    # Check tension (should be low - constraint satisfied)
    tension_sat = field.get_tension(system)
    curvature_sat = field.get_curvature(system)
    
    print(f"  Tension (satisfied): {tension_sat:.4f}")
    print(f"  Curvature (satisfied): {curvature_sat:.4f}")
    
    # Verify
    assert tension > 0.0, "Tension should be positive when violated"
    assert tension_sat < tension, "Tension should decrease when satisfied"
    assert curvature_sat > curvature, "Curvature should increase when satisfied"
    
    print("  ✓ Constraint correctly encoded as tension field")
    print()


def test_multiple_constraints():
    """Test multiple constraints creating tension landscape."""
    print("="*60)
    print("Test 2: Multiple Constraints → Tension Landscape")
    print("="*60)
    
    # Create system
    config = LivniumCoreConfig(lattice_size=3)
    system = LivniumCoreSystem(config)
    
    # Create encoder
    encoder = ConstraintEncoder(system)
    
    # Add multiple constraints
    field1 = encoder.encode_equality_constraint(
        "eq1", [(0,0,0)], [(1,0,0)], 0.0
    )
    field2 = encoder.encode_equality_constraint(
        "eq2", [(0,1,0)], [(1,1,0)], 0.0
    )
    
    # Set values (violate both)
    system.get_cell((0,0,0)).symbolic_weight = 20.0
    system.get_cell((1,0,0)).symbolic_weight = 10.0
    system.get_cell((0,1,0)).symbolic_weight = 20.0
    system.get_cell((1,1,0)).symbolic_weight = 10.0
    
    # Get individual tensions
    tensions = encoder.get_constraint_tensions(system)
    total_tension = encoder.get_total_tension(system)
    
    print(f"  Constraint 1 tension: {tensions['eq1']:.4f}")
    print(f"  Constraint 2 tension: {tensions['eq2']:.4f}")
    print(f"  Total tension: {total_tension:.4f}")
    
    # Verify
    assert total_tension == tensions['eq1'] + tensions['eq2']
    assert total_tension > 0.0
    
    print("  ✓ Multiple constraints create tension landscape")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("CONSTRAINT ENCODER - TEST SUITE")
    print("="*60)
    print()
    
    try:
        test_equality_constraint()
        test_multiple_constraints()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print()
        print("Summary:")
        print("  ✓ Constraints encoded as tension fields (NOT basins)")
        print("  ✓ Tension increases when constraint violated")
        print("  ✓ Tension decreases when constraint satisfied")
        print("  ✓ Multiple constraints create tension landscape")
        print()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()

