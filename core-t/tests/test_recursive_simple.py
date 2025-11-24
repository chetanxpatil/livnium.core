"""
Simple test script for Recursive Simplex Engine

Tests basic functionality without complex import setup.
"""

import sys
from pathlib import Path

# Add core-t directory to path
core_t_path = Path(__file__).parent.parent
sys.path.insert(0, str(core_t_path))

# Import directly from module files
from classical.livnium_t_system import LivniumTSystem

# Import recursive modules - need to handle relative imports
# Add parent directory to path so relative imports work
sys.path.insert(0, str(core_t_path.parent))
import importlib.util

# Load recursive modules manually to handle relative imports
recursive_dir = core_t_path / "recursive"
spec = importlib.util.spec_from_file_location(
    "recursive_simplex_engine", 
    recursive_dir / "recursive_simplex_engine.py"
)
recursive_simplex_engine_module = importlib.util.module_from_spec(spec)
# Set up the module's __package__ so relative imports work
recursive_simplex_engine_module.__package__ = "core_t.recursive"
sys.modules['core_t'] = type(sys)('core_t')
sys.modules['core_t.recursive'] = type(sys)('core_t.recursive')
sys.modules['core_t.classical'] = type(sys)('core_t.classical')
spec.loader.exec_module(recursive_simplex_engine_module)

spec2 = importlib.util.spec_from_file_location(
    "moksha_engine",
    recursive_dir / "moksha_engine.py"
)
moksha_engine_module = importlib.util.module_from_spec(spec2)
moksha_engine_module.__package__ = "core_t.recursive"
spec2.loader.exec_module(moksha_engine_module)

RecursiveSimplexEngine = recursive_simplex_engine_module.RecursiveSimplexEngine
MokshaEngine = moksha_engine_module.MokshaEngine
FixedPointState = moksha_engine_module.FixedPointState
ConvergenceState = moksha_engine_module.ConvergenceState


def test_basic_functionality():
    """Test basic recursive engine functionality."""
    print("Testing Recursive Simplex Engine...")
    
    # Create base geometry
    base_geometry = LivniumTSystem()
    print(f"✓ Created base geometry with {len(base_geometry.nodes)} nodes")
    
    # Create recursive engine
    engine = RecursiveSimplexEngine(
        base_geometry=base_geometry,
        max_depth=1
    )
    print(f"✓ Created recursive engine with {len(engine.levels)} levels")
    
    # Test components
    assert engine.subdivision is not None, "Subdivision should be initialized"
    print("✓ Subdivision component initialized")
    
    assert engine.projection is not None, "Projection should be initialized"
    print("✓ Projection component initialized")
    
    assert engine.conservation is not None, "Conservation should be initialized"
    print("✓ Conservation component initialized")
    
    assert engine.moksha is not None, "Moksha should be initialized"
    print("✓ Moksha component initialized")
    
    # Test moksha engine
    moksha = engine.moksha
    assert moksha.recursive_engine == engine, "Moksha should reference engine"
    print("✓ Moksha engine references recursive engine correctly")
    
    # Test state capture
    state = moksha._capture_full_state()
    assert 'levels' in state, "State should have levels"
    assert 0 in state['levels'], "State should have level 0"
    assert 'state_hash' in state, "State should have hash"
    print("✓ State capture works correctly")
    
    # Test convergence check
    convergence = moksha.check_convergence()
    assert convergence == ConvergenceState.SEARCHING, "Initial state should be searching"
    print("✓ Convergence check works correctly")
    
    # Test statistics
    stats = moksha.get_moksha_statistics()
    assert 'moksha_reached' in stats, "Stats should have moksha_reached"
    assert 'convergence_score' in stats, "Stats should have convergence_score"
    print("✓ Statistics retrieval works correctly")
    
    # Test capacity
    capacity = engine.get_total_capacity()
    assert capacity >= 5, f"Capacity should be at least 5, got {capacity}"
    print(f"✓ Total capacity: {capacity} nodes")
    
    print("\n✅ All basic tests passed!")


if __name__ == '__main__':
    try:
        test_basic_functionality()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

