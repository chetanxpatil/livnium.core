"""
Tests for Recursive Simplex Engine

Verifies recursive simplex functionality:
- RecursiveSimplexEngine initialization and hierarchy building
- SimplexSubdivision logic
- RecursiveProjection functionality
- RecursiveConservation invariants
- MokshaEngine convergence detection
"""

import unittest
import sys
from pathlib import Path

# Add core-t directory to path
core_t_path = Path(__file__).parent.parent
sys.path.insert(0, str(core_t_path))

import numpy as np

# Import directly from the module files
from classical.livnium_t_system import LivniumTSystem

# Import recursive modules directly (bypassing __init__.py to avoid relative import issues)
import sys
recursive_dir = core_t_path / "recursive"
sys.path.insert(0, str(recursive_dir))

from recursive_simplex_engine import RecursiveSimplexEngine
from simplex_subdivision import SimplexSubdivision
from recursive_projection import RecursiveProjection
from recursive_conservation import RecursiveConservation
from moksha_engine import (
    MokshaEngine,
    FixedPointState,
    ConvergenceState,
)


class TestRecursiveSimplexEngine(unittest.TestCase):
    """Test suite for Recursive Simplex Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_geometry = LivniumTSystem()
        self.engine = RecursiveSimplexEngine(
            base_geometry=self.base_geometry,
            max_depth=2  # Keep depth small for testing
        )
    
    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.base_geometry, self.base_geometry)
        self.assertEqual(self.engine.max_depth, 2)
        self.assertIsNotNone(self.engine.levels)
        self.assertIn(0, self.engine.levels)
    
    def test_hierarchy_building(self):
        """Test that hierarchy is built correctly."""
        # Should have at least level 0
        self.assertIn(0, self.engine.levels)
        level_0 = self.engine.levels[0]
        
        # Level 0 should have base geometry
        self.assertEqual(level_0.geometry, self.base_geometry)
        self.assertEqual(level_0.level_id, 0)
        self.assertIsNone(level_0.parent)
    
    def test_get_level(self):
        """Test getting level by ID."""
        level_0 = self.engine.get_level(0)
        self.assertIsNotNone(level_0)
        self.assertEqual(level_0.level_id, 0)
    
    def test_get_total_capacity(self):
        """Test total capacity calculation."""
        capacity = self.engine.get_total_capacity()
        self.assertGreater(capacity, 0)
        # Should have at least 5 nodes (base level)
        self.assertGreaterEqual(capacity, 5)
    
    def test_components_initialized(self):
        """Test that all components are initialized."""
        self.assertIsNotNone(self.engine.subdivision)
        self.assertIsNotNone(self.engine.projection)
        self.assertIsNotNone(self.engine.conservation)
        self.assertIsNotNone(self.engine.moksha)


class TestSimplexSubdivision(unittest.TestCase):
    """Test suite for Simplex Subdivision."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_geometry = LivniumTSystem()
        self.engine = RecursiveSimplexEngine(
            base_geometry=self.base_geometry,
            max_depth=1
        )
        self.subdivision = self.engine.subdivision
    
    def test_subdivision_initialization(self):
        """Test that subdivision initializes correctly."""
        self.assertIsNotNone(self.subdivision)
        self.assertEqual(self.subdivision.recursive_engine, self.engine)
    
    def test_should_subdivide(self):
        """Test subdivision decision logic."""
        # Core (node 0) should not subdivide
        should_subdivide_core = self.subdivision.should_subdivide(0, self.base_geometry)
        self.assertFalse(should_subdivide_core)
        
        # Vertices (nodes 1-4) should subdivide
        for node_id in range(1, 5):
            should_subdivide = self.subdivision.should_subdivide(node_id, self.base_geometry)
            self.assertTrue(should_subdivide)


class TestMokshaEngine(unittest.TestCase):
    """Test suite for Moksha Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_geometry = LivniumTSystem()
        self.engine = RecursiveSimplexEngine(
            base_geometry=self.base_geometry,
            max_depth=1
        )
        self.moksha = self.engine.moksha
    
    def test_moksha_initialization(self):
        """Test that Moksha engine initializes correctly."""
        self.assertIsNotNone(self.moksha)
        self.assertEqual(self.moksha.recursive_engine, self.engine)
        self.assertEqual(self.moksha.convergence_threshold, 0.999)
        self.assertEqual(self.moksha.stability_window, 10)
        self.assertFalse(self.moksha.moksha_reached)
        self.assertEqual(len(self.moksha.state_history), 0)
        self.assertEqual(len(self.moksha.fixed_points), 0)
    
    def test_check_convergence_initial_state(self):
        """Test convergence check on initial state."""
        # Initially should be searching (not enough history)
        state = self.moksha.check_convergence()
        self.assertEqual(state, ConvergenceState.SEARCHING)
    
    def test_capture_full_state(self):
        """Test state capture functionality."""
        state = self.moksha._capture_full_state()
        
        # Should have timestamp and levels
        self.assertIn('timestamp', state)
        self.assertIn('levels', state)
        self.assertIn('state_hash', state)
        
        # Should have level 0
        self.assertIn(0, state['levels'])
        level_state = state['levels'][0]
        
        # Should have total_sw, class_counts, node_states
        self.assertIn('total_sw', level_state)
        self.assertIn('class_counts', level_state)
        self.assertIn('node_states', level_state)
        
        # Should have 5 nodes
        self.assertEqual(len(level_state['node_states']), 5)
    
    def test_state_hash(self):
        """Test state hashing."""
        state1 = self.moksha._capture_full_state()
        state2 = self.moksha._capture_full_state()
        
        # Same state should produce same hash
        self.assertEqual(state1['state_hash'], state2['state_hash'])
    
    def test_is_state_stable(self):
        """Test state stability checking."""
        state = self.moksha._capture_full_state()
        
        # Single state is not stable (need at least 2)
        self.assertFalse(self.moksha._is_state_stable([state]))
        
        # Two identical states should be stable
        states = [state, state]
        self.assertTrue(self.moksha._is_state_stable(states))
    
    def test_get_convergence_score(self):
        """Test convergence score calculation."""
        # Initially should be 0
        score = self.moksha.get_convergence_score()
        self.assertEqual(score, 0.0)
        
        # After moksha reached, should be 1.0
        self.moksha.moksha_reached = True
        score = self.moksha.get_convergence_score()
        self.assertEqual(score, 1.0)
    
    def test_export_final_truth(self):
        """Test final truth export."""
        # Before moksha
        truth = self.moksha.export_final_truth()
        self.assertFalse(truth['moksha'])
        self.assertIn('message', truth)
        
        # After moksha
        self.moksha.moksha_reached = True
        self.moksha.moksha_state = self.moksha._capture_full_state()
        truth = self.moksha.export_final_truth()
        self.assertTrue(truth['moksha'])
        self.assertEqual(truth['convergence_score'], 1.0)
    
    def test_should_terminate(self):
        """Test termination check."""
        self.assertFalse(self.moksha.should_terminate())
        
        self.moksha.moksha_reached = True
        self.assertTrue(self.moksha.should_terminate())
    
    def test_reset(self):
        """Test reset functionality."""
        # Add some state
        self.moksha.state_history.append(self.moksha._capture_full_state())
        self.moksha.moksha_reached = True
        
        # Reset
        self.moksha.reset()
        
        # Should be cleared
        self.assertEqual(len(self.moksha.state_history), 0)
        self.assertFalse(self.moksha.moksha_reached)
        self.assertIsNone(self.moksha.moksha_state)
        self.assertEqual(len(self.moksha.fixed_points), 0)
    
    def test_get_moksha_statistics(self):
        """Test statistics retrieval."""
        stats = self.moksha.get_moksha_statistics()
        
        self.assertIn('moksha_reached', stats)
        self.assertIn('convergence_score', stats)
        self.assertIn('state_history_size', stats)
        self.assertIn('fixed_points_count', stats)
        self.assertIn('convergence_threshold', stats)


class TestRecursiveProjection(unittest.TestCase):
    """Test suite for Recursive Projection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_geometry = LivniumTSystem()
        self.engine = RecursiveSimplexEngine(
            base_geometry=self.base_geometry,
            max_depth=1
        )
        self.projection = self.engine.projection
    
    def test_projection_initialization(self):
        """Test that projection initializes correctly."""
        self.assertIsNotNone(self.projection)
        self.assertEqual(self.projection.recursive_engine, self.engine)


class TestRecursiveConservation(unittest.TestCase):
    """Test suite for Recursive Conservation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_geometry = LivniumTSystem()
        self.engine = RecursiveSimplexEngine(
            base_geometry=self.base_geometry,
            max_depth=1
        )
        self.conservation = self.engine.conservation
    
    def test_conservation_initialization(self):
        """Test that conservation initializes correctly."""
        self.assertIsNotNone(self.conservation)
        self.assertEqual(self.conservation.recursive_engine, self.engine)


class TestFixedPointState(unittest.TestCase):
    """Test suite for FixedPointState."""
    
    def test_fixed_point_creation(self):
        """Test FixedPointState creation."""
        fp = FixedPointState(
            level_id=0,
            node_id=0,
            state_hash="test_hash",
            convergence_score=1.0
        )
        
        self.assertEqual(fp.level_id, 0)
        self.assertEqual(fp.node_id, 0)
        self.assertEqual(fp.state_hash, "test_hash")
        self.assertEqual(fp.convergence_score, 1.0)
    
    def test_fixed_point_hashing(self):
        """Test FixedPointState hashing."""
        fp1 = FixedPointState(level_id=0, node_id=0, state_hash="hash1")
        fp2 = FixedPointState(level_id=0, node_id=0, state_hash="hash1")
        fp3 = FixedPointState(level_id=0, node_id=0, state_hash="hash2")
        
        # Same state should have same hash
        self.assertEqual(hash(fp1), hash(fp2))
        
        # Different state should have different hash
        self.assertNotEqual(hash(fp1), hash(fp3))


class TestConvergenceState(unittest.TestCase):
    """Test suite for ConvergenceState enum."""
    
    def test_convergence_states(self):
        """Test all convergence states exist."""
        self.assertEqual(ConvergenceState.SEARCHING.value, "searching")
        self.assertEqual(ConvergenceState.CONVERGING.value, "converging")
        self.assertEqual(ConvergenceState.MOKSHA.value, "moksha")
        self.assertEqual(ConvergenceState.DIVERGING.value, "diverging")


if __name__ == '__main__':
    unittest.main()

