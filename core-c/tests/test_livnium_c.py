"""
Test suite for Livnium-C System

Tests the canonical axioms and derived laws.
"""

import unittest
import numpy as np
from ..classical import (
    LivniumCSystem,
    CircleNode,
    Observer,
    NodeClass,
    CyclicRotationGroup,
)


class TestLivniumCSystem(unittest.TestCase):
    """Test the Livnium-C System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = LivniumCSystem(n_ring=8, radius=1.0)
    
    def test_structure(self):
        """Test S1: Circle structure forms correctly."""
        # Should have 1 core + N ring nodes
        self.assertEqual(self.system.total_nodes, 9)  # 1 + 8
        self.assertEqual(len(self.system.nodes), 9)
        
        # Core should be node 0
        core = self.system.get_core()
        self.assertEqual(core.node_id, 0)
        self.assertTrue(core.is_om)
        self.assertFalse(core.is_lo)
        self.assertEqual(core.exposure, 0)
        self.assertEqual(core.symbolic_weight, 0)
        
        # Should have N ring nodes
        ring_nodes = self.system.get_ring_nodes()
        self.assertEqual(len(ring_nodes), 8)
        
        for node in ring_nodes:
            self.assertFalse(node.is_om)
            self.assertTrue(node.is_lo)
            self.assertEqual(node.exposure, 1)
            self.assertEqual(node.symbolic_weight, 9)
    
    def test_exposure_classes(self):
        """Test S3: Two-class system verified."""
        core = self.system.get_core()
        ring_nodes = self.system.get_ring_nodes()
        
        # Core should have f=0
        self.assertEqual(core.exposure, 0)
        self.assertEqual(core.node_class, NodeClass.CORE)
        
        # All ring nodes should have f=1
        for node in ring_nodes:
            self.assertEqual(node.exposure, 1)
            self.assertEqual(node.node_class, NodeClass.RING)
    
    def test_symbolic_weight(self):
        """Test C-A4: Symbolic Weight Law."""
        # Total SW should be 9N
        expected_sw = 9 * self.system.n_ring
        self.assertEqual(self.system.total_symbolic_weight, expected_sw)
        
        # Core should have SW=0
        core = self.system.get_core()
        self.assertEqual(core.symbolic_weight, 0)
        
        # Ring nodes should have SW=9
        ring_nodes = self.system.get_ring_nodes()
        for node in ring_nodes:
            self.assertEqual(node.symbolic_weight, 9)
    
    def test_equilibrium_constant(self):
        """Test C-D1: Circle Equilibrium Constant."""
        # K_C should be 9
        self.assertEqual(self.system.equilibrium_constant, 9.0)
    
    def test_rotation_group(self):
        """Test C-A5: Cyclic Rotation Group."""
        rotation_group = self.system.rotation_group
        
        # Should have order N
        self.assertEqual(rotation_group.order, 8)
        
        # Should have N rotations
        self.assertEqual(len(rotation_group.rotations), 8)
        
        # Identity rotation (0) should map i -> i
        identity = rotation_group.get_rotation(0)
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(identity, expected)
    
    def test_rotation(self):
        """Test R1: Rotation bijection."""
        original_ring = self.system.get_ring_nodes()
        original_angles = [node.angle for node in original_ring]
        
        # Apply rotation
        rotated_system = self.system.rotate(rotation_id=1)
        rotated_ring = rotated_system.get_ring_nodes()
        rotated_angles = [node.angle for node in rotated_ring]
        
        # Should have same number of nodes
        self.assertEqual(len(rotated_angles), len(original_angles))
        
        # Angles should be rotated by 2π/N
        angle_step = 2 * np.pi / self.system.n_ring
        expected_angles = [(a + angle_step) % (2 * np.pi) for a in original_angles]
        
        for actual, expected in zip(rotated_angles, expected_angles):
            self.assertAlmostEqual(actual, expected, places=10)
    
    def test_rotation_inverse(self):
        """Test R1: Rotation inverse."""
        rotation_group = self.system.rotation_group
        
        # Test inverse for each rotation
        for k in range(rotation_group.order):
            inverse_k = rotation_group.get_inverse(k)
            # k + inverse_k should be 0 mod N
            self.assertEqual((k + inverse_k) % rotation_group.order, 0)
    
    def test_rotation_reversibility(self):
        """Test R1: Perfect reversibility."""
        original_ring = self.system.get_ring_nodes()
        original_angles = np.array([node.angle for node in original_ring])
        
        # Apply rotation and its inverse
        rotated_system = self.system.rotate(rotation_id=1)
        inverse_id = self.system.rotation_group.get_inverse(1)
        restored_system = rotated_system.rotate(rotation_id=inverse_id)
        
        restored_ring = restored_system.get_ring_nodes()
        restored_angles = np.array([node.angle for node in restored_ring])
        
        # Should restore original angles
        np.testing.assert_allclose(restored_angles, original_angles, rtol=1e-10)
    
    def test_ledger_conservation(self):
        """Test L1: Conservation Ledger."""
        original_ledger = self.system.get_ledger()
        
        # Apply rotation
        rotated_system = self.system.rotate(rotation_id=1)
        rotated_ledger = rotated_system.get_ledger()
        
        # Ledger should be preserved
        self.assertEqual(original_ledger['total_sw'], rotated_ledger['total_sw'])
        self.assertEqual(original_ledger['core_count'], rotated_ledger['core_count'])
        self.assertEqual(original_ledger['ring_count'], rotated_ledger['ring_count'])
        self.assertEqual(original_ledger['equilibrium_constant'], rotated_ledger['equilibrium_constant'])
        self.assertEqual(original_ledger['encoding_base'], rotated_ledger['encoding_base'])
    
    def test_encoding_base(self):
        """Test C-D5: Base-(N+1) Encoding Law."""
        # Encoding base should be N+1
        expected_base = self.system.n_ring + 1
        self.assertEqual(self.system.encoding_base, expected_base)
    
    def test_different_n(self):
        """Test system with different N values."""
        for n in [4, 6, 12, 16]:
            system = LivniumCSystem(n_ring=n, radius=1.0)
            self.assertEqual(system.n_ring, n)
            self.assertEqual(system.total_nodes, n + 1)
            self.assertEqual(system.total_symbolic_weight, 9 * n)
            self.assertEqual(system.encoding_base, n + 1)
            self.assertEqual(system.rotation_group.order, n)


if __name__ == "__main__":
    unittest.main()

