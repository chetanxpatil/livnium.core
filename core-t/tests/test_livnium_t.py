"""
Tests for Livnium-T System

Verifies all axioms and derived laws:
- T-A1: Canonical Simplex Alphabet (5-node topology)
- T-A2: Observer Anchor & Frame (Om-Simplex)
- T-A3: Exposure Law (Two-Class System)
- T-A4: Symbolic Weight Law (SW = 9·f, ΣSW = 108)
- T-A5: Dynamic Law (Tetrahedral Rotation Group A₄)
- T-A6: Connection & Activation Rule

- T-D1: Simplex Equilibrium Constant (K_T = 27)
- T-D2: Exposure Density Law
- T-D3: Conservation Ledger
- T-D4: Perfect Reversibility Law
"""

import unittest
import sys
from pathlib import Path

# Add core-t directory to path
core_t_path = Path(__file__).parent.parent
sys.path.insert(0, str(core_t_path))

import numpy as np

# Import directly from the module file
from classical.livnium_t_system import (
    LivniumTSystem,
    SimplexNode,
    Observer,
    NodeClass,
    TetrahedralRotationGroup,
)


class TestLivniumTSystem(unittest.TestCase):
    """Test suite for Livnium-T System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = LivniumTSystem()
    
    def test_t_a1_canonical_simplex_alphabet(self):
        """T-A1: Verify 5-node topology."""
        # Should have exactly 5 nodes
        self.assertEqual(len(self.system.nodes), 5)
        
        # Should have 1 core node
        core_nodes = [n for n in self.system.nodes.values() if n.node_class == NodeClass.CORE]
        self.assertEqual(len(core_nodes), 1)
        self.assertEqual(core_nodes[0].node_id, 0)
        
        # Should have 4 vertex nodes
        vertex_nodes = [n for n in self.system.nodes.values() if n.node_class == NodeClass.VERTEX]
        self.assertEqual(len(vertex_nodes), 4)
        self.assertEqual(set(n.node_id for n in vertex_nodes), {1, 2, 3, 4})
    
    def test_t_a2_observer_anchor(self):
        """T-A2: Verify Om observer anchor."""
        # Om observer should exist
        self.assertIsNotNone(self.system.om_observer)
        self.assertTrue(self.system.om_observer.is_om)
        self.assertFalse(self.system.om_observer.is_lo)
        self.assertEqual(self.system.om_observer.node_id, 0)
        
        # Om node should be immovable
        om_node = self.system.get_node(0)
        self.assertTrue(om_node.is_om)
        self.assertFalse(om_node.is_lo)
    
    def test_t_a3_exposure_law(self):
        """T-A3: Verify two-class system."""
        # Core should have f=0
        core_node = self.system.get_node(0)
        self.assertEqual(core_node.exposure, 0)
        self.assertEqual(core_node.node_class, NodeClass.CORE)
        
        # Vertices should have f=3
        for i in range(1, 5):
            vertex_node = self.system.get_node(i)
            self.assertEqual(vertex_node.exposure, 3)
            self.assertEqual(vertex_node.node_class, NodeClass.VERTEX)
        
        # Should only have 2 classes
        classes = set(n.node_class for n in self.system.nodes.values())
        self.assertEqual(classes, {NodeClass.CORE, NodeClass.VERTEX})
    
    def test_t_a4_symbolic_weight_law(self):
        """T-A4: Verify symbolic weight law."""
        # Core should have SW=0
        core_node = self.system.get_node(0)
        self.assertEqual(core_node.symbolic_weight, 0.0)
        self.assertEqual(core_node.symbolic_weight, 9 * core_node.exposure)
        
        # Vertices should have SW=27
        for i in range(1, 5):
            vertex_node = self.system.get_node(i)
            self.assertEqual(vertex_node.symbolic_weight, 27.0)
            self.assertEqual(vertex_node.symbolic_weight, 9 * vertex_node.exposure)
        
        # Total SW should be 108
        total_sw = self.system.get_total_sw()
        self.assertEqual(total_sw, 108.0)
        self.assertEqual(total_sw, LivniumTSystem.CANONICAL_TOTAL_SW)
    
    def test_t_a5_rotation_group(self):
        """T-A5: Verify tetrahedral rotation group."""
        rotation_group = self.system.rotation_group
        
        # Should have order 12
        self.assertEqual(rotation_group.order, 12)
        
        # All rotations should be valid
        for i in range(12):
            rotation_matrix = rotation_group.get_rotation(i)
            self.assertEqual(rotation_matrix.shape, (3, 3))
            
            # Should be orthogonal (rotation matrices preserve length)
            self.assertTrue(np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3)))
            
            # Should have determinant 1 (orientation-preserving)
            self.assertAlmostEqual(np.linalg.det(rotation_matrix), 1.0, places=6)
        
        # Identity should be rotation 0
        identity = rotation_group.get_rotation(0)
        self.assertTrue(np.allclose(identity, np.eye(3)))
    
    def test_t_d1_equilibrium_constant(self):
        """T-D1: Verify equilibrium constant K_T = 27."""
        k_t = self.system.get_equilibrium_constant()
        self.assertEqual(k_t, 27.0)
        self.assertEqual(k_t, LivniumTSystem.CANONICAL_EQUILIBRIUM_CONSTANT)
        
        # Verify derivation: 108 = 12 × (K_T / 3)
        total_sw = self.system.get_total_sw()
        total_faces = 4 * 3  # 4 vertices × 3 faces each
        concentration = k_t / 3
        self.assertAlmostEqual(total_sw, total_faces * concentration, places=6)
    
    def test_t_d3_conservation_ledger(self):
        """T-D3: Verify conservation ledger."""
        # Verify ledger is intact
        self.assertTrue(self.system.verify_ledger())
        
        # Check total SW
        total_sw = self.system.get_total_sw()
        self.assertEqual(total_sw, 108.0)
        
        # Check class counts
        counts = self.system.get_class_counts()
        self.assertEqual(counts[NodeClass.CORE], 1)
        self.assertEqual(counts[NodeClass.VERTEX], 4)
        
        # Check Om observer
        self.assertIsNotNone(self.system.om_observer)
        self.assertTrue(self.system.om_observer.is_om)
    
    def test_t_d4_reversibility(self):
        """T-D4: Verify perfect reversibility."""
        rotation_group = self.system.rotation_group
        
        # Every rotation should have an inverse
        for i in range(12):
            inverse_id = rotation_group.get_inverse(i)
            rotation = rotation_group.get_rotation(i)
            inverse_rotation = rotation_group.get_rotation(inverse_id)
            
            # Composition should be identity
            composition = rotation @ inverse_rotation
            self.assertTrue(np.allclose(composition, np.eye(3)))
            
            # Inverse of inverse should be original
            inverse_inverse_id = rotation_group.get_inverse(inverse_id)
            self.assertEqual(inverse_inverse_id, i)
    
    def test_node_validation(self):
        """Test node validation."""
        # Core node with wrong exposure should fail
        with self.assertRaises(ValueError):
            SimplexNode(
                node_id=0,
                node_class=NodeClass.CORE,
                exposure=3,  # Wrong!
                symbolic_weight=0.0
            )
        
        # Vertex node with wrong exposure should fail
        with self.assertRaises(ValueError):
            SimplexNode(
                node_id=1,
                node_class=NodeClass.VERTEX,
                exposure=0,  # Wrong!
                symbolic_weight=27.0
            )
    
    def test_observer_validation(self):
        """Test observer validation."""
        # Om observer with wrong node_id should fail
        with self.assertRaises(ValueError):
            Observer(node_id=1, is_om=True)
        
        # LO observer with wrong node_id should fail
        with self.assertRaises(ValueError):
            Observer(node_id=0, is_om=False)
        with self.assertRaises(ValueError):
            Observer(node_id=5, is_om=False)
    
    def test_t_d5_base5_encoding(self):
        """T-D5: Verify base-5 encoding law."""
        # Base should be 5
        self.assertEqual(self.system.BASE, 5)
        
        # Test encoding
        sequence = [2, 4, 1, 3]
        encoded = self.system.encode_base5(sequence)
        expected = 2 * 5**3 + 4 * 5**2 + 1 * 5**1 + 3 * 5**0
        self.assertEqual(encoded, expected)
        self.assertEqual(encoded, 358)
        
        # Test decoding
        decoded = self.system.decode_base5(encoded, length=len(sequence))
        self.assertEqual(decoded, sequence)
        
        # Test reversibility
        for test_seq in [[0], [1, 2], [3, 4, 0], [0, 1, 2, 3, 4]]:
            encoded = self.system.encode_base5(test_seq)
            decoded = self.system.decode_base5(encoded, length=len(test_seq))
            self.assertEqual(decoded, test_seq)
        
        # Test validation
        with self.assertRaises(ValueError):
            self.system.encode_base5([5])  # Invalid digit
        with self.assertRaises(ValueError):
            self.system.encode_base5([-1])  # Invalid digit
    
    def test_scaling_formula(self):
        """Test D-simplex scaling formula."""
        # D=3 (current tetrahedron)
        self.assertEqual(LivniumTSystem.compute_total_sw(3), 108.0)  # 9·3·4
        self.assertEqual(LivniumTSystem.compute_vertex_sw(3), 27.0)  # 9·3
        
        # D=4 (4-simplex)
        self.assertEqual(LivniumTSystem.compute_total_sw(4), 180.0)  # 9·4·5
        self.assertEqual(LivniumTSystem.compute_vertex_sw(4), 36.0)  # 9·4
        
        # D=5 (5-simplex)
        self.assertEqual(LivniumTSystem.compute_total_sw(5), 270.0)  # 9·5·6
        self.assertEqual(LivniumTSystem.compute_vertex_sw(5), 45.0)  # 9·5
        
        # D=6 (6-simplex)
        self.assertEqual(LivniumTSystem.compute_total_sw(6), 378.0)  # 9·6·7
        self.assertEqual(LivniumTSystem.compute_vertex_sw(6), 54.0)  # 9·6
        
        # Verify formula: ΣSW_T(D) = 9D(D+1)
        for D in [3, 4, 5, 6, 7]:
            expected_sw = 9 * D * (D + 1)
            computed_sw = LivniumTSystem.compute_total_sw(D)
            self.assertEqual(computed_sw, expected_sw)
            
            # Verify vertex SW: SW = 9D
            expected_vertex_sw = 9 * D
            computed_vertex_sw = LivniumTSystem.compute_vertex_sw(D)
            self.assertEqual(computed_vertex_sw, expected_vertex_sw)


if __name__ == '__main__':
    unittest.main()

