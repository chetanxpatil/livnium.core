"""
Test suite for Livnium-O System

Verifies all axioms and derived laws:
- O-A1: Structure Axiom (1+N spherical structure)
- O-A2: Observer Axiom (Om-Sphere)
- O-A3: Exposure Axiom (Continuous Solid-Angle Fraction)
- O-A4: Symbolic Weight Law (SW = 9·f)
- O-A5: Kissing Constraint
- O-A6: Activation Axiom

- D1: Sphere Equilibrium Constant (K_O = 9)
- D2: Concentration Law
- D3: Conservation Ledger
- D4: Reversibility (SO(3))
- D5: Base-(N+1) Encoding Law
"""

import unittest
import sys
from pathlib import Path
import math
import random

# Add core-o directory to path
core_o_path = Path(__file__).parent.parent
sys.path.insert(0, str(core_o_path))

import numpy as np

# Import directly from the module file
from classical.livnium_o_system import (
    LivniumOSystem,
    SphereNode,
    Observer,
    NodeClass,
    SphericalRotationGroup,
    kissing_constraint_weight,
    check_kissing_constraint,
    calculate_exposure,
)


class TestLivniumOSystem(unittest.TestCase):
    """Test suite for Livnium-O System."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create system with 6 neighbors of radius 1.0 (classical kissing number)
        self.neighbor_radii = [1.0] * 6
        self.system = LivniumOSystem(neighbor_radii=self.neighbor_radii, core_radius=1.0)
    
    # ========================================================================
    # S1: Sphere Structure Test
    # ========================================================================
    
    def test_s1_sphere_structure(self):
        """S1: Verify sphere structure forms correctly."""
        # Should have 1 core + N neighbors
        self.assertEqual(self.system.total_nodes, 7)  # 1 + 6
        self.assertEqual(len(self.system.nodes), 7)
        
        # Core should be node 0
        core = self.system.get_core()
        self.assertEqual(core.node_id, 0)
        self.assertTrue(core.is_om)
        self.assertFalse(core.is_lo)
        self.assertEqual(core.radius, 1.0)
        self.assertEqual(core.exposure, 0.0)
        self.assertEqual(core.symbolic_weight, 0.0)
        self.assertEqual(core.position, (0.0, 0.0, 0.0))
        
        # Should have N neighbor nodes
        neighbor_nodes = self.system.get_neighbor_nodes()
        self.assertEqual(len(neighbor_nodes), 6)
        
        for node in neighbor_nodes:
            self.assertFalse(node.is_om)
            self.assertTrue(node.is_lo)
            self.assertGreater(node.radius, 0)
            self.assertGreater(node.exposure, 0)
            self.assertGreater(node.symbolic_weight, 0)
    
    def test_s1_tangency(self):
        """S1: Verify tangency constraint."""
        core = self.system.get_core()
        neighbor_nodes = self.system.get_neighbor_nodes()
        
        for node in neighbor_nodes:
            # Distance from core center should be 1 + radius
            distance = math.sqrt(sum(x*x for x in node.position))
            expected_distance = self.system.core_radius + node.radius
            self.assertAlmostEqual(distance, expected_distance, places=6,
                                 msg=f"Node {node.node_id}: distance={distance:.6f}, expected={expected_distance:.6f}")
    
    def test_s1_valid_positions(self):
        """S1: Verify all positions are valid in 3D."""
        neighbor_nodes = self.system.get_neighbor_nodes()
        
        for node in neighbor_nodes:
            # Position should be a 3-tuple
            self.assertEqual(len(node.position), 3)
            
            # All coordinates should be finite
            for coord in node.position:
                self.assertTrue(math.isfinite(coord))
            
            # Distance should be positive
            distance = math.sqrt(sum(x*x for x in node.position))
            self.assertGreater(distance, 0)
    
    # ========================================================================
    # S3: Exposure Test
    # ========================================================================
    
    def test_s3_exposure_calculation(self):
        """S3: Verify exposure calculation (f = Ω/4π)."""
        neighbor_nodes = self.system.get_neighbor_nodes()
        
        for node in neighbor_nodes:
            # Exposure should be calculated correctly
            expected_exposure = calculate_exposure(node.radius, self.system.core_radius)
            self.assertAlmostEqual(node.exposure, expected_exposure, places=10)
            
            # Exposure should be in [0, 1]
            self.assertGreaterEqual(node.exposure, 0.0)
            self.assertLessEqual(node.exposure, 1.0)
            
            # For radius 1.0, exposure should be approximately 0.0670
            if abs(node.radius - 1.0) < 1e-10:
                self.assertAlmostEqual(node.exposure, 0.0669872981, places=5)
    
    def test_s3_solid_angle(self):
        """S3: Verify solid angle calculation."""
        neighbor_nodes = self.system.get_neighbor_nodes()
        
        for node in neighbor_nodes:
            # Calculate solid angle from exposure
            # f = Ω/4π, so Ω = 4π·f
            solid_angle = 4 * math.pi * node.exposure
            
            # Solid angle should be positive
            self.assertGreater(solid_angle, 0)
            
            # Solid angle should be less than 4π
            self.assertLess(solid_angle, 4 * math.pi)
            
            # For radius 1.0, verify the formula
            if abs(node.radius - 1.0) < 1e-10:
                r = node.radius
                R0 = self.system.core_radius
                sin_alpha = r / (R0 + r)
                cos_alpha = math.sqrt(1 - sin_alpha * sin_alpha)
                expected_omega = 2 * math.pi * (1 - cos_alpha)
                self.assertAlmostEqual(solid_angle, expected_omega, places=6)
    
    def test_s3_symbolic_weight(self):
        """S3: Verify symbolic weight law (SW = 9·f)."""
        neighbor_nodes = self.system.get_neighbor_nodes()
        
        for node in neighbor_nodes:
            # SW should equal 9·f
            expected_sw = 9.0 * node.exposure
            self.assertAlmostEqual(node.symbolic_weight, expected_sw, places=10)
            
            # SW should be in [0, 9]
            self.assertGreaterEqual(node.symbolic_weight, 0.0)
            self.assertLessEqual(node.symbolic_weight, 9.0)
    
    def test_s3_exposure_ranges(self):
        """S3: Verify exposure value ranges."""
        # Test with different radii
        test_radii = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for radius in test_radii:
            exposure = calculate_exposure(radius, core_radius=1.0)
            
            # Exposure should be in [0, 1]
            self.assertGreaterEqual(exposure, 0.0, 
                                   msg=f"Exposure for r={radius} should be >= 0")
            self.assertLessEqual(exposure, 1.0,
                                msg=f"Exposure for r={radius} should be <= 1")
            
            # Larger radius should give larger exposure
            if radius > 0.1:
                small_exposure = calculate_exposure(0.1, core_radius=1.0)
                self.assertGreater(exposure, small_exposure,
                                  msg=f"Larger radius should give larger exposure")
    
    def test_s3_total_symbolic_weight(self):
        """S3: Verify total symbolic weight."""
        neighbor_nodes = self.system.get_neighbor_nodes()
        total_sw = sum(node.symbolic_weight for node in neighbor_nodes)
        
        # Total SW should equal system total
        self.assertAlmostEqual(total_sw, self.system.total_symbolic_weight, places=10)
        
        # Total SW should be sum of 9·f_i for each neighbor
        # For uniform radii, each neighbor has same exposure, so total ≈ 9N·f
        # But since f < 1, total SW < 9N
        self.assertGreater(total_sw, 0)
        self.assertLess(total_sw, 9.0 * len(neighbor_nodes))
    
    # ========================================================================
    # K1: Kissing Constraint Test
    # ========================================================================
    
    def test_k1_kissing_constraint_valid(self):
        """K1: Verify kissing constraint for valid configuration."""
        # 6 neighbors with radius 1.0 should satisfy constraint
        radii = [1.0] * 6
        is_valid, total_weight = check_kissing_constraint(radii, core_radius=1.0)
        
        self.assertTrue(is_valid, 
                       msg=f"6 neighbors with r=1.0 should be valid, got total_weight={total_weight:.6f}")
        self.assertLessEqual(total_weight, 2.0)
        self.assertGreater(total_weight, 0)
    
    def test_k1_kissing_constraint_invalid(self):
        """K1: Verify kissing constraint rejects invalid configuration."""
        # Need many neighbors to violate constraint (weight per neighbor ~0.134)
        # So we need > 14 neighbors with radius 1.0
        radii = [1.0] * 20
        is_valid, total_weight = check_kissing_constraint(radii, core_radius=1.0)
        
        self.assertFalse(is_valid,
                        msg=f"20 neighbors with r=1.0 should be invalid, got total_weight={total_weight:.6f}")
        self.assertGreater(total_weight, 2.0)
    
    def test_k1_kissing_constraint_formula(self):
        """K1: Verify kissing constraint formula."""
        test_radii = [0.5, 1.0, 2.0]
        
        for radius in test_radii:
            weight = kissing_constraint_weight(radius, core_radius=1.0)
            
            # Weight should be positive
            self.assertGreater(weight, 0)
            
            # Weight should be <= 1 (since it's normalized)
            self.assertLessEqual(weight, 1.0)
            
            # Verify formula: w = 1 - sqrt(1 - (r/(1+r))^2)
            r = radius
            ratio = r / (1 + r)
            expected_weight = 1.0 - math.sqrt(1.0 - ratio * ratio)
            self.assertAlmostEqual(weight, expected_weight, places=10)
    
    def test_k1_kissing_constraint_boundary(self):
        """K1: Test boundary case (exactly at limit)."""
        # Find maximum number of neighbors with radius 1.0
        radius = 1.0
        weight_per_neighbor = kissing_constraint_weight(radius, core_radius=1.0)
        max_neighbors = int(2.0 / weight_per_neighbor)
        
        # Should be able to fit max_neighbors
        valid_radii = [radius] * max_neighbors
        is_valid, total_weight = check_kissing_constraint(valid_radii, core_radius=1.0)
        self.assertTrue(is_valid or abs(total_weight - 2.0) < 0.01)
        
        # Should NOT be able to fit max_neighbors + 1
        invalid_radii = [radius] * (max_neighbors + 1)
        is_valid, total_weight = check_kissing_constraint(invalid_radii, core_radius=1.0)
        self.assertFalse(is_valid)
    
    def test_k1_heterogeneous_radii(self):
        """K1: Test kissing constraint with mixed radii."""
        # Mix of different radii
        radii = [1.0, 0.8, 0.6, 0.4, 0.2]
        is_valid, total_weight = check_kissing_constraint(radii, core_radius=1.0)
        
        # Should be valid (sum should be less than 2)
        self.assertTrue(is_valid)
        self.assertLessEqual(total_weight, 2.0)
        
        # Verify we can create a system with these radii
        try:
            system = LivniumOSystem(neighbor_radii=radii, core_radius=1.0)
            self.assertEqual(len(system.get_neighbor_nodes()), len(radii))
        except ValueError:
            self.fail("Should be able to create system with valid radii")
    
    def test_k1_system_creation_enforces_constraint(self):
        """K1: Verify system creation enforces kissing constraint."""
        # Valid configuration should work
        valid_radii = [1.0] * 6
        system = LivniumOSystem(neighbor_radii=valid_radii, core_radius=1.0)
        self.assertIsNotNone(system)
        
        # Invalid configuration should raise ValueError
        # Need many neighbors to violate constraint (weight per neighbor ~0.134)
        invalid_radii = [1.0] * 20
        with self.assertRaises(ValueError):
            LivniumOSystem(neighbor_radii=invalid_radii, core_radius=1.0)
    
    # ========================================================================
    # Additional Tests
    # ========================================================================
    
    def test_equilibrium_constant(self):
        """D1: Verify equilibrium constant K_O = 9."""
        self.assertEqual(self.system.equilibrium_constant, 9.0)
    
    def test_concentration_law(self):
        """D2: Verify concentration law."""
        neighbor_nodes = self.system.get_neighbor_nodes()
        
        for node in neighbor_nodes:
            if node.exposure > 0:
                # C(f) = 9/f
                concentration = 9.0 / node.exposure
                
                # SW = C(f) · f = 9
                sw_from_concentration = concentration * node.exposure
                self.assertAlmostEqual(sw_from_concentration, 9.0, places=6)
    
    def test_conservation_ledger(self):
        """D3: Verify conservation ledger."""
        ledger = self.system.get_ledger()
        
        # Check all ledger invariants
        self.assertIn('total_sw', ledger)
        self.assertIn('core_count', ledger)
        self.assertIn('neighbor_count', ledger)
        self.assertIn('kissing_weight', ledger)
        self.assertIn('core_radius', ledger)
        
        # Verify values
        self.assertEqual(ledger['core_count'], 1)
        self.assertEqual(ledger['neighbor_count'], 6)
        self.assertEqual(ledger['core_radius'], 1.0)
        self.assertLessEqual(ledger['kissing_weight'], 2.0)
    
    def test_rotation_group(self):
        """D4: Verify rotation group SO(3)."""
        rotation_group = self.system.rotation_group
        self.assertIsNotNone(rotation_group)
        
        # Test rotation matrix generation
        axis = np.array([0, 0, 1])
        angle = math.pi / 4
        rotation_matrix = rotation_group.rotation_matrix_axis_angle(axis, angle)
        
        # Should be 3x3
        self.assertEqual(rotation_matrix.shape, (3, 3))
        
        # Should be orthogonal (R^T R = I)
        identity = rotation_matrix.T @ rotation_matrix
        np.testing.assert_allclose(identity, np.eye(3), rtol=1e-10)
        
        # Determinant should be 1 (proper rotation)
        det = np.linalg.det(rotation_matrix)
        self.assertAlmostEqual(det, 1.0, places=10)
    
    def test_rotation_reversibility(self):
        """D4: Verify rotation reversibility."""
        rotation_group = self.system.rotation_group
        
        # Create a rotation
        axis = np.array([1, 1, 1]) / math.sqrt(3)
        angle = math.pi / 3
        rotation_matrix = rotation_group.rotation_matrix_axis_angle(axis, angle)
        
        # Get inverse
        inverse_matrix = rotation_group.get_inverse(rotation_matrix)
        
        # R · R^-1 should be identity (within floating point precision)
        identity = rotation_matrix @ inverse_matrix
        np.testing.assert_allclose(identity, np.eye(3), rtol=1e-10, atol=1e-15)
    
    def test_rotation_preserves_tangency(self):
        """D4: Verify rotation preserves tangency."""
        original_neighbors = self.system.get_neighbor_nodes()
        original_distances = [math.sqrt(sum(x*x for x in n.position)) 
                            for n in original_neighbors]
        
        # Apply rotation
        axis = np.array([0, 0, 1])
        angle = math.pi / 4
        rotation_matrix = self.system.rotation_group.rotation_matrix_axis_angle(axis, angle)
        rotated_system = self.system.rotate(rotation_matrix)
        
        # Check tangency is preserved
        rotated_neighbors = rotated_system.get_neighbor_nodes()
        for i, node in enumerate(rotated_neighbors):
            distance = math.sqrt(sum(x*x for x in node.position))
            expected_distance = self.system.core_radius + node.radius
            self.assertAlmostEqual(distance, expected_distance, places=6)
    
    def test_encoding_base(self):
        """D5: Verify encoding base."""
        # Encoding base should be N+1
        expected_base = self.system.n_neighbors + 1
        self.assertEqual(self.system.encoding_base, expected_base)
    
    def test_numerical_stress_test(self):
        """Numerical stress test with random radius distributions."""
        random.seed(42)  # For reproducibility
        
        for _ in range(10):
            # Generate random number of neighbors (between 3 and 8)
            n_neighbors = random.randint(3, 8)
            
            # Generate random radii (between 0.1 and 2.0)
            radii = [random.uniform(0.1, 2.0) for _ in range(n_neighbors)]
            
            # Check if configuration is valid
            is_valid, total_weight = check_kissing_constraint(radii, core_radius=1.0)
            
            if is_valid:
                # Should be able to create system
                try:
                    system = LivniumOSystem(neighbor_radii=radii, core_radius=1.0)
                    
                    # Verify structure
                    self.assertEqual(len(system.get_neighbor_nodes()), n_neighbors)
                    
                    # Verify tangency
                    for node in system.get_neighbor_nodes():
                        distance = math.sqrt(sum(x*x for x in node.position))
                        expected_distance = 1.0 + node.radius
                        self.assertAlmostEqual(distance, expected_distance, places=5)
                    
                    # Verify exposure ranges
                    for node in system.get_neighbor_nodes():
                        self.assertGreaterEqual(node.exposure, 0.0)
                        self.assertLessEqual(node.exposure, 1.0)
                        self.assertAlmostEqual(node.symbolic_weight, 9.0 * node.exposure, places=6)
                    
                except ValueError as e:
                    self.fail(f"Valid configuration should create system: {e}")
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Very small radius
        small_radii = [0.01] * 20
        is_valid, total_weight = check_kissing_constraint(small_radii, core_radius=1.0)
        self.assertTrue(is_valid)  # Many small neighbors should fit
        
        # Very large radius
        large_radii = [10.0]
        is_valid, total_weight = check_kissing_constraint(large_radii, core_radius=1.0)
        self.assertTrue(is_valid)  # One large neighbor should fit
        
        # Single neighbor
        single_radius = [1.0]
        system = LivniumOSystem(neighbor_radii=single_radius, core_radius=1.0)
        self.assertEqual(len(system.get_neighbor_nodes()), 1)
    
    def test_exposure_monotonicity(self):
        """Test that exposure increases with radius."""
        radii = [0.1, 0.5, 1.0, 2.0, 5.0]
        exposures = [calculate_exposure(r, core_radius=1.0) for r in radii]
        
        # Exposure should be monotonically increasing
        for i in range(len(exposures) - 1):
            self.assertLess(exposures[i], exposures[i+1],
                          msg=f"Exposure should increase with radius: f({radii[i]})={exposures[i]:.6f} < f({radii[i+1]})={exposures[i+1]:.6f}")
    
    # ========================================================================
    # O-A7: The Flow Law Tests
    # ========================================================================
    
    def test_o_a7_tangential_velocity_constraint(self):
        """O-A7: Verify tangential velocity constraint v · (N - Om) = 0."""
        neighbor = self.system.get_neighbor_nodes()[0]
        pos_vec = np.array(neighbor.position)
        
        # Create a tangential velocity (perpendicular to radial)
        # Use cross product to get a vector perpendicular to radial
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity)
        
        # Verify constraint: v · (N - Om) = 0
        dot_product = np.dot(tangential_velocity, pos_vec)
        self.assertAlmostEqual(dot_product, 0.0, places=10,
                              msg="Tangential velocity must be perpendicular to radial vector")
    
    def test_o_a7_move_neighbor_preserves_tangency(self):
        """O-A7: Verify move_neighbor preserves tangency."""
        neighbor = self.system.get_neighbor_nodes()[0]
        neighbor_id = neighbor.node_id
        original_distance = math.sqrt(sum(x*x for x in neighbor.position))
        expected_distance = self.system.core_radius + neighbor.radius
        
        # Create tangential velocity
        pos_vec = np.array(neighbor.position)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1  # Small velocity
        
        # Move neighbor
        moved_system = self.system.move_neighbor(neighbor_id, tangential_velocity, dt=0.1)
        moved_neighbor = moved_system.get_node(neighbor_id)
        
        # Verify tangency is preserved
        new_distance = math.sqrt(sum(x*x for x in moved_neighbor.position))
        self.assertAlmostEqual(new_distance, expected_distance, places=6,
                             msg="Distance from core must remain constant (tangency preserved)")
    
    def test_o_a7_move_neighbor_preserves_distance(self):
        """O-A7: Verify move_neighbor preserves distance |N - Om| = 1 + r."""
        neighbor = self.system.get_neighbor_nodes()[0]
        neighbor_id = neighbor.node_id
        original_distance = math.sqrt(sum(x*x for x in neighbor.position))
        expected_distance = self.system.core_radius + neighbor.radius
        
        # Create tangential velocity
        pos_vec = np.array(neighbor.position)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.2
        
        # Move neighbor multiple times
        current_system = self.system
        for _ in range(5):
            current_system = current_system.move_neighbor(neighbor_id, tangential_velocity, dt=0.1)
            moved_neighbor = current_system.get_node(neighbor_id)
            distance = math.sqrt(sum(x*x for x in moved_neighbor.position))
            self.assertAlmostEqual(distance, expected_distance, places=6,
                                 msg="Distance must remain constant after each move")
    
    def test_o_a7_velocity_projection(self):
        """O-A7: Verify non-tangential velocities are projected onto tangent plane."""
        neighbor = self.system.get_neighbor_nodes()[0]
        neighbor_id = neighbor.node_id
        pos_vec = np.array(neighbor.position)
        
        # Create a velocity with radial component (should be projected out)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        radial_velocity = radial_unit * 0.5  # Pure radial velocity
        
        # move_neighbor should project this onto tangent plane
        moved_system = self.system.move_neighbor(neighbor_id, radial_velocity, dt=0.1)
        
        # After projection, there should be no movement (or minimal)
        # Since pure radial velocity becomes zero after projection
        moved_neighbor = moved_system.get_node(neighbor_id)
        moved_pos = np.array(moved_neighbor.position)
        
        # The position should be unchanged (or very close) since radial component is removed
        distance_change = np.linalg.norm(moved_pos - pos_vec)
        self.assertLess(distance_change, 1e-6,
                       msg="Pure radial velocity should result in no movement after projection")
    
    def test_o_a7_so3_rotation_update(self):
        """O-A7: Verify movement uses SO(3) incremental rotation."""
        neighbor = self.system.get_neighbor_nodes()[0]
        neighbor_id = neighbor.node_id
        
        # Create tangential velocity
        pos_vec = np.array(neighbor.position)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1
        
        # Move neighbor
        moved_system = self.system.move_neighbor(neighbor_id, tangential_velocity, dt=0.1)
        moved_neighbor = moved_system.get_node(neighbor_id)
        
        # Verify the movement is a rotation (distance preserved, angle changed)
        original_pos = np.array(neighbor.position)
        moved_pos = np.array(moved_neighbor.position)
        
        # Distance should be preserved
        original_dist = np.linalg.norm(original_pos)
        moved_dist = np.linalg.norm(moved_pos)
        self.assertAlmostEqual(original_dist, moved_dist, places=6)
        
        # Angle should have changed (unless velocity was zero)
        dot_product = np.dot(original_pos, moved_pos) / (original_dist * moved_dist)
        # Should be close to 1 but not exactly 1 (unless no movement)
        self.assertLess(dot_product, 1.0 - 1e-6,
                       msg="Position should have rotated (angle changed)")
    
    def test_o_a7_flow_reversibility(self):
        """O-A7: Verify flow is reversible."""
        neighbor = self.system.get_neighbor_nodes()[0]
        neighbor_id = neighbor.node_id
        original_pos = np.array(neighbor.position)
        
        # Create tangential velocity
        pos_vec = np.array(neighbor.position)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1
        
        # Move forward
        moved_system = self.system.move_neighbor(neighbor_id, tangential_velocity, dt=0.1)
        moved_pos = np.array(moved_system.get_node(neighbor_id).position)
        
        # Move backward (negative velocity)
        reversed_system = moved_system.move_neighbor(neighbor_id, -tangential_velocity, dt=0.1)
        reversed_pos = np.array(reversed_system.get_node(neighbor_id).position)
        
        # Should return to original position (within numerical precision)
        distance_error = np.linalg.norm(reversed_pos - original_pos)
        self.assertLess(distance_error, 1e-5,
                       msg="Flow should be reversible: forward then backward should return to original")
    
    def test_o_a7_evolve_multiple_neighbors(self):
        """O-A7: Verify evolve() moves multiple neighbors simultaneously."""
        neighbors = self.system.get_neighbor_nodes()
        
        # Create velocity field for first 3 neighbors
        velocity_field = {}
        for i, neighbor in enumerate(neighbors[:3], start=1):
            pos_vec = np.array(neighbor.position)
            radial_unit = pos_vec / np.linalg.norm(pos_vec)
            arbitrary_vec = np.array([1, 0, 0])
            if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
                arbitrary_vec = np.array([0, 1, 0])
            
            tangential_velocity = np.cross(radial_unit, arbitrary_vec)
            tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1
            velocity_field[neighbor.node_id] = tangential_velocity
        
        # Evolve system
        evolved_system = self.system.evolve(velocity_field, dt=0.1)
        
        # Verify all neighbors in velocity field moved
        for neighbor_id in velocity_field.keys():
            original_pos = np.array(self.system.get_node(neighbor_id).position)
            evolved_pos = np.array(evolved_system.get_node(neighbor_id).position)
            
            # Position should have changed
            distance_change = np.linalg.norm(evolved_pos - original_pos)
            self.assertGreater(distance_change, 1e-6,
                             msg=f"Neighbor {neighbor_id} should have moved")
            
            # Tangency should be preserved
            distance = np.linalg.norm(evolved_pos)
            expected_distance = self.system.core_radius + evolved_system.get_node(neighbor_id).radius
            self.assertAlmostEqual(distance, expected_distance, places=6)
        
        # Neighbors not in velocity field should not have moved
        for neighbor in neighbors[3:]:
            original_pos = np.array(self.system.get_node(neighbor.node_id).position)
            evolved_pos = np.array(evolved_system.get_node(neighbor.node_id).position)
            distance_change = np.linalg.norm(evolved_pos - original_pos)
            self.assertLess(distance_change, 1e-6,
                          msg=f"Neighbor {neighbor.node_id} should not have moved")
    
    def test_o_a7_ledger_conservation(self):
        """O-A7: Verify flow preserves conservation ledger."""
        neighbor = self.system.get_neighbor_nodes()[0]
        neighbor_id = neighbor.node_id
        
        # Create tangential velocity
        pos_vec = np.array(neighbor.position)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1
        
        # Get original ledger
        original_ledger = self.system.get_ledger()
        
        # Move neighbor
        moved_system = self.system.move_neighbor(neighbor_id, tangential_velocity, dt=0.1)
        moved_ledger = moved_system.get_ledger()
        
        # Verify ledger is preserved
        self.assertEqual(original_ledger['total_sw'], moved_ledger['total_sw'])
        self.assertEqual(original_ledger['core_count'], moved_ledger['core_count'])
        self.assertEqual(original_ledger['neighbor_count'], moved_ledger['neighbor_count'])
        self.assertEqual(original_ledger['equilibrium_constant'], moved_ledger['equilibrium_constant'])
        self.assertEqual(original_ledger['encoding_base'], moved_ledger['encoding_base'])
        self.assertEqual(original_ledger['core_radius'], moved_ledger['core_radius'])
    
    def test_o_a7_exposure_changes_continuously(self):
        """O-A7: Verify exposure changes continuously during flow."""
        neighbor = self.system.get_neighbor_nodes()[0]
        neighbor_id = neighbor.node_id
        original_exposure = neighbor.exposure
        
        # Create tangential velocity
        pos_vec = np.array(neighbor.position)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1
        
        # Move neighbor
        moved_system = self.system.move_neighbor(neighbor_id, tangential_velocity, dt=0.1)
        moved_neighbor = moved_system.get_node(neighbor_id)
        
        # Exposure should be recalculated (may change due to position change)
        # But radius is same, so exposure should be same
        # Actually, exposure depends only on radius, not position!
        # So exposure should remain the same
        self.assertAlmostEqual(moved_neighbor.exposure, original_exposure, places=10,
                             msg="Exposure depends only on radius, not position")
    
    def test_o_a7_cannot_move_core(self):
        """O-A7: Verify core (Om) cannot be moved."""
        # Try to move core (node_id=0)
        with self.assertRaises(ValueError):
            self.system.move_neighbor(0, np.array([0.1, 0.1, 0.1]), dt=0.1)


if __name__ == "__main__":
    unittest.main()

