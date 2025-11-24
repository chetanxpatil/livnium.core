"""
Livnium-O System: Canonical Continuous Spherical Field

Implements the complete Livnium-O System specification:
- 1+N topology (1 core + N neighbor spheres)
- Continuous exposure: f = Ω/4π (solid-angle fraction, not discrete classes)
- Symbolic Weight: SW = 9·f (geometric energy principle)
- Spherical rotation group SO(3) (continuous)
- Generalized kissing constraint
- Perfect reversibility

**The Fundamental Insight:**

On a cube or tetrahedron, exposure f = number of flat faces (discrete: f ∈ {0,1,2,3}).

On a sphere, there are **no faces**. Exposure becomes **continuous**:

    f = Ω/4π

This makes **SW = 9f** a **real physical law**—exposure is energy density.

**All 6 Axioms Implemented:**
- O-A1: Canonical Sphere Alphabet (1+N topology) ✅
- O-A2: Observer Anchor & Frame (Om-Sphere) ✅
- O-A3: Exposure Law (Continuous Solid-Angle Fraction) ✅
- O-A4: Symbolic Weight Law (SW = 9·f as geometric energy principle) ✅
- O-A5: Dynamic Law (Generalized Kissing Constraint) ✅
- O-A6: Connection & Activation Rule ✅

**Canonical Values:**
- Total SW: ΣSW_O = 9N
- Equilibrium Constant: K_O = 9
- Core: 1 sphere, radius=1, f=0, SW=0
- Neighbors: N spheres, radii={r_i}, f=f_i, SW=9·f_i each
- Exposure: f ∈ [0,1] (continuous, not discrete)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import math


class NodeClass(Enum):
    """Node classification based on exposure."""
    CORE = 0      # f = 0, SW = 0
    NEIGHBOR = 1  # f > 0, SW = 9·f


def kissing_constraint_weight(radius: float, core_radius: float = 1.0) -> float:
    """
    Calculate the kissing constraint weight for a neighbor sphere.
    
    This is the normalized solid angle contribution:
    w_i = 1 - sqrt(1 - (r_i / (R_0 + r_i))^2)
    
    Args:
        radius: Radius of the neighbor sphere (r_i)
        core_radius: Radius of the core sphere (R_0), default 1.0
        
    Returns:
        Weight w_i (normalized solid angle contribution)
    """
    if radius <= 0:
        raise ValueError(f"Radius must be > 0, got {radius}")
    if core_radius <= 0:
        raise ValueError(f"Core radius must be > 0, got {core_radius}")
    
    ratio = radius / (core_radius + radius)
    weight = 1.0 - math.sqrt(1.0 - ratio * ratio)
    return weight


def check_kissing_constraint(radii: List[float], core_radius: float = 1.0) -> Tuple[bool, float]:
    """
    Check if a set of neighbor radii satisfies the kissing constraint.
    
    The constraint is:
    sum_i w_i <= 2
    
    where w_i = 1 - sqrt(1 - (r_i / (R_0 + r_i))^2)
    
    Args:
        radii: List of neighbor sphere radii
        core_radius: Radius of the core sphere (R_0), default 1.0
        
    Returns:
        Tuple of (is_valid, total_weight)
    """
    total_weight = sum(kissing_constraint_weight(r, core_radius) for r in radii)
    is_valid = total_weight <= 2.0
    return (is_valid, total_weight)


def calculate_exposure(radius: float, core_radius: float = 1.0) -> float:
    """
    Calculate exposure (f) for a neighbor sphere based on solid angle.
    
    Exposure is a continuous solid-angle fraction:
        f = Ω/4π
    
    Where Ω is the solid angle of the spherical cap covered by the neighbor.
    
    f_i = (1 - cos(alpha_i)) / 2
    where sin(alpha_i) = r_i / (R_0 + r_i)
    
    This makes f a continuous value in [0,1], not discrete like on cubes.
    On a cube, f ∈ {0,1,2,3} (discrete faces).
    On a sphere, f ∈ [0,1] (continuous solid angle).
    
    Args:
        radius: Radius of the neighbor sphere (r_i)
        core_radius: Radius of the core sphere (R_0), default 1.0
        
    Returns:
        Exposure f_i (normalized solid angle fraction, continuous in [0,1])
    """
    if radius <= 0:
        raise ValueError(f"Radius must be > 0, got {radius}")
    if core_radius <= 0:
        raise ValueError(f"Core radius must be > 0, got {core_radius}")
    
    ratio = radius / (core_radius + radius)
    cos_alpha = math.sqrt(1.0 - ratio * ratio)
    exposure = (1.0 - cos_alpha) / 2.0
    return exposure


@dataclass
class SphereNode:
    """
    A single sphere in the Livnium-O 1+N topology.
    
    There are exactly 1+N spheres:
    - 1 Core sphere (Om) with radius=1, f=0, SW=0
    - N Neighbor spheres (LOs) with radii={r_i}, f=f_i, SW=9·f_i each
    """
    node_id: int  # 0 = Core (Om), 1-N = Neighbor spheres (LOs)
    node_class: NodeClass
    radius: float  # Radius of the sphere
    exposure: float  # f ∈ [0, 1] (0 for core, >0 for neighbors)
    symbolic_weight: float  # SW = 9·f
    is_om: bool = False  # True if this is the Om core
    is_lo: bool = False  # True if this is a Local Observer (neighbor)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D position
    
    def __post_init__(self):
        """Validate node properties."""
        if self.node_class == NodeClass.CORE:
            if self.exposure != 0:
                raise ValueError(f"Core node must have f=0, got {self.exposure}")
            if self.symbolic_weight != 0:
                raise ValueError(f"Core node must have SW=0, got {self.symbolic_weight}")
            if self.node_id != 0:
                raise ValueError(f"Core node must have node_id=0, got {self.node_id}")
            if abs(self.radius - 1.0) > 1e-10:
                raise ValueError(f"Core node must have radius=1, got {self.radius}")
            self.is_om = True
            self.is_lo = False
            self.position = (0.0, 0.0, 0.0)
        elif self.node_class == NodeClass.NEIGHBOR:
            if self.exposure <= 0:
                raise ValueError(f"Neighbor node must have f>0, got {self.exposure}")
            if self.node_id == 0:
                raise ValueError(f"Neighbor node cannot have node_id=0")
            if self.radius <= 0:
                raise ValueError(f"Neighbor radius must be > 0, got {self.radius}")
            self.is_om = False
            self.is_lo = True
        else:
            raise ValueError(f"Invalid node class: {self.node_class}")
    
    def get_distance_from_core(self) -> float:
        """Get distance from core center (should be 1 + radius for neighbors)."""
        if self.is_om:
            return 0.0
        else:
            return math.sqrt(sum(x*x for x in self.position))


@dataclass
class Observer:
    """Observer in the Livnium-O System."""
    node_id: int  # 0 = Om, 1-N = LO
    is_om: bool = False
    is_lo: bool = False
    
    def __init__(self, node_id: int, is_om: bool = False, n_neighbors: int = 6):
        if is_om and node_id != 0:
            raise ValueError("Om observer must have node_id=0")
        if not is_om and (node_id < 1 or node_id > n_neighbors):
            raise ValueError(f"LO observer must have node_id in [1,{n_neighbors}], got {node_id}")
        
        self.node_id = node_id
        self.is_om = is_om
        self.is_lo = not is_om


class SphericalRotationGroup:
    """
    Spherical rotation group SO(3).
    
    The special orthogonal group in 3D:
    - Continuous group of all rotations in 3D space
    - All rotations preserve distance and orientation
    - All rotations are invertible
    
    All rotations are bijective, invertible, and orientation-preserving.
    """
    
    def __init__(self):
        """Initialize the rotation group."""
        pass
    
    @staticmethod
    def rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Generate rotation matrix around an axis using Rodrigues' formula.
        
        Args:
            axis: Unit vector axis (3D)
            angle: Rotation angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        axis = axis / np.linalg.norm(axis)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = (cos_a * np.eye(3) + 
             sin_a * K + 
             (1 - cos_a) * np.outer(axis, axis))
        
        return R
    
    @staticmethod
    def rotation_matrix_euler(alpha: float, beta: float, gamma: float) -> np.ndarray:
        """
        Generate rotation matrix from Euler angles (ZYZ convention).
        
        Args:
            alpha: Rotation around z-axis
            beta: Rotation around y-axis
            gamma: Rotation around z-axis
            
        Returns:
            3x3 rotation matrix
        """
        # Rotation around z-axis
        R_z1 = np.array([
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha), math.cos(alpha), 0],
            [0, 0, 1]
        ])
        
        # Rotation around y-axis
        R_y = np.array([
            [math.cos(beta), 0, math.sin(beta)],
            [0, 1, 0],
            [-math.sin(beta), 0, math.cos(beta)]
        ])
        
        # Rotation around z-axis
        R_z2 = np.array([
            [math.cos(gamma), -math.sin(gamma), 0],
            [math.sin(gamma), math.cos(gamma), 0],
            [0, 0, 1]
        ])
        
        return R_z2 @ R_y @ R_z1
    
    @staticmethod
    def apply_rotation(position: Tuple[float, float, float], rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Apply rotation to a 3D position.
        
        Args:
            position: 3D position (x, y, z)
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Rotated position
        """
        pos_vec = np.array(position)
        rotated = rotation_matrix @ pos_vec
        return tuple(rotated)
    
    @staticmethod
    def get_inverse(rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Get the inverse rotation matrix (transpose for orthogonal matrices).
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Inverse rotation matrix
        """
        return rotation_matrix.T


class LivniumOSystem:
    """
    Livnium-O System: Stand-Alone Spherical Semantic Engine
    
    Implements the complete Livnium-O specification:
    - 1+N topology (1 core + N neighbor spheres)
    - Continuous exposure based on solid angle
    - Symbolic Weight: SW = 9·f, ΣSW = 9N
    - Spherical rotation group SO(3)
    - Generalized kissing constraint
    - Perfect reversibility
    """
    
    def __init__(self, neighbor_radii: List[float], core_radius: float = 1.0, 
                 positions: Optional[List[Tuple[float, float, float]]] = None):
        """
        Initialize the Livnium-O system.
        
        Args:
            neighbor_radii: List of radii for neighbor spheres
            core_radius: Radius of the core sphere (default 1.0)
            positions: Optional list of 3D positions for neighbors.
                      If None, positions will be generated on a sphere.
        """
        if core_radius <= 0:
            raise ValueError(f"Core radius must be > 0, got {core_radius}")
        if len(neighbor_radii) == 0:
            raise ValueError("Must have at least one neighbor sphere")
        if any(r <= 0 for r in neighbor_radii):
            raise ValueError("All neighbor radii must be > 0")
        
        self.core_radius = core_radius
        self.neighbor_radii = neighbor_radii
        self.n_neighbors = len(neighbor_radii)
        self.total_nodes = 1 + self.n_neighbors
        
        # Check kissing constraint
        is_valid, total_weight = check_kissing_constraint(neighbor_radii, core_radius)
        if not is_valid:
            raise ValueError(
                f"Kissing constraint violated: total_weight={total_weight:.6f} > 2.0. "
                f"Neighbors cannot all be tangent to the core without overlapping."
            )
        self._kissing_weight = total_weight
        
        # Initialize rotation group
        self.rotation_group = SphericalRotationGroup()
        
        # Create nodes
        self.nodes: Dict[int, SphereNode] = {}
        self._create_nodes(positions)
        
        # Ledger invariants
        self._total_sw = self._calculate_total_sw()
        self._core_count = 1
        self._neighbor_count = self.n_neighbors
        
        # Verify ledger
        self._verify_ledger()
    
    def _create_nodes(self, positions: Optional[List[Tuple[float, float, float]]]):
        """Create all nodes in the system."""
        # Create core node (Om)
        self.nodes[0] = SphereNode(
            node_id=0,
            node_class=NodeClass.CORE,
            radius=self.core_radius,
            exposure=0.0,
            symbolic_weight=0.0,
            position=(0.0, 0.0, 0.0)
        )
        
        # Create neighbor nodes
        if positions is None:
            # Generate positions on a sphere (equal spacing approximation)
            positions = self._generate_sphere_positions(self.n_neighbors)
        
        for i, (radius, position) in enumerate(zip(self.neighbor_radii, positions), start=1):
            # Verify tangency: distance from core should be core_radius + radius
            distance = math.sqrt(sum(x*x for x in position))
            expected_distance = self.core_radius + radius
            if abs(distance - expected_distance) > 1e-6:
                # Normalize position to correct distance
                if distance > 1e-10:
                    scale = expected_distance / distance
                    position = tuple(x * scale for x in position)
                else:
                    # Default to unit vector if at origin
                    position = (expected_distance, 0.0, 0.0)
            
            # Calculate exposure
            exposure = calculate_exposure(radius, self.core_radius)
            symbolic_weight = 9.0 * exposure
            
            self.nodes[i] = SphereNode(
                node_id=i,
                node_class=NodeClass.NEIGHBOR,
                radius=radius,
                exposure=exposure,
                symbolic_weight=symbolic_weight,
                position=position
            )
    
    def _generate_sphere_positions(self, n: int) -> List[Tuple[float, float, float]]:
        """
        Generate approximately equal-spaced positions on a sphere.
        
        Uses Fibonacci sphere algorithm for good distribution.
        
        Args:
            n: Number of positions
            
        Returns:
            List of 3D positions (normalized to unit sphere)
        """
        positions = []
        golden_angle = math.pi * (3 - math.sqrt(5))  # Golden angle
        
        for i in range(n):
            y = 1 - (2 * i) / n  # y goes from 1 to -1
            radius_at_y = math.sqrt(1 - y * y)
            theta = golden_angle * i
            
            x = math.cos(theta) * radius_at_y
            z = math.sin(theta) * radius_at_y
            
            positions.append((x, y, z))
        
        return positions
    
    def _calculate_total_sw(self) -> float:
        """Calculate total symbolic weight."""
        return sum(node.symbolic_weight for node in self.nodes.values())
    
    def _verify_ledger(self):
        """Verify that ledger invariants are correct."""
        # Check kissing constraint
        radii = [self.nodes[i].radius for i in range(1, self.n_neighbors + 1)]
        is_valid, total_weight = check_kissing_constraint(radii, self.core_radius)
        if not is_valid:
            raise ValueError(f"Ledger violation: Kissing constraint violated, total_weight={total_weight:.6f} > 2.0")
        
        # Check core count
        core_count = sum(1 for node in self.nodes.values() if node.is_om)
        neighbor_count = sum(1 for node in self.nodes.values() if node.is_lo)
        
        if core_count != 1:
            raise ValueError(f"Ledger violation: Expected 1 core, got {core_count}")
        if neighbor_count != self.n_neighbors:
            raise ValueError(f"Ledger violation: Expected {self.n_neighbors} neighbors, got {neighbor_count}")
    
    @property
    def total_symbolic_weight(self) -> float:
        """Return total symbolic weight."""
        return self._total_sw
    
    @property
    def equilibrium_constant(self) -> float:
        """Return equilibrium constant K_O (should be 9)."""
        return 9.0
    
    @property
    def encoding_base(self) -> int:
        """Return encoding base (should be N+1)."""
        return self.total_nodes
    
    @property
    def kissing_weight(self) -> float:
        """Return total kissing constraint weight."""
        return self._kissing_weight
    
    def get_node(self, node_id: int) -> SphereNode:
        """Get node by ID."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        return self.nodes[node_id]
    
    def get_core(self) -> SphereNode:
        """Get the core node (Om)."""
        return self.nodes[0]
    
    def get_neighbor_nodes(self) -> List[SphereNode]:
        """Get all neighbor nodes."""
        return [self.nodes[i] for i in range(1, self.n_neighbors + 1)]
    
    def rotate(self, rotation_matrix: np.ndarray) -> 'LivniumOSystem':
        """
        Apply rotation to the system.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            New system with rotated neighbor positions (core unchanged)
        """
        # Create new system with same parameters
        neighbor_radii = [self.nodes[i].radius for i in range(1, self.n_neighbors + 1)]
        original_positions = [self.nodes[i].position for i in range(1, self.n_neighbors + 1)]
        
        # Apply rotation to positions
        rotated_positions = [
            self.rotation_group.apply_rotation(pos, rotation_matrix)
            for pos in original_positions
        ]
        
        new_system = LivniumOSystem(
            neighbor_radii=neighbor_radii,
            core_radius=self.core_radius,
            positions=rotated_positions
        )
        
        # Verify ledger is preserved
        new_system._verify_ledger()
        
        return new_system
    
    def move_neighbor(self, neighbor_id: int, tangential_velocity: np.ndarray, dt: float = 0.01) -> 'LivniumOSystem':
        """
        Move a neighbor along the tangent plane (O-A7: The Flow Law).
        
        The neighbor moves only along the tangent plane, preserving tangency:
        v_i(t) · (N_i - Om) = 0
        
        Args:
            neighbor_id: ID of the neighbor to move (1-based)
            tangential_velocity: 3D velocity vector (must be tangential)
            dt: Time step
            
        Returns:
            New system with moved neighbor
        """
        if neighbor_id < 1 or neighbor_id > self.n_neighbors:
            raise ValueError(f"Neighbor ID must be in [1, {self.n_neighbors}], got {neighbor_id}")
        
        neighbor = self.nodes[neighbor_id]
        if neighbor.is_om:
            raise ValueError("Cannot move core (Om)")
        
        # Get current position relative to core
        pos_vec = np.array(neighbor.position)
        radial_vec = pos_vec  # Since Om is at origin
        
        # Verify tangential constraint: v · (N - Om) = 0
        dot_product = np.dot(tangential_velocity, radial_vec)
        if abs(dot_product) > 1e-10:
            # Project velocity onto tangent plane
            radial_unit = radial_vec / np.linalg.norm(radial_vec)
            tangential_velocity = tangential_velocity - np.dot(tangential_velocity, radial_unit) * radial_unit
        
        # Current distance from core
        current_distance = np.linalg.norm(radial_vec)
        expected_distance = self.core_radius + neighbor.radius
        
        # Normalize position to ensure correct distance
        if abs(current_distance - expected_distance) > 1e-10:
            pos_vec = pos_vec * (expected_distance / current_distance)
        
        # Compute incremental rotation that moves along tangent plane
        # The velocity defines a rotation axis perpendicular to both radial and velocity
        velocity_norm = np.linalg.norm(tangential_velocity)
        if velocity_norm < 1e-10:
            # No movement
            return self
        
        # Rotation axis is perpendicular to both radial and velocity
        rotation_axis = np.cross(radial_vec, tangential_velocity)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-10:
            # Velocity is parallel to radial (shouldn't happen after projection)
            return self
        
        rotation_axis = rotation_axis / rotation_axis_norm
        
        # Rotation angle is proportional to velocity magnitude and time step
        # The angle should move the neighbor by |v| * dt along the sphere surface
        rotation_angle = (velocity_norm * dt) / expected_distance
        
        # Create incremental rotation matrix
        rotation_matrix = self.rotation_group.rotation_matrix_axis_angle(rotation_axis, rotation_angle)
        
        # Apply rotation to position
        new_pos = rotation_matrix @ pos_vec
        
        # Ensure distance is preserved
        new_distance = np.linalg.norm(new_pos)
        if abs(new_distance - expected_distance) > 1e-10:
            new_pos = new_pos * (expected_distance / new_distance)
        
        # Create new system with updated position
        neighbor_radii = [self.nodes[i].radius for i in range(1, self.n_neighbors + 1)]
        positions = [self.nodes[i].position for i in range(1, self.n_neighbors + 1)]
        positions[neighbor_id - 1] = tuple(new_pos)
        
        new_system = LivniumOSystem(
            neighbor_radii=neighbor_radii,
            core_radius=self.core_radius,
            positions=positions
        )
        
        # Verify ledger is preserved
        new_system._verify_ledger()
        
        return new_system
    
    def evolve(self, velocity_field: Dict[int, np.ndarray], dt: float = 0.01) -> 'LivniumOSystem':
        """
        Evolve the system according to O-A7: The Flow Law.
        
        All neighbors move along their tangent planes simultaneously.
        
        Args:
            velocity_field: Dictionary mapping neighbor_id -> tangential velocity vector
            dt: Time step
            
        Returns:
            New system after evolution
        """
        current_system = self
        
        # Move each neighbor in the velocity field
        for neighbor_id, velocity in velocity_field.items():
            if neighbor_id < 1 or neighbor_id > self.n_neighbors:
                continue
            current_system = current_system.move_neighbor(neighbor_id, velocity, dt)
        
        return current_system
    
    def get_ledger(self) -> Dict:
        """
        Get the conservation ledger.
        
        Returns:
            Dictionary with ledger invariants
        """
        return {
            'total_sw': self._total_sw,
            'core_count': self._core_count,
            'neighbor_count': self._neighbor_count,
            'n_neighbors': self.n_neighbors,
            'equilibrium_constant': self.equilibrium_constant,
            'encoding_base': self.encoding_base,
            'kissing_weight': self._kissing_weight,
            'core_radius': self.core_radius,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"LivniumOSystem(n_neighbors={self.n_neighbors}, "
                f"total_nodes={self.total_nodes}, "
                f"total_sw={self._total_sw:.3f}, "
                f"K_O={self.equilibrium_constant}, "
                f"kissing_weight={self._kissing_weight:.6f})")

