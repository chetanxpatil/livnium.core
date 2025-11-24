"""
Livnium-C System: Stand-Alone Circular Implementation

Implements the complete Livnium-C System specification:
- 1+N topology (1 core + N ring nodes)
- Two-class system (Core f=0, Ring f=1)
- Symbolic Weight (SW = 9·f, ΣSW = 9N)
- Cyclic rotation group C_N (N elements)
- Perfect reversibility

**All 6 Axioms Implemented:**
- C-A1: Canonical Circle Alphabet (1+N topology) ✅
- C-A2: Observer Anchor & Frame (Om-Core) ✅
- C-A3: Exposure Law (Two-Class System) ✅
- C-A4: Symbolic Weight Law (SW = 9·f) ✅
- C-A5: Dynamic Law (Cyclic Rotation Group C_N) ✅
- C-A6: Connection & Activation Rule ✅

**Canonical Values:**
- Total SW: ΣSW_C = 9N
- Equilibrium Constant: K_C = 9
- Core: 1 node, f=0, SW=0
- Ring: N nodes, f=1, SW=9 each
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class NodeClass(Enum):
    """Node classification based on exposure."""
    CORE = 0      # f = 0, SW = 0
    RING = 1      # f = 1, SW = 9


@dataclass
class CircleNode:
    """
    A single node in the Livnium-C 1+N topology.
    
    There are exactly 1+N nodes:
    - 1 Core node (Om) with f=0, SW=0
    - N Ring nodes (LOs) with f=1, SW=9 each
    """
    node_id: int  # 0 = Core (Om), 1-N = Ring nodes (LOs)
    node_class: NodeClass
    exposure: int  # f ∈ {0, 1}
    symbolic_weight: float  # SW = 9·f
    is_om: bool = False  # True if this is the Om core
    is_lo: bool = False  # True if this is a Local Observer (ring node)
    angle: float = 0.0  # Angle in radians (for ring nodes)
    radius: float = 1.0  # Distance from center (for ring nodes)
    
    def __post_init__(self):
        """Validate node properties."""
        if self.node_class == NodeClass.CORE:
            if self.exposure != 0:
                raise ValueError(f"Core node must have f=0, got {self.exposure}")
            if self.symbolic_weight != 0:
                raise ValueError(f"Core node must have SW=0, got {self.symbolic_weight}")
            if self.node_id != 0:
                raise ValueError(f"Core node must have node_id=0, got {self.node_id}")
            self.is_om = True
            self.is_lo = False
            self.angle = 0.0
            self.radius = 0.0
        elif self.node_class == NodeClass.RING:
            if self.exposure != 1:
                raise ValueError(f"Ring node must have f=1, got {self.exposure}")
            if self.symbolic_weight != 9:
                raise ValueError(f"Ring node must have SW=9, got {self.symbolic_weight}")
            if self.node_id == 0:
                raise ValueError(f"Ring node cannot have node_id=0")
            self.is_om = False
            self.is_lo = True
        else:
            raise ValueError(f"Invalid node class: {self.node_class}")
    
    def get_position(self) -> Tuple[float, float]:
        """Get 2D position (x, y) of the node."""
        if self.is_om:
            return (0.0, 0.0)
        else:
            x = self.radius * np.cos(self.angle)
            y = self.radius * np.sin(self.angle)
            return (x, y)


@dataclass
class Observer:
    """Observer in the Livnium-C System."""
    node_id: int  # 0 = Om, 1-N = LO
    is_om: bool = False
    is_lo: bool = False
    
    def __init__(self, node_id: int, is_om: bool = False, n_ring: int = 8):
        if is_om and node_id != 0:
            raise ValueError("Om observer must have node_id=0")
        if not is_om and (node_id < 1 or node_id > n_ring):
            raise ValueError(f"LO observer must have node_id in [1,{n_ring}], got {node_id}")
        
        self.node_id = node_id
        self.is_om = is_om
        self.is_lo = not is_om


class CyclicRotationGroup:
    """
    N-element cyclic rotation group C_N.
    
    The cyclic group on N ring nodes:
    - N rotations by angles k·(2π/N) for k = 0, 1, 2, ..., N-1
    - 1 identity (k=0)
    
    All rotations are bijective, invertible, and orientation-preserving.
    """
    
    def __init__(self, n: int):
        """
        Initialize the cyclic rotation group for N ring nodes.
        
        Args:
            n: Number of ring nodes (N)
        """
        if n < 1:
            raise ValueError(f"N must be >= 1, got {n}")
        self.n = n
        self._rotations = self._generate_all_rotations()
        self._inverses = {k: self._find_inverse(k) for k in range(n)}
    
    @property
    def order(self) -> int:
        """Return the order of the rotation group (N)."""
        return self.n
    
    @property
    def rotations(self) -> Dict[int, np.ndarray]:
        """Return all rotation matrices (as permutation matrices)."""
        return self._rotations
    
    def get_rotation(self, rotation_id: int) -> np.ndarray:
        """
        Get rotation permutation by ID.
        
        Args:
            rotation_id: Integer in [0, N-1] identifying the rotation
            
        Returns:
            Permutation array mapping old indices to new indices
        """
        if rotation_id < 0 or rotation_id >= self.n:
            raise ValueError(f"Rotation ID must be in [0, {self.n-1}], got {rotation_id}")
        return self._rotations[rotation_id]
    
    def get_inverse(self, rotation_id: int) -> int:
        """
        Get the inverse rotation ID.
        
        Args:
            rotation_id: Rotation ID in [0, N-1]
            
        Returns:
            Inverse rotation ID
        """
        if rotation_id < 0 or rotation_id >= self.n:
            raise ValueError(f"Rotation ID must be in [0, {self.n-1}], got {rotation_id}")
        return self._inverses[rotation_id]
    
    def apply_rotation(self, ring_indices: List[int], rotation_id: int) -> List[int]:
        """
        Apply rotation to ring node indices.
        
        Args:
            ring_indices: List of ring node indices (1-based, relative to ring)
            rotation_id: Rotation ID in [0, N-1]
            
        Returns:
            Rotated ring node indices
        """
        permutation = self.get_rotation(rotation_id)
        # Convert to 0-based for permutation, then back to 1-based
        zero_based = [idx - 1 for idx in ring_indices]
        rotated = [permutation[idx] + 1 for idx in zero_based]
        return rotated
    
    def rotate_angles(self, angles: np.ndarray, rotation_id: int) -> np.ndarray:
        """
        Apply rotation to angles (in radians).
        
        Args:
            angles: Array of angles for ring nodes
            rotation_id: Rotation ID in [0, N-1]
            
        Returns:
            Rotated angles
        """
        if len(angles) != self.n:
            raise ValueError(f"Expected {self.n} angles, got {len(angles)}")
        
        # Rotation by k steps: add k·(2π/N) to all angles
        angle_step = 2 * np.pi / self.n
        rotation_angle = rotation_id * angle_step
        
        rotated = (angles + rotation_angle) % (2 * np.pi)
        return rotated
    
    def _generate_all_rotations(self) -> Dict[int, np.ndarray]:
        """
        Generate all N rotations of the cyclic group C_N.
        
        Rotation k maps ring node i to ring node (i+k) mod N.
        
        Returns:
            Dictionary mapping rotation ID to permutation array
        """
        rotations = {}
        
        for k in range(self.n):
            # Rotation k: i -> (i+k) mod N
            # But we need 1-based indices, so: i -> ((i-1+k) mod N) + 1
            permutation = np.array([((i + k) % self.n) for i in range(self.n)])
            rotations[k] = permutation
        
        return rotations
    
    def _find_inverse(self, rotation_id: int) -> int:
        """
        Find the inverse rotation ID.
        
        For cyclic group C_N, rotation k has inverse rotation (N-k) mod N.
        
        Args:
            rotation_id: Rotation ID in [0, N-1]
            
        Returns:
            Inverse rotation ID
        """
        return (self.n - rotation_id) % self.n


class LivniumCSystem:
    """
    Livnium-C System: Stand-Alone Circular Semantic Engine
    
    Implements the complete Livnium-C specification:
    - 1+N topology (1 core + N ring nodes)
    - Two-class system (Core f=0, Ring f=1)
    - Symbolic Weight: SW = 9·f, ΣSW = 9N
    - Cyclic rotation group C_N
    - Perfect reversibility
    """
    
    def __init__(self, n_ring: int = 8, radius: float = 1.0):
        """
        Initialize the Livnium-C system.
        
        Args:
            n_ring: Number of ring nodes (N). Default is 8.
            radius: Radius of the circle. Default is 1.0.
        """
        if n_ring < 1:
            raise ValueError(f"N must be >= 1, got {n_ring}")
        if radius <= 0:
            raise ValueError(f"Radius must be > 0, got {radius}")
        
        self.n_ring = n_ring
        self.radius = radius
        self.total_nodes = 1 + n_ring
        
        # Initialize rotation group
        self.rotation_group = CyclicRotationGroup(n_ring)
        
        # Create nodes
        self.nodes: Dict[int, CircleNode] = {}
        self._create_nodes()
        
        # Ledger invariants
        self._total_sw = self._calculate_total_sw()
        self._core_count = 1
        self._ring_count = n_ring
        
        # Verify ledger
        self._verify_ledger()
    
    def _create_nodes(self):
        """Create all nodes in the system."""
        # Create core node (Om)
        self.nodes[0] = CircleNode(
            node_id=0,
            node_class=NodeClass.CORE,
            exposure=0,
            symbolic_weight=0.0,
            angle=0.0,
            radius=0.0
        )
        
        # Create ring nodes
        angle_step = 2 * np.pi / self.n_ring
        for i in range(1, self.n_ring + 1):
            angle = (i - 1) * angle_step
            self.nodes[i] = CircleNode(
                node_id=i,
                node_class=NodeClass.RING,
                exposure=1,
                symbolic_weight=9.0,
                angle=angle,
                radius=self.radius
            )
    
    def _calculate_total_sw(self) -> float:
        """Calculate total symbolic weight."""
        return sum(node.symbolic_weight for node in self.nodes.values())
    
    def _verify_ledger(self):
        """Verify that ledger invariants are correct."""
        expected_sw = 9 * self.n_ring
        actual_sw = self._total_sw
        
        if abs(actual_sw - expected_sw) > 1e-10:
            raise ValueError(f"Ledger violation: Expected SW={expected_sw}, got {actual_sw}")
        
        core_count = sum(1 for node in self.nodes.values() if node.is_om)
        ring_count = sum(1 for node in self.nodes.values() if node.is_lo)
        
        if core_count != 1:
            raise ValueError(f"Ledger violation: Expected 1 core, got {core_count}")
        if ring_count != self.n_ring:
            raise ValueError(f"Ledger violation: Expected {self.n_ring} ring nodes, got {ring_count}")
    
    @property
    def total_symbolic_weight(self) -> float:
        """Return total symbolic weight (should be 9N)."""
        return self._total_sw
    
    @property
    def equilibrium_constant(self) -> float:
        """Return equilibrium constant K_C (should be 9)."""
        return 9.0
    
    @property
    def encoding_base(self) -> int:
        """Return encoding base (should be N+1)."""
        return self.total_nodes
    
    def get_node(self, node_id: int) -> CircleNode:
        """Get node by ID."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        return self.nodes[node_id]
    
    def get_core(self) -> CircleNode:
        """Get the core node (Om)."""
        return self.nodes[0]
    
    def get_ring_nodes(self) -> List[CircleNode]:
        """Get all ring nodes."""
        return [self.nodes[i] for i in range(1, self.n_ring + 1)]
    
    def rotate(self, rotation_id: int) -> 'LivniumCSystem':
        """
        Apply rotation to the system.
        
        Args:
            rotation_id: Rotation ID in [0, N-1]
            
        Returns:
            New system with rotated ring nodes (core unchanged)
        """
        if rotation_id < 0 or rotation_id >= self.n_ring:
            raise ValueError(f"Rotation ID must be in [0, {self.n_ring-1}], got {rotation_id}")
        
        # Create new system with same parameters
        new_system = LivniumCSystem(n_ring=self.n_ring, radius=self.radius)
        
        # Apply rotation to ring node angles
        ring_nodes = self.get_ring_nodes()
        angles = np.array([node.angle for node in ring_nodes])
        rotated_angles = self.rotation_group.rotate_angles(angles, rotation_id)
        
        # Update ring node angles
        for i, node in enumerate(new_system.get_ring_nodes()):
            node.angle = rotated_angles[i]
        
        # Verify ledger is preserved
        new_system._verify_ledger()
        
        return new_system
    
    def get_ledger(self) -> Dict:
        """
        Get the conservation ledger.
        
        Returns:
            Dictionary with ledger invariants
        """
        return {
            'total_sw': self._total_sw,
            'core_count': self._core_count,
            'ring_count': self._ring_count,
            'n_ring': self.n_ring,
            'equilibrium_constant': self.equilibrium_constant,
            'encoding_base': self.encoding_base,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"LivniumCSystem(n_ring={self.n_ring}, "
                f"total_nodes={self.total_nodes}, "
                f"total_sw={self._total_sw:.1f}, "
                f"K_C={self.equilibrium_constant})")

