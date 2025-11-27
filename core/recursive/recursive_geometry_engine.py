"""
Recursive Geometry Engine: The Core Fractal Machine

Implements geometry → geometry → geometry recursion.

This is Layer 0 - the structural foundation that makes all other layers scalable.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field

from ..classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from .geometry_subdivision import GeometrySubdivision
from .recursive_projection import RecursiveProjection
from .recursive_conservation import RecursiveConservation
from .inheritance import fabricate_child_universe


@dataclass
class GeometryLevel:
    """
    A level in the recursive geometry hierarchy.
    
    Each level contains:
    - A LivniumCoreSystem (geometry)
    - Reference to parent level
    - List of child levels (one per cell)
    """
    level_id: int
    geometry: LivniumCoreSystem
    parent: Optional['GeometryLevel'] = None
    children: Dict[Tuple[int, int, int], 'GeometryLevel'] = field(default_factory=dict)
    scale_factor: int = 1  # How much smaller than parent
    
    def get_total_cells(self) -> int:
        """Get total cells at this level."""
        return len(self.geometry.lattice)
    
    def get_total_cells_recursive(self) -> int:
        """Get total cells including all child levels."""
        total = self.get_total_cells()
        for child in self.children.values():
            total += child.get_total_cells_recursive()
        return total


class RecursiveGeometryEngine:
    """
    Recursive Geometry Engine: Layer 0
    
    The fractal engine that creates geometry from geometry.
    
    Features:
    1. Subdivide geometry into smaller geometry (N×N×N → M×M×M)
    2. Project high-dimensional states downward
    3. Conservation recursion (ΣSW preserved per scale)
    4. Recursive entanglement (compressed into lower scale)
    5. Recursive observer (macro → micro)
    6. Recursive motion (rotation at macro → rotation in micro)
    7. Recursive problem solving (search across layers)
    """
    
    def __init__(self, 
                 base_geometry: LivniumCoreSystem,
                 max_depth: int = 3,
                 subdivision_rule: Optional[Callable] = None,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize recursive geometry engine.
        
        Args:
            base_geometry: Base Livnium Core System (Level 0)
            max_depth: Maximum recursion depth
            subdivision_rule: Optional custom subdivision rule
            rng: Optional random number generator for inheritance
        """
        self.base_geometry = base_geometry
        self.max_depth = max_depth
        self.subdivision_rule = subdivision_rule or self._default_subdivision_rule
        self.rng = rng or np.random.default_rng()
        
        # Create hierarchy
        self.levels: Dict[int, GeometryLevel] = {}
        self._build_hierarchy()
        
        # Initialize components (lazy import to avoid circular dependency)
        from .geometry_subdivision import GeometrySubdivision
        from .recursive_projection import RecursiveProjection
        from .recursive_conservation import RecursiveConservation
        from .moksha_engine import MokshaEngine
        
        self.subdivision = GeometrySubdivision(self)
        self.projection = RecursiveProjection(self)
        self.conservation = RecursiveConservation(self)
        self.moksha = MokshaEngine(self)  # Fixed-point convergence engine
        
        # Optional: Initialize Hamiltonian dynamics (lazy import)
        self.hamiltonian = None
        self._hamiltonian_enabled = False
    
    def _build_hierarchy(self):
        """Build recursive geometry hierarchy."""
        # Level 0: Base geometry
        level_0 = GeometryLevel(
            level_id=0,
            geometry=self.base_geometry,
            scale_factor=1
        )
        self.levels[0] = level_0
        
        # Recursively create child levels
        self._create_child_levels(level_0, depth=1)
    
    def _create_child_levels(self, parent_level: GeometryLevel, depth: int):
        """Recursively create child levels with inheritance."""
        if depth > self.max_depth:
            return
        
        # Create child geometry for each cell in parent
        for coords, cell in parent_level.geometry.lattice.items():
            # Subdivide: create smaller geometry inside this cell with inheritance
            child_geometry = self.subdivision_rule(parent_level.geometry, coords, depth)
            
            if child_geometry:
                # Use depth as level_id (children at same depth share level_id)
                # This matches the existing structure where levels are grouped by depth
                child_level = GeometryLevel(
                    level_id=depth,
                    geometry=child_geometry,
                    parent=parent_level,
                    scale_factor=parent_level.geometry.config.lattice_size // child_geometry.config.lattice_size
                )
                parent_level.children[coords] = child_level
                
                # Register in levels dict (store first child as representative for this depth)
                # We can iterate through parent.children to get all children at a given depth
                if depth not in self.levels:
                    self.levels[depth] = child_level  # Store first child as representative
                
                # Recursively create grandchildren
                self._create_child_levels(child_level, depth + 1)
    
    def _default_subdivision_rule(self, 
                                  parent_geometry: LivniumCoreSystem,
                                  coords: Tuple[int, int, int],
                                  depth: int) -> Optional[LivniumCoreSystem]:
        """
        Default subdivision rule: Create smaller geometry inside cell.
        
        Rule: Each cell contains a geometry of size (parent_size - 2) or minimum 3.
        
        Args:
            parent_geometry: Parent geometry
            coords: Coordinates of cell to subdivide
            depth: Current depth
            
        Returns:
            Child geometry or None
        """
        parent_size = parent_geometry.config.lattice_size
        
        # Subdivide: create smaller geometry
        # Rule: child_size = max(3, parent_size - 2) for depth > 0
        if depth == 1:
            child_size = max(3, parent_size - 2)
        else:
            # Further subdivision: keep minimum 3
            child_size = 3
        
        # Only subdivide if child_size >= 3 and odd
        if child_size < 3 or child_size % 2 == 0:
            return None
        
        # Get parent cell
        parent_cell = parent_geometry.lattice.get(coords)
        if parent_cell is None:
            return None
        
        # Create child universe with inheritance law
        # This gives the child real physics - inherits parent's SW as energy budget
        child_geometry = fabricate_child_universe(
            parent_cell=parent_cell,
            parent_geometry=parent_geometry,
            child_lattice_size=child_size,
            rng=self.rng,
            noise_scale=0.10  # 10% noise in SW distribution
        )
        
        return child_geometry
    
    def subdivide_cell(self, level_id: int, coords: Tuple[int, int, int]) -> bool:
        """
        Subdivide a cell at given level into smaller geometry.
        
        Args:
            level_id: Level to subdivide at
            coords: Coordinates of cell to subdivide
            
        Returns:
            True if subdivision successful
        """
        if level_id not in self.levels:
            return False
        
        level = self.levels[level_id]
        if coords not in level.geometry.lattice:
            return False
        
        # Already subdivided?
        if coords in level.children:
            return False
        
        # Create child geometry
        child_geometry = self.subdivision_rule(level.geometry, coords, level_id + 1)
        if not child_geometry:
            return False
        
        # Add child level
        child_level = GeometryLevel(
            level_id=level_id + 1,
            geometry=child_geometry,
            parent=level,
            scale_factor=level.geometry.config.lattice_size // child_geometry.config.lattice_size
        )
        level.children[coords] = child_level
        self.levels[level_id + 1] = child_level
        
        return True
    
    def project_state_downward(self, 
                              source_level: int,
                              target_level: int,
                              state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project state from higher level to lower level.
        
        Args:
            source_level: Source level ID
            target_level: Target level ID (must be > source_level)
            state: State to project
            
        Returns:
            Projected state
        """
        return self.projection.project_downward(source_level, target_level, state)
    
    def project_state_upward(self,
                            source_level: int,
                            target_level: int,
                            state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project state from lower level to higher level.
        
        Args:
            source_level: Source level ID
            target_level: Target level ID (must be < source_level)
            state: State to project
            
        Returns:
            Projected state
        """
        return self.projection.project_upward(source_level, target_level, state)
    
    def apply_recursive_rotation(self, 
                                level_id: int,
                                axis: RotationAxis,
                                quarter_turns: int):
        """
        Apply rotation recursively: macro → all micro levels.
        
        Args:
            level_id: Level to rotate at
            axis: Rotation axis
            quarter_turns: Number of quarter turns
        """
        if level_id not in self.levels:
            return
        
        level = self.levels[level_id]
        
        # Rotate this level
        level.geometry.rotate(axis, quarter_turns)
        
        # Recursively rotate all children
        for child_level in level.children.values():
            self.apply_recursive_rotation(
                child_level.level_id,
                axis,
                quarter_turns
            )
    
    def enable_hamiltonian_dynamics(
        self,
        temp: float = 0.1,
        friction: float = 0.05,
        dt: float = 0.01
    ):
        """
        Enable Hamiltonian dynamics for recursive evolution.
        
        This integrates core-o Hamiltonian core into recursive geometry,
        making the system evolve with momentum and forces.
        
        Args:
            temp: Temperature for thermal bath
            friction: Friction coefficient
            dt: Time step
        """
        from .recursive_hamiltonian import RecursiveHamiltonian
        
        self.hamiltonian = RecursiveHamiltonian(
            recursive_engine=self,
            temp=temp,
            friction=friction,
            dt=dt,
            enable_dynamics=True
        )
        self._hamiltonian_enabled = True
    
    def evolve_step(self) -> Dict[int, Dict[str, Any]]:
        """
        Evolve the recursive system one step using Hamiltonian dynamics.
        
        Returns:
            Dictionary mapping level_id to evolution statistics
        """
        if not self._hamiltonian_enabled or self.hamiltonian is None:
            raise RuntimeError(
                "Hamiltonian dynamics not enabled. Call enable_hamiltonian_dynamics() first."
            )
        
        return self.hamiltonian.evolve_all_levels()
    
    def get_recursive_observer(self, level_id: int) -> Optional[Tuple[int, int, int]]:
        """
        Get observer at level, derived from parent observer.
        
        Args:
            level_id: Level ID
            
        Returns:
            Observer coordinates at this level
        """
        if level_id not in self.levels:
            return None
        
        level = self.levels[level_id]
        
        if level.level_id == 0:
            # Base level: use global observer
            return (0, 0, 0)
        
        # Derived observer: map parent observer to child coordinates
        if level.parent:
            parent_observer = self.get_recursive_observer(level.parent.level_id)
            if parent_observer:
                # Map parent observer to child scale
                # Simplified: center of child geometry
                child_size = level.geometry.config.lattice_size
                return (0, 0, 0)  # Center of child geometry
        
        return None
    
    def get_total_capacity(self) -> int:
        """
        Get total cell capacity across all levels.
        
        This is the "magic number" - how many states can exist.
        
        Returns:
            Total cells across all levels
        """
        if 0 not in self.levels:
            return 0
        
        return self.levels[0].get_total_cells_recursive()
    
    def get_level_statistics(self) -> Dict:
        """Get statistics for all levels."""
        stats = {}
        for level_id, level in self.levels.items():
            stats[level_id] = {
                'total_cells': level.get_total_cells(),
                'total_cells_recursive': level.get_total_cells_recursive(),
                'num_children': len(level.children),
                'scale_factor': level.scale_factor,
                'lattice_size': level.geometry.config.lattice_size,
            }
        return stats
    
    def check_moksha(self) -> bool:
        """
        Check if system has reached moksha (fixed point).
        
        Returns:
            True if moksha reached
        """
        from .moksha_engine import ConvergenceState
        convergence = self.moksha.check_convergence()
        return convergence == ConvergenceState.MOKSHA
    
    def get_final_truth(self) -> Dict[str, Any]:
        """
        Get final truth when moksha is reached.
        
        Returns:
            Final truth dictionary
        """
        return self.moksha.export_final_truth()
    
    def compress_entanglement(self, 
                            level_id: int,
                            entangled_pairs: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]) -> Dict:
        """
        Compress entanglement into lower scale geometry.
        
        Args:
            level_id: Level to compress at
            entangled_pairs: List of (coords1, coords2) pairs
            
        Returns:
            Compressed representation
        """
        if level_id not in self.levels:
            return {}
        
        level = self.levels[level_id]
        
        # Compress: map entangled pairs to child geometry
        compressed = {}
        for coords1, coords2 in entangled_pairs:
            # Find which child geometries contain these coordinates
            child1 = self._find_child_containing(level, coords1)
            child2 = self._find_child_containing(level, coords2)
            
            if child1 and child2:
                # Map to child coordinates
                child_coords1 = self._map_to_child_coords(child1, coords1)
                child_coords2 = self._map_to_child_coords(child2, coords2)
                
                compressed[(child1.level_id, child_coords1, child_coords2)] = {
                    'original_pair': (coords1, coords2),
                    'compressed_level': child1.level_id,
                }
        
        return compressed
    
    def _find_child_containing(self, level: GeometryLevel, coords: Tuple[int, int, int]) -> Optional[GeometryLevel]:
        """Find child level containing coordinates."""
        for child_coords, child_level in level.children.items():
            # Check if coords are within child's geometry bounds
            child_size = child_level.geometry.config.lattice_size
            half = child_size // 2
            
            # Simplified: check if coords are near child_coords
            if abs(coords[0] - child_coords[0]) <= half and \
               abs(coords[1] - child_coords[1]) <= half and \
               abs(coords[2] - child_coords[2]) <= half:
                return child_level
        
        return None
    
    def _map_to_child_coords(self, child_level: GeometryLevel, parent_coords: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Map parent coordinates to child coordinates."""
        # Simplified: relative to child's center
        child_size = child_level.geometry.config.lattice_size
        half = child_size // 2
        
        # Map to child's coordinate system
        child_coords = (
            parent_coords[0] % child_size - half,
            parent_coords[1] % child_size - half,
            parent_coords[2] % child_size - half,
        )
        
        return child_coords

