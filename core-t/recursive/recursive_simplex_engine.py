"""
Recursive Simplex Engine: The Core Fractal Machine for Livnium-T

Implements simplex → simplex → simplex recursion.

This is Layer 0 - the structural foundation that makes all other layers scalable.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from ..classical.livnium_t_system import LivniumTSystem
from .simplex_subdivision import SimplexSubdivision
from .recursive_projection import RecursiveProjection
from .recursive_conservation import RecursiveConservation


@dataclass
class SimplexLevel:
    """
    A level in the recursive simplex hierarchy.
    
    Each level contains:
    - A LivniumTSystem (simplex geometry)
    - Reference to parent level
    - List of child levels (one per node)
    """
    level_id: int
    geometry: LivniumTSystem
    parent: Optional['SimplexLevel'] = None
    children: Dict[int, 'SimplexLevel'] = field(default_factory=dict)  # node_id -> child
    scale_factor: int = 1  # How much smaller than parent (for D-simplex scaling)
    
    def get_total_nodes(self) -> int:
        """Get total nodes at this level."""
        return len(self.geometry.nodes)
    
    def get_total_nodes_recursive(self) -> int:
        """Get total nodes including all child levels."""
        total = self.get_total_nodes()
        for child in self.children.values():
            total += child.get_total_nodes_recursive()
        return total


class RecursiveSimplexEngine:
    """
    Recursive Simplex Engine: Layer 0 for Livnium-T
    
    The fractal engine that creates simplex from simplex.
    
    Features:
    1. Subdivide simplex into smaller simplex (D-simplex → (D-1)-simplex or smaller D-simplex)
    2. Project high-dimensional states downward
    3. Conservation recursion (ΣSW preserved per scale)
    4. Recursive entanglement (compressed into lower scale)
    5. Recursive observer (macro → micro)
    6. Recursive motion (rotation at macro → rotation in micro)
    7. Recursive problem solving (search across layers)
    """
    
    def __init__(self, 
                 base_geometry: LivniumTSystem,
                 max_depth: int = 3,
                 subdivision_rule: Optional[Callable] = None):
        """
        Initialize recursive simplex engine.
        
        Args:
            base_geometry: Base Livnium-T System (Level 0)
            max_depth: Maximum recursion depth
            subdivision_rule: Optional custom subdivision rule
        """
        self.base_geometry = base_geometry
        self.max_depth = max_depth
        self.subdivision_rule = subdivision_rule or self._default_subdivision_rule
        
        # Create hierarchy
        self.levels: Dict[int, SimplexLevel] = {}
        self._build_hierarchy()
        
        # Initialize components
        self.subdivision = SimplexSubdivision(self)
        self.projection = RecursiveProjection(self)
        self.conservation = RecursiveConservation(self)
        from .moksha_engine import MokshaEngine
        self.moksha = MokshaEngine(self)  # Fixed-point convergence engine
    
    def _build_hierarchy(self):
        """Build recursive simplex hierarchy."""
        # Level 0: Base geometry
        level_0 = SimplexLevel(
            level_id=0,
            geometry=self.base_geometry,
            scale_factor=1
        )
        self.levels[0] = level_0
        
        # Recursively create child levels
        self._create_child_levels(level_0, depth=1)
    
    def _create_child_levels(self, parent_level: SimplexLevel, depth: int):
        """Recursively create child levels."""
        if depth > self.max_depth:
            return
        
        # Create child geometry for each node in parent
        for node_id in range(5):  # 5 nodes: 0 (core) + 1-4 (vertices)
            # Subdivide: create smaller simplex inside this node
            child_geometry = self.subdivision_rule(parent_level.geometry, node_id, depth)
            
            if child_geometry:
                child_level = SimplexLevel(
                    level_id=depth,
                    geometry=child_geometry,
                    parent=parent_level,
                    scale_factor=1  # For now, same dimension (can be adapted)
                )
                parent_level.children[node_id] = child_level
                
                # Recursively create grandchildren
                self._create_child_levels(child_level, depth + 1)
    
    def _default_subdivision_rule(self, 
                                  parent_geometry: LivniumTSystem,
                                  node_id: int,
                                  depth: int) -> Optional[LivniumTSystem]:
        """
        Default subdivision rule: Create smaller simplex inside node.
        
        Rule: Each node contains a smaller Livnium-T system.
        For simplicity, we create another D=3 simplex (tetrahedron).
        
        Args:
            parent_geometry: Parent geometry
            node_id: Node ID to subdivide
            depth: Current depth
            
        Returns:
            Child geometry or None
        """
        # For now, create another D=3 simplex (same as parent)
        # Future: could create D=2 (triangle) or D=4 (4-simplex) based on node class
        child_geometry = LivniumTSystem()
        return child_geometry
    
    def get_level(self, level_id: int) -> SimplexLevel:
        """Get geometry level by ID."""
        if level_id not in self.levels:
            raise ValueError(f"Level {level_id} not found")
        return self.levels[level_id]
    
    def get_node_path(self, path: List[int]) -> SimplexLevel:
        """
        Get node at path through hierarchy.
        
        Args:
            path: List of node IDs [level0_node, level1_node, ...]
            
        Returns:
            SimplexLevel at path
        """
        current = self.levels[0]
        for node_id in path:
            if node_id not in current.children:
                raise ValueError(f"Path {path} invalid: node {node_id} not found")
            current = current.children[node_id]
        return current
    
    def get_total_capacity(self) -> int:
        """Get total capacity (nodes) across all levels."""
        return self.levels[0].get_total_nodes_recursive()
    
    def __repr__(self) -> str:
        """String representation."""
        total_nodes = self.get_total_capacity()
        return (
            f"RecursiveSimplexEngine("
            f"levels={len(self.levels)}, "
            f"max_depth={self.max_depth}, "
            f"total_nodes={total_nodes}"
            f")"
        )

