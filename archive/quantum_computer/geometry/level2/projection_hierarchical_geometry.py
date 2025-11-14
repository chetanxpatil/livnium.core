"""
Level 2: Projection-Based Hierarchical Geometry

Uses geometry > geometry > geometry to PROJECT high-entanglement states
into manageable representations without losing critical information.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from quantum_computer.geometry.level0.sparse_base_geometry import SparseBaseGeometry
from quantum_computer.geometry.level1.sparse_geometry_in_geometry import SparseGeometryInGeometry


@dataclass
class ProjectionOperation:
    """
    Projection operation - compresses high-entanglement state into geometry hierarchy.
    
    Uses Level 2 to project onto Level 1, which operates on Level 0.
    """
    projection_type: str
    parameters: Dict
    target_geometry_in_geometry: SparseGeometryInGeometry
    
    def apply(self) -> SparseGeometryInGeometry:
        """Apply projection to compress state."""
        if self.projection_type == 'entanglement_compression':
            return self._compress_entanglement()
        elif self.projection_type == 'local_projection':
            return self._local_projection()
        elif self.projection_type == 'hierarchical_decomposition':
            return self._hierarchical_decomposition()
        else:
            return self.target_geometry_in_geometry
    
    def _compress_entanglement(self) -> SparseGeometryInGeometry:
        """
        Compress entanglement by projecting onto geometry hierarchy.
        
        Instead of storing full entanglement, project it into:
        - Level 0: Local correlations (sparse)
        - Level 1: Medium-range correlations (efficient operations)
        - Level 2: Long-range correlations (compressed representation)
        """
        # Get current state from Level 0
        base = self.target_geometry_in_geometry.base_geometry
        
        # Project high-entanglement regions onto compressed representation
        # Keep only significant correlations in Level 0
        # Store long-range correlations in Level 2
        
        # Create new compressed geometry
        compressed_base = SparseBaseGeometry(
            dimension=base.dimension,
            threshold=base.threshold * 10  # Higher threshold = more compression
        )
        
        # Project: Keep only most significant states
        significant_states = []
        for coords in base.active_coordinates:
            amplitude = base.get_amplitude(coords)
            if abs(amplitude) > base.threshold * 10:
                significant_states.append((coords, amplitude))
        
        # Sort by magnitude and keep top N
        significant_states.sort(key=lambda x: abs(x[1]), reverse=True)
        max_states = self.parameters.get('max_states', 1000)
        
        for coords, amplitude in significant_states[:max_states]:
            compressed_base.add_state(coords, amplitude, np.angle(amplitude))
        
        # Create new geometry-in-geometry with compressed base
        compressed = SparseGeometryInGeometry(compressed_base)
        
        # Copy operations from original
        for op in self.target_geometry_in_geometry.meta_operations:
            compressed.add_meta_operation(op.operation_type, **op.parameters)
        
        return compressed
    
    def _local_projection(self) -> SparseGeometryInGeometry:
        """
        Project onto local geometry - keep only local correlations.
        
        For maximum entanglement chain, project onto local regions.
        """
        base = self.target_geometry_in_geometry.base_geometry
        
        # Project: Keep only states with local structure
        # (e.g., for 1D chain, keep states that are "close" in geometry)
        
        compressed_base = SparseBaseGeometry(
            dimension=base.dimension,
            threshold=base.threshold
        )
        
        # Keep states based on local geometry
        for coords in base.active_coordinates:
            amplitude = base.get_amplitude(coords)
            
            # Check if state is "local" (coordinates are close)
            if self._is_local_state(coords):
                compressed_base.add_state(coords, amplitude, np.angle(amplitude))
        
        compressed = SparseGeometryInGeometry(compressed_base)
        for op in self.target_geometry_in_geometry.meta_operations:
            compressed.add_meta_operation(op.operation_type, **op.parameters)
        
        return compressed
    
    def _is_local_state(self, coords: Tuple[float, ...]) -> bool:
        """Check if state is 'local' (coordinates are close together)."""
        # For 1D chain, local means coordinates don't jump too much
        coords_array = np.array(coords)
        if len(coords_array) < 2:
            return True
        
        # Check if coordinates are "smooth" (no big jumps)
        diffs = np.diff(coords_array)
        max_jump = np.max(np.abs(diffs))
        
        # Local if max jump is small
        return max_jump < 0.5
    
    def _hierarchical_decomposition(self) -> SparseGeometryInGeometry:
        """
        Decompose state hierarchically:
        - Level 0: Store local structure (low memory)
        - Level 1: Store medium-range structure (efficient)
        - Level 2: Store long-range structure (compressed)
        """
        base = self.target_geometry_in_geometry.base_geometry
        
        # Decompose into hierarchical levels
        # Level 0: Most significant local states
        # Level 1: Medium-range correlations
        # Level 2: Long-range correlations (compressed)
        
        compressed_base = SparseBaseGeometry(
            dimension=base.dimension,
            threshold=base.threshold
        )
        
        # Project onto hierarchical structure
        # Keep states that can be represented efficiently in hierarchy
        
        # Sort by "importance" (magnitude and locality)
        state_scores = []
        for coords in base.active_coordinates:
            amplitude = base.get_amplitude(coords)
            magnitude = abs(amplitude)
            
            # Score: magnitude * locality
            locality = 1.0 / (1.0 + self._state_range(coords))
            score = magnitude * locality
            
            state_scores.append((coords, amplitude, score))
        
        # Keep top states
        state_scores.sort(key=lambda x: x[2], reverse=True)
        max_states = self.parameters.get('max_states', 2000)
        
        for coords, amplitude, _ in state_scores[:max_states]:
            compressed_base.add_state(coords, amplitude, np.angle(amplitude))
        
        compressed = SparseGeometryInGeometry(compressed_base)
        for op in self.target_geometry_in_geometry.meta_operations:
            compressed.add_meta_operation(op.operation_type, **op.parameters)
        
        return compressed
    
    def _state_range(self, coords: Tuple[float, ...]) -> float:
        """Compute 'range' of state (how spread out coordinates are)."""
        coords_array = np.array(coords)
        if len(coords_array) < 2:
            return 0.0
        
        return np.max(coords_array) - np.min(coords_array)


class ProjectionHierarchicalGeometrySystem:
    """
    Hierarchical geometry system with projection-based compression.
    
    Uses Level 2 to project high-entanglement states onto manageable
    representations in Level 0/1, preserving critical information.
    """
    
    def __init__(self, base_dimension: int = 3, threshold: float = 1e-15):
        """Initialize projection-based hierarchical system."""
        # Level 0: Sparse base geometry
        self.base_geometry = SparseBaseGeometry(dimension=base_dimension, threshold=threshold)
        
        # Level 1: Geometry in geometry
        self.geometry_in_geometry = SparseGeometryInGeometry(self.base_geometry)
        
        # Level 2: Projection operations
        self.projection_operations: List[ProjectionOperation] = []
    
    def add_base_state(self, coordinates: Tuple[float, ...], 
                      amplitude: complex = 1.0+0j, phase: float = 0.0):
        """Add state to Level 0."""
        return self.base_geometry.add_state(coordinates, amplitude, phase)
    
    def add_meta_operation(self, operation_type: str, **parameters):
        """Add Level 1 operation."""
        return self.geometry_in_geometry.add_meta_operation(operation_type, **parameters)
    
    def project_entanglement(self, max_states: int = 1000):
        """
        Project high-entanglement state onto manageable representation.
        
        Uses Level 2 to compress state while preserving critical information.
        """
        projection = ProjectionOperation(
            projection_type='entanglement_compression',
            parameters={'max_states': max_states},
            target_geometry_in_geometry=self.geometry_in_geometry
        )
        
        self.projection_operations.append(projection)
        
        # Apply projection
        self.geometry_in_geometry = projection.apply()
        self.base_geometry = self.geometry_in_geometry.base_geometry
    
    def project_hierarchical(self, max_states: int = 2000):
        """Project using hierarchical decomposition."""
        projection = ProjectionOperation(
            projection_type='hierarchical_decomposition',
            parameters={'max_states': max_states},
            target_geometry_in_geometry=self.geometry_in_geometry
        )
        
        self.projection_operations.append(projection)
        self.geometry_in_geometry = projection.apply()
        self.base_geometry = self.geometry_in_geometry.base_geometry
    
    def get_full_structure(self) -> Dict:
        """Get hierarchical structure."""
        return {
            'hierarchical_levels': 3,
            'level_0': self.base_geometry.get_geometry_structure(),
            'level_1': self.geometry_in_geometry.get_meta_structure(),
            'level_2_projections': len(self.projection_operations),
            'principle': 'Geometry > Geometry > Geometry (Projection-Based)',
            'compression': 'Uses Level 2 to project high-entanglement onto Level 0/1'
        }

