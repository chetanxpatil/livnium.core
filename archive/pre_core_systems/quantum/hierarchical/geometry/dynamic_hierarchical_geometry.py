"""
Dynamic Hierarchical Geometry System

Allows N-level geometry stacking: geometry > geometry > geometry > ... > geometry
Can be initialized with any number of levels (1, 2, 3, 4, 5, 6...)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from quantum.hierarchical.geometry.level0.base_geometry import BaseGeometry, BaseGeometricState


@dataclass
class HierarchicalOperation:
    """
    A hierarchical operation at any level.
    
    Each level can have operations that transform the geometry below it.
    """
    level: int
    operation_type: str
    parameters: Dict[str, Any]
    
    def apply(self, target_geometry: Any) -> Any:
        """Apply operation to target geometry."""
        # Transform based on operation type
        if self.operation_type == 'rotation':
            return self._apply_rotation(target_geometry)
        elif self.operation_type == 'scale':
            return self._apply_scale(target_geometry)
        elif self.operation_type == 'entangle':
            return self._apply_entangle(target_geometry)
        elif self.operation_type == 'transform':
            return self._apply_transform(target_geometry)
        else:
            return target_geometry
    
    def _apply_rotation(self, geometry: Any) -> Any:
        """Apply rotation transformation."""
        # Simplified: return geometry unchanged (full implementation would rotate)
        return geometry
    
    def _apply_scale(self, geometry: Any) -> Any:
        """Apply scale transformation."""
        scale = self.parameters.get('scale', 1.0)
        # Simplified: return geometry unchanged (full implementation would scale)
        return geometry
    
    def _apply_entangle(self, geometry: Any) -> Any:
        """Apply entanglement operation."""
        # Simplified: return geometry unchanged (full implementation would entangle)
        return geometry
    
    def _apply_transform(self, geometry: Any) -> Any:
        """Apply general transformation."""
        # Simplified: return geometry unchanged (full implementation would transform)
        return geometry


class HierarchicalLevel:
    """
    A single level in the hierarchical geometry system.
    
    Each level wraps the level below it and can operate on it.
    """
    
    def __init__(self, level: int, base_geometry: Any):
        """
        Initialize hierarchical level.
        
        Args:
            level: Level number (0 = base, 1 = first meta, 2 = second meta, etc.)
            base_geometry: The geometry this level operates on
        """
        self.level = level
        self.base_geometry = base_geometry
        self.operations: List[HierarchicalOperation] = []
    
    def add_operation(self, operation_type: str, **parameters) -> HierarchicalOperation:
        """
        Add an operation at this level.
        
        Args:
            operation_type: Type of operation
            **parameters: Operation parameters
            
        Returns:
            Created operation
        """
        operation = HierarchicalOperation(
            level=self.level,
            operation_type=operation_type,
            parameters=parameters
        )
        self.operations.append(operation)
        return operation
    
    def apply_all_operations(self) -> Any:
        """Apply all operations at this level."""
        result = self.base_geometry
        for operation in self.operations:
            result = operation.apply(result)
        return result
    
    def get_structure(self) -> Dict:
        """Get structure information for this level."""
        return {
            'level': self.level,
            'num_operations': len(self.operations),
            'base_structure': self._get_base_structure()
        }
    
    def _get_base_structure(self) -> Dict:
        """Get structure of base geometry."""
        if hasattr(self.base_geometry, 'get_geometry_structure'):
            return self.base_geometry.get_geometry_structure()
        elif hasattr(self.base_geometry, 'get_structure'):
            return self.base_geometry.get_structure()
        else:
            return {'type': 'unknown'}


class DynamicHierarchicalGeometrySystem:
    """
    Dynamic hierarchical geometry system with N levels.
    
    Can be initialized with any number of levels:
    - 1 level: Just base geometry
    - 2 levels: Base + 1 meta level
    - 3 levels: Base + 2 meta levels
    - N levels: Base + (N-1) meta levels
    
    Example:
        # 3 levels (like current system)
        system = DynamicHierarchicalGeometrySystem(base_dimension=3, num_levels=3)
        
        # 5 levels
        system = DynamicHierarchicalGeometrySystem(base_dimension=3, num_levels=5)
        
        # 6 levels
        system = DynamicHierarchicalGeometrySystem(base_dimension=3, num_levels=6)
    """
    
    def __init__(self, base_dimension: int = 3, num_levels: int = 3):
        """
        Initialize dynamic hierarchical geometry system.
        
        Args:
            base_dimension: Dimension of base geometric space
            num_levels: Number of hierarchical levels (1, 2, 3, 4, 5, 6...)
                       Level 0 is always base geometry
                       Levels 1+ are meta-levels
        """
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        
        self.base_dimension = base_dimension
        self.num_levels = num_levels
        
        # Level 0: Base geometry
        self.base_geometry = BaseGeometry(dimension=base_dimension)
        
        # Build hierarchy: each level wraps the previous one
        self.levels: List[HierarchicalLevel] = []
        current_geometry = self.base_geometry
        
        # Create meta-levels (1, 2, 3, ..., num_levels-1)
        for level_num in range(1, num_levels):
            level = HierarchicalLevel(level=level_num, base_geometry=current_geometry)
            self.levels.append(level)
            current_geometry = level
        
        # Store reference to top level
        self.top_level = self.levels[-1] if self.levels else None
    
    def add_base_state(self, coordinates: Tuple[float, ...], 
                      amplitude: complex = 1.0+0j, phase: float = 0.0) -> BaseGeometricState:
        """
        Add state to base geometry (Level 0).
        
        Args:
            coordinates: Geometric coordinates
            amplitude: Quantum amplitude
            phase: Phase angle
            
        Returns:
            Created geometric state
        """
        return self.base_geometry.add_state(coordinates, amplitude, phase)
    
    def add_operation(self, level: int, operation_type: str, **parameters) -> HierarchicalOperation:
        """
        Add operation at specified level.
        
        Args:
            level: Level number (1 = first meta, 2 = second meta, etc.)
                   Level 0 is base geometry (use add_base_state instead)
            operation_type: Type of operation
            **parameters: Operation parameters
            
        Returns:
            Created operation
            
        Raises:
            ValueError: If level is invalid
        """
        if level < 1:
            raise ValueError(f"Level must be >= 1 (use add_base_state for level 0)")
        if level >= self.num_levels:
            raise ValueError(f"Level {level} exceeds maximum level {self.num_levels - 1}")
        
        # Get the level (levels are 1-indexed in the list, but 0-indexed in hierarchy)
        level_index = level - 1
        return self.levels[level_index].add_operation(operation_type, **parameters)
    
    def add_meta_operation(self, operation_type: str, **parameters) -> HierarchicalOperation:
        """
        Add operation at Level 1 (convenience method).
        
        Args:
            operation_type: Type of operation
            **parameters: Operation parameters
            
        Returns:
            Created operation
        """
        return self.add_operation(level=1, operation_type=operation_type, **parameters)
    
    def add_meta_meta_operation(self, operation_type: str, **parameters) -> HierarchicalOperation:
        """
        Add operation at Level 2 (convenience method).
        
        Args:
            operation_type: Type of operation
            **parameters: Operation parameters
            
        Returns:
            Created operation
        """
        if self.num_levels < 3:
            raise ValueError(f"System has only {self.num_levels} levels, need at least 3 for meta-meta operations")
        return self.add_operation(level=2, operation_type=operation_type, **parameters)
    
    def apply_all_operations(self) -> BaseGeometry:
        """
        Apply all operations at all levels, starting from top.
        
        Returns:
            Transformed base geometry
        """
        result = self.base_geometry
        
        # Apply operations from bottom to top (level 1, then 2, then 3, etc.)
        for level in self.levels:
            result = level.apply_all_operations()
        
        return result
    
    def get_full_structure(self) -> Dict:
        """
        Get complete hierarchical structure information.
        
        Returns:
            Dictionary describing all levels
        """
        structure = {
            'num_levels': self.num_levels,
            'base_dimension': self.base_dimension,
            'level_0': self.base_geometry.get_geometry_structure(),
            'meta_levels': []
        }
        
        for level in self.levels:
            structure['meta_levels'].append(level.get_structure())
        
        # Build description
        if self.num_levels == 1:
            structure['principle'] = 'Base Geometry'
        elif self.num_levels == 2:
            structure['principle'] = 'Geometry > Geometry'
        elif self.num_levels == 3:
            structure['principle'] = 'Geometry > Geometry > Geometry'
        else:
            structure['principle'] = f'Geometry > ' * (self.num_levels - 1) + 'Geometry'
        
        return structure
    
    def get_level_info(self, level: int) -> Dict:
        """
        Get information about a specific level.
        
        Args:
            level: Level number (0 = base, 1+ = meta levels)
            
        Returns:
            Level information
        """
        if level == 0:
            return self.base_geometry.get_geometry_structure()
        elif 1 <= level < self.num_levels:
            return self.levels[level - 1].get_structure()
        else:
            raise ValueError(f"Invalid level: {level} (max: {self.num_levels - 1})")

