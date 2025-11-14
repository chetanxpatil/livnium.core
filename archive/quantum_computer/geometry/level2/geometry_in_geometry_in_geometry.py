"""
Level 2: Geometry in Geometry in Geometry

Meta-meta-geometric operations that operate ON geometry-in-geometry.
This is geometry > geometry > geometry - the highest level.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from quantum_computer.geometry.level1.geometry_in_geometry import GeometryInGeometry, MetaGeometricOperation
from quantum_computer.geometry.level0.base_geometry import BaseGeometry


@dataclass
class MetaMetaGeometricOperation:
    """
    Meta-meta-geometric operation - operates on geometry-in-geometry.
    
    This is geometry operating on geometry operating on geometry.
    """
    operation_type: str
    parameters: Dict
    target_geometry_in_geometry: GeometryInGeometry
    
    def apply(self) -> GeometryInGeometry:
        """Apply meta-meta operation to geometry-in-geometry."""
        # Transform the geometry-in-geometry system
        transformed_base = self.target_geometry_in_geometry.base_geometry
        
        # Create new geometry-in-geometry with transformed base
        transformed = GeometryInGeometry(transformed_base)
        
        # Apply meta-meta transformation to operations
        for op in self.target_geometry_in_geometry.meta_operations:
            # Transform the operation parameters
            new_params = self._transform_parameters(op.parameters)
            transformed.add_meta_operation(op.operation_type, **new_params)
        
        return transformed
    
    def _transform_parameters(self, params: Dict) -> Dict:
        """Transform operation parameters."""
        if self.operation_type == 'scale_operations':
            scale = self.parameters.get('scale', 1.0)
            return {k: v * scale if isinstance(v, (int, float)) else v 
                   for k, v in params.items()}
        elif self.operation_type == 'compose':
            # Compose operations
            return params
        else:
            return params


class GeometryInGeometryInGeometry:
    """
    Level 2: Geometry operating on geometry operating on geometry.
    
    This is the highest meta-geometric level.
    """
    
    def __init__(self, geometry_in_geometry: GeometryInGeometry):
        """
        Initialize geometry-in-geometry-in-geometry system.
        
        Args:
            geometry_in_geometry: The geometry-in-geometry to operate on
        """
        self.geometry_in_geometry = geometry_in_geometry
        self.meta_meta_operations: List[MetaMetaGeometricOperation] = []
    
    def add_meta_meta_operation(self, operation_type: str, **parameters) -> MetaMetaGeometricOperation:
        """
        Add a meta-meta-geometric operation.
        
        Args:
            operation_type: Type of operation
            **parameters: Operation parameters
            
        Returns:
            Created meta-meta operation
        """
        operation = MetaMetaGeometricOperation(
            operation_type=operation_type,
            parameters=parameters,
            target_geometry_in_geometry=self.geometry_in_geometry
        )
        self.meta_meta_operations.append(operation)
        return operation
    
    def apply_all_operations(self) -> GeometryInGeometry:
        """
        Apply all meta-meta operations.
        
        Returns:
            Transformed geometry-in-geometry
        """
        result = self.geometry_in_geometry
        
        for operation in self.meta_meta_operations:
            result = operation.apply()
        
        return result
    
    def get_hierarchical_structure(self) -> Dict:
        """
        Get the full hierarchical structure.
        
        Returns:
            Dictionary describing all levels
        """
        return {
            'level': 2,
            'type': 'geometry_in_geometry_in_geometry',
            'description': 'Geometry operating on geometry operating on geometry',
            'num_meta_meta_operations': len(self.meta_meta_operations),
            'geometry_in_geometry': self.geometry_in_geometry.get_meta_structure()
        }


class HierarchicalGeometrySystem:
    """
    Complete hierarchical geometry system.
    
    Manages all three levels:
    - Level 0: Base geometry
    - Level 1: Geometry in geometry
    - Level 2: Geometry in geometry in geometry
    """
    
    def __init__(self, base_dimension: int = 3):
        """
        Initialize hierarchical geometry system.
        
        Args:
            base_dimension: Dimension of base geometry
        """
        # Level 0: Base geometry
        self.base_geometry = BaseGeometry(dimension=base_dimension)
        
        # Level 1: Geometry in geometry
        self.geometry_in_geometry = GeometryInGeometry(self.base_geometry)
        
        # Level 2: Geometry in geometry in geometry
        self.geometry_in_geometry_in_geometry = GeometryInGeometryInGeometry(
            self.geometry_in_geometry
        )
    
    def add_base_state(self, coordinates: Tuple[float, ...], 
                      amplitude: complex = 1.0+0j, phase: float = 0.0):
        """Add state to base geometry (Level 0)."""
        return self.base_geometry.add_state(coordinates, amplitude, phase)
    
    def add_meta_operation(self, operation_type: str, **parameters):
        """Add meta-geometric operation (Level 1)."""
        return self.geometry_in_geometry.add_meta_operation(operation_type, **parameters)
    
    def add_meta_meta_operation(self, operation_type: str, **parameters):
        """Add meta-meta-geometric operation (Level 2)."""
        return self.geometry_in_geometry_in_geometry.add_meta_meta_operation(
            operation_type, **parameters
        )
    
    def get_full_structure(self) -> Dict:
        """Get complete hierarchical structure."""
        return {
            'hierarchical_levels': 3,
            'level_0': self.base_geometry.get_geometry_structure(),
            'level_1': self.geometry_in_geometry.get_meta_structure(),
            'level_2': self.geometry_in_geometry_in_geometry.get_hierarchical_structure(),
            'principle': 'Geometry > Geometry in Geometry'
        }

