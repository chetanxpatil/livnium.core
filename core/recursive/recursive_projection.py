"""
Recursive Projection: Project States Across Geometry Levels

Projects high-dimensional states downward and aggregates upward.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .recursive_geometry_engine import RecursiveGeometryEngine, GeometryLevel


class RecursiveProjection:
    """
    Handles projection of states across geometry levels.
    
    Features:
    - Project downward: macro → micro
    - Project upward: micro → macro
    - State compression
    - Constraint propagation
    """
    
    def __init__(self, recursive_engine: 'RecursiveGeometryEngine'):
        """
        Initialize recursive projection.
        
        Args:
            recursive_engine: Recursive geometry engine
        """
        self.recursive_engine = recursive_engine
    
    def project_downward(self,
                        source_level: int,
                        target_level: int,
                        state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project state from higher level to lower level.
        
        Rule: Macro constraints → micro constraints
        
        Args:
            source_level: Source level ID
            target_level: Target level ID (must be > source_level)
            state: State to project
            
        Returns:
            Projected state
        """
        if source_level >= target_level:
            raise ValueError("target_level must be > source_level for downward projection")
        
        if source_level not in self.recursive_engine.levels:
            return {}
        
        if target_level not in self.recursive_engine.levels:
            return {}
        
        source_level_obj = self.recursive_engine.levels[source_level]
        target_level_obj = self.recursive_engine.levels[target_level]
        
        # Project: map macro state to micro constraints
        projected = {}
        
        # Extract constraints from source state
        if 'constraints' in state:
            # Project constraints to target level
            projected['constraints'] = self._project_constraints(
                source_level_obj,
                target_level_obj,
                state['constraints']
            )
        
        # Extract values from source state
        if 'values' in state:
            # Project values (e.g., SW, face exposure)
            projected['values'] = self._project_values(
                source_level_obj,
                target_level_obj,
                state['values']
            )
        
        return projected
    
    def project_upward(self,
                      source_level: int,
                      target_level: int,
                      state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project state from lower level to higher level.
        
        Rule: Micro results → macro aggregation
        
        Args:
            source_level: Source level ID
            target_level: Target level ID (must be < source_level)
            state: State to project
            
        Returns:
            Projected state
        """
        if source_level <= target_level:
            raise ValueError("target_level must be < source_level for upward projection")
        
        if source_level not in self.recursive_engine.levels:
            return {}
        
        if target_level not in self.recursive_engine.levels:
            return {}
        
        source_level_obj = self.recursive_engine.levels[source_level]
        target_level_obj = self.recursive_engine.levels[target_level]
        
        # Aggregate: combine micro states into macro state
        aggregated = {}
        
        # Aggregate values
        if 'values' in state:
            aggregated['values'] = self._aggregate_values(
                source_level_obj,
                target_level_obj,
                state['values']
            )
        
        # Aggregate constraints
        if 'constraints' in state:
            aggregated['constraints'] = self._aggregate_constraints(
                source_level_obj,
                target_level_obj,
                state['constraints']
            )
        
        return aggregated
    
    def _project_constraints(self,
                            source_level: 'GeometryLevel',
                            target_level: 'GeometryLevel',
                            constraints: Dict) -> Dict:
        """Project constraints from source to target level."""
        # Simplified: map constraints to child geometry
        projected = {}
        
        # Find which child of source contains target
        for parent_coords, child_level in source_level.children.items():
            if child_level.level_id == target_level.level_id:
                # Map constraints to child coordinates
                projected[parent_coords] = constraints
        
        return projected
    
    def _project_values(self,
                       source_level: 'GeometryLevel',
                       target_level: 'GeometryLevel',
                       values: Dict) -> Dict:
        """Project values from source to target level."""
        # Simplified: distribute values to child geometry
        projected = {}
        
        for parent_coords, child_level in source_level.children.items():
            if child_level.level_id == target_level.level_id:
                # Distribute values proportionally
                child_size = child_level.geometry.config.lattice_size
                parent_size = source_level.geometry.config.lattice_size
                scale = child_size / parent_size
                
                projected[parent_coords] = {
                    k: v * scale for k, v in values.items()
                }
        
        return projected
    
    def _aggregate_values(self,
                         source_level: 'GeometryLevel',
                         target_level: 'GeometryLevel',
                         values: Dict) -> Dict:
        """Aggregate values from source to target level."""
        # Aggregate: sum values from children
        aggregated = {}
        
        # Find children of target that are at source level
        for parent_coords, child_level in target_level.children.items():
            if child_level.level_id == source_level.level_id:
                # Aggregate child values
                if parent_coords not in aggregated:
                    aggregated[parent_coords] = {}
                
                for key, value in values.items():
                    if key not in aggregated[parent_coords]:
                        aggregated[parent_coords][key] = 0.0
                    aggregated[parent_coords][key] += value
        
        return aggregated
    
    def _aggregate_constraints(self,
                              source_level: 'GeometryLevel',
                              target_level: 'GeometryLevel',
                              constraints: Dict) -> Dict:
        """Aggregate constraints from source to target level."""
        # Aggregate: combine constraints from children
        aggregated = {}
        
        # Find children of target that are at source level
        for parent_coords, child_level in target_level.children.items():
            if child_level.level_id == source_level.level_id:
                # Combine constraints (AND operation)
                if parent_coords not in aggregated:
                    aggregated[parent_coords] = {}
                
                for key, value in constraints.items():
                    if key not in aggregated[parent_coords]:
                        aggregated[parent_coords][key] = True
                    aggregated[parent_coords][key] = aggregated[parent_coords][key] and value
        
        return aggregated

