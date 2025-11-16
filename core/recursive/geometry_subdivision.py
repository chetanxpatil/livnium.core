"""
Geometry Subdivision: Subdivide Geometry into Smaller Geometry

Implements the core rule: N×N×N → each cell contains an M×M×M
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, TYPE_CHECKING

from ..classical.livnium_core_system import LivniumCoreSystem

if TYPE_CHECKING:
    from .recursive_geometry_engine import RecursiveGeometryEngine


class GeometrySubdivision:
    """
    Handles subdivision of geometry into smaller geometry.
    
    Core rule: Each cell can contain a smaller geometry.
    This creates fractal compression.
    """
    
    def __init__(self, recursive_engine: 'RecursiveGeometryEngine'):
        """
        Initialize geometry subdivision.
        
        Args:
            recursive_engine: Recursive geometry engine
        """
        self.recursive_engine = recursive_engine
    
    def subdivide_by_face_exposure(self, 
                                   level_id: int,
                                   min_exposure: int = 2) -> int:
        """
        Subdivide cells based on face exposure.
        
        Rule: Cells with high face exposure (corners, edges) get subdivided.
        
        Args:
            level_id: Level to subdivide at
            min_exposure: Minimum face exposure to subdivide
            
        Returns:
            Number of cells subdivided
        """
        if level_id not in self.recursive_engine.levels:
            return 0
        
        level = self.recursive_engine.levels[level_id]
        subdivided = 0
        
        for coords, cell in level.geometry.lattice.items():
            if cell.face_exposure >= min_exposure:
                if self.recursive_engine.subdivide_cell(level_id, coords):
                    subdivided += 1
        
        return subdivided
    
    def subdivide_by_symbolic_weight(self,
                                     level_id: int,
                                     min_sw: float = 18.0) -> int:
        """
        Subdivide cells based on symbolic weight.
        
        Rule: Cells with high SW get subdivided (more "important" cells).
        
        Args:
            level_id: Level to subdivide at
            min_sw: Minimum symbolic weight to subdivide
            
        Returns:
            Number of cells subdivided
        """
        if level_id not in self.recursive_engine.levels:
            return 0
        
        level = self.recursive_engine.levels[level_id]
        subdivided = 0
        
        for coords, cell in level.geometry.lattice.items():
            if cell.symbolic_weight >= min_sw:
                if self.recursive_engine.subdivide_cell(level_id, coords):
                    subdivided += 1
        
        return subdivided
    
    def subdivide_all(self, level_id: int) -> int:
        """
        Subdivide all cells at level.
        
        Args:
            level_id: Level to subdivide at
            
        Returns:
            Number of cells subdivided
        """
        if level_id not in self.recursive_engine.levels:
            return 0
        
        level = self.recursive_engine.levels[level_id]
        subdivided = 0
        
        for coords in level.geometry.lattice.keys():
            if self.recursive_engine.subdivide_cell(level_id, coords):
                subdivided += 1
        
        return subdivided
    
    def get_subdivision_statistics(self, level_id: int) -> Dict:
        """
        Get subdivision statistics for level.
        
        Args:
            level_id: Level ID
            
        Returns:
            Statistics dictionary
        """
        if level_id not in self.recursive_engine.levels:
            return {}
        
        level = self.recursive_engine.levels[level_id]
        
        total_cells = len(level.geometry.lattice)
        subdivided_cells = len(level.children)
        
        return {
            'level_id': level_id,
            'total_cells': total_cells,
            'subdivided_cells': subdivided_cells,
            'subdivision_ratio': subdivided_cells / total_cells if total_cells > 0 else 0.0,
            'total_child_cells': sum(
                child.get_total_cells_recursive() 
                for child in level.children.values()
            ),
        }

