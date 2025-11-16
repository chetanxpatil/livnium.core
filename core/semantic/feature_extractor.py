"""
Feature Extractor: Extract Features from Geometric and Quantum States

Extracts meaningful features for semantic processing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from ..classical.livnium_core_system import LivniumCoreSystem


class FeatureExtractor:
    """
    Extracts features from Livnium Core System cells.
    
    Features include geometric, symbolic, and contextual properties.
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize feature extractor.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
    
    def extract_geometric_features(self, coords: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Extract geometric features from cell.
        
        Args:
            coords: Cell coordinates
            
        Returns:
            Feature dictionary
        """
        cell = self.core_system.get_cell(coords)
        if not cell:
            return {}
        
        # Calculate distance from observer
        observer_coords = (0, 0, 0)
        distance_from_observer = np.linalg.norm(
            np.array(coords) - np.array(observer_coords)
        )
        
        # Calculate distance from center
        center_distance = np.linalg.norm(np.array(coords))
        
        # Get neighbors (simplified)
        neighbors = self._get_neighbors(coords)
        
        return {
            'distance_from_observer': float(distance_from_observer),
            'center_distance': float(center_distance),
            'num_neighbors': len(neighbors),
            'is_boundary': self._is_boundary(coords),
            'is_corner': cell.cell_class.value == 3 if cell.cell_class else False,
            'is_core': cell.cell_class.value == 0 if cell.cell_class else False,
        }
    
    def extract_symbolic_features(self, coords: Tuple[int, int, int]) -> Dict[str, Any]:
        """Extract symbolic features."""
        cell = self.core_system.get_cell(coords)
        if not cell:
            return {}
        
        symbol = self.core_system.get_symbol(coords)
        
        return {
            'symbol': symbol,
            'symbolic_weight': cell.symbolic_weight,
            'face_exposure': cell.face_exposure,
            'sw_normalized': cell.symbolic_weight / 27.0,
        }
    
    def extract_contextual_features(self, coords: Tuple[int, int, int]) -> Dict[str, Any]:
        """Extract contextual features (neighborhood)."""
        neighbors = self._get_neighbors(coords)
        
        neighbor_classes = []
        neighbor_sws = []
        
        for neighbor_coords in neighbors:
            neighbor_cell = self.core_system.get_cell(neighbor_coords)
            if neighbor_cell:
                neighbor_classes.append(neighbor_cell.cell_class.value if neighbor_cell.cell_class else 0)
                neighbor_sws.append(neighbor_cell.symbolic_weight)
        
        return {
            'neighbor_count': len(neighbors),
            'avg_neighbor_sw': float(np.mean(neighbor_sws)) if neighbor_sws else 0.0,
            'neighbor_class_diversity': len(set(neighbor_classes)),
        }
    
    def extract_all_features(self, coords: Tuple[int, int, int]) -> Dict[str, Any]:
        """Extract all features."""
        features = {}
        features.update(self.extract_geometric_features(coords))
        features.update(self.extract_symbolic_features(coords))
        features.update(self.extract_contextual_features(coords))
        return features
    
    def _get_neighbors(self, coords: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get neighboring cells."""
        x, y, z = coords
        neighbors = []
        
        # 6 face-adjacent neighbors
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in self.core_system.lattice:
                neighbors.append(neighbor)
        
        return neighbors
    
    def _is_boundary(self, coords: Tuple[int, int, int]) -> bool:
        """Check if cell is on boundary."""
        cell = self.core_system.get_cell(coords)
        if cell:
            return cell.face_exposure > 0
        return False
    
    def get_statistics(self) -> Dict:
        """Get feature extractor statistics."""
        return {
            'total_cells': len(self.core_system.lattice),
        }

