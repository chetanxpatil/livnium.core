"""
Recursive Conservation: Preserve Invariants Across All Levels

Enforces conservation recursion: ΣSW and class counts preserved per scale.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .recursive_geometry_engine import RecursiveGeometryEngine


class RecursiveConservation:
    """
    Handles conservation of invariants across recursive levels.
    
    Rules:
    - ΣSW is preserved per scale
    - Class counts preserved per scale
    - Symbolic invariants propagate
    """
    
    def __init__(self, recursive_engine: 'RecursiveGeometryEngine'):
        """
        Initialize recursive conservation.
        
        Args:
            recursive_engine: Recursive geometry engine
        """
        self.recursive_engine = recursive_engine
    
    def verify_level_conservation(self, level_id: int) -> Dict[str, bool]:
        """
        Verify conservation at a specific level.
        
        Args:
            level_id: Level ID
            
        Returns:
            Dictionary of conservation checks
        """
        if level_id not in self.recursive_engine.levels:
            return {}
        
        level = self.recursive_engine.levels[level_id]
        geometry = level.geometry
        
        # Check SW conservation
        actual_sw = geometry.get_total_symbolic_weight()
        expected_sw = geometry.get_expected_total_sw()
        sw_conserved = abs(actual_sw - expected_sw) < 1e-6
        
        # Check class count conservation
        actual_counts = geometry.get_class_counts()
        expected_counts = geometry.get_expected_class_counts()
        counts_conserved = True
        
        for cls, expected_count in expected_counts.items():
            actual_count = actual_counts.get(cls, 0)
            if actual_count != expected_count:
                counts_conserved = False
                break
        
        return {
            'sw_conserved': sw_conserved,
            'class_counts_conserved': counts_conserved,
            'actual_sw': actual_sw,
            'expected_sw': expected_sw,
        }
    
    def verify_recursive_conservation(self) -> Dict[int, Dict[str, bool]]:
        """
        Verify conservation across all levels.
        
        Returns:
            Dictionary mapping level_id to conservation checks
        """
        results = {}
        
        for level_id in self.recursive_engine.levels.keys():
            results[level_id] = self.verify_level_conservation(level_id)
        
        return results
    
    def propagate_conservation_downward(self, level_id: int) -> bool:
        """
        Propagate conservation constraints downward.
        
        Rule: Parent invariants → child constraints
        
        Args:
            level_id: Level to propagate from
            
        Returns:
            True if propagation successful
        """
        if level_id not in self.recursive_engine.levels:
            return False
        
        level = self.recursive_engine.levels[level_id]
        
        # Get parent invariants
        parent_sw = level.geometry.get_total_symbolic_weight()
        parent_counts = level.geometry.get_class_counts()
        
        # Propagate to children
        for child_level in level.children.values():
            # Set child constraints based on parent
            # Simplified: child should maintain its own invariants
            child_sw = child_level.geometry.get_total_symbolic_weight()
            child_expected_sw = child_level.geometry.get_expected_total_sw()
            
            # Verify child maintains its own conservation
            if abs(child_sw - child_expected_sw) > 1e-6:
                # Child violates conservation - this is an error
                return False
        
        return True
    
    def aggregate_conservation_upward(self, level_id: int) -> Dict[str, float]:
        """
        Aggregate conservation values upward.
        
        Rule: Sum child invariants → parent totals
        
        Args:
            level_id: Level to aggregate from
            
        Returns:
            Aggregated values
        """
        if level_id not in self.recursive_engine.levels:
            return {}
        
        level = self.recursive_engine.levels[level_id]
        
        # Aggregate from children
        total_child_sw = 0.0
        total_child_counts = {}
        
        for child_level in level.children.values():
            child_sw = child_level.geometry.get_total_symbolic_weight()
            total_child_sw += child_sw
            
            child_counts = child_level.geometry.get_class_counts()
            for cls, count in child_counts.items():
                if cls not in total_child_counts:
                    total_child_counts[cls] = 0
                total_child_counts[cls] += count
        
        return {
            'total_child_sw': total_child_sw,
            'total_child_counts': total_child_counts,
        }
    
    def get_conservation_statistics(self) -> Dict:
        """Get conservation statistics across all levels."""
        stats = {}
        
        for level_id in self.recursive_engine.levels.keys():
            conservation = self.verify_level_conservation(level_id)
            aggregated = self.aggregate_conservation_upward(level_id)
            
            stats[level_id] = {
                'conservation': conservation,
                'aggregated_from_children': aggregated,
            }
        
        return stats

