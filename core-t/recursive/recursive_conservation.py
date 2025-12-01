"""
Recursive Conservation: Invariant Preservation Across Scales for Livnium-T

Ensures conservation laws hold at every recursive level.
"""

from typing import Dict, List, Optional
from ..classical.livnium_t_system import LivniumTSystem, NodeClass


class RecursiveConservation:
    """
    Conservation engine for recursive simplex hierarchy.
    
    Verifies that invariants are preserved at every scale:
    - ΣSW conservation
    - Class count conservation
    - Ledger conservation
    """
    
    def __init__(self, recursive_engine):
        """
        Initialize conservation engine.
        
        Args:
            recursive_engine: RecursiveSimplexEngine instance
        """
        self.recursive_engine = recursive_engine
    
    def verify_level_conservation(self, level_id: int) -> bool:
        """
        Verify conservation at a specific level.
        
        Args:
            level_id: Level ID
            
        Returns:
            True if all invariants conserved
        """
        level = self.recursive_engine.get_level(level_id)
        geometry = level.geometry
        
        # Check ledger
        if not geometry.verify_ledger():
            return False
        
        # Check total SW (should be 108 for D=3)
        total_sw = geometry.get_total_sw()
        expected_sw = 108.0  # For D=3 tetrahedron
        if abs(total_sw - expected_sw) > 1e-6:
            return False
        
        # Check class counts
        counts = geometry.get_class_counts()
        if counts[NodeClass.CORE] != 1:
            return False
        if counts[NodeClass.VERTEX] != 4:
            return False
        
        return True
    
    def verify_recursive_conservation(self) -> bool:
        """
        Verify conservation at all levels.
        
        Returns:
            True if all levels conserve invariants
        """
        for level_id in self.recursive_engine.levels.keys():
            if not self.verify_level_conservation(level_id):
                return False
        return True
    
    def get_level_sw(self, level_id: int) -> float:
        """
        Get total SW at a level (including children).
        
        Args:
            level_id: Level ID
            
        Returns:
            Total SW including recursive children
        """
        level = self.recursive_engine.get_level(level_id)
        
        # Base SW at this level
        total_sw = level.geometry.get_total_sw()
        
        # Add SW from all children
        for child in level.children.values():
            total_sw += self.get_level_sw(child.level_id)
        
        return total_sw
    
    def get_level_node_count(self, level_id: int) -> int:
        """
        Get total node count at a level (including children).
        
        Args:
            level_id: Level ID
            
        Returns:
            Total node count including recursive children
        """
        level = self.recursive_engine.get_level(level_id)
        return level.get_total_nodes_recursive()

















