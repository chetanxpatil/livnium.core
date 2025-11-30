"""
Recursive Projection: Macro ↔ Micro State Transfer for Livnium-T

Projects states between recursive levels (downward constraints, upward aggregation).
"""

from typing import Dict, List, Optional
import numpy as np

from ..classical.livnium_t_system import LivniumTSystem, NodeClass


class RecursiveProjection:
    """
    Projection engine for recursive simplex hierarchy.
    
    Handles:
    - Downward projection: Macro constraints → micro geometry
    - Upward projection: Micro results → macro aggregation
    """
    
    def __init__(self, recursive_engine):
        """
        Initialize projection engine.
        
        Args:
            recursive_engine: RecursiveSimplexEngine instance
        """
        self.recursive_engine = recursive_engine
    
    def project_downward(self, 
                        parent_level,
                        node_id: int,
                        constraints: Optional[Dict] = None) -> Dict:
        """
        Project macro-level constraints downward to child level.
        
        Args:
            parent_level: Parent SimplexLevel
            node_id: Node ID in parent
            constraints: Optional constraints dict
            
        Returns:
            Constraints for child level
        """
        if node_id not in parent_level.children:
            return {}
        
        child_level = parent_level.children[node_id]
        parent_node = parent_level.geometry.get_node(node_id)
        
        # Project based on node class
        constraints = constraints or {}
        
        # Core: Project stability constraints
        if parent_node.node_class == NodeClass.CORE:
            constraints['stability'] = True
            constraints['exposure'] = 0
        
        # Vertex: Project activity constraints
        elif parent_node.node_class == NodeClass.VERTEX:
            constraints['stability'] = False
            constraints['exposure'] = 3
            constraints['symbolic_weight'] = 27
        
        return constraints
    
    def project_upward(self, 
                      child_level,
                      aggregation: Optional[Dict] = None) -> Dict:
        """
        Project micro-level results upward to parent level.
        
        Args:
            child_level: Child SimplexLevel
            aggregation: Optional aggregation dict
            
        Returns:
            Aggregated results for parent
        """
        aggregation = aggregation or {}
        
        # Aggregate total SW from child
        child_sw = child_level.geometry.get_total_sw()
        aggregation['total_sw'] = aggregation.get('total_sw', 0) + child_sw
        
        # Aggregate class counts
        child_counts = child_level.geometry.get_class_counts()
        for node_class, count in child_counts.items():
            key = f'count_{node_class.name.lower()}'
            aggregation[key] = aggregation.get(key, 0) + count
        
        # Aggregate node count
        aggregation['total_nodes'] = aggregation.get('total_nodes', 0) + child_level.get_total_nodes()
        
        return aggregation
    
    def project_all_upward(self, level_id: int) -> Dict:
        """
        Project all children upward from a level.
        
        Args:
            level_id: Level ID
            
        Returns:
            Aggregated results
        """
        level = self.recursive_engine.get_level(level_id)
        aggregation = {}
        
        for child in level.children.values():
            child_agg = self.project_upward(child)
            for key, value in child_agg.items():
                aggregation[key] = aggregation.get(key, 0) + value
        
        return aggregation















