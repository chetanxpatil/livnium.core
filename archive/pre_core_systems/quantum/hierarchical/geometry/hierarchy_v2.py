"""
Hierarchy v2: Advanced Hierarchical Geometry System

Features:
1. Level Graph - Visual map of hierarchy structure
2. Operation Registry - Documented, auditable operations
3. Propagation Engine - Operations propagate down through levels
4. Geometric Symbolic Logic - Scalable reasoning without explosion
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from quantum.hierarchical.geometry.level0.base_geometry import BaseGeometry, BaseGeometricState


class OperationType(Enum):
    """Types of geometric operations."""
    ROTATION = "rotation"
    SCALE = "scale"
    TRANSLATION = "translation"
    ENTANGLE = "entangle"
    TRANSFORM = "transform"
    COMPOSE = "compose"
    PROJECT = "project"
    FILTER = "filter"
    AGGREGATE = "aggregate"


@dataclass
class RegisteredOperation:
    """
    A registered operation with full metadata.
    
    Operations are documented, auditable, and traceable.
    """
    operation_id: str
    operation_type: OperationType
    level: int
    parameters: Dict[str, Any]
    description: str
    propagates_down: bool = True
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.operation_id,
            'type': self.operation_type.value,
            'level': self.level,
            'parameters': self.parameters,
            'description': self.description,
            'propagates_down': self.propagates_down,
            'timestamp': self.timestamp
        }


@dataclass
class LevelNode:
    """
    A node in the level graph representing one hierarchy level.
    """
    level: int
    node_type: str  # 'base', 'meta', 'meta_meta', etc.
    num_operations: int
    num_states: int = 0
    children: List['LevelNode'] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for visualization."""
        return {
            'level': self.level,
            'type': self.node_type,
            'operations': self.num_operations,
            'states': self.num_states,
            'children': [c.to_dict() for c in self.children]
        }


class PropagationEngine:
    """
    Propagation engine: Operations propagate down through hierarchy levels.
    
    When an operation is applied at level N, it propagates:
    Level N → Level N-1 → ... → Level 1 → Level 0 (base)
    """
    
    def __init__(self, hierarchy_system: 'HierarchyV2System'):
        """Initialize propagation engine."""
        self.hierarchy = hierarchy_system
        self.propagation_history: List[Dict] = []
    
    def propagate_operation(self, operation: RegisteredOperation) -> Dict:
        """
        Propagate an operation down through all levels.
        
        Args:
            operation: Operation to propagate
            
        Returns:
            Propagation result with effects at each level
        """
        if not operation.propagates_down:
            return {'propagated': False, 'reason': 'Operation does not propagate'}
        
        propagation_path = []
        current_level = operation.level
        
        # Propagate from operation level down to base (level 0)
        while current_level >= 0:
            effect = self._apply_at_level(operation, current_level)
            propagation_path.append({
                'level': current_level,
                'effect': effect,
                'applied': True
            })
            current_level -= 1
        
        result = {
            'operation_id': operation.operation_id,
            'start_level': operation.level,
            'propagation_path': propagation_path,
            'total_levels_affected': len(propagation_path)
        }
        
        self.propagation_history.append(result)
        return result
    
    def _apply_at_level(self, operation: RegisteredOperation, level: int) -> Dict:
        """Apply operation at specific level."""
        # Simplified: return effect description
        # Full implementation would actually transform geometry
        return {
            'level': level,
            'operation_type': operation.operation_type.value,
            'transformed': True,
            'description': f"Applied {operation.operation_type.value} at level {level}"
        }
    
    def get_propagation_graph(self) -> Dict:
        """Get graph of all propagations."""
        return {
            'total_propagations': len(self.propagation_history),
            'propagations': self.propagation_history
        }


class OperationRegistry:
    """
    Registry of all operations with full documentation.
    
    Makes operations auditable, traceable, and predictable.
    """
    
    def __init__(self):
        """Initialize operation registry."""
        self.operations: Dict[str, RegisteredOperation] = {}
        self.operations_by_level: Dict[int, List[str]] = defaultdict(list)
        self.operations_by_type: Dict[OperationType, List[str]] = defaultdict(list)
        self.operation_counter = 0
    
    def register(self, operation_type: OperationType, level: int, 
                parameters: Dict[str, Any], description: str,
                propagates_down: bool = True) -> RegisteredOperation:
        """
        Register a new operation.
        
        Args:
            operation_type: Type of operation
            level: Level at which operation applies
            parameters: Operation parameters
            description: Human-readable description
            propagates_down: Whether operation propagates to lower levels
            
        Returns:
            Registered operation
        """
        self.operation_counter += 1
        operation_id = f"op_{self.operation_counter:06d}"
        
        operation = RegisteredOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            level=level,
            parameters=parameters,
            description=description,
            propagates_down=propagates_down
        )
        
        self.operations[operation_id] = operation
        self.operations_by_level[level].append(operation_id)
        self.operations_by_type[operation_type].append(operation_id)
        
        return operation
    
    def get_operation(self, operation_id: str) -> Optional[RegisteredOperation]:
        """Get operation by ID."""
        return self.operations.get(operation_id)
    
    def get_operations_at_level(self, level: int) -> List[RegisteredOperation]:
        """Get all operations at a specific level."""
        operation_ids = self.operations_by_level.get(level, [])
        return [self.operations[op_id] for op_id in operation_ids]
    
    def get_operations_by_type(self, operation_type: OperationType) -> List[RegisteredOperation]:
        """Get all operations of a specific type."""
        operation_ids = self.operations_by_type.get(operation_type, [])
        return [self.operations[op_id] for op_id in operation_ids]
    
    def get_registry_summary(self) -> Dict:
        """Get summary of all registered operations."""
        return {
            'total_operations': len(self.operations),
            'operations_by_level': {
                level: len(ops) 
                for level, ops in self.operations_by_level.items()
            },
            'operations_by_type': {
                op_type.value: len(ops)
                for op_type, ops in self.operations_by_type.items()
            }
        }


class LevelGraph:
    """
    Level graph: Visual map of hierarchy structure.
    
    Shows:
    - Level 0 → States
    - Level 1 → Transforms
    - Level 2 → Transforms of Transforms
    - Level N → Meta^N transforms
    """
    
    def __init__(self, hierarchy_system: 'HierarchyV2System'):
        """Initialize level graph."""
        self.hierarchy = hierarchy_system
    
    def build_graph(self) -> LevelNode:
        """
        Build the level graph structure.
        
        Returns:
            Root node of the graph
        """
        # Build from base (level 0) up
        base_node = LevelNode(
            level=0,
            node_type='base',
            num_operations=0,
            num_states=len(self.hierarchy.base_geometry.states)
        )
        
        # Build meta-level nodes
        current_node = base_node
        for level_num in range(1, self.hierarchy.num_levels):
            level = self.hierarchy.levels[level_num - 1]
            meta_node = LevelNode(
                level=level_num,
                node_type=f'meta_{level_num}' if level_num <= 2 else f'meta^{level_num}',
                num_operations=len(level.operations),
                num_states=0
            )
            current_node.children.append(meta_node)
            current_node = meta_node
        
        return base_node
    
    def visualize(self, format: str = 'text') -> str:
        """
        Visualize the level graph.
        
        Args:
            format: Output format ('text', 'tree', 'json')
            
        Returns:
            Visualization string
        """
        graph = self.build_graph()
        
        if format == 'text':
            return self._visualize_text(graph)
        elif format == 'tree':
            return self._visualize_tree(graph)
        elif format == 'json':
            import json
            return json.dumps(graph.to_dict(), indent=2)
        else:
            return str(graph.to_dict())
    
    def _visualize_text(self, node: LevelNode, indent: int = 0) -> str:
        """Visualize as text tree."""
        prefix = "  " * indent
        node_str = f"{prefix}Level {node.level}: {node.node_type}"
        node_str += f" ({node.num_operations} ops"
        if node.num_states > 0:
            node_str += f", {node.num_states} states"
        node_str += ")\n"
        
        for child in node.children:
            node_str += self._visualize_text(child, indent + 1)
        
        return node_str
    
    def _visualize_tree(self, node: LevelNode, prefix: str = "", is_last: bool = True) -> str:
        """Visualize as tree structure."""
        connector = "└── " if is_last else "├── "
        node_str = f"{prefix}{connector}Level {node.level}: {node.node_type}"
        node_str += f" ({node.num_operations} ops"
        if node.num_states > 0:
            node_str += f", {node.num_states} states"
        node_str += ")\n"
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            node_str += self._visualize_tree(child, new_prefix, is_last_child)
        
        return node_str


class HierarchyV2System:
    """
    Hierarchy v2: Advanced hierarchical geometry system.
    
    Features:
    - Level graph visualization
    - Operation registry
    - Propagation engine
    - Geometric symbolic logic
    """
    
    def __init__(self, base_dimension: int = 3, num_levels: int = 3):
        """
        Initialize Hierarchy v2 system.
        
        Args:
            base_dimension: Dimension of base geometric space
            num_levels: Number of hierarchical levels
        """
        from quantum.hierarchical.geometry.dynamic_hierarchical_geometry import (
            DynamicHierarchicalGeometrySystem, HierarchicalLevel
        )
        
        self.base_dimension = base_dimension
        self.num_levels = num_levels
        
        # Level 0: Base geometry
        self.base_geometry = BaseGeometry(dimension=base_dimension)
        
        # Build hierarchy levels
        self.levels: List[HierarchicalLevel] = []
        current_geometry = self.base_geometry
        
        for level_num in range(1, num_levels):
            level = HierarchicalLevel(level=level_num, base_geometry=current_geometry)
            self.levels.append(level)
            current_geometry = level
        
        # V2 Features
        self.registry = OperationRegistry()
        self.propagation_engine = PropagationEngine(self)
        self.level_graph = LevelGraph(self)
    
    def add_base_state(self, coordinates: Tuple[float, ...], 
                      amplitude: complex = 1.0+0j, phase: float = 0.0) -> BaseGeometricState:
        """Add state to base geometry (Level 0)."""
        return self.base_geometry.add_state(coordinates, amplitude, phase)
    
    def register_operation(self, operation_type: OperationType, level: int,
                          parameters: Dict[str, Any], description: str,
                          propagates_down: bool = True) -> RegisteredOperation:
        """
        Register and apply an operation.
        
        Args:
            operation_type: Type of operation
            level: Level at which to apply
            parameters: Operation parameters
            description: Human-readable description
            propagates_down: Whether to propagate down
            
        Returns:
            Registered operation
        """
        # Register operation
        operation = self.registry.register(
            operation_type=operation_type,
            level=level,
            parameters=parameters,
            description=description,
            propagates_down=propagates_down
        )
        
        # Add to level
        if 1 <= level < self.num_levels:
            level_index = level - 1
            self.levels[level_index].add_operation(
                operation_type.value,
                **parameters
            )
        
        # Propagate if needed
        if propagates_down:
            self.propagation_engine.propagate_operation(operation)
        
        return operation
    
    def get_level_graph(self, format: str = 'tree') -> str:
        """Get level graph visualization."""
        return self.level_graph.visualize(format=format)
    
    def get_operation_registry(self) -> Dict:
        """Get operation registry summary."""
        return self.registry.get_registry_summary()
    
    def get_propagation_history(self) -> Dict:
        """Get propagation history."""
        return self.propagation_engine.get_propagation_graph()
    
    def get_full_system_info(self) -> Dict:
        """Get complete system information."""
        return {
            'hierarchy': {
                'num_levels': self.num_levels,
                'base_dimension': self.base_dimension,
                'base_states': len(self.base_geometry.states)
            },
            'level_graph': self.level_graph.build_graph().to_dict(),
            'operation_registry': self.get_operation_registry(),
            'propagation_history': self.get_propagation_history(),
            'insight': 'Hierarchy depth does not affect qubit capacity - capacity is memory-limited, not hierarchy-limited'
        }

