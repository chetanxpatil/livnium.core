"""
Geometry ↔ Quantum Coupling: Livnium-T Specific Integration

Maps geometric properties (exposure f, symbolic weight SW) to quantum state.
Adapted for the 5-node simplex topology.
"""

import numpy as np
from typing import Dict, Optional

from .quantum_node import QuantumNode
from .quantum_gates import QuantumGates
from classical.livnium_t_system import LivniumTSystem, NodeClass


class GeometryQuantumCoupling:
    """
    Couples geometric properties to quantum state for Livnium-T.
    
    Rules:
    - Exposure f → superposition strength
    - Symbolic Weight SW → amplitude magnitude
    - Node class (Core/Vertex) → initial state
    - Om observer → measurement basis
    """
    
    def __init__(self, t_system: LivniumTSystem):
        """
        Initialize geometry-quantum coupling.
        
        Args:
            t_system: Livnium-T System instance
        """
        self.t_system = t_system
    
    def initialize_quantum_state_from_geometry(self, 
                                              node: QuantumNode,
                                              geometric_node) -> QuantumNode:
        """
        Initialize quantum state based on geometric properties.
        
        Rules:
        - Core (f=0, SW=0) → |0⟩ state (ground)
        - Vertex (f=3, SW=27) → superposition |+⟩ = (|0⟩ + |1⟩)/√2
        
        Args:
            node: Quantum node to initialize
            geometric_node: SimplexNode from T system
            
        Returns:
            Initialized quantum node
        """
        if geometric_node.node_class == NodeClass.CORE:
            # Core: ground state |0⟩
            node.amplitudes = np.array([1.0, 0.0], dtype=complex)
        elif geometric_node.node_class == NodeClass.VERTEX:
            # Vertex: superposition |+⟩ = (|0⟩ + |1⟩)/√2
            node.amplitudes = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        else:
            raise ValueError(f"Unknown node class: {geometric_node.node_class}")
        
        node.normalize()
        return node
    
    def exposure_to_entanglement_strength(self, exposure: int) -> float:
        """
        Map exposure to entanglement strength.
        
        Rule: Higher exposure → stronger entanglement
        - Core (f=0) → 0.0 (no entanglement)
        - Vertex (f=3) → 1.0 (maximum entanglement)
        
        Args:
            exposure: Exposure value (0 or 3)
            
        Returns:
            Entanglement strength [0, 1]
        """
        if exposure == 0:
            return 0.0
        elif exposure == 3:
            return 1.0
        else:
            # Should not happen in Livnium-T, but handle gracefully
            return exposure / 3.0
    
    def symbolic_weight_to_amplitude_modulation(self, sw: float) -> float:
        """
        Map symbolic weight to amplitude modulation factor.
        
        Rule: Higher SW → stronger amplitudes
        - Core (SW=0) → 0.0
        - Vertex (SW=27) → 1.0
        
        Args:
            sw: Symbolic weight value
            
        Returns:
            Amplitude modulation factor [0, 1]
        """
        # Normalize SW (max = 27 for vertices)
        return min(1.0, sw / 27.0)
    
    def update_quantum_from_geometry(self, quantum_nodes: Dict[int, QuantumNode]):
        """
        Update all quantum nodes based on current geometry.
        
        Args:
            quantum_nodes: Dictionary of quantum nodes by node_id
        """
        for node_id, quantum_node in quantum_nodes.items():
            geometric_node = self.t_system.get_node(node_id)
            self.initialize_quantum_state_from_geometry(quantum_node, geometric_node)

