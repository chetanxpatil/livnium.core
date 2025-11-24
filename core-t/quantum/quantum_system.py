"""
Quantum System: Quantum Layer for Livnium-T

Integrates quantum states, gates, entanglement, and measurement with the 5-node topology.
"""

import numpy as np
from typing import Dict, List, Optional

from .quantum_node import QuantumNode
from .quantum_gates import QuantumGates, GateType
from .entanglement_manager import EntanglementManager, EntangledPair
from .measurement_engine import MeasurementEngine, MeasurementResult, MeasurementBasis
from .geometry_quantum_coupling import GeometryQuantumCoupling
from classical.livnium_t_system import LivniumTSystem


class QuantumSystem:
    """
    Quantum layer for Livnium-T System.
    
    Manages quantum states, gates, entanglement, and measurement for the 5-node topology.
    """
    
    def __init__(self, t_system: LivniumTSystem, 
                 enable_entanglement: bool = True,
                 enable_geometry_coupling: bool = True):
        """
        Initialize quantum system.
        
        Args:
            t_system: Livnium-T System instance
            enable_entanglement: Enable entanglement features
            enable_geometry_coupling: Enable geometry-quantum coupling
        """
        self.t_system = t_system
        
        # Initialize quantum nodes (one per geometric node)
        self.quantum_nodes: Dict[int, QuantumNode] = {}
        self._initialize_quantum_nodes()
        
        # Entanglement manager
        if enable_entanglement:
            self.entanglement_manager = EntanglementManager()
        else:
            self.entanglement_manager = None
        
        # Measurement engine
        self.measurement_engine = MeasurementEngine()
        
        # Geometry-quantum coupling
        if enable_geometry_coupling:
            self.coupling = GeometryQuantumCoupling(t_system)
            self._sync_with_geometry()
        else:
            self.coupling = None
    
    def _initialize_quantum_nodes(self):
        """Initialize quantum nodes for all 5 nodes."""
        for node_id in range(5):
            # Default: |0⟩ state
            quantum_node = QuantumNode(
                node_id=node_id,
                amplitudes=np.array([1.0, 0.0], dtype=complex),
                num_levels=2
            )
            self.quantum_nodes[node_id] = quantum_node
    
    def _sync_with_geometry(self):
        """Sync quantum states with geometric properties."""
        if self.coupling:
            self.coupling.update_quantum_from_geometry(self.quantum_nodes)
    
    def apply_gate(self, node_id: int, gate_type: GateType, **params):
        """
        Apply quantum gate to a node.
        
        Args:
            node_id: Node ID (0-4)
            gate_type: Type of gate
            **params: Gate parameters
        """
        if node_id not in self.quantum_nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.quantum_nodes[node_id]
        gate = QuantumGates.get_gate(gate_type, **params)
        node.apply_unitary(gate)
    
    def apply_gate_to_all(self, gate_type: GateType, **params):
        """Apply gate to all nodes."""
        for node_id in range(5):
            self.apply_gate(node_id, gate_type, **params)
    
    def entangle_nodes(self, node1: int, node2: int, bell_type: str = "phi_plus"):
        """
        Create Bell pair between two nodes.
        
        Args:
            node1: First node ID (0-4)
            node2: Second node ID (0-4)
            bell_type: Type of Bell state
        """
        if not self.entanglement_manager:
            raise ValueError("Entanglement not enabled")
        
        self.entanglement_manager.create_bell_pair(node1, node2, bell_type)
    
    def measure_node(self, node_id: int, 
                    basis: MeasurementBasis = MeasurementBasis.COMPUTATIONAL) -> MeasurementResult:
        """
        Measure a quantum node.
        
        Args:
            node_id: Node ID to measure
            basis: Measurement basis
            
        Returns:
            MeasurementResult
        """
        if node_id not in self.quantum_nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.quantum_nodes[node_id]
        result = self.measurement_engine.measure(node.amplitudes, node_id, basis)
        
        # Collapse node state
        node.amplitudes = result.collapsed_state.copy()
        
        return result
    
    def get_node_state(self, node_id: int) -> QuantumNode:
        """
        Get quantum node state.
        
        Args:
            node_id: Node ID
            
        Returns:
            QuantumNode
        """
        if node_id not in self.quantum_nodes:
            raise ValueError(f"Node {node_id} not found")
        return self.quantum_nodes[node_id]
    
    def get_all_states(self) -> Dict[int, QuantumNode]:
        """Get all quantum node states."""
        return self.quantum_nodes.copy()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QuantumSystem("
            f"nodes={len(self.quantum_nodes)}, "
            f"entanglement={self.entanglement_manager is not None}, "
            f"coupling={self.coupling is not None}"
            f")"
        )

