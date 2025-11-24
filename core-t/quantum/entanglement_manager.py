"""
Entanglement Manager: Multi-Node Quantum Correlations for Livnium-T

Handles entanglement between nodes using simplex structure.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EntangledPair:
    """
    Entangled pair of nodes.
    
    For 2-qubit entanglement: |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
    """
    node1: int  # Node ID (0-4)
    node2: int  # Node ID (0-4)
    state_vector: np.ndarray  # 4-element vector for 2-qubit
    entanglement_strength: float = 1.0  # 0 = separable, 1 = maximally entangled
    
    def __post_init__(self):
        """Normalize entangled state."""
        if self.node1 == self.node2:
            raise ValueError("Cannot entangle a node with itself")
        if self.node1 < 0 or self.node1 > 4 or self.node2 < 0 or self.node2 > 4:
            raise ValueError(f"Node IDs must be in [0, 4], got {self.node1}, {self.node2}")
        
        norm = np.sqrt(np.sum(np.abs(self.state_vector) ** 2))
        if norm > 1e-10:
            self.state_vector /= norm
        else:
            # Default: Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            self.state_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    
    def get_concurrence(self) -> float:
        """
        Calculate concurrence (entanglement measure).
        
        Returns:
            Concurrence value [0, 1]
        """
        # For Bell states: concurrence = 1
        bell_plus = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        bell_minus = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], dtype=complex)
        
        if np.allclose(self.state_vector, bell_plus) or np.allclose(self.state_vector, bell_minus):
            return 1.0
        
        # For separable states: concurrence = 0
        nonzero = np.nonzero(np.abs(self.state_vector))[0]
        if len(nonzero) == 1:
            return 0.0
        
        # Approximate: use off-diagonal terms
        off_diag = np.abs(self.state_vector[1]) + np.abs(self.state_vector[2])
        return min(1.0, off_diag * 2)
    
    def is_maximally_entangled(self) -> bool:
        """Check if pair is maximally entangled (Bell state)."""
        return self.get_concurrence() > 0.99


class EntanglementManager:
    """
    Manages entanglement between nodes in Livnium-T.
    
    Uses simplex structure to determine entanglement topology.
    """
    
    def __init__(self):
        """Initialize entanglement manager for 5-node topology."""
        self.entangled_pairs: Dict[Tuple[int, int], EntangledPair] = {}
        self.entanglement_graph: Dict[int, Set[int]] = defaultdict(set)
    
    def create_bell_pair(self, node1: int, node2: int,
                        bell_type: str = "phi_plus") -> EntangledPair:
        """
        Create Bell pair between two nodes.
        
        Args:
            node1: First node ID (0-4)
            node2: Second node ID (0-4)
            bell_type: Type of Bell state ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
            
        Returns:
            EntangledPair
        """
        if node1 == node2:
            raise ValueError("Cannot entangle a node with itself")
        
        # Create Bell state vector
        if bell_type == "phi_plus":
            state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        elif bell_type == "phi_minus":
            state = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], dtype=complex)
        elif bell_type == "psi_plus":
            state = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
        elif bell_type == "psi_minus":
            state = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
        else:
            raise ValueError(f"Unknown Bell type: {bell_type}")
        
        pair = EntangledPair(node1, node2, state, entanglement_strength=1.0)
        
        # Store pair (order-independent)
        key = tuple(sorted([node1, node2]))
        self.entangled_pairs[key] = pair
        
        # Update graph
        self.entanglement_graph[node1].add(node2)
        self.entanglement_graph[node2].add(node1)
        
        return pair
    
    def get_entangled_nodes(self, node_id: int) -> Set[int]:
        """
        Get all nodes entangled with given node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Set of entangled node IDs
        """
        return self.entanglement_graph.get(node_id, set())
    
    def is_entangled(self, node1: int, node2: int) -> bool:
        """
        Check if two nodes are entangled.
        
        Args:
            node1: First node ID
            node2: Second node ID
            
        Returns:
            True if entangled
        """
        key = tuple(sorted([node1, node2]))
        return key in self.entangled_pairs
    
    def remove_entanglement(self, node1: int, node2: int):
        """
        Remove entanglement between two nodes.
        
        Args:
            node1: First node ID
            node2: Second node ID
        """
        key = tuple(sorted([node1, node2]))
        if key in self.entangled_pairs:
            del self.entangled_pairs[key]
            self.entanglement_graph[node1].discard(node2)
            self.entanglement_graph[node2].discard(node1)
    
    def get_all_pairs(self) -> List[EntangledPair]:
        """Get all entangled pairs."""
        return list(self.entangled_pairs.values())

