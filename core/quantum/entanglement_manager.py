"""
Entanglement Manager: Multi-Cell Quantum Correlations

Handles entanglement between lattice cells using geometric structure.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EntangledPair:
    """
    DEPRECATED: This class represents "fake" entanglement metadata.
    
    It stores a 4-element vector that acts like a label but does NOT affect
    the actual QuantumCell amplitudes, creating a disconnect.
    
    ⚠️ DO NOT USE FOR TRUE QUANTUM PROTOCOLS ⚠️
    
    For true quantum entanglement, use `TrueQuantumRegister` from
    `true_quantum_layer.py` which implements proper tensor product mechanics.
    
    This class is kept for backward compatibility with geometry-quantum coupling,
    but should not be used for quantum teleportation, Bell tests, or other
    protocols requiring true multi-qubit entanglement.
    
    For 2-qubit entanglement: |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
    """
    cell1: Tuple[int, int, int]
    cell2: Tuple[int, int, int]
    state_vector: np.ndarray  # 4-element vector for 2-qubit
    entanglement_strength: float = 1.0  # 0 = separable, 1 = maximally entangled
    
    def __post_init__(self):
        """Normalize entangled state."""
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
        # Simplified: use state vector structure
        # For Bell states: concurrence = 1
        if np.allclose(self.state_vector, [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]):
            return 1.0
        # For separable states: concurrence = 0
        if len(np.nonzero(np.abs(self.state_vector))[0]) == 1:
            return 0.0
        # Approximate: use off-diagonal terms
        off_diag = np.abs(self.state_vector[1]) + np.abs(self.state_vector[2])
        return min(1.0, off_diag * 2)
    
    def is_maximally_entangled(self) -> bool:
        """Check if pair is maximally entangled (Bell state)."""
        return self.get_concurrence() > 0.99


class EntanglementManager:
    """
    Manages entanglement between lattice cells.
    
    Uses geometric structure to determine entanglement topology.
    """
    
    def __init__(self, lattice_size: int):
        """
        Initialize entanglement manager.
        
        Args:
            lattice_size: Size of lattice (N)
        """
        self.lattice_size = lattice_size
        self.entangled_pairs: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], EntangledPair] = {}
        self.entanglement_graph: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = defaultdict(set)
    
    def create_bell_pair(self, cell1: Tuple[int, int, int], 
                        cell2: Tuple[int, int, int],
                        bell_type: str = "phi_plus") -> EntangledPair:
        """
        DEPRECATED: Creates "fake" Bell pair metadata.
        
        ⚠️ WARNING: This does NOT create true quantum entanglement! ⚠️
        
        This method only creates metadata that tracks entanglement but does NOT
        affect the actual QuantumCell amplitudes. For true quantum protocols
        (teleportation, Bell tests), use `TrueQuantumRegister` from
        `true_quantum_layer.py`.
        
        This method is kept for backward compatibility with geometry-quantum
        coupling, but should not be used for quantum protocols.
        
        Create Bell state between two cells.
        
        Args:
            cell1: First cell coordinates
            cell2: Second cell coordinates
            bell_type: Type of Bell state ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
            
        Returns:
            Entangled pair (metadata only, not true entanglement)
        """
        # Bell states
        bell_states = {
            "phi_plus": np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex),  # (|00⟩ + |11⟩)/√2
            "phi_minus": np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], dtype=complex),  # (|00⟩ - |11⟩)/√2
            "psi_plus": np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex),  # (|01⟩ + |10⟩)/√2
            "psi_minus": np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex),  # (|01⟩ - |10⟩)/√2
        }
        
        state = bell_states.get(bell_type, bell_states["phi_plus"])
        pair = EntangledPair(cell1, cell2, state, entanglement_strength=1.0)
        
        # Store pair (both directions)
        self.entangled_pairs[(cell1, cell2)] = pair
        self.entangled_pairs[(cell2, cell1)] = pair
        self.entanglement_graph[cell1].add(cell2)
        self.entanglement_graph[cell2].add(cell1)
        
        return pair
    
    def entangle_by_distance(self, cell: Tuple[int, int, int], 
                           max_distance: float = 1.5,
                           bell_type: str = "phi_plus") -> List[EntangledPair]:
        """
        Entangle cell with nearby cells (geometric entanglement).
        
        Args:
            cell: Cell coordinates
            max_distance: Maximum distance for entanglement
            bell_type: Type of Bell state
            
        Returns:
            List of created entangled pairs
        """
        pairs = []
        cell_array = np.array(cell)
        
        # Check all cells within distance
        for other_cell in self._get_all_cells():
            if other_cell == cell:
                continue
            
            other_array = np.array(other_cell)
            distance = np.linalg.norm(cell_array - other_array)
            
            if distance <= max_distance:
                # Check if already entangled
                if (cell, other_cell) not in self.entangled_pairs:
                    pair = self.create_bell_pair(cell, other_cell, bell_type)
                    pairs.append(pair)
        
        return pairs
    
    def entangle_by_face_exposure(self, cell: Tuple[int, int, int],
                                 face_exposure: int,
                                 bell_type: str = "phi_plus") -> List[EntangledPair]:
        """
        Entangle cell based on face exposure (Livnium-specific).
        
        Higher face exposure → more entanglement connections.
        
        Args:
            cell: Cell coordinates
            face_exposure: Face exposure value (0-3)
            bell_type: Type of Bell state
            
        Returns:
            List of created entangled pairs
        """
        # Number of entanglements = face_exposure + 1
        num_entanglements = face_exposure + 1
        
        # Find nearest neighbors
        neighbors = self._get_nearest_neighbors(cell)
        
        pairs = []
        for i, neighbor in enumerate(neighbors[:num_entanglements]):
            if (cell, neighbor) not in self.entangled_pairs:
                pair = self.create_bell_pair(cell, neighbor, bell_type)
                pairs.append(pair)
        
        return pairs
    
    def get_entangled_cells(self, cell: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get all cells entangled with given cell."""
        return list(self.entanglement_graph.get(cell, set()))
    
    def is_entangled(self, cell1: Tuple[int, int, int], 
                    cell2: Tuple[int, int, int]) -> bool:
        """Check if two cells are entangled."""
        return (cell1, cell2) in self.entangled_pairs
    
    def get_entangled_pair(self, cell1: Tuple[int, int, int],
                          cell2: Tuple[int, int, int]) -> Optional[EntangledPair]:
        """Get entangled pair between two cells."""
        return self.entangled_pairs.get((cell1, cell2))
    
    def break_entanglement(self, cell1: Tuple[int, int, int],
                          cell2: Tuple[int, int, int]):
        """Break entanglement between two cells."""
        if (cell1, cell2) in self.entangled_pairs:
            del self.entangled_pairs[(cell1, cell2)]
            del self.entangled_pairs[(cell2, cell1)]
            self.entanglement_graph[cell1].discard(cell2)
            self.entanglement_graph[cell2].discard(cell1)
    
    def get_entanglement_statistics(self) -> Dict:
        """Get statistics about entanglement in the system."""
        total_pairs = len(self.entangled_pairs) // 2  # Each pair counted twice
        max_entangled = max(len(neighbors) for neighbors in self.entanglement_graph.values()) if self.entanglement_graph else 0
        
        return {
            'total_entangled_pairs': total_pairs,
            'max_connections_per_cell': max_entangled,
            'entangled_cells': len(self.entanglement_graph),
        }
    
    def _get_all_cells(self) -> List[Tuple[int, int, int]]:
        """Get all cell coordinates in lattice."""
        cells = []
        coord_range = list(range(-(self.lattice_size - 1) // 2, 
                                 (self.lattice_size - 1) // 2 + 1))
        for x in coord_range:
            for y in coord_range:
                for z in coord_range:
                    cells.append((x, y, z))
        return cells
    
    def _get_nearest_neighbors(self, cell: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get nearest neighbor cells (distance = 1)."""
        x, y, z = cell
        neighbors = []
        
        # 6 nearest neighbors (face-adjacent)
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            neighbor = (x + dx, y + dy, z + dz)
            # Check if within lattice bounds
            max_coord = (self.lattice_size - 1) // 2
            if all(abs(c) <= max_coord for c in neighbor):
                neighbors.append(neighbor)
        
        return neighbors

