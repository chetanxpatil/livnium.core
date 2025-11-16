"""
Memory Coupling: Cross-Layer Memory Integration

Couples memory with geometry and quantum layers (the "haircut" mechanism).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .memory_lattice import MemoryLattice
from ..classical.livnium_core_system import LivniumCoreSystem


class MemoryCoupling:
    """
    Couples memory with geometric and quantum properties.
    
    Rules:
    - Face exposure → memory persistence
    - Symbolic Weight → memory importance
    - Quantum state → memory encoding
    - Observer → memory retrieval context
    """
    
    def __init__(self, core_system: LivniumCoreSystem, memory_lattice: MemoryLattice):
        """
        Initialize memory coupling.
        
        Args:
            core_system: Livnium Core System
            memory_lattice: Memory Lattice
        """
        self.core_system = core_system
        self.memory_lattice = memory_lattice
    
    def encode_geometric_state(self, coords: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Encode geometric state into memory format.
        
        Args:
            coords: Cell coordinates
            
        Returns:
            Encoded state dictionary
        """
        geometric_cell = self.core_system.get_cell(coords)
        if not geometric_cell:
            return {}
        
        return {
            'coordinates': coords,
            'face_exposure': geometric_cell.face_exposure,
            'symbolic_weight': geometric_cell.symbolic_weight,
            'cell_class': geometric_cell.cell_class.value if geometric_cell.cell_class else None,
            'symbol': self.core_system.get_symbol(coords),
        }
    
    def encode_quantum_state(self, coords: Tuple[int, int, int],
                            quantum_cell) -> Dict[str, Any]:
        """
        Encode quantum state into memory format.
        
        Args:
            coords: Cell coordinates
            quantum_cell: Quantum cell
            
        Returns:
            Encoded state dictionary
        """
        if not quantum_cell:
            return {}
        
        probs = quantum_cell.get_probabilities()
        return {
            'quantum_state': quantum_cell.get_state_vector().tolist(),
            'probabilities': probs.tolist(),
            'entropy': float(-np.sum(probs * np.log(probs + 1e-10))),
        }
    
    def calculate_memory_importance(self, coords: Tuple[int, int, int]) -> float:
        """
        Calculate memory importance based on geometric properties.
        
        Rule: Higher SW → higher importance
        
        Args:
            coords: Cell coordinates
            
        Returns:
            Importance value [0, 1]
        """
        geometric_cell = self.core_system.get_cell(coords)
        if not geometric_cell:
            return 0.0
        
        # SW normalized to [0, 1] (max SW = 27 for corners)
        sw_normalized = geometric_cell.symbolic_weight / 27.0
        
        # Face exposure adds persistence
        f_factor = geometric_cell.face_exposure / 3.0
        
        # Combined importance
        importance = 0.7 * sw_normalized + 0.3 * f_factor
        return float(np.clip(importance, 0.0, 1.0))
    
    def apply_memory_decay_by_geometry(self):
        """
        Apply memory decay based on geometric properties.
        
        Rule: Lower face exposure → faster decay (core cells forget faster)
        """
        for coords, memory_cell in self.memory_lattice.memory_cells.items():
            geometric_cell = self.core_system.get_cell(coords)
            if geometric_cell:
                # Core cells (f=0) decay faster, corners (f=3) decay slower
                f = geometric_cell.face_exposure
                decay_factor = 1.0 - (f / 3.0) * 0.5  # 0.5x to 1.0x decay rate
                adjusted_decay = self.memory_lattice.decay_rate * decay_factor
                memory_cell.capsule.decay_memory(adjusted_decay)
    
    def sync_memory_with_geometry(self):
        """Sync memory with current geometric state."""
        for coords in self.memory_lattice.memory_cells.keys():
            state = self.encode_geometric_state(coords)
            importance = self.calculate_memory_importance(coords)
            self.memory_lattice.remember(coords, state, importance)
    
    def retrieve_by_context(self, context: Dict[str, Any]) -> List[Tuple[int, int, int]]:
        """
        Retrieve memory cells matching context.
        
        Args:
            context: Context dictionary (e.g., {'cell_class': CellClass.CORNER})
            
        Returns:
            List of matching cell coordinates
        """
        matches = []
        for coords, memory_cell in self.memory_lattice.memory_cells.items():
            # Check if memory matches context
            recent = memory_cell.recall()
            if recent:
                match = True
                for key, value in context.items():
                    if key not in recent or recent[key] != value:
                        match = False
                        break
                if match:
                    matches.append(coords)
        return matches

