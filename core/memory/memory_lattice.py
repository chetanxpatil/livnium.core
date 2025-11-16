"""
Memory Lattice: Global Memory Management

Manages memory across all cells with decay, consolidation, and cross-cell associations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .memory_cell import MemoryCell, MemoryCapsule, MemoryState
from ..classical.livnium_core_system import LivniumCoreSystem


class MemoryLattice:
    """
    Global memory lattice for Livnium Core System.
    
    Features:
    - Per-cell memory capsules
    - Global memory state
    - Memory decay over time
    - Cross-cell associations
    - Memory consolidation
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize memory lattice.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
        self.memory_cells: Dict[Tuple[int, int, int], MemoryCell] = {}
        self._initialize_memory_cells()
        
        # Global memory parameters
        self.decay_rate = 0.01  # Per timestep decay
        self.consolidation_threshold = 0.7  # Importance threshold for consolidation
        self.max_working_memory = 10  # Max items in working memory per cell
    
    def _initialize_memory_cells(self):
        """Initialize memory cells for all lattice cells."""
        for coords in self.core_system.lattice.keys():
            self.memory_cells[coords] = MemoryCell(coords)
    
    def remember(self, coords: Tuple[int, int, int], 
                state: Dict[str, Any], 
                importance: float = 0.5):
        """
        Remember state at cell.
        
        Args:
            coords: Cell coordinates
            state: State to remember
            importance: Importance (0-1)
        """
        if coords in self.memory_cells:
            self.memory_cells[coords].remember(state, importance)
    
    def recall(self, coords: Tuple[int, int, int], 
              key: Optional[str] = None) -> Optional[Any]:
        """Recall from memory at cell."""
        if coords in self.memory_cells:
            return self.memory_cells[coords].recall(key)
        return None
    
    def associate_cells(self, coords1: Tuple[int, int, int],
                       coords2: Tuple[int, int, int]):
        """Create bidirectional association between cells."""
        if coords1 in self.memory_cells and coords2 in self.memory_cells:
            self.memory_cells[coords1].associate(coords2)
            self.memory_cells[coords2].associate(coords1)
    
    def apply_decay(self):
        """Apply memory decay to all cells."""
        for cell in self.memory_cells.values():
            cell.capsule.decay_memory(self.decay_rate)
    
    def consolidate_memories(self):
        """Consolidate important memories across lattice."""
        # Find high-importance memories and strengthen associations
        for coords, cell in self.memory_cells.items():
            for key, memory in cell.capsule.long_term_memory.items():
                if memory['importance'] > self.consolidation_threshold:
                    # Strengthen associations with nearby cells
                    geometric_cell = self.core_system.get_cell(coords)
                    if geometric_cell:
                        # Associate with cells of same class
                        for other_coords, other_cell in self.memory_cells.items():
                            if other_coords != coords:
                                other_geometric = self.core_system.get_cell(other_coords)
                                if other_geometric and other_geometric.cell_class == geometric_cell.cell_class:
                                    self.associate_cells(coords, other_coords)
    
    def get_memory_graph(self) -> Dict:
        """Get memory association graph."""
        graph = {}
        for coords, cell in self.memory_cells.items():
            graph[coords] = cell.get_associations()
        return graph
    
    def get_memory_statistics(self) -> Dict:
        """Get global memory statistics."""
        total_working = sum(len(cell.capsule.working_memory) for cell in self.memory_cells.values())
        total_long_term = sum(len(cell.capsule.long_term_memory) for cell in self.memory_cells.values())
        avg_strength = np.mean([cell.capsule.memory_strength for cell in self.memory_cells.values()])
        
        state_counts = {}
        for cell in self.memory_cells.values():
            state = cell.capsule.get_memory_state()
            state_counts[state.value] = state_counts.get(state.value, 0) + 1
        
        return {
            'total_cells': len(self.memory_cells),
            'total_working_memory': total_working,
            'total_long_term_memory': total_long_term,
            'average_memory_strength': float(avg_strength),
            'memory_states': state_counts,
        }

