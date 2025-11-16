"""
Memory Cell: Per-Cell Memory Capsule

Stores working memory and long-term memory for each lattice cell.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class MemoryState(Enum):
    """Memory state types."""
    ACTIVE = "active"      # Currently in working memory
    STORED = "stored"      # In long-term memory
    DECAYING = "decaying"  # Fading from memory
    FORGOTTEN = "forgotten" # Below threshold


@dataclass
class MemoryCapsule:
    """
    Memory capsule for a single cell.
    
    Stores:
    - Working memory (recent states)
    - Long-term memory (important patterns)
    - Memory strength (decay rate)
    - Associations (links to other cells)
    """
    coordinates: Tuple[int, int, int]
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    long_term_memory: Dict[str, Any] = field(default_factory=dict)
    memory_strength: float = 1.0  # Decays over time
    associations: List[Tuple[int, int, int]] = field(default_factory=list)
    access_count: int = 0
    last_accessed: float = field(default_factory=lambda: __import__('time').time())
    
    def add_to_working_memory(self, state: Dict[str, Any], max_size: int = 10):
        """Add state to working memory (FIFO)."""
        self.working_memory.append(state)
        if len(self.working_memory) > max_size:
            self.working_memory.pop(0)
        self.access_count += 1
        self.last_accessed = __import__('time').time()
    
    def consolidate_to_long_term(self, key: str, value: Any, importance: float = 1.0):
        """Consolidate important information to long-term memory."""
        if key not in self.long_term_memory:
            self.long_term_memory[key] = {
                'value': value,
                'importance': importance,
                'timestamp': __import__('time').time(),
                'access_count': 0
            }
        else:
            # Update existing memory (weighted average)
            existing = self.long_term_memory[key]
            existing['value'] = (existing['value'] * existing['importance'] + value * importance) / (existing['importance'] + importance)
            existing['importance'] = min(1.0, existing['importance'] + importance * 0.1)
            existing['access_count'] += 1
    
    def decay_memory(self, decay_rate: float = 0.01):
        """Apply memory decay."""
        self.memory_strength *= (1.0 - decay_rate)
        self.memory_strength = max(0.0, self.memory_strength)
        
        # Decay long-term memory importance
        for key, memory in self.long_term_memory.items():
            memory['importance'] *= (1.0 - decay_rate * 0.1)
            if memory['importance'] < 0.01:
                # Forget if importance too low
                memory['importance'] = 0.0
    
    def get_memory_state(self) -> MemoryState:
        """Get current memory state."""
        if self.memory_strength > 0.7:
            return MemoryState.ACTIVE
        elif self.memory_strength > 0.3:
            return MemoryState.STORED
        elif self.memory_strength > 0.01:
            return MemoryState.DECAYING
        else:
            return MemoryState.FORGOTTEN


class MemoryCell:
    """
    Memory cell wrapper that integrates with lattice cell.
    
    Provides memory operations for a single cell.
    """
    
    def __init__(self, coordinates: Tuple[int, int, int]):
        """Initialize memory cell."""
        self.coordinates = coordinates
        self.capsule = MemoryCapsule(coordinates=coordinates)
    
    def remember(self, state: Dict[str, Any], importance: float = 0.5):
        """Remember a state (adds to working memory, may consolidate)."""
        self.capsule.add_to_working_memory(state)
        
        if importance > 0.7:
            # High importance: consolidate to long-term
            self.capsule.consolidate_to_long_term(
                key=f"state_{len(self.capsule.long_term_memory)}",
                value=state,
                importance=importance
            )
    
    def recall(self, key: Optional[str] = None) -> Optional[Any]:
        """Recall from memory."""
        if key:
            return self.capsule.long_term_memory.get(key)
        else:
            # Return most recent working memory
            return self.capsule.working_memory[-1] if self.capsule.working_memory else None
    
    def associate(self, other_coords: Tuple[int, int, int]):
        """Create association with another cell."""
        if other_coords not in self.capsule.associations:
            self.capsule.associations.append(other_coords)
    
    def get_associations(self) -> List[Tuple[int, int, int]]:
        """Get all associated cells."""
        return self.capsule.associations.copy()
    
    def forget(self, key: Optional[str] = None):
        """Forget memory (clear working or long-term)."""
        if key:
            if key in self.capsule.long_term_memory:
                del self.capsule.long_term_memory[key]
        else:
            self.capsule.working_memory.clear()
    
    def get_memory_summary(self) -> Dict:
        """Get memory summary."""
        return {
            'coordinates': self.coordinates,
            'working_memory_size': len(self.capsule.working_memory),
            'long_term_memory_size': len(self.capsule.long_term_memory),
            'memory_strength': self.capsule.memory_strength,
            'associations': len(self.capsule.associations),
            'state': self.capsule.get_memory_state().value,
        }

