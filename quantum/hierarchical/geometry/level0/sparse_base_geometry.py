"""
Level 0: Sparse Base Geometry

Enhanced base geometry that uses sparse storage for efficient quantum state representation.
This is geometry > geometry > geometry at the foundation level.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Set
from dataclasses import dataclass


@dataclass
class SparseBaseGeometricState:
    """
    Sparse base geometric state - only stores non-zero amplitudes.
    
    This is the foundation for efficient quantum simulation in geometry.
    """
    coordinates: Tuple[float, ...]  # Geometric coordinates
    amplitude: complex  # Quantum amplitude (only non-zero states stored)
    phase: float  # Phase angle
    
    def __post_init__(self):
        """Initialize sparse geometric state."""
        if self.amplitude is None:
            self.amplitude = 0.0 + 0j


class SparseBaseGeometry:
    """
    Sparse base geometry system - Level 0 with optimization.
    
    Only stores states with non-zero amplitudes, making it efficient
    for large quantum systems.
    """
    
    def __init__(self, dimension: int = 3, threshold: float = 1e-15):
        """
        Initialize sparse base geometry.
        
        Args:
            dimension: Geometric dimension
            threshold: Minimum amplitude to store (below this = zero)
        """
        self.dimension = dimension
        self.threshold = threshold
        
        # Sparse storage: only non-zero states
        # Map: coordinates -> state
        self.states: Dict[Tuple[float, ...], SparseBaseGeometricState] = {}
        
        # Track active coordinates (for fast iteration)
        self.active_coordinates: Set[Tuple[float, ...]] = set()
    
    def add_state(self, coordinates: Tuple[float, ...], 
                  amplitude: complex = 1.0+0j, phase: float = 0.0) -> Optional[SparseBaseGeometricState]:
        """
        Add a quantum state - only if amplitude is significant.
        
        Args:
            coordinates: Geometric coordinates
            amplitude: Quantum amplitude
            phase: Phase angle
            
        Returns:
            Created state if stored, None if below threshold
        """
        # Only store if amplitude is significant
        if abs(amplitude) < self.threshold:
            # Remove if exists
            if coordinates in self.states:
                del self.states[coordinates]
                self.active_coordinates.discard(coordinates)
            return None
        
        state = SparseBaseGeometricState(
            coordinates=coordinates,
            amplitude=amplitude,
            phase=phase
        )
        
        self.states[coordinates] = state
        self.active_coordinates.add(coordinates)
        
        return state
    
    def get_state(self, coordinates: Tuple[float, ...]) -> Optional[SparseBaseGeometricState]:
        """Get state by coordinates."""
        return self.states.get(coordinates)
    
    def get_amplitude(self, coordinates: Tuple[float, ...]) -> complex:
        """Get amplitude of state (returns 0 if not stored)."""
        state = self.states.get(coordinates)
        if state:
            return state.amplitude * np.exp(1j * state.phase)
        return 0.0 + 0j
    
    def set_amplitude(self, coordinates: Tuple[float, ...], amplitude: complex):
        """Set amplitude of state (automatically handles sparse storage)."""
        amp_magnitude = abs(amplitude)
        phase = np.angle(amplitude)
        
        if amp_magnitude < self.threshold:
            # Remove if exists
            if coordinates in self.states:
                del self.states[coordinates]
                self.active_coordinates.discard(coordinates)
        else:
            # Add or update
            if coordinates in self.states:
                state = self.states[coordinates]
                state.amplitude = amp_magnitude
                state.phase = phase
            else:
                self.add_state(coordinates, amplitude, phase)
    
    def get_all_states(self) -> List[SparseBaseGeometricState]:
        """Get all active states."""
        return [self.states[coords] for coords in self.active_coordinates]
    
    def get_geometry_structure(self) -> Dict:
        """Get the geometric structure."""
        return {
            'level': 0,
            'dimension': self.dimension,
            'num_states': len(self.states),
            'num_active_states': len(self.active_coordinates),
            'type': 'sparse_base_geometry',
            'description': 'Sparse geometric structure (only non-zero states)',
            'threshold': self.threshold,
            'efficiency': f"{len(self.states) / (2 ** self.dimension) * 100:.2f}%" if self.dimension <= 20 else "sparse"
        }

