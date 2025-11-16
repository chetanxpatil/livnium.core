"""
Dual Cube System: Positive and Anti-Semantic Spaces

This implements the "−3×−3×−3" as an anti-cube concept:
- +3×+3×+3 cube = positive semantic space (stable meanings, attractors)
- −3×−3×−3 cube = anti-semantic space (contradictions, conflicts, decohered states)

This is NOT just a sign flip - it's two linked lattices with cross-cube dynamics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from quantum.hierarchical.geometry.level0.base_geometry import BaseGeometry, BaseGeometricState


@dataclass
class DualState:
    """
    A state that can exist in both positive and negative cubes.
    """
    positive_state: Optional[BaseGeometricState] = None
    negative_state: Optional[BaseGeometricState] = None
    energy: float = 0.0
    contradiction_score: float = 0.0
    
    def get_total_amplitude(self) -> complex:
        """Get combined amplitude from both cubes."""
        pos_amp = self.positive_state.amplitude if self.positive_state else 0.0
        neg_amp = self.negative_state.amplitude if self.negative_state else 0.0
        return pos_amp - neg_amp  # Negative cancels positive
    
    def is_balanced(self) -> bool:
        """Check if positive and negative cancel out."""
        if not self.positive_state or not self.negative_state:
            return False
        pos_amp = abs(self.positive_state.amplitude)
        neg_amp = abs(self.negative_state.amplitude)
        return abs(pos_amp - neg_amp) < 0.01  # Nearly balanced


class DualCubeSystem:
    """
    Dual cube system: positive semantic space + anti-semantic space.
    
    This implements the "−3×−3×−3" as an anti-cube with dual semantics:
    - Positive cube: stable meanings, attractors, verified patterns
    - Negative cube: contradictions, conflicts, decohered states, "what this is not"
    """
    
    def __init__(self, base_dimension: int = 3, num_levels: int = 3):
        """
        Initialize dual cube system.
        
        Args:
            base_dimension: Dimension of each cube (3 = 3×3×3 and −3×−3×−3)
            num_levels: Number of hierarchy levels
        """
        self.base_dimension = base_dimension
        self.num_levels = num_levels
        
        # Positive cube: stable semantic space
        self.positive_cube = BaseGeometry(dimension=base_dimension)
        
        # Negative cube: anti-semantic space (contradictions, conflicts)
        self.negative_cube = BaseGeometry(dimension=base_dimension)
        
        # Track dual states (states that exist in both cubes)
        self.dual_states: List[DualState] = []
        
        # Cross-cube dynamics parameters
        self.contradiction_threshold = 0.5  # When to move to negative cube
        self.decoherence_rate = 0.1  # Rate of drift from + to −
        self.cancellation_threshold = 0.01  # When states cancel out
    
    def add_positive_state(self, coordinates: Tuple[float, ...], 
                          amplitude: complex = 1.0+0j, phase: float = 0.0) -> BaseGeometricState:
        """Add state to positive cube (stable semantic space)."""
        return self.positive_cube.add_state(coordinates, amplitude, phase)
    
    def add_negative_state(self, coordinates: Tuple[float, ...], 
                          amplitude: complex = 1.0+0j, phase: float = 0.0) -> BaseGeometricState:
        """Add state to negative cube (anti-semantic space)."""
        return self.negative_cube.add_state(coordinates, amplitude, phase)
    
    def add_dual_state(self, coordinates: Tuple[float, ...],
                      positive_amplitude: complex = 1.0+0j,
                      negative_amplitude: complex = 0.0+0j) -> DualState:
        """
        Add state that exists in both cubes (dual semantic state).
        
        Args:
            coordinates: Geometric coordinates
            positive_amplitude: Amplitude in positive cube
            negative_amplitude: Amplitude in negative cube
        """
        pos_state = self.positive_cube.add_state(coordinates, positive_amplitude, 0.0)
        neg_state = self.negative_cube.add_state(coordinates, negative_amplitude, 0.0)
        
        dual = DualState(
            positive_state=pos_state,
            negative_state=neg_state,
            energy=abs(positive_amplitude) - abs(negative_amplitude),
            contradiction_score=abs(negative_amplitude)
        )
        
        self.dual_states.append(dual)
        return dual
    
    def detect_contradiction(self, state: BaseGeometricState, 
                           context: List[BaseGeometricState]) -> float:
        """
        Detect contradiction score for a state.
        
        Higher score = more contradictory / conflicting.
        
        Args:
            state: State to check
            context: Other states in the system
            
        Returns:
            Contradiction score (0.0 = no contradiction, 1.0 = maximum contradiction)
        """
        # Simple contradiction detection: check for conflicting coordinates/amplitudes
        contradiction = 0.0
        
        for other in context:
            # Check if states are close but have opposite amplitudes
            coords_diff = np.linalg.norm(
                np.array(state.coordinates) - np.array(other.coordinates)
            )
            
            if coords_diff < 0.1:  # Close coordinates
                amp_diff = abs(state.amplitude - other.amplitude)
                if amp_diff > 1.0:  # Conflicting amplitudes
                    contradiction += 0.2
        
        return min(1.0, contradiction)
    
    def move_to_negative_cube(self, state: BaseGeometricState, 
                            contradiction_score: float) -> Optional[BaseGeometricState]:
        """
        Move a contradictory state to the negative cube.
        
        This implements: "When a state hits a contradiction rule → move to −cube"
        
        Args:
            state: State to move
            contradiction_score: How contradictory it is
            
        Returns:
            New state in negative cube, or None if not moved
        """
        if contradiction_score < self.contradiction_threshold:
            return None
        
        # Create corresponding state in negative cube
        neg_state = self.negative_cube.add_state(
            state.coordinates,
            amplitude=state.amplitude * contradiction_score,  # Scale by contradiction
            phase=state.phase + np.pi  # Phase shift (opposite)
        )
        
        return neg_state
    
    def apply_decoherence_drift(self, rate: float = None):
        """
        Apply semantic decoherence as cross-cube drift.
        
        As operations overload states, gradually migrate amplitude from +cube → −cube.
        Decoherence = "meaning drains into the anti-cube"
        
        Args:
            rate: Decoherence rate (defaults to self.decoherence_rate)
        """
        if rate is None:
            rate = self.decoherence_rate
        
        # For each positive state, drift some amplitude to negative
        for pos_state in self.positive_cube.states:
            # Calculate drift amount
            drift = abs(pos_state.amplitude) * rate
            
            if drift > 0.01:  # Only drift if significant
                # Reduce positive amplitude
                pos_state.amplitude *= (1.0 - rate)
                
                # Add to negative cube
                self.negative_cube.add_state(
                    pos_state.coordinates,
                    amplitude=drift + 0j,
                    phase=pos_state.phase + np.pi  # Opposite phase
                )
    
    def apply_cancellation(self):
        """
        Apply cancellation: if pattern appears in both cubes with opposite sign, cancel.
        
        Implements: "If a pattern appears in both cubes with opposite sign,
        they cancel → neutral / forgotten"
        """
        to_remove_positive = []
        to_remove_negative = []
        
        for pos_state in self.positive_cube.states:
            # Find corresponding negative state
            for neg_state in self.negative_cube.states:
                coords_match = np.allclose(
                    pos_state.coordinates,
                    neg_state.coordinates,
                    atol=0.01
                )
                
                if coords_match:
                    # Check if they cancel
                    pos_amp = abs(pos_state.amplitude)
                    neg_amp = abs(neg_state.amplitude)
                    
                    if abs(pos_amp - neg_amp) < self.cancellation_threshold:
                        # They cancel - mark for removal
                        to_remove_positive.append(pos_state)
                        to_remove_negative.append(neg_state)
        
        # Remove cancelled states
        for state in to_remove_positive:
            if state in self.positive_cube.states:
                self.positive_cube.states.remove(state)
        
        for state in to_remove_negative:
            if state in self.negative_cube.states:
                self.negative_cube.states.remove(state)
    
    def get_decoherence_measure(self) -> Dict:
        """
        Measure decoherence: "how much has leaked to −cube?"
        
        Returns:
            Dictionary with decoherence metrics
        """
        pos_energy = sum(abs(s.amplitude) for s in self.positive_cube.states)
        neg_energy = sum(abs(s.amplitude) for s in self.negative_cube.states)
        total_energy = pos_energy + neg_energy
        
        if total_energy == 0:
            return {
                'decoherence_fraction': 0.0,
                'positive_energy': 0.0,
                'negative_energy': 0.0,
                'total_energy': 0.0
            }
        
        decoherence_fraction = neg_energy / total_energy
        
        return {
            'decoherence_fraction': decoherence_fraction,
            'positive_energy': pos_energy,
            'negative_energy': neg_energy,
            'total_energy': total_energy,
            'positive_states': len(self.positive_cube.states),
            'negative_states': len(self.negative_cube.states)
        }
    
    def diagnose_confusion(self, coordinates: Tuple[float, ...]) -> Dict:
        """
        Diagnose confusion: check if input lives mostly in −cube.
        
        Args:
            coordinates: Coordinates to check
            
        Returns:
            Diagnosis dictionary
        """
        # Find states near these coordinates
        pos_states = [
            s for s in self.positive_cube.states
            if np.linalg.norm(np.array(s.coordinates) - np.array(coordinates)) < 0.1
        ]
        neg_states = [
            s for s in self.negative_cube.states
            if np.linalg.norm(np.array(s.coordinates) - np.array(coordinates)) < 0.1
        ]
        
        pos_energy = sum(abs(s.amplitude) for s in pos_states)
        neg_energy = sum(abs(s.amplitude) for s in neg_states)
        total_energy = pos_energy + neg_energy
        
        if total_energy == 0:
            confusion_score = 0.0
        else:
            confusion_score = neg_energy / total_energy
        
        return {
            'coordinates': coordinates,
            'confusion_score': confusion_score,
            'positive_energy': pos_energy,
            'negative_energy': neg_energy,
            'diagnosis': 'confused' if confusion_score > 0.5 else 'clear'
        }
    
    def get_system_summary(self) -> Dict:
        """Get summary of dual cube system."""
        decoherence = self.get_decoherence_measure()
        
        return {
            'base_dimension': self.base_dimension,
            'num_levels': self.num_levels,
            'positive_cube': {
                'states': len(self.positive_cube.states),
                'energy': decoherence['positive_energy']
            },
            'negative_cube': {
                'states': len(self.negative_cube.states),
                'energy': decoherence['negative_energy']
            },
            'dual_states': len(self.dual_states),
            'decoherence_fraction': decoherence['decoherence_fraction'],
            'interpretation': {
                'positive': 'Stable meanings, attractors, verified patterns',
                'negative': 'Contradictions, conflicts, decohered states, "what this is not"'
            }
        }

