"""
Dual Cube System with Trapped φ (Cube⁰)

Extends the dual-cube system with trapped φ support:
- Cube⁺ (+1): Active meaning
- Cube⁻ (-1): Active contradiction
- Cube⁰ (0): Trapped/frozen φ (structural obstacles)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from quantum.hierarchical.geometry.dual_cube_system import DualCubeSystem
from quantum.hierarchical.geometry.trapped_phi import (
    ExtendedGeometricState,
    TrappedPhiSystem
)
from quantum.hierarchical.geometry.level0.base_geometry import BaseGeometry


class DualCubeWithTrappedSystem(DualCubeSystem):
    """
    Dual cube system extended with trapped φ (Cube⁰).
    
    Three-state semantic universe:
    - Cube⁺ (+1): Active meaning (fluid)
    - Cube⁻ (-1): Active contradiction (fluid)
    - Cube⁰ (0): Trapped/frozen φ (solid structure)
    """
    
    def __init__(self, base_dimension: int = 3, num_levels: int = 3,
                 **trapped_params):
        """
        Initialize dual cube system with trapped φ support.
        
        Args:
            base_dimension: Dimension of each cube
            num_levels: Number of hierarchy levels
            **trapped_params: Parameters for TrappedPhiSystem
        """
        super().__init__(base_dimension, num_levels)
        
        # Trapped φ system
        self.trapped_system = TrappedPhiSystem(**trapped_params)
        
        # Track extended states (with state_type, age, etc.)
        # We'll maintain compatibility with base system while adding trapped support
        self.extended_states: List[ExtendedGeometricState] = []
        
        # Map base states to extended states
        self.state_map: Dict[object, ExtendedGeometricState] = {}
    
    def add_positive_state(self, coordinates: Tuple[float, ...], 
                          amplitude: complex = 1.0+0j, phase: float = 0.0) -> ExtendedGeometricState:
        """Add state to positive cube (stable semantic space)."""
        base_state = super().add_positive_state(coordinates, amplitude, phase)
        
        # Create extended state
        extended = ExtendedGeometricState(
            coordinates=coordinates,
            amplitude=amplitude,
            phase=phase,
            state_type=+1,  # Cube⁺
            contradiction_score=0.0,
            decoherence_score=0.0
        )
        
        self.extended_states.append(extended)
        self.state_map[base_state] = extended
        
        return extended
    
    def add_negative_state(self, coordinates: Tuple[float, ...], 
                          amplitude: complex = 1.0+0j, phase: float = 0.0) -> ExtendedGeometricState:
        """Add state to negative cube (anti-semantic space)."""
        base_state = super().add_negative_state(coordinates, amplitude, phase)
        
        # Create extended state
        extended = ExtendedGeometricState(
            coordinates=coordinates,
            amplitude=amplitude,
            phase=phase,
            state_type=-1,  # Cube⁻
            contradiction_score=1.0,  # Starts with some contradiction
            decoherence_score=0.5
        )
        
        self.extended_states.append(extended)
        self.state_map[base_state] = extended
        
        return extended
    
    def update_scores(self):
        """
        Update contradiction and decoherence scores for all states.
        
        This should be called periodically to track state health.
        """
        # Update scores based on context
        for extended in self.extended_states:
            if extended.is_trapped():
                continue  # Trapped states don't update scores
            
            # Update contradiction score
            if extended.state_type == +1:
                # Check against negative cube states
                context = [s for s in self.extended_states 
                          if s.state_type == -1 and not s.is_trapped()]
                extended.contradiction_score = self._calculate_contradiction(extended, context)
            else:
                # Negative cube states have high contradiction
                extended.contradiction_score = 0.8
            
            # Update decoherence score (simplified: based on position in negative cube)
            if extended.state_type == -1:
                extended.decoherence_score = 0.7
            else:
                # Positive cube states have low decoherence initially
                extended.decoherence_score = 0.2
    
    def _calculate_contradiction(self, state: ExtendedGeometricState,
                               context: List[ExtendedGeometricState]) -> float:
        """Calculate contradiction score for a state."""
        contradiction = 0.0
        
        for other in context:
            coords_diff = np.linalg.norm(
                np.array(state.coordinates) - np.array(other.coordinates)
            )
            
            if coords_diff < 0.1:  # Close coordinates
                amp_diff = abs(state.amplitude - other.amplitude)
                if amp_diff > 1.0:  # Conflicting amplitudes
                    contradiction += 0.2
        
        return min(1.0, contradiction)
    
    def apply_trapping(self) -> int:
        """
        Apply trapping rules: check all states and trap those that meet criteria.
        
        Returns:
            Number of states trapped
        """
        trapped_count = 0
        
        # Update active age
        self.trapped_system.update_active_age(self.extended_states)
        
        # Check each state
        for extended in self.extended_states:
            if extended.is_active() and self.trapped_system.should_trap(extended):
                self.trapped_system.trap_state(extended)
                trapped_count += 1
        
        return trapped_count
    
    def apply_decay(self) -> int:
        """
        Apply decay to trapped states (leak back to Cube⁻).
        
        Returns:
            Number of states that decayed
        """
        return self.trapped_system.apply_decay(self.extended_states)
    
    def enforce_capacity_limits(self) -> Dict:
        """
        Enforce capacity limits (prevent universe from turning to stone).
        
        Returns:
            Dictionary with actions taken
        """
        return self.trapped_system.enforce_capacity_limit(self.extended_states)
    
    def step(self):
        """
        Perform one step of evolution:
        1. Update scores
        2. Apply trapping
        3. Apply decay
        4. Enforce capacity limits
        """
        # Update scores
        self.update_scores()
        
        # Apply trapping
        trapped_count = self.apply_trapping()
        
        # Apply decay
        decayed_count = self.apply_decay()
        
        # Enforce capacity limits
        capacity_actions = self.enforce_capacity_limits()
        
        return {
            'trapped': trapped_count,
            'decayed': decayed_count,
            'capacity_actions': capacity_actions
        }
    
    def get_trapped_statistics(self) -> Dict:
        """
        Get statistics about trapped states.
        
        Returns:
            Dictionary with trapped state statistics
        """
        total_states = len(self.extended_states)
        trapped_states = [s for s in self.extended_states if s.is_trapped()]
        positive_states = [s for s in self.extended_states if s.state_type == +1]
        negative_states = [s for s in self.extended_states if s.state_type == -1]
        
        trapped_fraction = len(trapped_states) / total_states if total_states > 0 else 0.0
        
        # Analyze scars
        scar_analysis = self.trapped_system.analyze_scars(self.extended_states)
        
        return {
            'total_states': total_states,
            'trapped_count': len(trapped_states),
            'positive_count': len(positive_states),
            'negative_count': len(negative_states),
            'trapped_fraction': trapped_fraction,
            'scar_analysis': scar_analysis
        }
    
    def get_system_summary(self) -> Dict:
        """Get summary of dual cube system with trapped φ."""
        base_summary = super().get_system_summary()
        trapped_stats = self.get_trapped_statistics()
        
        base_summary['trapped_phi'] = {
            'trapped_count': trapped_stats['trapped_count'],
            'trapped_fraction': trapped_stats['trapped_fraction'],
            'num_scars': trapped_stats['scar_analysis']['num_scars'],
            'largest_scar_size': trapped_stats['scar_analysis']['largest_scar_size']
        }
        
        base_summary['three_state_universe'] = {
            'positive': 'Active meaning (fluid)',
            'negative': 'Active contradiction (fluid)',
            'trapped': 'Frozen residue/structure (solid)'
        }
        
        return base_summary

