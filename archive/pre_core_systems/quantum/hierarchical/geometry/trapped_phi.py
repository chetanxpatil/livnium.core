"""
Trapped φ (Cube⁰): Fossilized Contradictions as Structural Memory

Extends the dual-cube system with a third state: frozen/fossilized contradictions
that become structural obstacles in the geometry.

Three-State Semantic Universe:
- Cube⁺ (+1): Active meaning (fluid)
- Cube⁻ (-1): Active contradiction/decoherence (fluid)
- Cube⁰ (0): Trapped/frozen φ (solid structure)
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from quantum.hierarchical.geometry.level0.base_geometry import BaseGeometricState


# Trapping parameters
CONTRA_TRAP_THRESHOLD = 0.7  # High contradiction threshold
DECO_TRAP_THRESHOLD = 0.6     # High decoherence threshold
MIN_ACTIVE_AGE = 10           # Must be contradictory for N ticks

# Decay parameters
TRAP_HALF_LIFE = 100          # Decay starts after N ticks
LEAK_PROB_PER_TICK = 0.01     # 1% chance per tick after half-life

# Capacity limit
MAX_TRAPPED_FRACTION = 0.25   # Max 25% trapped


@dataclass
class ExtendedGeometricState(BaseGeometricState):
    """
    Extended geometric state with trapped φ support.
    
    Adds state_type, phi_energy, age, and tracking scores.
    """
    state_type: int = +1  # +1 (Cube⁺), -1 (Cube⁻), or 0 (Cube⁰)
    phi_energy: float = 0.0  # Energy when trapped
    age: int = 0  # Age in current state
    contradiction_score: float = 0.0  # Current contradiction score
    decoherence_score: float = 0.0  # Current decoherence score
    active_age: int = 0  # How long in active state before trapping
    
    def is_trapped(self) -> bool:
        """Check if state is trapped (Cube⁰)."""
        return self.state_type == 0
    
    def is_active(self) -> bool:
        """Check if state is active (Cube⁺ or Cube⁻)."""
        return self.state_type != 0
    
    def can_participate_in_operations(self) -> bool:
        """Check if state can participate in normal operations."""
        return not self.is_trapped()


class TrappedPhiSystem:
    """
    System for managing trapped φ states.
    
    Handles:
    - Trapping rules (when to trap)
    - Decay mechanism (leak back to Cube⁻)
    - Capacity limits (prevent over-crystallization)
    """
    
    def __init__(self,
                 contra_trap_threshold: float = CONTRA_TRAP_THRESHOLD,
                 deco_trap_threshold: float = DECO_TRAP_THRESHOLD,
                 min_active_age: int = MIN_ACTIVE_AGE,
                 trap_half_life: int = TRAP_HALF_LIFE,
                 leak_prob: float = LEAK_PROB_PER_TICK,
                 max_trapped_fraction: float = MAX_TRAPPED_FRACTION):
        """
        Initialize trapped φ system.
        
        Args:
            contra_trap_threshold: Contradiction threshold for trapping
            deco_trap_threshold: Decoherence threshold for trapping
            min_active_age: Minimum age before trapping
            trap_half_life: Half-life for trapped states
            leak_prob: Probability of leak per tick after half-life
            max_trapped_fraction: Maximum fraction of trapped states
        """
        self.contra_trap_threshold = contra_trap_threshold
        self.deco_trap_threshold = deco_trap_threshold
        self.min_active_age = min_active_age
        self.trap_half_life = trap_half_life
        self.leak_prob = leak_prob
        self.max_trapped_fraction = max_trapped_fraction
        
        # Dynamic thresholds (adjust if too much trapping)
        self.current_contra_threshold = contra_trap_threshold
        self.current_deco_threshold = deco_trap_threshold
    
    def should_trap(self, state: ExtendedGeometricState) -> bool:
        """
        Check if a state should be trapped.
        
        Args:
            state: State to check
            
        Returns:
            True if state should be trapped
        """
        if state.is_trapped():
            return False  # Already trapped
        
        # Check trapping conditions
        if (state.contradiction_score > self.current_contra_threshold
            and state.decoherence_score > self.current_deco_threshold
            and state.active_age > self.min_active_age):
            return True
        
        return False
    
    def trap_state(self, state: ExtendedGeometricState) -> ExtendedGeometricState:
        """
        Trap a state (move to Cube⁰).
        
        Args:
            state: State to trap
            
        Returns:
            Trapped state
        """
        state.state_type = 0  # Cube⁰
        state.phi_energy = abs(state.amplitude)
        state.age = 0  # Reset age counter
        state.active_age = 0  # Reset active age
        
        return state
    
    def apply_decay(self, states: List[ExtendedGeometricState]) -> int:
        """
        Apply decay to trapped states (leak back to Cube⁻).
        
        Args:
            states: List of states to check for decay
            
        Returns:
            Number of states that decayed
        """
        decayed_count = 0
        
        for state in states:
            if state.is_trapped():
                state.age += 1
                
                # Check if past half-life
                if state.age > self.trap_half_life:
                    # Random decay
                    if random.random() < self.leak_prob:
                        state.state_type = -1  # Leak back to Cube⁻
                        state.age = 0
                        state.active_age = 0
                        decayed_count += 1
        
        return decayed_count
    
    def get_trapped_fraction(self, states: List[ExtendedGeometricState]) -> float:
        """
        Get fraction of states that are trapped.
        
        Args:
            states: List of states
            
        Returns:
            Fraction of trapped states (0.0 to 1.0)
        """
        if not states:
            return 0.0
        
        trapped_count = sum(1 for s in states if s.is_trapped())
        return trapped_count / len(states)
    
    def enforce_capacity_limit(self, states: List[ExtendedGeometricState]) -> Dict:
        """
        Enforce capacity limit (prevent over-crystallization).
        
        If too many states are trapped:
        - Raise thresholds dynamically
        - Force decay of oldest trapped states
        
        Args:
            states: List of states
            
        Returns:
            Dictionary with actions taken
        """
        trapped_fraction = self.get_trapped_fraction(states)
        
        actions = {
            'trapped_fraction': trapped_fraction,
            'thresholds_raised': False,
            'forced_decay': 0
        }
        
        if trapped_fraction > self.max_trapped_fraction:
            # Raise thresholds dynamically
            self.current_contra_threshold *= 1.1
            self.current_deco_threshold *= 1.1
            actions['thresholds_raised'] = True
            
            # Force decay of oldest trapped states
            trapped_states = [s for s in states if s.is_trapped()]
            trapped_states.sort(key=lambda s: s.age, reverse=True)  # Oldest first
            
            # Decay top 10% of oldest
            num_to_decay = max(1, int(len(trapped_states) * 0.1))
            for state in trapped_states[:num_to_decay]:
                state.state_type = -1  # Leak back to Cube⁻
                state.age = 0
                state.active_age = 0
                actions['forced_decay'] += 1
        
        return actions
    
    def update_active_age(self, states: List[ExtendedGeometricState]):
        """
        Update active age for all active states.
        
        Args:
            states: List of states
        """
        for state in states:
            if state.is_active():
                state.active_age += 1
            else:
                # Reset active age for trapped states
                state.active_age = 0
    
    def analyze_scars(self, states: List[ExtendedGeometricState], 
                     cluster_threshold: float = 0.1) -> Dict:
        """
        Analyze scar patterns (clusters of trapped states).
        
        Args:
            states: List of states
            cluster_threshold: Distance threshold for clustering
            
        Returns:
            Dictionary with scar analysis
        """
        trapped_states = [s for s in states if s.is_trapped()]
        
        if not trapped_states:
            return {
                'num_scars': 0,
                'largest_scar_size': 0,
                'total_trapped': 0,
                'clusters': []
            }
        
        # Simple clustering: group by proximity
        clusters = []
        used = set()
        
        for i, state1 in enumerate(trapped_states):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            for j, state2 in enumerate(trapped_states):
                if j in used or j == i:
                    continue
                
                # Check distance
                coords1 = np.array(state1.coordinates)
                coords2 = np.array(state2.coordinates)
                distance = np.linalg.norm(coords1 - coords2)
                
                if distance < cluster_threshold:
                    cluster.append(j)
                    used.add(j)
            
            clusters.append(cluster)
        
        # Analyze clusters
        cluster_sizes = [len(c) for c in clusters]
        largest_scar_size = max(cluster_sizes) if cluster_sizes else 0
        
        return {
            'num_scars': len(clusters),
            'largest_scar_size': largest_scar_size,
            'total_trapped': len(trapped_states),
            'clusters': clusters,
            'cluster_sizes': cluster_sizes
        }

