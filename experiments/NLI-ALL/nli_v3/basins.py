"""
Basin System: Energy Wells for Each Class

Creates "gravity wells" that pull correct patterns inward.
Deeper basins = stronger attractors = more confident predictions.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class Basin:
    """Represents an energy well (basin) for one class."""
    class_idx: int  # 0=entailment, 1=contradiction, 2=neutral
    depth: float = 1.0  # Basin depth (deeper = stronger attractor)
    curvature: float = 1.0  # Basin curvature (sharper = more confident)
    energy: float = 0.0  # Current energy level (lower = stronger)
    
    def __init__(self, class_idx: int, initial_depth: float = 1.0):
        self.class_idx = class_idx
        self.depth = initial_depth
        self.curvature = 1.0
        self.energy = 0.0


class BasinSystem:
    """
    Basin system for 3-class NLI (Entailment, Contradiction, Neutral).
    
    When a class is reinforced, its basin deepens (becomes a stronger attractor).
    This creates "gravity wells" that pull correct patterns inward.
    """
    
    def __init__(self, initial_depth: float = 1.0):
        """
        Initialize basin system.
        
        Args:
            initial_depth: Initial depth for all basins
        """
        # Two basins only: E and C (no Neutral basin - it's the valley)
        self.basins = {
            0: Basin(0, initial_depth),  # Entailment
            1: Basin(1, initial_depth),  # Contradiction
            # Neutral (2) has NO basin - it's the valley, not a peak
        }
        
        # Learning parameters
        self.reinforcement_rate = 0.3  # How fast basins deepen
        self.decay_rate = 0.01  # Natural decay (prevents infinite growth)
        self.capacity = 200.0  # Maximum basin depth (stabilization point)
    
    def reinforce(self, correct_class: int, strength: float = 1.0):
        """
        Reinforce correct class basin (deepen the gravity well).
        
        Uses logistic growth: growth slows as basin approaches capacity.
        This prevents infinite growth while allowing strong basins to form.
        
        Args:
            correct_class: Index of correct class (0=E, 1=C, 2=N has no basin)
            strength: Reinforcement strength (0-1)
        """
        # Only reinforce E and C (no Neutral basin)
        if correct_class not in self.basins:
            return
        
        basin = self.basins[correct_class]
        
        # Logistic growth: dH/dt = r * (1 - H/K)
        # Growth slows as depth approaches capacity
        current_depth = basin.depth
        growth = self.reinforcement_rate * strength * (1.0 - current_depth / self.capacity)
        basin.depth += max(0.01, growth)  # Always grow a little
        
        # Lower energy = stronger attractor
        basin.energy = max(-5.0, basin.energy - 0.1 * strength)
        
        # Update curvature (deeper basins = sharper = more confident)
        basin.curvature = 1.0 + (basin.depth / 5.0)
        
        # Decay other basins (proportional decay prevents infinite growth)
        # Only decay E and C (no Neutral basin)
        for i in [0, 1]:
            if i != correct_class and i in self.basins:
                other_basin = self.basins[i]
                other_basin.depth *= (1.0 - self.decay_rate)
                other_basin.depth = max(0.1, other_basin.depth)  # Minimum depth
                other_basin.curvature = 1.0 + (other_basin.depth / 5.0)
    
    def sculpt_valley(self, resonance: float, variance: float, strength: float = 1.0):
        """
        Sculpt the valley for Neutral examples.
        
        Valley creator: doesn't reinforce peaks, but:
        - Adds local noise around this resonance for both basins
        - Slightly flattens gradients trying to pull into E or C
        - Pushes peaks away from this region
        
        Args:
            resonance: Resonance value where Neutral lives
            variance: Variance value (high variance = noisy region)
            strength: Valley sculpting strength
        """
        # Flatten both basins near this resonance (add noise, reduce attractor strength)
        noise_factor = 0.15 * strength * variance  # More variance = more flattening
        
        # Flatten E basin (reduce depth slightly if resonance is in E's region)
        if resonance > 0:  # Positive resonance = closer to E
            if 0 in self.basins:
                self.basins[0].depth *= (1.0 - noise_factor)
                self.basins[0].depth = max(0.1, self.basins[0].depth)  # Minimum depth
                self.basins[0].curvature = 1.0 + (self.basins[0].depth / 5.0)
        
        # Flatten C basin (reduce depth slightly if resonance is in C's region)
        if resonance < 0:  # Negative resonance = closer to C
            if 1 in self.basins:
                self.basins[1].depth *= (1.0 - noise_factor)
                self.basins[1].depth = max(0.1, self.basins[1].depth)  # Minimum depth
                self.basins[1].curvature = 1.0 + (self.basins[1].depth / 5.0)
        
        # If resonance is near zero (middle valley), flatten both
        if abs(resonance) < 0.3:
            if 0 in self.basins:
                self.basins[0].depth *= (1.0 - noise_factor * 0.5)
                self.basins[0].depth = max(0.1, self.basins[0].depth)
            if 1 in self.basins:
                self.basins[1].depth *= (1.0 - noise_factor * 0.5)
                self.basins[1].depth = max(0.1, self.basins[1].depth)
    
    def get_weights(self) -> np.ndarray:
        """
        Get basin weights (normalized depths).
        
        Deeper basins = stronger attractors = higher weight.
        Neutral has no basin - returns 0.0 for N.
        
        Returns:
            Array of weights [E, C, N] where N=0.0
        """
        depths = np.array([self.basins[i].depth for i in [0, 1]])
        total = np.sum(depths)
        if total > 0:
            weights = depths / total
            # Return [E, C, N] with N=0 (valley has no weight)
            return np.array([weights[0], weights[1], 0.0])
        return np.array([0.5, 0.5, 0.0])  # Equal E/C, zero N
    
    def get_depths(self) -> Dict[int, float]:
        """Get raw basin depths (E and C only - no Neutral)."""
        return {i: self.basins[i].depth for i in [0, 1]}
    
    def get_curvatures(self) -> Dict[int, float]:
        """Get basin curvatures (sharper = more confident) - E and C only."""
        return {i: self.basins[i].curvature for i in [0, 1]}

