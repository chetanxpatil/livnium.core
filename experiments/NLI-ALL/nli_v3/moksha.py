"""
Moksha Router: Meta-Controller for Convergence

Detects when the system has reached a stable state (Moksha).
Tunes the system dynamically for optimal routing.
"""

from typing import List
from dataclasses import dataclass


@dataclass
class MokshaState:
    """State of Moksha convergence."""
    is_moksha: bool  # Has convergence been reached?
    resonance: float  # Current resonance score
    stability_window: int  # Number of stable steps
    convergence_threshold: float  # Threshold for convergence


class MokshaRouter:
    """
    Moksha router - detects convergence and tunes system dynamically.
    
    Moksha = fixed-point convergence = system has reached stable state.
    High resonance = stable state = patterns have "fallen inward" correctly.
    """
    
    def __init__(self, 
                 convergence_threshold: float = 0.7,
                 stability_window: int = 10):
        """
        Initialize Moksha router.
        
        Args:
            convergence_threshold: Resonance threshold for convergence (0-1)
            stability_window: Number of steps to maintain stability
        """
        self.convergence_threshold = convergence_threshold
        self.stability_window = stability_window
        self.resonance_history: List[float] = []
    
    def check_moksha(self, resonance: float) -> MokshaState:
        """
        Check if system has reached Moksha (convergence).
        
        Args:
            resonance: Current resonance score
            
        Returns:
            MokshaState with convergence status
        """
        self.resonance_history.append(resonance)
        
        # Keep only recent history
        if len(self.resonance_history) > self.stability_window:
            self.resonance_history.pop(0)
        
        # Check if resonance is consistently high (stable state)
        if len(self.resonance_history) >= self.stability_window:
            avg_resonance = sum(self.resonance_history) / len(self.resonance_history)
            is_moksha = avg_resonance >= self.convergence_threshold
        else:
            is_moksha = resonance >= self.convergence_threshold
        
        return MokshaState(
            is_moksha=is_moksha,
            resonance=resonance,
            stability_window=len(self.resonance_history),
            convergence_threshold=self.convergence_threshold
        )
    
    def reset(self):
        """Reset Moksha state (for new training run)."""
        self.resonance_history = []

