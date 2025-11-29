"""
Quantum Collapse: 3-Way Decision Making

Converts scores → probabilities → amplitudes → collapse to one class.
Creates stability - patterns get pulled into right semantic regions.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class CollapseResult:
    """Result of quantum collapse."""
    label: str  # 'entailment', 'contradiction', or 'neutral'
    class_idx: int  # 0, 1, or 2
    probabilities: Dict[str, float]  # Probabilities before collapse
    confidence: float  # Probability of collapsed state
    amplitudes: np.ndarray  # Quantum amplitudes before collapse


class QuantumCollapse:
    """
    Quantum collapse for 3-class NLI.
    
    Converts geometric scores into probabilities, then amplitudes,
    then collapses to one class. This creates stability.
    """
    
    def __init__(self, temperature: float = 3.0):
        """
        Initialize quantum collapse.
        
        Args:
            temperature: Temperature for softmax (higher = more uniform)
        """
        self.temperature = temperature
    
    def collapse(self, 
                 scores: np.ndarray,
                 basin_weights: np.ndarray) -> CollapseResult:
        """
        Perform quantum collapse.
        
        Args:
            scores: Raw scores [E, C, N]
            basin_weights: Basin weights [E, C, N] (from BasinSystem)
            
        Returns:
            CollapseResult with collapsed class
        """
        # 1. Apply basin weights (deeper basins = stronger attractors)
        weighted_scores = scores * (1.0 + basin_weights * 0.3)
        
        # 2. Convert to probabilities (softmax with temperature)
        exp_scores = np.exp(weighted_scores / self.temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        # 3. Convert to quantum amplitudes (square root for normalization)
        amplitudes = np.sqrt(probabilities)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)  # Normalize
        
        # 4. Collapse to one class (argmax)
        class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[class_idx])
        
        # Map to label
        label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
        label = label_map[class_idx]
        
        return CollapseResult(
            label=label,
            class_idx=class_idx,
            probabilities={
                'entailment': float(probabilities[0]),
                'contradiction': float(probabilities[1]),
                'neutral': float(probabilities[2])
            },
            confidence=confidence,
            amplitudes=amplitudes
        )

