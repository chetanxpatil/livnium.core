"""
Layer 1: Cold & Distance Curvature

Forms ON TOP of Layer 0. Doesn't fight the core.
Just shapes how the energy flows.

PLANET METAPHOR:
- Entailment = Cold Region (dense, stable, pulls inward)
- Contradiction = Far Lands (distance-based, edge of semantic space)
- Neutral = The City (balance point where forces overlap)

This layer computes:
- Cold density (E): How dense/stable the semantic field is
- Distance (C): How far from the cold region (edge of semantic continent)
"""

import numpy as np
from typing import Dict, List


class Layer1Curvature:
    """
    Layer 1: Cold & Distance Curvature.
    
    Builds on Layer 0 resonance.
    Computes cold density (E) and distance (C) to understand semantic geography.
    
    Cold = dense air, low energy, stable, pulls inward
    Far = high distance, different climate, edge of map
    """
    
    def __init__(self):
        """Initialize Layer 1 (cold & distance)."""
        self.resonance_history: List[float] = []
        self.history_window = 200  # Long memory for curvature (was 10 - too short, caused collapse)
        self.entropy_scale = 0.02  # Thermal noise injection (prevents 0K freeze)
    
    def compute(self, layer0_output: Dict[str, float]) -> Dict[str, float]:
        """
        Compute cold density and distance from resonance.
        
        Args:
            layer0_output: Output from Layer 0 (contains 'resonance')
            
        Returns:
            Dict with:
            - 'cold_density': How dense/stable the cold region (E) is
            - 'distance': How far from cold (C) - edge of semantic continent
            - 'curvature': Overall field curvature (for backward compatibility)
        """
        resonance = layer0_output['resonance']
        
        # ============================================================
        # ENTROPY INJECTION: Thermal noise to prevent 0K freeze
        # ============================================================
        # Small stochastic jiggle prevents phase-lock collapse
        thermal_noise = np.random.normal(0.0, self.entropy_scale)
        resonance_with_entropy = resonance + thermal_noise
        
        # Track resonance history for temporal stability (long memory)
        self.resonance_history.append(resonance_with_entropy)
        if len(self.resonance_history) > self.history_window:
            self.resonance_history.pop(0)
        
        # ============================================================
        # COLD REGION (Entailment): Dense, stable, pulls inward
        # ============================================================
        # Cold density = positive resonance + stability (low variance)
        # Higher density = stronger pull, more stable, lower energy
        # Use entropy-modified resonance for calculations (thermal effects)
        if len(self.resonance_history) >= 2:
            resonance_variance = np.var(self.resonance_history[-5:]) if len(self.resonance_history) >= 5 else 0.0
            stability = 1.0 - min(resonance_variance, 1.0)  # 1.0 = perfectly stable
        else:
            stability = 1.0
        
        # Cold density: positive resonance × stability (with entropy)
        # Dense air = high positive resonance + high stability
        # Entropy adds thermal fluctuations (prevents freeze)
        cold_density = max(0.0, resonance_with_entropy) * stability
        
        # ============================================================
        # FAR LANDS (Contradiction): Distance-based, edge of continent
        # ============================================================
        # Distance = how far from the cold region
        # Not just negative resonance - it's about semantic distance
        # Far = opposite pole + edge of map (high distance metric)
        
        # Distance from cold: negative resonance + how far from zero
        # Edge of continent = far from cold, different climate
        # Use entropy-modified resonance (thermal effects matter for distance too)
        distance_from_cold = max(0.0, -resonance_with_entropy)  # Distance when resonance is negative
        
        # Also measure "edge distance" - how far from the semantic center
        # This captures the "far lands" concept (edge of map)
        edge_distance = abs(resonance_with_entropy) if resonance_with_entropy < 0 else 0.0  # Only count when negative
        
        # Total distance = combination of both
        # Far lands = far from cold + at the edge
        total_distance = distance_from_cold + 0.3 * edge_distance
        
        # ============================================================
        # FIELD CURVATURE (for backward compatibility)
        # ============================================================
        # Compute overall curvature (second derivative approximation)
        if len(self.resonance_history) >= 3:
            r_t = self.resonance_history[-1]
            r_t1 = self.resonance_history[-2]
            r_t2 = self.resonance_history[-3]
            curvature = r_t - 2.0 * r_t1 + r_t2
        else:
            if len(self.resonance_history) >= 2:
                curvature = abs(self.resonance_history[-1] - self.resonance_history[-2])
            else:
                curvature = 0.0
        
        curvature_magnitude = abs(curvature)
        
        return {
            'resonance': resonance,  # Pass through (original, not entropy-modified)
            'resonance_with_entropy': float(resonance_with_entropy),  # With thermal noise
            'curvature': float(curvature_magnitude),  # Backward compatibility
            'cold_density': float(cold_density),  # E: dense, stable, pulls inward
            'distance': float(total_distance),  # C: far from cold, edge of continent
            'stability': float(stability),  # How stable the field is
            'thermal_noise': float(thermal_noise)  # Entropy injection amount
        }
    
    def get_state(self) -> Dict:
        """Get current state."""
        return {
            'resonance_history': self.resonance_history.copy()
        }

