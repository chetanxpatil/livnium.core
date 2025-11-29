"""
Layer 2: Cold & Far Basins

Forms from Layer 1. More structure, more refinement.
Still follows gravity.

PLANET METAPHOR:
- Cold Basin (E): Dense air, low energy, stable, pulls inward
- Far Basin (C): High distance, edge of continent, different climate
- City (N): Forms in Layer 3 where cold and far forces balance

Turns cold density and distance into attraction wells.
"""

import numpy as np
from typing import Dict


class Layer2Basin:
    """
    Layer 2: Cold & Far Basin system.
    
    Builds on Layer 1 cold/distance curvature.
    Creates gravity wells for:
    - Cold region (E): Dense, stable, pulls inward
    - Far lands (C): Distance-based, edge of semantic continent
    
    CRITICAL: Must be shared across all instances to learn from all examples.
    """
    
    # Shared state (class-level singleton)
    _shared_cold_basin_depth = 1.0  # Cold region depth (E)
    _shared_far_basin_depth = 1.0   # Far lands depth (C)
    _shared_prediction_counts = {'entailment': 0, 'contradiction': 0, 'neutral': 0}  # For anti-monopoly
    _shared_total_predictions = 0  # Total predictions made
    
    def __init__(self):
        """Initialize Layer 2 (cold & far basins)."""
        # Use shared state (all instances share the same basins)
        # Learning parameters
        self.reinforcement_rate = 0.3
        self.decay_rate = 0.01
        self.capacity = 200.0
        self.monopoly_threshold = 0.6  # If one class > 60%, trigger heat wave
        self.heat_wave_strength = 0.15  # How much to flatten overconfident basin
    
    @classmethod
    def reset_shared_state(cls):
        """Reset shared state (for new training run)."""
        cls._shared_cold_basin_depth = 1.0
        cls._shared_far_basin_depth = 1.0
        cls._shared_prediction_counts = {'entailment': 0, 'contradiction': 0, 'neutral': 0}
        cls._shared_total_predictions = 0
    
    @property
    def cold_basin_depth(self):
        """Get shared cold basin depth (E)."""
        return Layer2Basin._shared_cold_basin_depth
    
    @property
    def far_basin_depth(self):
        """Get shared far basin depth (C)."""
        return Layer2Basin._shared_far_basin_depth
    
    def compute(self, layer1_output: Dict[str, float]) -> Dict[str, float]:
        """
        Compute basin attractions from cold density and distance.
        
        Args:
            layer1_output: Output from Layer 1 (contains 'resonance', 'cold_density', 'distance')
            
        Returns:
            Dict with basin weights, depths, and attractions
        """
        resonance = layer1_output['resonance']
        cold_density = layer1_output.get('cold_density', max(0.0, resonance))  # Fallback for compatibility
        distance = layer1_output.get('distance', max(0.0, -resonance))  # Fallback for compatibility
        curvature = layer1_output.get('curvature', 0.0)  # For backward compatibility
        
        # Use shared basin depths
        cold_depth = self.cold_basin_depth
        far_depth = self.far_basin_depth
        
        # ============================================================
        # BASIN TEMPERATURE: Lower depth when overconfident
        # ============================================================
        # If a basin is too deep relative to the other, add "heat" to flatten it
        # This prevents one basin from monopolizing all gravity
        total_depth = cold_depth + far_depth
        if total_depth > 0:
            cold_ratio = cold_depth / total_depth
            far_ratio = far_depth / total_depth
            
            # If one basin dominates, add thermal agitation
            if cold_ratio > 0.7:  # Cold is overconfident
                temperature_penalty = (cold_ratio - 0.7) * 0.3  # Flatten by up to 30%
                cold_depth *= (1.0 - temperature_penalty)
            if far_ratio > 0.7:  # Far is overconfident
                temperature_penalty = (far_ratio - 0.7) * 0.3
                far_depth *= (1.0 - temperature_penalty)
        
        # Basin weights (normalized depths)
        total_depth = cold_depth + far_depth
        if total_depth > 0:
            cold_weight = cold_depth / total_depth
            far_weight = far_depth / total_depth
        else:
            cold_weight = far_weight = 0.5
        
        # ============================================================
        # COLD ATTRACTION (E): Dense air pulls inward
        # ============================================================
        # Cold region = dense, stable, low energy, pulls things together
        # Attraction = cold density × basin depth × stability
        # Higher density = stronger pull (like dense air creates pressure)
        cold_attraction = cold_weight * (1.0 + curvature) * (1.0 + cold_density)
        
        # ============================================================
        # FAR ATTRACTION (C): Distance-based, edge of continent
        # ============================================================
        # Far lands = high distance, edge of map, different climate
        # Attraction = distance × basin depth × edge factor
        # Further = stronger separation (like being at the edge of the continent)
        
        # REPULSION FIELD: Far lands push away from cold
        # This creates the "continent of contradiction" effect
        # Repulsion = negative resonance creates push force
        repulsion = max(0.0, -resonance) * (distance ** 2)  # Distance amplifies repulsion
        repulsion_boost = 1.0 + (repulsion * 0.3)  # Up to 30% boost from repulsion
        
        far_attraction = far_weight * (1.0 + curvature) * (1.0 + distance) * repulsion_boost
        
        return {
            'resonance': resonance,  # Pass through
            'curvature': curvature,  # Pass through
            'cold_density': float(cold_density),  # Pass through
            'distance': float(distance),  # Pass through
            # Basin depths (for learning)
            'e_basin_depth': float(cold_depth),  # Backward compatibility
            'c_basin_depth': float(far_depth),  # Backward compatibility
            # Basin weights
            'e_weight': float(cold_weight),  # Backward compatibility
            'c_weight': float(far_weight),  # Backward compatibility
            # Attractions (cold = E, far = C)
            'e_attraction': float(cold_attraction),  # Backward compatibility
            'c_attraction': float(far_attraction),  # Backward compatibility
            'cold_attraction': float(cold_attraction),  # New: explicit cold
            'far_attraction': float(far_attraction)  # New: explicit far
        }
    
    def reinforce(self, class_idx: int, strength: float = 1.0):
        """
        Reinforce basin (deepen gravity well) with anti-monopoly protection.
        
        Args:
            class_idx: 0=Cold (E), 1=Far (C) (no Neutral basin - it's the city)
            strength: Reinforcement strength
        """
        if class_idx == 0:  # Cold region (Entailment)
            growth = self.reinforcement_rate * strength * (1.0 - Layer2Basin._shared_cold_basin_depth / self.capacity)
            Layer2Basin._shared_cold_basin_depth += max(0.01, growth)
            # Decay far lands (opposite force)
            Layer2Basin._shared_far_basin_depth *= (1.0 - self.decay_rate)
            Layer2Basin._shared_far_basin_depth = max(0.1, Layer2Basin._shared_far_basin_depth)
        elif class_idx == 1:  # Far lands (Contradiction)
            growth = self.reinforcement_rate * strength * (1.0 - Layer2Basin._shared_far_basin_depth / self.capacity)
            Layer2Basin._shared_far_basin_depth += max(0.01, growth)
            # Decay cold region (opposite force)
            Layer2Basin._shared_cold_basin_depth *= (1.0 - self.decay_rate)
            Layer2Basin._shared_cold_basin_depth = max(0.1, Layer2Basin._shared_cold_basin_depth)
        # Neutral (2) has no basin - it's the city, not a force
    
    @classmethod
    def track_prediction(cls, predicted_label: str):
        """Track prediction for anti-monopoly rule."""
        cls._shared_total_predictions += 1
        if predicted_label in cls._shared_prediction_counts:
            cls._shared_prediction_counts[predicted_label] += 1
    
    @classmethod
    def apply_anti_monopoly_heat(cls, instance):
        """
        Anti-monopoly rule: If one class > 60%, add heat wave to flatten it.
        
        This prevents thermal death - one class can't monopolize forever.
        """
        if cls._shared_total_predictions < 100:  # Need some history first
            return
        
        # Check if any class is monopolizing
        for label, count in cls._shared_prediction_counts.items():
            ratio = count / cls._shared_total_predictions
            if ratio > instance.monopoly_threshold:
                # Heat wave: flatten the overconfident basin
                heat_strength = (ratio - instance.monopoly_threshold) * instance.heat_wave_strength
                
                if label == 'entailment':
                    # Flatten cold basin
                    Layer2Basin._shared_cold_basin_depth *= (1.0 - heat_strength)
                    Layer2Basin._shared_cold_basin_depth = max(0.5, Layer2Basin._shared_cold_basin_depth)
                elif label == 'contradiction':
                    # Flatten far basin
                    Layer2Basin._shared_far_basin_depth *= (1.0 - heat_strength)
                    Layer2Basin._shared_far_basin_depth = max(0.5, Layer2Basin._shared_far_basin_depth)
                
                # Reset counts to prevent continuous heat waves
                cls._shared_prediction_counts = {'entailment': 0, 'contradiction': 0, 'neutral': 0}
                cls._shared_total_predictions = 0
                break
    
    def get_state(self) -> Dict:
        """Get current state."""
        return {
            'cold_basin_depth': float(Layer2Basin._shared_cold_basin_depth),
            'far_basin_depth': float(Layer2Basin._shared_far_basin_depth),
            # Backward compatibility
            'e_basin_depth': float(Layer2Basin._shared_cold_basin_depth),
            'c_basin_depth': float(Layer2Basin._shared_far_basin_depth)
        }

