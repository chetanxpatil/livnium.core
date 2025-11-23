"""
Layer 3: The City (Neutral Valley)

Natural neutral emerges automatically from force balance.
The city forms where cold and far forces overlap - gravity decides it.

PLANET METAPHOR:
- Cold Region (E): Dense air, stable, pulls inward
- Far Lands (C): High distance, edge of continent
- The City (N): Mixed temperatures, crowded, all roads cross here
  → Forms where cold and far gravitational pulls cancel
"""

import numpy as np
from typing import Dict


class Layer3Valley:
    """
    Layer 3: The City (natural neutral from force balance).
    
    Builds on Layer 2 cold/far basins.
    The city forms automatically where cold and far attractions overlap.
    
    The city is NOT a third force - it's the balance point between forces.
    Like a real city between cold north and far desert.
    """
    
    def __init__(self):
        """Initialize Layer 3 (the city)."""
        pass
    
    def compute(self, layer2_output: Dict[str, float]) -> Dict[str, float]:
        """
        Compute the city (natural neutral) from cold/far force balance.
        
        Args:
            layer2_output: Output from Layer 2 (contains cold_attraction, far_attraction)
            
        Returns:
            Dict with city score and class probabilities
        """
        resonance = layer2_output['resonance']
        # Support both new (cold/far) and old (e/c) naming for compatibility
        cold_attraction = layer2_output.get('cold_attraction', layer2_output.get('e_attraction', 0.0))
        far_attraction = layer2_output.get('far_attraction', layer2_output.get('c_attraction', 0.0))
        
        # ============================================================
        # THE CITY: Where cold and far forces balance
        # ============================================================
        # City = where attractions overlap (forces cancel)
        # Use ratio-based threshold (scale-invariant)
        max_attraction = max(cold_attraction, far_attraction)
        if max_attraction > 1e-6:
            attraction_ratio = abs(cold_attraction - far_attraction) / max_attraction
        else:
            attraction_ratio = 0.0
        
        # City threshold: when cold and far are close, city forms
        city_threshold = 0.15  # When forces are within 15%, city appears
        
        # City gravity: when cold and far overlap, geometry falls into the city
        # This gives the city real curvature without making it a peak
        # The city has gravitational mass - it's where all roads cross
        overlap_strength = 1.0 - min(attraction_ratio, 1.0)  # Higher when cold ≈ far (0 to 1)
        city_gravity = 0.7  # City gravitational constant (stronger than before)
        
        # City pull: real gravitational force when cold and far balance
        # This is the key: city has MASS, not just absence of forces
        # Mixed temperatures, crowded, all roads cross here
        min_attraction = min(cold_attraction, far_attraction)
        city_pull = overlap_strength * city_gravity * (min_attraction + 0.1)  # +0.1 ensures minimum pull
        
        # ============================================================
        # COMPUTE SCORES WITH CITY GRAVITY (NO LABEL ASSIGNMENT)
        # ============================================================
        # Layer 3 ONLY computes forces and scores - NO label decisions
        # Layer 7 will be the authoritative decision layer
        
        # Base scores from attractions
        e_score = cold_attraction
        c_score = far_attraction
        
        # City score based on force balance
        if attraction_ratio < city_threshold and max_attraction > 0.05:
            # Forces balance → The City (Neutral) with real gravitational pull
            base_city_score = 1.0 - (attraction_ratio / city_threshold)  # How close to perfect balance
            base_city_score = max(0.0, min(1.0, base_city_score))
            n_score = min(1.0, base_city_score + city_pull)
            
            # Cold and far scores reduced in city (they cancel, city pulls)
            reduction = 0.4 + city_pull * 0.4  # More reduction when city pulls strongly
            e_score = cold_attraction * (1.0 - reduction)
            c_score = far_attraction * (1.0 - reduction)
        else:
            # One force dominates, but city still has pull if cold and far are close
            if attraction_ratio < 0.35:  # Forces are somewhat close
                n_score = city_pull * 0.8  # City has gravitational pull
                # Slight reduction from city when forces are close
                if cold_attraction > far_attraction:
                    e_score = cold_attraction * (1.0 - city_pull * 0.2)
                    c_score = far_attraction * 0.3
                else:
                    e_score = cold_attraction * 0.3
                    c_score = far_attraction * (1.0 - city_pull * 0.2)
            else:
                # Forces are far apart - city has minimal pull
                n_score = max(0.0, city_pull * 0.3)  # Minimal city pull
        
        # Ensure non-negative
        e_score = max(0.0, e_score)
        c_score = max(0.0, c_score)
        n_score = max(0.0, n_score)
        
        # Normalize scores to sum to 1.0 (for probability interpretation)
        total = e_score + c_score + n_score
        if total > 0:
            e_score /= total
            c_score /= total
            n_score /= total
        
        # NO LABEL ASSIGNMENT - Layer 7 will decide based on these forces
        
        return {
            'resonance': resonance,  # Pass through
            'e_score': float(e_score),  # Backward compatibility
            'c_score': float(c_score),  # Backward compatibility
            'n_score': float(n_score),
            # REMOVED: 'label' and 'confidence' - Layer 7 will decide
            'valley_score': float(n_score),  # Backward compatibility
            'city_score': float(n_score),  # New: explicit city
            'city_pull': float(city_pull),  # New: city gravitational pull
            'attraction_ratio': float(attraction_ratio),
            'cold_attraction': float(cold_attraction),  # New: explicit cold
            'far_attraction': float(far_attraction)  # New: explicit far
        }
    
    def get_state(self) -> Dict:
        """Get current state (Layer 3 has no state - it's pure computation)."""
        return {}

