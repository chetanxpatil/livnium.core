"""
Layer 4: Meta Routing

Reads the geometry. No override.
Routes information based on lower layer states.
"""

from typing import Dict


class Layer4MetaRouting:
    """
    Layer 4: Meta routing (reads geometry, no override).
    
    Builds on Layer 3 valley.
    Routes information but never overrides lower layers.
    """
    
    def __init__(self):
        """Initialize Layer 4 (meta routing)."""
        self.routing_history = []
    
    def compute(self, layer3_output: Dict[str, float]) -> Dict[str, float]:
        """
        Route based on geometry (read-only, no override).
        
        Args:
            layer3_output: Output from Layer 3 (contains scores, forces - NO label)
            
        Returns:
            Dict with routing information
        """
        # Compute route from forces/scores, not from label
        valley_score = layer3_output.get('valley_score', layer3_output.get('n_score', 0.0))
        city_pull = layer3_output.get('city_pull', 0.0)
        cold_attraction = layer3_output.get('cold_attraction', 0.0)
        far_attraction = layer3_output.get('far_attraction', 0.0)
        
        # Compute confidence from forces (max force strength)
        max_force = max(cold_attraction, far_attraction, city_pull)
        
        # Route based on geometry (read-only)
        if valley_score > 0.7 or city_pull > 0.5:
            route = 'valley'  # Strong valley/city signal
        elif max_force > 0.8:
            route = 'peak'  # Strong peak signal
        else:
            route = 'transition'  # Between peak and valley
        
        # Track routing
        self.routing_history.append(route)
        if len(self.routing_history) > 100:
            self.routing_history.pop(0)
        
        return {
            **layer3_output,  # Pass through all
            'route': route,
            'is_peak': route == 'peak',
            'is_valley': route == 'valley',
            'is_transition': route == 'transition'
        }
    
    def get_state(self) -> Dict:
        """Get current state."""
        return {
            'routing_history': self.routing_history.copy()
        }

