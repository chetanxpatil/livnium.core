"""
Layer 5: Temporal Stability & Thermodynamic Cycle

Tracks stability across steps.
Detects convergence (Moksha).
Monitors thermodynamic health (entropy, heat, temperature).

This is the climate system that prevents thermal death.
"""

from typing import Dict, List
import numpy as np


class Layer5TemporalStability:
    """
    Layer 5: Temporal stability + thermodynamic monitoring.
    
    Builds on Layer 4 routing.
    Detects convergence and stability patterns.
    Monitors system health (entropy, heat, prevents 0K freeze).
    """
    
    def __init__(self, stability_window: int = 50):
        """
        Initialize Layer 5 (temporal stability + thermodynamics).
        
        Args:
            stability_window: Number of steps to track for stability (increased from 10)
        """
        self.stability_window = stability_window
        self.confidence_history: List[float] = []
        self.label_history: List[str] = []
        self.entropy_history: List[float] = []  # Track entropy injection
        self.temperature = 1.0  # System temperature (1.0 = healthy, 0.0 = frozen)
    
    def compute(self, layer4_output: Dict[str, float]) -> Dict[str, float]:
        """
        Compute temporal stability.
        
        Args:
            layer4_output: Output from Layer 4 (contains forces, scores, route - NO label)
            
        Returns:
            Dict with stability information
        """
        # Compute confidence from forces (max force strength)
        cold_attraction = layer4_output.get('cold_attraction', 0.0)
        far_attraction = layer4_output.get('far_attraction', 0.0)
        city_pull = layer4_output.get('city_pull', 0.0)
        max_force = max(cold_attraction, far_attraction, city_pull)
        
        # Use max_force as confidence proxy (or compute from scores)
        e_score = layer4_output.get('e_score', 0.0)
        c_score = layer4_output.get('c_score', 0.0)
        n_score = layer4_output.get('n_score', 0.0)
        confidence = max(e_score, c_score, n_score)
        
        # Track history (use force signature instead of label)
        self.confidence_history.append(confidence)
        # Create a "force signature" for tracking stability (which force dominates)
        if city_pull > max(cold_attraction, far_attraction):
            force_signature = 'city'
        elif cold_attraction > far_attraction:
            force_signature = 'cold'
        else:
            force_signature = 'far'
        self.label_history.append(force_signature)  # Track force patterns, not labels
        
        # Track entropy (thermal noise) from Layer 1
        thermal_noise = layer4_output.get('thermal_noise', 0.0)
        self.entropy_history.append(abs(thermal_noise))
        
        if len(self.confidence_history) > self.stability_window:
            self.confidence_history.pop(0)
            self.label_history.pop(0)
            self.entropy_history.pop(0)
        
        # ============================================================
        # THERMODYNAMIC HEALTH MONITORING
        # ============================================================
        # System temperature = measure of entropy/activity
        # Low temperature = frozen (0K), high temperature = healthy
        if len(self.entropy_history) >= 10:
            avg_entropy = np.mean(self.entropy_history[-10:])
            # Temperature scales with entropy (more entropy = higher temp)
            self.temperature = min(1.0, max(0.0, avg_entropy * 50.0))  # Scale entropy to [0, 1]
        else:
            self.temperature = 0.5  # Default moderate temperature
        
        # Check for thermal death (frozen system)
        is_frozen = self.temperature < 0.1  # Too cold = frozen
        is_overheated = self.temperature > 0.9  # Too hot = unstable
        
        # Compute stability
        if len(self.confidence_history) >= self.stability_window:
            # Check if label is consistent
            label_stable = len(set(self.label_history)) == 1
            
            # Check if confidence is high and stable
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
            confidence_stable = avg_confidence > 0.7
            
            # Stability requires good temperature (not frozen, not overheated)
            is_stable = label_stable and confidence_stable and not is_frozen and not is_overheated
            is_moksha = is_stable and avg_confidence > 0.8  # Convergence
        else:
            is_stable = False
            is_moksha = False
            avg_confidence = confidence
        
        return {
            **layer4_output,  # Pass through all
            'is_stable': is_stable,
            'is_moksha': is_moksha,
            'avg_confidence': float(avg_confidence),
            'stability_window': len(self.confidence_history),
            # Thermodynamic health
            'temperature': float(self.temperature),
            'is_frozen': is_frozen,
            'is_overheated': is_overheated,
            'avg_entropy': float(np.mean(self.entropy_history[-10:])) if len(self.entropy_history) >= 10 else 0.0
        }
    
    def get_state(self) -> Dict:
        """Get current state."""
        return {
            'confidence_history': self.confidence_history.copy(),
            'label_history': self.label_history.copy()
        }

