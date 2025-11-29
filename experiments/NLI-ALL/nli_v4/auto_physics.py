"""
AutoPhysicsEngine: Self-Organizing Universe

Closes the thermodynamic loop:
curvature → basins → gravity → memory → resonance → curvature

Three Laws:
1. Automatic Entropy Injection (scales with class imbalance)
2. Repulsion Field for Contradiction (far lands push away)
3. Dynamic Basin Depth (anti-monopoly, prevents collapse)

This is how meaning emerges automatically from physics.
"""

import numpy as np
from typing import Dict
from pathlib import Path
import json
from .layer2_basin import Layer2Basin


# Load physics overrides (set by run_auto_universe.py)
BASE_DIR = Path(__file__).resolve().parent
OVERRIDES_PATH = BASE_DIR / "auto_physics_overrides.json"

# Cache for overrides (loaded once, reused)
_OVERRIDES_CACHE = None
_OVERRIDES_LOADED = False


def load_overrides() -> Dict:
    """Load physics parameter overrides from file (cached)."""
    global _OVERRIDES_CACHE, _OVERRIDES_LOADED
    
    if _OVERRIDES_LOADED:
        return _OVERRIDES_CACHE or {}
    
    _OVERRIDES_LOADED = True
    if OVERRIDES_PATH.exists():
        try:
            with OVERRIDES_PATH.open() as f:
                _OVERRIDES_CACHE = json.load(f)
                # Only print once when first loaded
                print(f"[AutoPhysics] Loaded overrides: {_OVERRIDES_CACHE}")
                return _OVERRIDES_CACHE
        except Exception:
            pass
    
    _OVERRIDES_CACHE = {}
    return {}


OVERRIDES = load_overrides()


class AutoPhysicsEngine:
    """
    AutoPhysicsEngine: Self-organizing universe controller.
    
    Runs automatically after each batch to maintain thermodynamic balance.
    No manual tuning - the universe runs itself.
    """
    
    def __init__(self):
        """Initialize auto-physics engine."""
        # Load overrides (set by autonomous universe driver)
        overrides = load_overrides()
        
        self.entropy_base = 0.01  # Base entropy (always present)
        self.entropy_scale = overrides.get("entropy_scale", 0.02)  # Entropy scales with imbalance
        self.repulsion_strength = overrides.get("repulsion_strength", 0.3)  # How strong repulsion is for far lands
        self.dominance_threshold = 0.6  # When to trigger anti-monopoly
    
    def compute_class_imbalance(self) -> float:
        """
        Compute class imbalance from prediction distribution.
        
        Returns:
            Imbalance value [0, 1] where 1 = perfect imbalance (one class dominates)
        """
        counts = Layer2Basin._shared_prediction_counts
        total = Layer2Basin._shared_total_predictions
        
        if total < 10:  # Need some history
            return 0.0
        
        ratios = {k: v / total for k, v in counts.items()}
        max_ratio = max(ratios.values())
        min_ratio = min(ratios.values())
        
        # Imbalance = how much one class dominates
        imbalance = max_ratio - min_ratio
        
        return float(imbalance)
    
    def step(self, classifier) -> Dict[str, float]:
        """
        Run one auto-physics step.
        
        This closes the loop:
        - Updates entropy based on class imbalance
        - Applies repulsion field for contradiction
        - Adjusts basin depths dynamically
        - Maintains thermodynamic balance
        
        Args:
            classifier: LayeredLivniumClassifier instance
            
        Returns:
            Dict with physics state
        """
        # ============================================================
        # LAW 1: Automatic Entropy Injection
        # ============================================================
        # Entropy = base + scale * imbalance
        # More imbalance → more heat → more entropy → prevents freeze
        class_imbalance = self.compute_class_imbalance()
        dynamic_entropy = self.entropy_base + self.entropy_scale * class_imbalance
        
        # Update Layer 1 entropy scale
        classifier.layer1.entropy_scale = dynamic_entropy
        
        # ============================================================
        # LAW 2: Repulsion Field for Contradiction
        # ============================================================
        # Far lands (contradiction) need distance-based repulsion
        # This creates the "continent of contradiction"
        self._apply_repulsion_field(classifier)
        
        # ============================================================
        # LAW 3: Dynamic Basin Depth (Anti-Monopoly)
        # ============================================================
        # When one class dominates, flatten its basin
        # This prevents collapse and maintains exploration
        self._apply_dynamic_basin_depth(classifier, class_imbalance)
        
        return {
            'entropy': float(dynamic_entropy),
            'class_imbalance': float(class_imbalance),
            'cold_basin_depth': float(Layer2Basin._shared_cold_basin_depth),
            'far_basin_depth': float(Layer2Basin._shared_far_basin_depth),
            'temperature': float(class_imbalance * 2.0)  # Temperature scales with imbalance
        }
    
    def _apply_repulsion_field(self, classifier):
        """
        LAW 2: Repulsion Field for Contradiction.
        
        Far lands (contradiction) push away from cold region.
        Distance amplifies the push.
        """
        # Get current basin depths
        cold_depth = Layer2Basin._shared_cold_basin_depth
        far_depth = Layer2Basin._shared_far_basin_depth
        
        # Repulsion = how much far basin pushes away
        # When cold is strong, far needs more repulsion to maintain distance
        if cold_depth > 0:
            # Repulsion strength scales with cold dominance
            cold_ratio = cold_depth / (cold_depth + far_depth + 1e-6)
            
            # Far lands push away more when cold dominates
            # This creates the "continent of contradiction" effect
            repulsion_boost = self.repulsion_strength * cold_ratio
            
            # Boost far basin depth (repulsion = stronger separation)
            # But don't let it grow unbounded
            far_boost = 1.0 + repulsion_boost
            Layer2Basin._shared_far_basin_depth *= min(far_boost, 1.2)  # Cap at 20% boost
            Layer2Basin._shared_far_basin_depth = min(
                Layer2Basin._shared_far_basin_depth,
                Layer2Basin._shared_cold_basin_depth * 1.5  # Far can't exceed cold by too much
            )
    
    def _apply_dynamic_basin_depth(self, classifier, class_imbalance: float):
        """
        LAW 3: Dynamic Basin Depth (Anti-Monopoly).
        
        When one class dominates, flatten its basin.
        This prevents collapse and maintains exploration.
        """
        if class_imbalance < 0.3:  # Balanced system, no adjustment needed
            return
        
        # Get prediction distribution
        counts = Layer2Basin._shared_prediction_counts
        total = Layer2Basin._shared_total_predictions
        
        if total < 10:
            return
        
        ratios = {k: v / total for k, v in counts.items()}
        max_ratio = max(ratios.values())
        max_class = max(ratios, key=ratios.get)
        
        # If one class dominates too much, flatten it
        if max_ratio > self.dominance_threshold:
            # Dominance factor: how much over threshold
            dominance = (max_ratio - self.dominance_threshold) / (1.0 - self.dominance_threshold)
            dominance = min(dominance, 1.0)  # Cap at 1.0
            
            # Flatten the dominant basin
            if max_class == 'entailment':
                # Flatten cold basin
                flatten_factor = 1.0 - (dominance * 0.2)  # Up to 20% reduction
                Layer2Basin._shared_cold_basin_depth *= flatten_factor
                Layer2Basin._shared_cold_basin_depth = max(0.5, Layer2Basin._shared_cold_basin_depth)
                
                # Boost far and city (exploration)
                Layer2Basin._shared_far_basin_depth *= (1.0 + dominance * 0.1)
                
            elif max_class == 'contradiction':
                # Flatten far basin
                flatten_factor = 1.0 - (dominance * 0.2)
                Layer2Basin._shared_far_basin_depth *= flatten_factor
                Layer2Basin._shared_far_basin_depth = max(0.5, Layer2Basin._shared_far_basin_depth)
                
                # Boost cold (exploration)
                Layer2Basin._shared_cold_basin_depth *= (1.0 + dominance * 0.1)
            
            # Reset counts after adjustment (prevent continuous flattening)
            Layer2Basin._shared_prediction_counts = {'entailment': 0, 'contradiction': 0, 'neutral': 0}
            Layer2Basin._shared_total_predictions = 0

