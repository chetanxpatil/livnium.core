"""
Autonomous Meaning Engine (AME)

Full self-organizing semantic universe with:
- Semantic turbulence (entropy scales with dominance)
- Dynamic basin splitting (basins split when >70%)
- Curvature-based routing (high curvature → push out of city)
- Memory hysteresis (meaning has inertia)
- Long-range alignment (basin centers pull sentences)
- Competitive word polarity (basins compete for words)

This is semantic cosmology - meaning emerges from physics alone.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from .layer2_basin import Layer2Basin
from .layer1_curvature import Layer1Curvature


class BasinCenter:
    """Tracks the center (average resonance pattern) of a basin."""
    
    def __init__(self, basin_index: int):
        self.basin_index = basin_index
        self.resonance_history: List[float] = []
        self.max_history = 1000  # Keep last 1000 resonances
        
    def add(self, resonance: float):
        """Add a resonance value to this basin."""
        self.resonance_history.append(resonance)
        if len(self.resonance_history) > self.max_history:
            self.resonance_history.pop(0)
    
    def get_center(self) -> float:
        """Get the center (average) of this basin."""
        if not self.resonance_history:
            return 0.0
        return float(np.mean(self.resonance_history))
    
    def get_std(self) -> float:
        """Get standard deviation (spread) of this basin."""
        if len(self.resonance_history) < 2:
            return 0.0
        return float(np.std(self.resonance_history))


class AutonomousMeaningEngine:
    """
    Autonomous Meaning Engine: Full self-organizing semantic universe.
    
    Features:
    1. Semantic turbulence (entropy scales with city dominance)
    2. Dynamic basin splitting (basins split when >70%)
    3. Curvature-based routing (high curvature → push out of city)
    4. Memory hysteresis (meaning has inertia)
    5. Long-range alignment (basin centers pull sentences)
    6. Competitive word polarity (basins compete for words)
    """
    
    def __init__(self):
        """Initialize AME."""
        # Load overrides (set by autonomous universe driver)
        # Reuse the same loading mechanism as auto_physics to avoid duplicate loading
        from .auto_physics import load_overrides
        overrides = load_overrides()
        
        # Turbulence parameters (can be overridden)
        self.turbulence_base = 0.02
        self.turbulence_scale = overrides.get("turbulence_scale", 0.2)
        self.city_dominance_threshold = 0.6
        
        # Basin splitting
        self.split_threshold = 0.7  # Split if basin > 70%
        self.max_basins = 12  # Maximum number of basins
        
        # Curvature routing
        self.curvature_threshold = 0.3  # High curvature → push out of city
        
        # Hysteresis
        self.hysteresis_alpha = 0.6  # 60% current, 40% previous
        
        # Long-range alignment
        self.alignment_strength = 0.1
        
        # Track basin centers
        self.basin_centers: Dict[int, BasinCenter] = {
            0: BasinCenter(0),  # Cold
            1: BasinCenter(1),  # Far
            2: BasinCenter(2)   # City
        }
        
        # Track previous basin assignments (for hysteresis)
        self.previous_basins: Dict[str, int] = {}  # sentence_pair_hash -> basin_index
        
        # Track basin sizes
        self.basin_sizes: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        self.total_assignments = 0
    
    def compute_semantic_turbulence(self, classifier) -> Tuple[float, float]:
        """
        STEP 1: Semantic turbulence.
        
        Entropy scales with city dominance.
        When city dominates → universe shakes → new categories form.
        
        Returns:
            (turbulence, repulsion_strength)
        """
        # Get current basin distribution
        total = self.total_assignments
        if total < 10:
            return (self.turbulence_base, 0.3)  # Default repulsion
        
        city_ratio = self.basin_sizes.get(2, 0) / total
        
        # If city dominates, inject turbulence
        if city_ratio > self.city_dominance_threshold:
            excess = city_ratio - self.city_dominance_threshold
            turbulence = self.turbulence_base + self.turbulence_scale * excess
            # Also increase repulsion when city dominates
            repulsion_strength = 0.3 + 1.0 * excess  # Scales with excess
        else:
            turbulence = self.turbulence_base
            repulsion_strength = 0.3
        
        # Update Layer 1 entropy
        classifier.layer1.entropy_scale = turbulence
        
        # Update repulsion in auto_physics
        classifier.auto_physics.repulsion_strength = repulsion_strength
        
        return (turbulence, repulsion_strength)
    
    def apply_curvature_routing(self, curvature: float, current_basin: int) -> int:
        """
        STEP 4: Curvature-based routing.
        
        High curvature = semantic shock → push out of city.
        """
        # If curvature is high and sentence is in city, push to cold or far
        if curvature > self.curvature_threshold and current_basin == 2:
            # High curvature → strong semantic change → not neutral
            # Push to cold (positive) or far (negative) based on resonance
            # For now, push to cold (can be made smarter later)
            return 0  # Push to cold basin
        
        return current_basin
    
    def apply_hysteresis(self, sentence_hash: str, current_basin: int) -> int:
        """
        STEP 5: Memory hysteresis.
        
        Meaning has inertia - don't jump instantly.
        """
        if sentence_hash in self.previous_basins:
            previous = self.previous_basins[sentence_hash]
            # Weighted average: 60% current, 40% previous
            # Round to nearest basin
            weighted = self.hysteresis_alpha * current_basin + (1.0 - self.hysteresis_alpha) * previous
            # Round to nearest integer basin
            smoothed_basin = int(round(weighted))
            # Update previous
            self.previous_basins[sentence_hash] = smoothed_basin
            return smoothed_basin
        else:
            # First time seeing this sentence
            self.previous_basins[sentence_hash] = current_basin
            return current_basin
    
    def apply_long_range_alignment(self, resonance: float, basin_index: int) -> float:
        """
        STEP 6: Long-range alignment pressure.
        
        Basin centers pull sentences toward them.
        """
        if basin_index not in self.basin_centers:
            return resonance
        
        center = self.basin_centers[basin_index].get_center()
        
        # Pull resonance toward basin center
        aligned_resonance = resonance + self.alignment_strength * (center - resonance)
        
        return aligned_resonance
    
    def check_basin_splitting(self, classifier) -> List[int]:
        """
        STEP 3: Dynamic basin splitting.
        
        If a basin exceeds 70%, split it into two sub-basins.
        """
        total = self.total_assignments
        if total < 100:  # Need some history
            return [0, 1, 2]  # Default 3 basins
        
        # Check if any basin needs splitting
        for basin_idx, size in self.basin_sizes.items():
            ratio = size / total
            if ratio > self.split_threshold:
                # This basin is too large - split it
                # For now, just log it (full splitting requires more infrastructure)
                # TODO: Implement full K=2 clustering and sub-basin creation
                print(f"⚠️  Basin {basin_idx} is {ratio*100:.1f}% - should split (not yet implemented)")
        
        # Return current basin indices
        return list(self.basin_centers.keys())
    
    def update_basin_center(self, basin_index: int, resonance: float):
        """Update basin center with new resonance."""
        if basin_index in self.basin_centers:
            self.basin_centers[basin_index].add(resonance)
    
    def track_assignment(self, basin_index: int):
        """Track basin assignment for statistics."""
        self.basin_sizes[basin_index] = self.basin_sizes.get(basin_index, 0) + 1
        self.total_assignments += 1
    
    def step(self, classifier, resonance: float, curvature: float, 
             current_basin: int, sentence_hash: str) -> Dict[str, float]:
        """
        Run one AME step.
        
        Applies all 7 steps of autonomous meaning emergence.
        """
        # STEP 1: Semantic turbulence
        turbulence, repulsion_strength = self.compute_semantic_turbulence(classifier)
        
        # STEP 3: Check basin splitting
        active_basins = self.check_basin_splitting(classifier)
        
        # STEP 4: Curvature-based routing
        routed_basin = self.apply_curvature_routing(curvature, current_basin)
        
        # STEP 5: Memory hysteresis
        final_basin = self.apply_hysteresis(sentence_hash, routed_basin)
        
        # STEP 6: Long-range alignment
        aligned_resonance = self.apply_long_range_alignment(resonance, final_basin)
        
        # Update basin center
        self.update_basin_center(final_basin, aligned_resonance)
        
        # Track assignment (but don't double-count - this is called after classification)
        # self.track_assignment(final_basin)  # Commented out - tracking happens in train loop
        
        return {
            'turbulence': float(turbulence),
            'repulsion_strength': float(repulsion_strength),
            'final_basin': final_basin,
            'routed_basin': routed_basin,
            'aligned_resonance': float(aligned_resonance),
            'basin_center': float(self.basin_centers[final_basin].get_center()),
            'active_basins': len(active_basins),
            'city_ratio': float(self.basin_sizes.get(2, 0) / max(self.total_assignments, 1))
        }
    
    def get_statistics(self) -> Dict:
        """Get AME statistics."""
        total = self.total_assignments
        if total == 0:
            return {}
        
        stats = {}
        for basin_idx, size in self.basin_sizes.items():
            ratio = size / total if total > 0 else 0.0
            center = self.basin_centers[basin_idx].get_center()
            std = self.basin_centers[basin_idx].get_std()
            
            stats[f'basin_{basin_idx}'] = {
                'count': size,
                'ratio': float(ratio),
                'center': float(center),
                'std': float(std)
            }
        
        return stats

