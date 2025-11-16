"""
Dual Cube Monitor: Lightweight wrapper around DualCubeSystem for diagnostics.

This module provides a clean API for using the dual cube system as a
semantic sensor/thermometer/logger without coupling to specific implementations.

Key principle: dual cube = diagnostic layer, not main engine.
Use it as:
- a sensor: where is meaning / contradiction flowing?
- a thermometer: how decohered / confused is the system?
- a logger: what patterns ended up in the negative cube?
"""

from typing import Tuple, Dict, Optional
import numpy as np
from collections import defaultdict

from quantum.hierarchical.geometry.dual_cube_system import DualCubeSystem


class DualCubeMonitor:
    """
    Lightweight monitor wrapping DualCubeSystem for semantic diagnostics.
    
    This is a read-only diagnostic layer - it observes and reports,
    but doesn't directly control the main system behavior.
    """
    
    def __init__(self, base_dimension: int = 3, num_levels: int = 3):
        """
        Initialize dual cube monitor.
        
        Args:
            base_dimension: Dimension of each cube (3 = 3×3×3 and −3×−3×−3)
            num_levels: Number of hierarchy levels
        """
        self.system = DualCubeSystem(base_dimension=base_dimension, num_levels=num_levels)
        
        # Track coordinate neighborhoods for cancellation detection
        # Grid-binned coordinates to detect overused invalid regions
        self._coordinate_bins: Dict[Tuple[int, ...], Dict[str, int]] = defaultdict(
            lambda: {'positive': 0, 'negative': 0}
        )
        self._bin_size = 0.1  # Grid resolution for binning coordinates
        
    def record_positive(self, coords: Tuple[float, ...], weight: float = 1.0):
        """
        Record a positive (valid/promising) state.
        
        Args:
            coords: Geometric coordinates
            weight: Weight/importance of this recording (e.g., edge count)
        """
        self.system.add_positive_state(coords, amplitude=weight + 0j)
        self._record_bin(coords, 'positive', weight)
    
    def record_negative(self, coords: Tuple[float, ...], weight: float = 1.0):
        """
        Record a negative (invalid/contradictory) state.
        
        Args:
            coords: Geometric coordinates
            weight: Weight/importance of this recording
        """
        self.system.add_negative_state(coords, amplitude=weight + 0j)
        self._record_bin(coords, 'negative', weight)
    
    def record_dual(self, coords: Tuple[float, ...], pos_weight: float, neg_weight: float):
        """
        Record a dual state (exists in both cubes).
        
        Args:
            coords: Geometric coordinates
            pos_weight: Weight in positive cube
            neg_weight: Weight in negative cube
        """
        self.system.add_dual_state(
            coords,
            positive_amplitude=pos_weight + 0j,
            negative_amplitude=neg_weight + 0j
        )
        self._record_bin(coords, 'positive', pos_weight)
        self._record_bin(coords, 'negative', neg_weight)
    
    def _record_bin(self, coords: Tuple[float, ...], cube_type: str, weight: float):
        """Record coordinate in grid bin for cancellation detection."""
        bin_coords = tuple(int(c / self._bin_size) for c in coords)
        self._coordinate_bins[bin_coords][cube_type] += int(weight)
    
    def measure_decoherence(self) -> Dict:
        """
        Measure decoherence: how much has leaked to −cube?
        
        Returns:
            Dictionary with:
            - decoherence_fraction: fraction of energy in negative cube [0, 1]
            - positive_energy: total energy in positive cube
            - negative_energy: total energy in negative cube
        """
        return self.system.get_decoherence_measure()
    
    def diagnose_confusion(self, coords: Tuple[float, ...]) -> Dict:
        """
        Diagnose confusion: check if coordinates live mostly in −cube.
        
        Args:
            coords: Coordinates to check
            
        Returns:
            Dictionary with:
            - confusion_score: how much lives in negative cube [0, 1]
            - diagnosis: 'confused' or 'clear'
            - positive_energy: energy in positive cube near coords
            - negative_energy: energy in negative cube near coords
        """
        return self.system.diagnose_confusion(coords)
    
    def get_cancellation_zones(self, threshold: float = 0.7) -> Dict[Tuple[int, ...], Dict]:
        """
        Get coordinate neighborhoods that are frequently negative, rarely positive.
        
        These are "cancellation zones" - regions where the system keeps
        generating invalid states, suggesting they should be avoided.
        
        Args:
            threshold: Minimum ratio of negative/(negative+positive) to mark as zone
            
        Returns:
            Dictionary mapping bin coordinates to zone info:
            {
                (bin_x, bin_y, bin_z): {
                    'negative_count': int,
                    'positive_count': int,
                    'ratio': float,
                    'is_zone': bool
                }
            }
        """
        zones = {}
        for bin_coords, counts in self._coordinate_bins.items():
            neg_count = counts['negative']
            pos_count = counts['positive']
            total = neg_count + pos_count
            
            if total == 0:
                continue
            
            ratio = neg_count / total if total > 0 else 0.0
            is_zone = ratio >= threshold and neg_count > 5  # Need minimum samples
            
            zones[bin_coords] = {
                'negative_count': neg_count,
                'positive_count': pos_count,
                'ratio': ratio,
                'is_zone': is_zone
            }
        
        return zones
    
    def is_in_cancellation_zone(self, coords: Tuple[float, ...], threshold: float = 0.7) -> bool:
        """
        Check if coordinates fall into a cancellation zone.
        
        Args:
            coords: Coordinates to check
            threshold: Minimum ratio to consider a zone
            
        Returns:
            True if coordinates are in a cancellation zone
        """
        bin_coords = tuple(int(c / self._bin_size) for c in coords)
        zones = self.get_cancellation_zones(threshold)
        
        if bin_coords in zones:
            return zones[bin_coords]['is_zone']
        return False
    
    def get_system_summary(self) -> Dict:
        """
        Get overall system summary.
        
        Returns:
            Dictionary with system state summary
        """
        return self.system.get_system_summary()
    
    def reset(self):
        """Reset monitor state (clear all recordings)."""
        self.system = DualCubeSystem(
            base_dimension=self.system.base_dimension,
            num_levels=self.system.num_levels
        )
        self._coordinate_bins.clear()

