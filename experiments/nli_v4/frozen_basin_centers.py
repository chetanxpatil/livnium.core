"""
Frozen Basin Centers: Stabilize the Three-Phase Universe

Freezes the center of each class (Cold=E, Far=C, City=N) by averaging
across cycles. This stabilizes meaning and prevents drift.

The key insight: Labels are just names for stable attractors.
By freezing the attractor centers, we create a self-stabilizing system.
"""

import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict


class FrozenBasinCenters:
    """
    Freezes basin centers to stabilize the three-phase universe.
    
    Every cycle, computes:
    - cold_center = mean(vector representations of Cold sentences)
    - far_center = mean(vector representations of Far sentences)
    - city_center = mean(vector representations of City sentences)
    
    Then stores frozen centers and uses them in routing.
    """
    
    def __init__(self, vector_size: int = 27, ema_alpha: float = 0.1):
        """
        Initialize frozen basin centers.
        
        Args:
            vector_size: Size of sentence vectors
            ema_alpha: Exponential moving average decay factor (0.1 = slow update)
        """
        self.vector_size = vector_size
        self.ema_alpha = ema_alpha
        
        # Frozen centers (normalized vectors)
        self.frozen_cold_center: Optional[np.ndarray] = None
        self.frozen_far_center: Optional[np.ndarray] = None
        self.frozen_city_center: Optional[np.ndarray] = None
        
        # Track counts for initialization
        self.cold_count = 0
        self.far_count = 0
        self.city_count = 0
        
        # Temporary storage for current cycle vectors
        self.current_cycle_vectors: Dict[int, List[np.ndarray]] = {
            0: [],  # Cold
            1: [],  # Far
            2: []   # City
        }
    
    def add_vector(self, basin_index: int, sentence_vector: np.ndarray):
        """
        Add a sentence vector to the current cycle's collection.
        
        Args:
            basin_index: Which basin (0=Cold, 1=Far, 2=City)
            sentence_vector: Normalized sentence vector representation
        """
        if basin_index in [0, 1, 2]:
            # Normalize vector
            norm = np.linalg.norm(sentence_vector)
            if norm > 0:
                normalized = sentence_vector / norm
            else:
                normalized = sentence_vector
            
            self.current_cycle_vectors[basin_index].append(normalized)
    
    def update_frozen_centers(self):
        """
        Update frozen centers using EMA (Exponential Moving Average).
        
        Called at the end of each cycle to stabilize the attractors.
        """
        # Compute mean vectors for this cycle
        for basin_idx in [0, 1, 2]:
            vectors = self.current_cycle_vectors[basin_idx]
            
            if not vectors:
                continue
            
            # Compute mean vector for this cycle
            cycle_mean = np.mean(vectors, axis=0)
            norm = np.linalg.norm(cycle_mean)
            if norm > 0:
                cycle_mean = cycle_mean / norm
            
            # Update frozen center using EMA
            if basin_idx == 0:  # Cold
                if self.frozen_cold_center is None:
                    self.frozen_cold_center = cycle_mean.copy()
                    self.cold_count = len(vectors)
                else:
                    # EMA update: new = alpha * cycle_mean + (1 - alpha) * old
                    self.frozen_cold_center = (
                        self.ema_alpha * cycle_mean + 
                        (1.0 - self.ema_alpha) * self.frozen_cold_center
                    )
                    # Renormalize
                    norm = np.linalg.norm(self.frozen_cold_center)
                    if norm > 0:
                        self.frozen_cold_center = self.frozen_cold_center / norm
                    self.cold_count += len(vectors)
            
            elif basin_idx == 1:  # Far
                if self.frozen_far_center is None:
                    self.frozen_far_center = cycle_mean.copy()
                    self.far_count = len(vectors)
                else:
                    self.frozen_far_center = (
                        self.ema_alpha * cycle_mean + 
                        (1.0 - self.ema_alpha) * self.frozen_far_center
                    )
                    norm = np.linalg.norm(self.frozen_far_center)
                    if norm > 0:
                        self.frozen_far_center = self.frozen_far_center / norm
                    self.far_count += len(vectors)
            
            else:  # City
                if self.frozen_city_center is None:
                    self.frozen_city_center = cycle_mean.copy()
                    self.city_count = len(vectors)
                else:
                    self.frozen_city_center = (
                        self.ema_alpha * cycle_mean + 
                        (1.0 - self.ema_alpha) * self.frozen_city_center
                    )
                    norm = np.linalg.norm(self.frozen_city_center)
                    if norm > 0:
                        self.frozen_city_center = self.frozen_city_center / norm
                    self.city_count += len(vectors)
        
        # Clear current cycle vectors for next cycle
        self.current_cycle_vectors = {0: [], 1: [], 2: []}
    
    def get_frozen_center(self, basin_index: int) -> Optional[np.ndarray]:
        """
        Get frozen center for a basin.
        
        Args:
            basin_index: Which basin (0=Cold, 1=Far, 2=City)
            
        Returns:
            Frozen center vector or None if not initialized
        """
        if basin_index == 0:
            return self.frozen_cold_center.copy() if self.frozen_cold_center is not None else None
        elif basin_index == 1:
            return self.frozen_far_center.copy() if self.frozen_far_center is not None else None
        elif basin_index == 2:
            return self.frozen_city_center.copy() if self.frozen_city_center is not None else None
        return None
    
    def compute_attraction_to_frozen_center(self, sentence_vector: np.ndarray, 
                                            basin_index: int) -> float:
        """
        Compute attraction strength to frozen center.
        
        Args:
            sentence_vector: Normalized sentence vector
            basin_index: Which basin to compute attraction to
            
        Returns:
            Attraction strength (0 to 1)
        """
        frozen_center = self.get_frozen_center(basin_index)
        if frozen_center is None:
            return 0.0
        
        # Normalize input vector
        norm = np.linalg.norm(sentence_vector)
        if norm > 0:
            normalized = sentence_vector / norm
        else:
            normalized = sentence_vector
        
        # Compute cosine similarity (attraction)
        similarity = np.dot(normalized, frozen_center)
        
        # Convert to attraction strength (0 to 1)
        attraction = (similarity + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        
        return float(attraction)
    
    def get_statistics(self) -> Dict:
        """Get statistics about frozen centers."""
        return {
            'cold_center_initialized': self.frozen_cold_center is not None,
            'far_center_initialized': self.frozen_far_center is not None,
            'city_center_initialized': self.frozen_city_center is not None,
            'cold_count': self.cold_count,
            'far_count': self.far_count,
            'city_count': self.city_count,
            'ema_alpha': self.ema_alpha
        }
    
    def reset(self):
        """Reset all frozen centers (for clean start)."""
        self.frozen_cold_center = None
        self.frozen_far_center = None
        self.frozen_city_center = None
        self.cold_count = 0
        self.far_count = 0
        self.city_count = 0
        self.current_cycle_vectors = {0: [], 1: [], 2: []}

