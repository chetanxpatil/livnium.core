#!/usr/bin/env python3
"""
Livnium Force Module (Phase 6 - Minimal Version)

This is a minimal geometric influence operator for Rule 30 Shadow.
It is NOT the full Livnium system - it's a simple steering function.

Purpose: Add a small 8D bias vector to guide PCA trajectory.
"""

import numpy as np
from typing import Optional, Union


class LivniumForce:
    """
    Minimal Livnium geometric influence operator.
    
    Applies a small directional bias to PCA states:
        y_t' = y_t + Ω(y_t)
    
    Where Ω is either:
    - A fixed 8×8 matrix: Ω(y_t) = matrix @ y_t
    - A fixed 8D vector: Ω(y_t) = vector
    - A learned function (future extension)
    """
    
    def __init__(self, 
                 force_scale: float = 0.01,
                 force_type: str = 'vector',
                 force_matrix: Optional[np.ndarray] = None,
                 force_vector: Optional[np.ndarray] = None,
                 n_components: int = 8):
        """
        Initialize Livnium force.
        
        Args:
            force_scale: Scaling factor for the force (default: 0.01 = 1% influence)
            force_type: Type of force - 'matrix', 'vector', or 'learned'
            force_matrix: Optional 8×8 matrix (if force_type='matrix')
            force_vector: Optional 8D vector (if force_type='vector')
            n_components: Dimensionality of PCA space (default: 8)
        """
        self.force_scale = force_scale
        self.force_type = force_type
        self.n_components = n_components
        
        if force_type == 'matrix':
            if force_matrix is not None:
                assert force_matrix.shape == (n_components, n_components), \
                    f"Matrix must be {n_components}×{n_components}"
                self.force_matrix = force_matrix
            else:
                # Default: small random matrix
                np.random.seed(42)
                self.force_matrix = np.random.randn(n_components, n_components) * 0.1
        elif force_type == 'vector':
            if force_vector is not None:
                assert force_vector.shape == (n_components,), \
                    f"Vector must be shape ({n_components},)"
                self.force_vector = force_vector
            else:
                # Default: small random vector
                np.random.seed(42)
                self.force_vector = np.random.randn(n_components) * 0.1
        else:
            raise ValueError(f"Unknown force_type: {force_type}")
    
    def apply_livnium_force(self, y_t: np.ndarray) -> np.ndarray:
        """
        Apply Livnium geometric influence to PCA state.
        
        This is the main function that Cursor will call.
        
        Args:
            y_t: Current PCA coordinates of shape (n_components,)
        
        Returns:
            8D bias vector to add to y_t
        """
        y_t = y_t.flatten()
        assert y_t.shape == (self.n_components,), \
            f"y_t must be shape ({self.n_components},), got {y_t.shape}"
        
        if self.force_type == 'matrix':
            # Matrix multiplication: Ω(y_t) = matrix @ y_t
            bias = self.force_matrix @ y_t
        elif self.force_type == 'vector':
            # Constant vector: Ω(y_t) = vector
            bias = self.force_vector.copy()
        else:
            raise ValueError(f"Unknown force_type: {self.force_type}")
        
        # Scale the force
        bias = bias * self.force_scale
        
        return bias
    
    def set_force_scale(self, scale: float):
        """Update the force scaling factor."""
        self.force_scale = scale
    
    def get_force_scale(self) -> float:
        """Get current force scaling factor."""
        return self.force_scale


def create_default_livnium(n_components: int = 8, 
                           force_scale: float = 0.01,
                           force_type: str = 'vector') -> LivniumForce:
    """
    Create a default Livnium force instance.
    
    This is a convenience function for quick setup.
    """
    return LivniumForce(
        force_scale=force_scale,
        force_type=force_type,
        n_components=n_components
    )

