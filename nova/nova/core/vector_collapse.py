"""
Vector Collapse: Nonlinear Transform Functions

Pure vector-based collapse without 3D lattice.
Creates basins, stability, and nonlinearity through algebraic transforms.
"""

import numpy as np
from typing import Callable


def tanh_collapse(vector: np.ndarray, steps: int = 12, strength: float = 1.0) -> np.ndarray:
    """
    Tanh collapse: Creates smooth basins.
    
    v' = tanh(v * (1 + step * strength))
    
    Properties:
    - Smooth, differentiable
    - Creates soft basins
    - Preserves sign
    """
    v = vector.copy()
    
    for step in range(steps):
        # Apply tanh with increasing strength
        scale = 1.0 + step * strength * 0.1
        v = np.tanh(v * scale)
        
        # Add thermal jitter
        jitter_scale = 0.01 * (1.0 - step / max(steps, 1))
        if jitter_scale > 0:
            jitter = np.random.normal(0, jitter_scale, v.shape)
            v = v + jitter
        
        # Normalize
        v = v / (np.linalg.norm(v) + 1e-10)
    
    return v


def power_collapse(vector: np.ndarray, steps: int = 12, power: float = 3.0) -> np.ndarray:
    """
    Power collapse: Creates sharper basins.
    
    v' = sign(v) * |v|^p
    
    Properties:
    - Sharper basins than tanh
    - Preserves sign
    - Creates stronger attractors
    """
    v = vector.copy()
    
    for step in range(steps):
        # Apply element-wise power
        p = power * (1.0 + step * 0.05)
        v = np.sign(v) * np.power(np.abs(v), p)
        
        # Add thermal jitter
        jitter_scale = 0.01 * (1.0 - step / max(steps, 1))
        if jitter_scale > 0:
            jitter = np.random.normal(0, jitter_scale, v.shape)
            v = v + jitter
        
        # Normalize
        v = v / (np.linalg.norm(v) + 1e-10)
    
    return v


def relu_collapse(vector: np.ndarray, steps: int = 12, decay: float = 0.98) -> np.ndarray:
    """
    ReLU collapse: Creates sparse basins.
    
    v' = ReLU(v) * decay^step
    
    Properties:
    - Creates sparse representations
    - Gentle decay preserves patterns
    - Good for feature selection
    """
    v = vector.copy()
    
    for step in range(steps):
        # Apply ReLU with decay
        v = np.maximum(0, v) * (decay ** step)
        
        # Add thermal jitter
        jitter_scale = 0.01 * (1.0 - step / max(steps, 1))
        if jitter_scale > 0:
            jitter = np.random.normal(0, jitter_scale, v.shape)
            v = v + jitter
        
        # Normalize
        v = v / (np.linalg.norm(v) + 1e-10)
    
    return v


def sigmoid_collapse(vector: np.ndarray, steps: int = 12, steepness: float = 1.0) -> np.ndarray:
    """
    Sigmoid collapse: Creates smooth S-curve basins.
    
    v' = sigmoid(v * steepness)
    
    Properties:
    - Smooth, bounded
    - Creates soft thresholds
    - Good for probability-like outputs
    """
    v = vector.copy()
    
    for step in range(steps):
        # Apply sigmoid with increasing steepness
        scale = steepness * (1.0 + step * 0.1)
        v = 1.0 / (1.0 + np.exp(-v * scale))
        
        # Add thermal jitter
        jitter_scale = 0.01 * (1.0 - step / max(steps, 1))
        if jitter_scale > 0:
            jitter = np.random.normal(0, jitter_scale, v.shape)
            v = v + jitter
        
        # Normalize
        v = v / (np.linalg.norm(v) + 1e-10)
    
    return v


def get_collapse_function(collapse_type: str) -> Callable:
    """Get collapse function by name."""
    collapse_functions = {
        'tanh': tanh_collapse,
        'power3': lambda v, s: power_collapse(v, s, power=3.0),
        'power5': lambda v, s: power_collapse(v, s, power=5.0),
        'relu': relu_collapse,
        'sigmoid': sigmoid_collapse
    }
    
    return collapse_functions.get(collapse_type, tanh_collapse)

