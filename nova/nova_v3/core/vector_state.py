"""
Vector State: Core State Representation

The fundamental state container for Livnium Core v1.0.
A single vector h ∈ ℝ^D represents the "energy configuration" of the system.
"""

import torch
from typing import Optional


class VectorState:
    """
    Core state representation for Livnium.
    
    This is the minimal container: just a vector h.
    No cells, no lattices, no tokens.
    """
    
    def __init__(self, dim: int):
        """
        Initialize vector state container.
        
        Args:
            dim: Dimension of the vector space (e.g., 256)
        """
        self.dim = dim
    
    def zero(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Create zero state vector.
        
        Args:
            device: Optional torch device
            
        Returns:
            Zero vector of shape (dim,)
        """
        return torch.zeros(self.dim, device=device)
    
    def normalize(self, h: torch.Tensor) -> torch.Tensor:
        """
        Normalize vector to unit length.
        
        Args:
            h: Input vector
            
        Returns:
            Normalized vector (unit length)
        """
        return h / (h.norm(p=2) + 1e-8)
    
    def random(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Create random state vector (for initialization).
        
        Args:
            device: Optional torch device
            
        Returns:
            Random vector of shape (dim,)
        """
        return torch.randn(self.dim, device=device)
    
    def from_numpy(self, arr, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert numpy array to torch tensor.
        
        Args:
            arr: Numpy array
            device: Optional torch device
            
        Returns:
            Torch tensor
        """
        return torch.from_numpy(arr).to(device=device)

