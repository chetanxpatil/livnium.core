"""
Physics Laws: Core Livnium Laws (Frozen)

These are the fundamental laws that define Livnium's physics.
They operate on vectors, not cells.

Core Law: divergence = 0.38 - alignment
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def alignment(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute alignment (cosine similarity) between two vectors.
    
    Alignment is the core geometric measure:
    - alignment = 1.0 → vectors point in same direction
    - alignment = -1.0 → vectors point in opposite directions
    - alignment = 0.0 → vectors are orthogonal
    
    Args:
        a: First vector (any shape, will be flattened and normalized)
        b: Second vector (same shape as a)
        
    Returns:
        Scalar alignment value (cosine similarity)
    """
    # Flatten and normalize
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    a_n = F.normalize(a_flat, dim=0, p=2)
    b_n = F.normalize(b_flat, dim=0, p=2)
    
    return torch.dot(a_n, b_n)


def divergence_from_alignment(align_value: torch.Tensor) -> torch.Tensor:
    """
    Compute divergence from alignment using the core Livnium law.
    
    Law: divergence = 0.38 - alignment
    
    Physical meaning:
    - divergence < 0 → vectors pull inward (entailment)
    - divergence ≈ 0 → forces balance (neutral)
    - divergence > 0 → vectors push apart (contradiction)
    
    Args:
        align_value: Alignment (cosine similarity) value
        
    Returns:
        Divergence value
    """
    return 0.38 - align_value


def tension(divergence: torch.Tensor) -> torch.Tensor:
    """
    Compute tension from divergence.
    
    Tension measures the "stress" in the geometric field.
    High tension = system is under stress (high divergence magnitude).
    
    Args:
        divergence: Divergence value
        
    Returns:
        Tension (absolute value of divergence)
    """
    return divergence.abs()


def compute_om_lo_physics(om: torch.Tensor, lo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute full physics from OM and LO vectors.
    
    This is the complete physics computation:
    1. Alignment (cosine similarity)
    2. Divergence (0.38 - alignment)
    3. Tension (|divergence|)
    
    Args:
        om: OM vector (premise direction)
        lo: LO vector (hypothesis direction)
        
    Returns:
        Tuple of (alignment, divergence, tension)
    """
    align = alignment(om, lo)
    div = divergence_from_alignment(align)
    tens = tension(div)
    
    return align, div, tens

