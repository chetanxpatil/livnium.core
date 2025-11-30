"""
Vector Collapse Engine: Multi-Basin Collapse Dynamics

Evolves a state vector h through L steps with multiple anchors (E/C/N) to
encourage three basins. Each anchor uses the Livnium divergence law.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .physics_laws import alignment, divergence_from_alignment, tension


class VectorCollapseEngine(nn.Module):
    """
    Core collapse engine for Livnium.
    
    Takes an initial state h0 and evolves it through L collapse steps with
    multiple anchors (entailment/contradiction/neutral).
    At each step, it:
    1. Computes alignment/divergence/tension to each anchor
    2. Applies state update + anchor forces
    3. Logs trace
    
    The trace is what watchdogs inspect.
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 6,
        strength_entail: float = 0.1,
        strength_contra: float = 0.1,
        strength_neutral: float = 0.05,
    ):
        """
        Initialize collapse engine.
        
        Args:
            dim: Dimension of state vector
            num_layers: Number of collapse steps
            strength_entail: Force strength for entail anchor
            strength_contra: Force strength for contradiction anchor
            strength_neutral: Force strength for neutral anchor
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.strength_entail = strength_entail
        self.strength_contra = strength_contra
        self.strength_neutral = strength_neutral
        
        # State update network
        # This learns how to evolve the state based on current configuration
        self.update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )
        
        # Three anchors to create multi-basin geometry
        self.anchor_entail = nn.Parameter(torch.randn(dim))
        self.anchor_contra = nn.Parameter(torch.randn(dim))
        self.anchor_neutral = nn.Parameter(torch.randn(dim))
    
    def collapse(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Collapse initial state h0 through L steps.
        
        Args:
            h0: Initial state vector (dim,) or batch of vectors [B, dim]
            
        Returns:
            Tuple of (h_final, trace)
            trace: Dict with per-anchor align/div/tension lists
        """
        squeeze = False
        h = h0
        if h.dim() == 1:
            h = h.unsqueeze(0)
            squeeze = True
        h = h.clone()
        trace = {
            "alignment_entail": [],
            "alignment_contra": [],
            "alignment_neutral": [],
            "divergence_entail": [],
            "divergence_contra": [],
            "divergence_neutral": [],
            "tension_entail": [],
            "tension_contra": [],
            "tension_neutral": [],
        }
        
        # Normalize anchor directions
        e_dir = F.normalize(self.anchor_entail, dim=0)
        c_dir = F.normalize(self.anchor_contra, dim=0)
        n_dir = F.normalize(self.anchor_neutral, dim=0)
        
        for step in range(self.num_layers):
            # Normalize current state along feature dim
            h_n = F.normalize(h, dim=-1)
            
            # Compute physics to each anchor
            a_e = (h_n * e_dir).sum(dim=-1)
            a_c = (h_n * c_dir).sum(dim=-1)
            a_n = (h_n * n_dir).sum(dim=-1)
            d_e = divergence_from_alignment(a_e)
            d_c = divergence_from_alignment(a_c)
            d_n = divergence_from_alignment(a_n)
            t_e = tension(d_e)
            t_c = tension(d_c)
            t_n = tension(d_n)
            
            # Log trace
            trace["alignment_entail"].append(a_e.detach())
            trace["alignment_contra"].append(a_c.detach())
            trace["alignment_neutral"].append(a_n.detach())
            trace["divergence_entail"].append(d_e.detach())
            trace["divergence_contra"].append(d_c.detach())
            trace["divergence_neutral"].append(d_n.detach())
            trace["tension_entail"].append(t_e.detach())
            trace["tension_contra"].append(t_c.detach())
            trace["tension_neutral"].append(t_n.detach())
            
            # State update
            delta = self.update(h)
            # Anchor forces: move toward/away each anchor along their difference vector
            e_vec = F.normalize(h - e_dir.unsqueeze(0), dim=-1)
            c_vec = F.normalize(h - c_dir.unsqueeze(0), dim=-1)
            n_vec = F.normalize(h - n_dir.unsqueeze(0), dim=-1)
            h = (
                h
                + delta
                - self.strength_entail * d_e.unsqueeze(-1) * e_vec
                - self.strength_contra * d_c.unsqueeze(-1) * c_vec
                - self.strength_neutral * d_n.unsqueeze(-1) * n_vec
            )
            
            # Soft norm control (conservation-ish)
            # Keep ||h|| roughly bounded to prevent explosion
            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)
        
        if squeeze:
            h = h.squeeze(0)
        return h, trace
    
    def forward(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Forward pass (alias for collapse).
        
        Args:
            h0: Initial state vector
            
        Returns:
            Tuple of (h_final, trace)
        """
        return self.collapse(h0)
