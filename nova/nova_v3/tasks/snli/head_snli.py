"""
SNLI Head: Classification Head

Takes collapsed state h_final and outputs logits for E/N/C.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNLIHead(nn.Module):
    """
    SNLI classification head.
    
    Adds explicit directional and radial signals:
    - alignment between premise (OM) and hypothesis (LO)
    - opposition between -premise and hypothesis
    - energy feature from alignment (scaled exposure)
    - distance and norms capturing radial geometry
    - neutral alignment signals to give neutral its own basin
    
    Final features: [h_final, alignment, opposition, energy, expose_neg,
                     dist_p_h, r_p, r_h, r_final,
                     align_neutral_p, align_neutral_h] → logits (E, N, C)
    """
    
    def __init__(self, dim: int):
        """
        Initialize SNLI head.
        
        Args:
            dim: Dimension of input state vector
        """
        super().__init__()
        # Learned neutral anchor to give neutral its own geometric signal
        self.neutral_dir = nn.Parameter(torch.randn(dim))
        # Linear stack to allow mild feature interaction
        self.fc = nn.Sequential(
            nn.Linear(dim + 10, dim + 10),
            nn.ReLU(),
            nn.Linear(dim + 10, 3)
        )
        # Learnable scale for neutral alignment contribution
        self.neutral_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, h_final: torch.Tensor, v_p: torch.Tensor, v_h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state → logits.
        
        Args:
            h_final: Collapsed state vector [batch, dim] or [dim]
            v_p: Premise vector (OM) [batch, dim] or [dim]
            v_h: Hypothesis vector (LO) [batch, dim] or [dim]
            
        Returns:
            Logits tensor (batch, 3) for [entailment, neutral, contradiction]
        """
        # Ensure batch dimension
        if h_final.dim() == 1:
            h_final = h_final.unsqueeze(0)
            v_p = v_p.unsqueeze(0)
            v_h = v_h.unsqueeze(0)
        
        # Normalize OM/LO
        v_p_n = F.normalize(v_p, dim=-1)
        v_h_n = F.normalize(v_h, dim=-1)
        neutral_dir_n = F.normalize(self.neutral_dir, dim=0)
        
        # Alignment and opposition signals
        align = (v_p_n * v_h_n).sum(dim=-1, keepdim=True)  # cos(OM, LO)
        opp = (-v_p_n * v_h_n).sum(dim=-1, keepdim=True)   # cos(-OM, LO)
        # Neutral alignments: how close each vector is to the neutral anchor
        align_neutral_p = (v_p_n * neutral_dir_n.unsqueeze(0)).sum(dim=-1, keepdim=True)
        align_neutral_h = (v_h_n * neutral_dir_n.unsqueeze(0)).sum(dim=-1, keepdim=True)
        align_neutral_p = self.neutral_scale * align_neutral_p
        align_neutral_h = self.neutral_scale * align_neutral_h
        
        # Exposure/energy from alignment: map [-1,1] → [0,1], then scale
        energy = 9 * ((1 + align) / 2)
        expose_neg = (1 - align) / 2
        
        # Radial geometry: distance and norms
        dist_p_h = (v_h - v_p).norm(p=2, dim=-1, keepdim=True)
        r_p = v_p.norm(p=2, dim=-1, keepdim=True)
        r_h = v_h.norm(p=2, dim=-1, keepdim=True)
        r_final = h_final.norm(p=2, dim=-1, keepdim=True)
        
        # Minimal, less-redundant feature set:
        # features = torch.cat([h_final, align, dist_p_h, r_final], dim=-1)
        # Using full set for now (alignment/energy/opposition/radials) for continuity.
        features = torch.cat([
            h_final,
            align,
            opp,
            energy,
            expose_neg,
            dist_p_h,
            r_p,
            r_h,
            r_final,
            align_neutral_p,
            align_neutral_h
        ], dim=-1)
        return self.fc(features)
