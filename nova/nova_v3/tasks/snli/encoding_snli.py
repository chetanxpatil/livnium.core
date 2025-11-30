"""
SNLI Encoding: Premise/Hypothesis → Initial State

Builds initial state h0 from premise and hypothesis.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

# Use absolute import so module works when run as a script entry point
from text.encoder import TextEncoder


class SNLIEncoder(nn.Module):
    """
    SNLI-specific encoder.
    
    Takes premise and hypothesis, builds initial state h0.
    Also returns OM and LO vectors for physics computation.
    """
    
    def __init__(self, vocab_size: int, dim: int = 256, pad_idx: int = 0):
        """
        Initialize SNLI encoder.
        
        Args:
            vocab_size: Vocabulary size
            dim: Embedding dimension
            pad_idx: Padding token index
        """
        super().__init__()
        self.dim = dim
        
        # Use the task-agnostic text encoder
        self.text_encoder = TextEncoder(vocab_size, dim, pad_idx)
    
    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode sentence to vector.
        
        Args:
            token_ids: Token IDs tensor (seq_len,)
            
        Returns:
            Sentence vector (dim,)
        """
        return self.text_encoder.encode_sentence(token_ids)
    
    def build_initial_state(self, 
                           prem_ids: torch.Tensor, 
                           hyp_ids: torch.Tensor,
                           add_noise: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build initial state from premise and hypothesis.
        
        Args:
            prem_ids: Premise token IDs (seq_len_p,)
            hyp_ids: Hypothesis token IDs (seq_len_h,)
            add_noise: If True, add symmetry-breaking noise
            
        Returns:
            Tuple of (h0, v_p, v_h)
            - h0: Initial state vector (dim,)
            - v_p: Premise vector (OM) (dim,)
            - v_h: Hypothesis vector (LO) (dim,)
        """
        # Encode premise and hypothesis
        v_p = self.encode_sentence(prem_ids)  # OM vector
        v_h = self.encode_sentence(hyp_ids)   # LO vector
        
        # Build initial state: difference between hypothesis and premise
        # This captures the "semantic gap" between them
        h0 = v_h - v_p
        
        # Optional: add tiny symmetry-breaking noise
        # This ensures OM ≠ LO even for similar inputs
        if add_noise:
            noise = 0.01 * torch.randn_like(h0)
            h0 = h0 + noise
        
        return h0, v_p, v_h
