"""
Text Encoder: Task-Agnostic Text Encoding

Converts tokens to vectors. Simple and clean.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import re


class TextEncoder(nn.Module):
    """
    Simple text encoder: tokens → embeddings → sentence vector.
    
    This is task-agnostic. It doesn't know about SNLI or dialogue.
    It just converts text to vectors.
    """
    
    def __init__(self, vocab_size: int, dim: int = 256, pad_idx: int = 0):
        """
        Initialize text encoder.
        
        Args:
            vocab_size: Vocabulary size
            dim: Embedding dimension
            pad_idx: Padding token index
        """
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        
        # Simple embedding layer
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: split on whitespace and punctuation.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Simple regex-based tokenization
        pattern = r"(\w+|\s+|[^\w\s])"
        tokens = [t for t in re.split(pattern, text) if t.strip()]
        return tokens
    
    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode sentence from token IDs to vector.
        
        Simple average pooling for now.
        
        Args:
            token_ids: Token IDs tensor (seq_len,)
            
        Returns:
            Sentence vector (dim,)
        """
        # Get embeddings
        emb = self.embed(token_ids)  # [seq_len, dim] or [batch, seq_len, dim]
        
        # Average pooling (simple baseline)
        # Mask out padding
        mask = (token_ids != self.pad_idx).float().unsqueeze(-1)  # [seq_len, 1] or [batch, seq_len, 1]
        masked_emb = emb * mask
        
        # Sum over sequence dimension
        if token_ids.dim() == 1:
            sum_emb = masked_emb.sum(dim=0)  # [dim]
            count = mask.sum(dim=0).clamp(min=1.0)  # [dim]
        else:
            sum_emb = masked_emb.sum(dim=1)  # [batch, dim]
            count = mask.sum(dim=1).clamp(min=1.0)  # [batch, 1]
        
        return sum_emb / count  # [dim] or [batch, dim]
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (alias for encode_sentence).
        
        Args:
            token_ids: Token IDs tensor
            
        Returns:
            Sentence vector
        """
        return self.encode_sentence(token_ids)
