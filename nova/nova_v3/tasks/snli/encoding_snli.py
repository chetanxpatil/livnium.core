"""
SNLI Encoding: Premise/Hypothesis → Initial State

Builds initial state h0 from premise and hypothesis.
"""

import torch
import torch.nn as nn
from typing import List, Sequence, Tuple, Optional

# Use absolute import so module works when run as a script entry point
from text.encoder import TextEncoder
from text.geom_encoder import GeometricTextEncoder, tokenize
from text import SanskritTextEncoder
from quantum_embed.text_encoder_quantum import QuantumTextEncoder


class SNLIEncoder(nn.Module):
    """
    SNLI-specific encoder.
    
    Takes premise and hypothesis, builds initial state h0.
    Also returns OM and LO vectors for physics computation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        pad_idx: int = 0,
    ):
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
        self.text_encoder = TextEncoder(
            vocab_size,
            dim,
            pad_idx,
        )
    
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


class SanskritSNLIEncoder(nn.Module):
    """
    SNLI encoder backed by the Sanskrit phoneme geometry TextEncoder (no embeddings).
    """

    def __init__(self, vocab_size: int, dim: int = 256, pad_idx: int = 0, id_to_token=None):
        super().__init__()
        self.dim = dim
        self.text_encoder = SanskritTextEncoder(
            vocab_size=vocab_size,
            dim=dim,
            pad_idx=pad_idx,
            id_to_token=id_to_token,
        )

    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder.encode_sentence(token_ids)

    def build_initial_state(
        self, prem_ids: torch.Tensor, hyp_ids: torch.Tensor, add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_p = self.encode_sentence(prem_ids)
        v_h = self.encode_sentence(hyp_ids)
        h0 = v_h - v_p
        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)
        return h0, v_p, v_h


class GeometricSNLIEncoder(nn.Module):
    """
    SNLI encoder that uses the geometric text encoder (no embedding table).
    Operates directly on raw text strings.
    """

    def __init__(
        self,
        dim: int = 256,
        norm_target: float = None,
        use_transformer: bool = True,
        nhead: int = 4,
        num_layers: int = 1,
        ff_mult: int = 2,
        dropout: float = 0.1,
        use_attention_pooling: bool = True,
        token_norm_cap: float = 3.0,
    ):
        super().__init__()
        self.dim = dim
        self.geom_encoder = GeometricTextEncoder(
            dim=dim,
            norm_target=norm_target,
            use_transformer=use_transformer,
            nhead=nhead,
            num_layers=num_layers,
            ff_mult=ff_mult,
            dropout=dropout,
            use_attention_pooling=use_attention_pooling,
            token_norm_cap=token_norm_cap,
        )

    def encode_sentence(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        """
        Encode a batch of texts to vectors.
        """
        tokenized = [tokenize(t) for t in texts]
        return self.geom_encoder.encode_batch(tokenized, device=device)

    def build_initial_state(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        device: torch.device,
        add_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build initial state from raw text sequences.
        """
        v_p = self.encode_sentence(premises, device)  # OM vectors
        v_h = self.encode_sentence(hypotheses, device)  # LO vectors

        h0 = v_h - v_p
        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)
        return h0, v_p, v_h


class QuantumSNLIEncoder(nn.Module):
    """
    SNLI encoder that uses the pretrained Livnium quantum embeddings.
    """

    def __init__(self, ckpt_path: str):
        super().__init__()
        self.text_encoder = QuantumTextEncoder(ckpt_path)
        self.dim = self.text_encoder.dim
        self.pad_idx = self.text_encoder.pad_idx

    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder.encode_sentence(token_ids)

    def build_initial_state(
        self, prem_ids: torch.Tensor, hyp_ids: torch.Tensor, add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_p = self.encode_sentence(prem_ids)
        v_h = self.encode_sentence(hyp_ids)
        h0 = v_h - v_p
        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)
        return h0, v_p, v_h
