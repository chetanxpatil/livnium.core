"""
Geometric Text Encoder (Nova-3 style) with lightweight context.

Pure-geometry token conversion:
- Token → deterministic base-27 integer signature
- Integer → 3D coordinate in {-1, 0, +1}^3
- Derived features: exposure, distance, polarity
- Project 6D feature → model space with a small MLP + tanh
- Optional transformer layer for token interaction
- Attention or masked mean pooling with norm control
"""

import math
import re
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def tokenize(text: str) -> List[str]:
    """
    Simple, deterministic tokenizer.
    Splits on whitespace and keeps alphabetic/apostrophe spans.
    """
    return re.findall(r"[A-Za-z']+", text.lower())


class GeometricTextEncoder(nn.Module):
    """
    Geometry-first encoder: no embeddings, no lookup tables.
    Converts tokens to geometric feature vectors and projects to ℝ^D with light context.
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
        """
        Args:
            dim: Projection dimension (matches collapse engine dim)
            norm_target: Target L2 norm for sentence vectors (default sqrt(dim))
            use_transformer: If True, apply a small Transformer encoder over token projections
            nhead: Attention heads for the transformer
            num_layers: Number of transformer encoder layers
            ff_mult: Multiplier for feedforward hidden size inside transformer
            dropout: Dropout applied after projection and inside transformer
            use_attention_pooling: If True, pool with learned attention; else masked mean
            token_norm_cap: Clamp per-token projection norm to this value (None to disable)
        """
        super().__init__()
        self.dim = dim
        self.norm_target = norm_target if norm_target is not None else dim**0.5
        self.use_transformer = use_transformer
        self.use_attention_pooling = use_attention_pooling
        self.token_norm_cap = token_norm_cap

        # Projection MLP from 6D geometry to model space
        self.proj = nn.Sequential(
            nn.Linear(6, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)

        # Optional transformer encoder for token interaction
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=nhead,
                dim_feedforward=ff_mult * dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.transformer = None

        # Attention pooling scorer
        if use_attention_pooling:
            self.attn_score = nn.Linear(dim, 1)

    @staticmethod
    def _char_value(ch: str) -> int:
        """
        Map character to base-27 digit (a-z -> 1..26, apostrophe -> 0).
        Unknown characters map to 0.
        """
        if ch == "'":
            return 0
        if "a" <= ch <= "z":
            return ord(ch) - ord("a") + 1
        return 0

    @classmethod
    def token_signature(cls, token: str) -> int:
        """
        Deterministic base-27 integer signature for a token.
        """
        sig = 0
        for i, ch in enumerate(token):
            sig += cls._char_value(ch) * (27**i)
        return sig

    @staticmethod
    def integer_to_coords(sig: int) -> Tuple[int, int, int]:
        """
        Map integer signature to 3D coordinates in {-1, 0, +1}.
        """
        x = ((sig // 9) % 3) - 1
        y = ((sig // 3) % 3) - 1
        z = (sig % 3) - 1
        return x, y, z

    @staticmethod
    def _token_feature(sig: int) -> Tuple[float, float, float, float, float, float]:
        """
        Compute 6D geometric feature for a token signature.
        """
        x, y, z = GeometricTextEncoder.integer_to_coords(sig)
        exposure = (abs(x) == 1) + (abs(y) == 1) + (abs(z) == 1)
        exposure = exposure / 3.0
        dist = (x * x + y * y + z * z) ** 0.5
        dist /= 3 ** 0.5  # normalize to [0, 1]
        polarity = (1.0 if sig >= 0 else -1.0) * (1.0 - dist)
        return float(x), float(y), float(z), float(exposure), float(dist), float(polarity)

    @classmethod
    def tokens_to_features(cls, tokens: Sequence[str]) -> torch.Tensor:
        """
        Convert a sequence of tokens to a tensor of shape [T, 6].
        """
        feats = [cls._token_feature(cls.token_signature(tok)) for tok in tokens]
        if not feats:
            feats.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        return torch.tensor(feats, dtype=torch.float)

    @staticmethod
    def _pad_features(batch_feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a list of [Ti, 6] tensors to [B, T_max, 6] and return mask [B, T_max] (True for real tokens).
        """
        max_len = max(f.shape[0] for f in batch_feats)
        batch_size = len(batch_feats)
        padded = batch_feats[0].new_zeros((batch_size, max_len, 6))
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=batch_feats[0].device)
        for i, f in enumerate(batch_feats):
            t = f.shape[0]
            padded[i, :t] = f
            mask[i, :t] = True
        return padded, mask

    def encode_tokens(self, tokens: Sequence[str], device: torch.device) -> torch.Tensor:
        """
        Encode a single token sequence → sentence vector on the given device.
        """
        return self.encode_batch([tokens], device=device)[0]

    def encode_batch(self, batch_tokens: Sequence[Sequence[str]], device: torch.device) -> torch.Tensor:
        """
        Encode a batch of token sequences → [B, dim] tensor.
        """
        feats_list = [self.tokens_to_features(tokens).to(device) for tokens in batch_tokens]
        feats_padded, mask = self._pad_features(feats_list)  # [B, T, 6], [B, T]

        # Project and regularize token vectors
        proj = self.proj(feats_padded)  # [B, T, dim]
        proj = self.dropout(proj)

        if self.token_norm_cap is not None:
            token_norm = proj.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            proj = proj * (self.token_norm_cap / token_norm).clamp(max=1.0)

        # Optional transformer for token interactions
        if self.transformer is not None:
            proj = self.transformer(proj, src_key_padding_mask=~mask)

        # Pool
        if self.use_attention_pooling:
            scores = self.attn_score(proj).squeeze(-1)  # [B, T]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)  # [B, T]
            sent = torch.bmm(attn.unsqueeze(1), proj).squeeze(1)  # [B, dim]
        else:
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            sent = (proj * mask.unsqueeze(-1)).sum(dim=1) / lengths

        # Norm control
        norm = sent.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        sent = sent * (self.norm_target / norm)
        return sent

    def forward(self, batch_tokens: Sequence[Sequence[str]], device: torch.device) -> torch.Tensor:
        """
        Forward pass alias for encode_batch.
        """
        return self.encode_batch(batch_tokens, device)
