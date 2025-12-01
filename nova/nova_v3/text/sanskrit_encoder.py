"""
SNLI-Optimized Phoneme Geometry Encoder

A lightweight, deterministic, Sanskrit-inspired phoneme geometry encoder
designed specifically for nova_v3 + SNLI. This version removes heavy
phonology and keeps only the features that help stable semantic geometry:
place, manner, voiced/nasal flags, and simple C/V classification.

Produces an 8-D base phoneme vector → linear projection → mean-pool.
"""

from typing import List, Sequence, Optional
import math
import re
import torch
import torch.nn as nn

# Base dimensionality of the phoneme vector
BASE_DIM = 8
TAU = 2.0 * math.pi

# ----------------------------
#  Phoneme Feature Tables
# ----------------------------

# Simplified 5-point place (Sanskrit-style buckets)
PLACE = {
    "k": 0, "g": 0, "h": 0,        # guttural
    "c": 1, "j": 1, "y": 1,        # palatal
    "r": 2, "s": 2,                # retroflex/sibilant(ish)
    "t": 3, "d": 3, "n": 3, "l": 3,# dental
    "p": 4, "b": 4, "m": 4         # labial
}

# Simplified manner buckets
MANNER = {
    "a": 0, "e": 0, "i": 0, "o": 0, "u": 0,   # vowels
    "k": 1, "t": 1, "p": 1, "s": 1,           # unvoiced stops/sibilants
    "g": 2, "d": 2, "b": 2, "j": 2,           # voiced stops
    "h": 3,                                   # aspirate
    "m": 4, "n": 4                            # nasals
}

# Flags
VOWELS = {"a", "e", "i", "o", "u"}
VOICED = {"g", "d", "b", "j", "m", "n", "l", "r", "v"}
NASALS = {"m", "n"}

# Fallback phoneme when unknown
FALLBACK = "a"


def _phoneme_vector(ch: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Map a single character to an 8-D geometric phoneme vector."""
    ch = ch.lower()

    place_idx = PLACE.get(ch, 2)      # midpoint fallback
    manner_idx = MANNER.get(ch, 2)

    # Normalize to [0,1] range for cyclic encoding
    place = place_idx / 5.0
    manner = manner_idx / 5.0

    theta_place = TAU * place
    theta_manner = TAU * manner

    consonant_flag = 0.0 if ch in VOWELS else 1.0
    vowel_flag = 1.0 if ch in VOWELS else 0.0
    voiced_flag = 1.0 if ch in VOICED else 0.0
    nasal_flag = 1.0 if ch in NASALS else 0.0

    return torch.tensor(
        [
            math.sin(theta_place),
            math.cos(theta_place),
            math.sin(theta_manner),
            math.cos(theta_manner),
            consonant_flag,
            vowel_flag,
            voiced_flag,
            nasal_flag,
        ],
        device=device,
        dtype=dtype,
    )


class TextEncoder(nn.Module):
    """
    SNLI-optimized phoneme encoder.
    Keeps the interface of the legacy TextEncoder:
    - tokenize(text) → list of tokens
    - forward(token_ids) → vector
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        pad_idx: int = 0,
        id_to_token: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.pad_idx = pad_idx
        self.id_to_token = list(id_to_token) if id_to_token else None

        # Linear projection: 8 → 256
        self.expand = nn.Linear(BASE_DIM, dim)

    def set_id_to_token(self, id_to_token: Sequence[str]) -> None:
        self.id_to_token = list(id_to_token)

    def tokenize(self, text: str) -> List[str]:
        patt = r"(\w+|\s+|[^\w\s])"
        return [t for t in re.split(patt, text) if t.strip()]

    def _fallback_char(self, token_id: int) -> str:
        return FALLBACK

    def _chars_for_token(self, token_id: int) -> List[str]:
        if self.id_to_token and 0 <= token_id < len(self.id_to_token):
            token = self.id_to_token[token_id]
            if token:
                return list(token)
        return [self._fallback_char(token_id)]

    def _encode_single(self, token_ids: torch.Tensor) -> torch.Tensor:
        device = token_ids.device
        dtype = self.expand.weight.dtype

        phon_vecs = []

        for tid in token_ids.tolist():
            if tid == self.pad_idx:
                continue
            for ch in self._chars_for_token(tid):
                phon_vecs.append(_phoneme_vector(ch, device, dtype))

        if not phon_vecs:
            return torch.zeros(self.dim, device=device, dtype=dtype)

        base = torch.stack(phon_vecs, dim=0)
        projected = self.expand(base)
        return projected.mean(dim=0)

    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() == 1:
            return self._encode_single(token_ids)
        return torch.stack([self._encode_single(seq) for seq in token_ids], dim=0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_sentence(token_ids)
