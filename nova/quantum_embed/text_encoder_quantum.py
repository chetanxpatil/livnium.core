"""
QuantumEmbeddingTextEncoder

Uses a pre-trained Livnium quantum embedding table produced by
train_quantum_embeddings.py. Drop-in replacement for the legacy
TextEncoder in nova_v3 (same forward signature).
"""

from typing import List, Dict, Any
import re
import torch
import torch.nn as nn


class QuantumTextEncoder(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()

        data = torch.load(ckpt_path, map_location="cpu")
        emb = data["embeddings"]          # [vocab_size, dim]
        vocab = data["vocab"]
        self.idx2word = vocab["idx2word"]
        self.word2idx: Dict[str, int] = {w: i for i, w in enumerate(self.idx2word)}
        self.pad_idx = vocab["pad_idx"]
        self.unk_idx = vocab["unk_idx"]
        self.dim = emb.size(1)

        self.embed = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=self.pad_idx)

    def tokenize(self, text: str) -> List[str]:
        pattern = r"(\w+|\s+|[^\w\s])"
        return [t for t in re.split(pattern, text) if t.strip()]

    def encode_tokens(self, tokens: List[str]) -> torch.Tensor:
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [seq_len] or [batch, seq_len]
        Returns: [dim] or [batch, dim]
        """
        emb = self.embed(token_ids)
        mask = (token_ids != self.pad_idx).float().unsqueeze(-1)

        if token_ids.dim() == 1:
            masked = emb * mask
            denom = mask.sum(dim=0).clamp(min=1.0)
            return (masked.sum(dim=0) / denom)
        else:
            masked = emb * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (masked.sum(dim=1) / denom)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_sentence(token_ids)
