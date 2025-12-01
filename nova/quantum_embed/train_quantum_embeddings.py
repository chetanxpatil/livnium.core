"""
Quantum-Inspired Embedding Trainer (Livnium v0.1)

Trains word embeddings on WikiText-103 using a Livnium-style energy:
    alignment a = cos(v_i, v_j)
    divergence d = 0.38 - a
    positive energy  E_pos = d^2
    negative energy  E_neg = max(0, d_margin - d_neg)^2

This produces an embedding matrix that is "shaped" by the divergence law,
ready to be plugged into nova_v3's encoder as a pretrained table.
"""

import math
import os
import argparse
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------
#  Utility: Vocab
# -----------------------

class Vocab:
    def __init__(self, max_size: int = 50000, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.freqs: Dict[str, int] = {}
        self.special_tokens = ["<pad>", "<unk>"]
        for tok in self.special_tokens:
            self._add(tok)

    def _add(self, token: str):
        if token not in self.word2idx:
            idx = len(self.idx2word)
            self.word2idx[token] = idx
            self.idx2word.append(token)

    def add_tokens_from_line(self, line: str):
        for tok in line.strip().split():
            if not tok:
                continue
            self.freqs[tok] = self.freqs.get(tok, 0) + 1

    def build(self):
        # Sort by frequency
        sorted_items = sorted(self.freqs.items(), key=lambda x: -x[1])
        for tok, freq in sorted_items:
            if freq < self.min_freq:
                continue
            if tok in self.word2idx:
                continue
            if len(self.idx2word) >= self.max_size:
                break
            self._add(tok)

    @property
    def pad_idx(self) -> int:
        return self.word2idx["<pad>"]

    @property
    def unk_idx(self) -> int:
        return self.word2idx["<unk>"]

    def __len__(self) -> int:
        return len(self.idx2word)

    def encode_line(self, line: str) -> List[int]:
        return [self.word2idx.get(tok, self.unk_idx) for tok in line.strip().split() if tok]


# -----------------------
#  Dataset: Skip-gram pairs
# -----------------------

class SkipGramDataset(Dataset):
    """
    Simple Skip-Gram style dataset over tokenized WikiText.

    For each sentence:
        tokens = [w0, w1, ..., wn]
    For each index i:
        center = tokens[i]
        context = tokens[j] for j in window around i

    Returns (center_idx, context_idx) pairs.
    """

    def __init__(self, sequences: List[List[int]], window_size: int = 2):
        self.pairs: List[Tuple[int, int]] = []
        for seq in sequences:
            for i, c in enumerate(seq):
                if c == 0:
                    continue
                left = max(0, i - window_size)
                right = min(len(seq), i + window_size + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    self.pairs.append((c, seq[j]))
        # store as tensors later
        print(f"[SkipGramDataset] total pairs: {len(self.pairs)}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.pairs[idx]


# -----------------------
#  Model: Embedding Table
# -----------------------

class QuantumEmbeddingModel(nn.Module):
    """
    Simple embedding matrix trained with Livnium energy.

    - embeddings: [vocab_size, dim]
    """

    def __init__(self, vocab_size: int, dim: int = 256, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)

        # Initialize with small norm to make early divergence stable
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.05)

    def forward(self, idxs: torch.Tensor) -> torch.Tensor:
        return self.emb(idxs)


# -----------------------
#  Livnium Energy Loss
# -----------------------

def livnium_energy_loss(
    model: QuantumEmbeddingModel,
    centers: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    d_margin: float = 0.4,
    neg_weight: float = 5.0,
    norm_reg_weight: float = 1e-4,
) -> torch.Tensor:
    """
    Quantum-inspired Skip-gram loss using Livnium divergence.

    centers, positives, negatives: [batch]
    """

    # [batch, dim]
    v_c = model(centers)
    v_p = model(positives)
    v_n = model(negatives)

    # Normalize for cosine
    v_c_n = F.normalize(v_c, dim=-1)
    v_p_n = F.normalize(v_p, dim=-1)
    v_n_n = F.normalize(v_n, dim=-1)

    # Alignment
    a_pos = (v_c_n * v_p_n).sum(dim=-1)  # [batch]
    a_neg = (v_c_n * v_n_n).sum(dim=-1)

    # Livnium divergence
    d_pos = 0.38 - a_pos
    d_neg = 0.38 - a_neg

    # Positive energy: want d_pos → 0
    E_pos = d_pos.pow(2)

    # Negative energy: want d_neg ≥ d_margin
    # If d_neg < d_margin, penalize
    diff = torch.clamp(d_margin - d_neg, min=0.0)
    E_neg = diff.pow(2)

    # Norm regularization: keep embeddings from exploding
    norm = v_c.norm(dim=-1) + v_p.norm(dim=-1) + v_n.norm(dim=-1)
    norm_reg = norm.mean()

    loss = E_pos.mean() + neg_weight * E_neg.mean() + norm_reg_weight * norm_reg
    return loss


# -----------------------
#  Training Loop
# -----------------------

def build_vocab_and_sequences(path: str, max_lines: int, max_size: int) -> Tuple[Vocab, List[List[int]]]:
    vocab = Vocab(max_size=max_size, min_freq=2)

    print(f"[build_vocab] scanning {path}...")
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            vocab.add_tokens_from_line(line)

    vocab.build()
    print(f"[build_vocab] vocab size: {len(vocab)}")

    sequences: List[List[int]] = [vocab.encode_line(line) for line in lines]
    print(f"[build_vocab] sequences: {len(sequences)}")
    return vocab, sequences


def sample_negative(batch_size: int, vocab_size: int, pad_idx: int, device: torch.device) -> torch.Tensor:
    # Uniform negatives excluding pad_idx
    neg = torch.randint(low=0, high=vocab_size, size=(batch_size,), device=device)
    neg = torch.where(neg == pad_idx, (neg + 1) % vocab_size, neg)
    return neg


def train(
    train_path: str,
    output_dir: str,
    dim: int = 256,
    max_vocab: int = 50000,
    max_lines: int = 200000,
    window_size: int = 2,
    batch_size: int = 1024,
    epochs: int = 3,
    lr: float = 3e-4,
    device: str = "cpu",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    # 1) Build vocab + sequences
    vocab, sequences = build_vocab_and_sequences(
        train_path, max_lines=max_lines, max_size=max_vocab
    )

    # 2) Build dataset + dataloader
    dataset = SkipGramDataset(sequences, window_size=window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3) Model
    model = QuantumEmbeddingModel(vocab_size=len(vocab), dim=dim, pad_idx=vocab.pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[train] device={device}, vocab={len(vocab)}, dim={dim}, batches={len(loader)}")

    # 4) Train
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for centers, contexts in loader:
            centers = centers.to(device)
            contexts = contexts.to(device)
            negatives = sample_negative(
                batch_size=centers.size(0),
                vocab_size=len(vocab),
                pad_idx=vocab.pad_idx,
                device=device,
            )

            optimizer.zero_grad()
            loss = livnium_energy_loss(model, centers, contexts, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % 200 == 0:
                avg = total_loss / 200
                print(f"[epoch {epoch}] step {global_step} avg_loss={avg:.4f}")
                total_loss = 0.0

        # Save checkpoint per epoch
        ckpt_path = os.path.join(output_dir, f"quantum_embed_epoch{epoch}.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab": {
                    "idx2word": vocab.idx2word,
                    "pad_idx": vocab.pad_idx,
                    "unk_idx": vocab.unk_idx,
                },
                "dim": dim,
            },
            ckpt_path,
        )
        print(f"[train] saved {ckpt_path}")

    # final embedding table
    final_path = os.path.join(output_dir, "quantum_embeddings_final.pt")
    torch.save(
        {
            "embeddings": model.emb.weight.detach().cpu(),
            "vocab": {
                "idx2word": vocab.idx2word,
                "pad_idx": vocab.pad_idx,
                "unk_idx": vocab.unk_idx,
            },
            "dim": dim,
        },
        final_path,
    )
    print(f"[train] saved final embeddings to {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path to wiki.train.tokens")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--max-lines", type=int, default=200000,
                        help="Max lines from WikiText to load (0 = all)")
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(
        train_path=args.train_path,
        output_dir=args.output_dir,
        dim=args.dim,
        max_vocab=args.max_vocab,
        max_lines=args.max_lines,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
