"""
SNLI Training Script - Clean Architecture

Uses:
- Layer 0: VectorCollapseEngine (core physics)
- Layer 1: SNLIEncoder + SNLIHead (task-specific)
- Layer 2: This script (data loading, training loop)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

# Add nova_v3 to path
nova_v3_root = Path(__file__).parent.parent
repo_root = nova_v3_root.parent  # adds /nova so we can import quantum_embed
sys.path.insert(0, str(nova_v3_root))
sys.path.insert(0, str(repo_root))

from core import VectorCollapseEngine
from tasks.snli import SNLIEncoder, GeometricSNLIEncoder, SanskritSNLIEncoder, QuantumSNLIEncoder, SNLIHead
from quantum_embed.text_encoder_quantum import QuantumTextEncoder
from utils.vocab import build_vocab_from_snli


class SNLIDataset(Dataset):
    """SNLI dataset."""
    
    def __init__(self, samples: List[Dict], vocab=None, max_len: int = 128, encode_fn=None):
        """
        Args:
            samples: SNLI examples
            vocab: vocabulary object with .encode(...) (optional if encode_fn supplied)
            max_len: max sequence length for padding/truncation
            encode_fn: callable(text: str, max_len: int) -> List[int] (overrides vocab.encode)
        """
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len
        self.encode_fn = encode_fn
        
        # Label mapping
        self.label_map = {
            'entailment': 0,
            'contradiction': 1,
            'neutral': 2
        }
        if self.vocab is None and self.encode_fn is None:
            raise ValueError("SNLIDataset needs either a vocab or encode_fn")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Encode premise and hypothesis
        encode = self.encode_fn if self.encode_fn is not None else self.vocab.encode
        prem_ids = encode(sample['premise'], max_len=self.max_len)
        hyp_ids = encode(sample['hypothesis'], max_len=self.max_len)
        
        # Label
        label = self.label_map.get(sample['gold_label'], 2)  # Default to neutral
        
        return {
            'prem_ids': torch.tensor(prem_ids, dtype=torch.long),
            'hyp_ids': torch.tensor(hyp_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'premise': sample['premise'],
            'hypothesis': sample['hypothesis'],
            'gold_label': sample['gold_label']
        }


def load_snli_data(jsonl_path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Load SNLI data from JSONL file."""
    samples = []
    label_by_pair = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            data = json.loads(line.strip())
            gold_label = data.get('gold_label', '').strip()
            
            # Skip invalid labels
            if gold_label not in ['entailment', 'contradiction', 'neutral'] or gold_label == '-':
                continue
            
            premise = data.get('sentence1', '').strip()
            hypothesis = data.get('sentence2', '').strip()
            
            if not premise or not hypothesis:
                continue
            
            pair = (premise, hypothesis)
            # Skip ambiguous pairs that appear with conflicting labels
            if pair in label_by_pair and label_by_pair[pair] != gold_label:
                continue
            label_by_pair[pair] = gold_label
            
            samples.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'gold_label': gold_label
            })
            
            # Stop once we have collected the desired number of VALID samples
            if max_samples and len(samples) >= max_samples:
                break
    
    return samples


def train_epoch(model, encoder, head, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    encoder.train()
    head.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        labels = batch['label'].to(device)
        if isinstance(encoder, GeometricSNLIEncoder):
            # Geometric encoder consumes raw text
            h0, v_p, v_h = encoder.build_initial_state(
                batch['premise'],
                batch['hypothesis'],
                device=device
            )
        else:
            prem_ids = batch['prem_ids'].to(device)
            hyp_ids = batch['hyp_ids'].to(device)
            h0, v_p, v_h = encoder.build_initial_state(prem_ids, hyp_ids)
        
        # Collapse
        h_final, trace = model.collapse(h0)
        
        # Classify with directional signals
        logits = head(h_final, v_p, v_h)
        
        # Loss
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def evaluate(model, encoder, head, dataloader, device):
    """Evaluate model."""
    model.eval()
    encoder.eval()
    head.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch['label'].to(device)
            if isinstance(encoder, GeometricSNLIEncoder):
                h0, v_p, v_h = encoder.build_initial_state(
                    batch['premise'],
                    batch['hypothesis'],
                    device=device
                )
            else:
                prem_ids = batch['prem_ids'].to(device)
                hyp_ids = batch['hyp_ids'].to(device)
                h0, v_p, v_h = encoder.build_initial_state(prem_ids, hyp_ids)
            
            # Collapse
            h_final, trace = model.collapse(h0)
            
            # Classify with directional signals
            logits = head(h_final, v_p, v_h)
            pred = logits.argmax(dim=-1)
            
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Confusion matrix
    confusion = np.zeros((3, 3), dtype=int)
    for p, l in zip(all_predictions, all_labels):
        confusion[l, p] += 1
    
    return accuracy, confusion


def main():
    parser = argparse.ArgumentParser(description='Train SNLI with Livnium Core v1.0')
    parser.add_argument('--snli-train', type=str, required=True,
                       help='Path to SNLI training JSONL file')
    parser.add_argument('--snli-dev', type=str, default=None,
                       help='Path to SNLI dev JSONL file')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of training samples')
    parser.add_argument('--dim', type=int, default=256,
                       help='Vector dimension')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of collapse layers')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--strength-entail', type=float, default=0.1,
                       help='Force strength for entail anchor')
    parser.add_argument('--strength-contra', type=float, default=0.1,
                       help='Force strength for contradiction anchor')
    parser.add_argument('--strength-neutral', type=float, default=0.05,
                       help='Force strength for neutral anchor')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for model')
    parser.add_argument('--max-len', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                       help='Label smoothing for CrossEntropyLoss (e.g., 0.05)')
    parser.add_argument('--neutral-weight', type=float, default=1.0,
                       help='Class weight multiplier for neutral to emphasize that class')
    parser.add_argument('--neutral-oversample', type=float, default=1.0,
                       help='>1.0 to oversample neutral examples (e.g., 1.5)')
    parser.add_argument('--encoder-type', choices=['legacy', 'geom', 'sanskrit', 'quantum'], default='geom',
                       help='Sentence encoder: geom (geometric), legacy (embedding mean-pool), sanskrit (phoneme geometry), quantum (pretrained quantum embeddings)')
    parser.add_argument('--quantum-ckpt', type=str, default=None,
                       help='Path to quantum_embeddings_final.pt (required if encoder-type=quantum)')
    # Geometric encoder knobs
    parser.add_argument('--geom-disable-transformer', action='store_true',
                       help='Disable transformer interaction layer in geometric encoder')
    parser.add_argument('--geom-disable-attn-pool', action='store_true',
                       help='Disable attention pooling in geometric encoder (use masked mean)')
    parser.add_argument('--geom-nhead', type=int, default=4,
                       help='Attention heads for geometric encoder transformer')
    parser.add_argument('--geom-num-layers', type=int, default=1,
                       help='Transformer layers for geometric encoder')
    parser.add_argument('--geom-ff-mult', type=int, default=2,
                       help='Feedforward multiplier for geometric encoder transformer')
    parser.add_argument('--geom-dropout', type=float, default=0.1,
                       help='Dropout for geometric encoder projection/transformer')
    parser.add_argument('--geom-token-norm-cap', type=float, default=3.0,
                       help='Per-token norm cap after projection (set <=0 to disable)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading SNLI data...")
    train_samples = load_snli_data(Path(args.snli_train), max_samples=args.max_samples)
    print(f"Loaded {len(train_samples)} training samples")
    
    quantum_encode_fn = None
    vocab = None
    vocab_id_to_token = None

    if args.encoder_type == 'quantum':
        if not args.quantum_ckpt:
            raise ValueError("encoder-type=quantum requires --quantum-ckpt pointing to quantum_embeddings_final.pt")
        print(f"Loading quantum encoder vocab from {args.quantum_ckpt} ...")
        quantum_tokenizer = QuantumTextEncoder(args.quantum_ckpt)
        if args.dim != quantum_tokenizer.dim:
            print(f"Overriding dim {args.dim} -> {quantum_tokenizer.dim} to match quantum checkpoint")
            args.dim = quantum_tokenizer.dim

        def quantum_encode(text: str, max_len: int = args.max_len):
            tokens = quantum_tokenizer.tokenize(text)
            ids = [quantum_tokenizer.word2idx.get(t, quantum_tokenizer.unk_idx) for t in tokens]
            ids = ids[:max_len]
            if len(ids) < max_len:
                ids.extend([quantum_tokenizer.pad_idx] * (max_len - len(ids)))
            return ids

        quantum_encode_fn = quantum_encode
    else:
        # Build vocabulary from SNLI
        print("Building vocabulary...")
        vocab = build_vocab_from_snli(train_samples, min_count=2)
        print(f"Vocabulary size: {len(vocab)}")
        vocab_id_to_token = vocab.id_to_token_list()
    
    # Create datasets
    train_dataset = SNLIDataset(train_samples, vocab, max_len=args.max_len, encode_fn=quantum_encode_fn)
    # Optional oversampling of neutral class
    sampler = None
    if args.neutral_oversample > 1.0:
        labels = [s['label'].item() for s in (train_dataset[i] for i in range(len(train_dataset)))]
        # base weights inverse-frequency
        counts = np.bincount(labels, minlength=3)
        base_weights = [1.0 / max(c, 1) for c in counts]
        weights = [base_weights[l] for l in labels]
        # amplify neutral
        weights = [w * (args.neutral_oversample if l == 2 else 1.0) for w, l in zip(weights, labels)]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Dev set
    dev_loader = None
    if args.snli_dev:
        dev_samples = load_snli_data(Path(args.snli_dev))
        dev_dataset = SNLIDataset(dev_samples, vocab, max_len=args.max_len, encode_fn=quantum_encode_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Loaded {len(dev_samples)} dev samples")
    
    # Create models
    print("Creating models...")
    collapse_engine = VectorCollapseEngine(
        dim=args.dim,
        num_layers=args.num_layers,
        strength_entail=args.strength_entail,
        strength_contra=args.strength_contra,
        strength_neutral=args.strength_neutral
    ).to(device)
    if args.encoder_type == 'geom':
        encoder = GeometricSNLIEncoder(
            dim=args.dim,
            norm_target=None,
            use_transformer=not args.geom_disable_transformer,
            nhead=args.geom_nhead,
            num_layers=args.geom_num_layers,
            ff_mult=args.geom_ff_mult,
            dropout=args.geom_dropout,
            use_attention_pooling=not args.geom_disable_attn_pool,
            token_norm_cap=args.geom_token_norm_cap if args.geom_token_norm_cap > 0 else None,
        ).to(device)
    elif args.encoder_type == 'sanskrit':
        encoder = SanskritSNLIEncoder(
            vocab_size=len(vocab),
            dim=args.dim,
            pad_idx=vocab.pad_idx,
            id_to_token=vocab_id_to_token,
        ).to(device)
    elif args.encoder_type == 'quantum':
        encoder = QuantumSNLIEncoder(
            ckpt_path=args.quantum_ckpt,
        ).to(device)
    else:
        encoder = SNLIEncoder(
            vocab_size=len(vocab),
            dim=args.dim,
            pad_idx=vocab.pad_idx,
        ).to(device)
    head = SNLIHead(dim=args.dim).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        list(collapse_engine.parameters()) + 
        list(encoder.parameters()) + 
        list(head.parameters()),
        lr=args.lr
    )
    
    # Loss with optional class weighting and label smoothing
    class_weights = torch.tensor([1.0, 1.0, args.neutral_weight], device=device, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            collapse_engine, encoder, head, train_loader, 
            optimizer, criterion, device
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Evaluate
        if dev_loader:
            dev_acc, confusion = evaluate(collapse_engine, encoder, head, dev_loader, device)
            print(f"Dev Acc: {dev_acc:.4f}")
            print("\nConfusion Matrix:")
            print("      E    N    C")
            for i, label in enumerate(['E', 'N', 'C']):
                print(f"{label}  {confusion[i]}")
            
            # Save best model
            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save({
                    'collapse_engine': collapse_engine.state_dict(),
                    'encoder': encoder.state_dict(),
                    'head': head.state_dict(),
                    'vocab': vocab,
                    'args': args
                }, output_dir / 'best_model.pt')
                print(f"✓ Saved best model (acc: {best_acc:.4f})")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best Dev Accuracy: {best_acc:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
