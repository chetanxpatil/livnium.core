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

# Add nova_v2 to path
nova_v2_root = Path(__file__).parent.parent
sys.path.insert(0, str(nova_v2_root))

from core import VectorCollapseEngine
from tasks.snli import SNLIEncoder, SNLIHead
from utils.vocab import build_vocab_from_snli


class SNLIDataset(Dataset):
    """SNLI dataset."""
    
    def __init__(self, samples: List[Dict], vocab, max_len: int = 128):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len
        
        # Label mapping
        self.label_map = {
            'entailment': 0,
            'contradiction': 1,
            'neutral': 2
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Encode premise and hypothesis
        prem_ids = self.vocab.encode(sample['premise'], max_len=self.max_len)
        hyp_ids = self.vocab.encode(sample['hypothesis'], max_len=self.max_len)
        
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
        prem_ids = batch['prem_ids'].to(device)
        hyp_ids = batch['hyp_ids'].to(device)
        labels = batch['label'].to(device)
        
        # Build initial state (returns h0, OM, LO)
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
            prem_ids = batch['prem_ids'].to(device)
            hyp_ids = batch['hyp_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Build initial state (returns h0, OM, LO)
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
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for model')
    parser.add_argument('--max-len', type=int, default=128,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading SNLI data...")
    train_samples = load_snli_data(Path(args.snli_train), max_samples=args.max_samples)
    print(f"Loaded {len(train_samples)} training samples")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab_from_snli(train_samples, min_count=2)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = SNLIDataset(train_samples, vocab, max_len=args.max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Dev set
    dev_loader = None
    if args.snli_dev:
        dev_samples = load_snli_data(Path(args.snli_dev))
        dev_dataset = SNLIDataset(dev_samples, vocab, max_len=args.max_len)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Loaded {len(dev_samples)} dev samples")
    
    # Create models
    print("Creating models...")
    collapse_engine = VectorCollapseEngine(dim=args.dim, num_layers=args.num_layers).to(device)
    encoder = SNLIEncoder(vocab_size=len(vocab), dim=args.dim).to(device)
    head = SNLIHead(dim=args.dim).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        list(collapse_engine.parameters()) + 
        list(encoder.parameters()) + 
        list(head.parameters()),
        lr=args.lr
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
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
