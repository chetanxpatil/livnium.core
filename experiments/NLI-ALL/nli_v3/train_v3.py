"""
Livnium NLI v3 Training: Complete System

Combines chain structure + Livnium physics for maximum accuracy.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional
from collections import Counter

import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.nli_v3.chain_encoder import ChainEncoder
from experiments.nli_v3.classifier import LivniumNLIClassifier
from experiments.nli_simple.native_chain import SimpleLexicon


def load_snli_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load SNLI dataset from JSONL file."""
    examples = []
    
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: Data file not found: {file_path}")
        return examples
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            try:
                example = json.loads(line.strip())
                gold_label = example.get('gold_label', '').lower()
                
                if gold_label in ['entailment', 'contradiction', 'neutral']:
                    examples.append({
                        'sentence1': example.get('sentence1', ''),
                        'sentence2': example.get('sentence2', ''),
                        'gold_label': gold_label
                    })
            except json.JSONDecodeError:
                continue
    
    return examples


def print_confusion_matrix(y_true: List[str], y_pred: List[str], title: str):
    """Print confusion matrix with proper formatting."""
    labels = ['entailment', 'contradiction', 'neutral']
    label_short = {'entailment': 'E', 'contradiction': 'C', 'neutral': 'N'}
    
    # Build confusion matrix
    matrix = {}
    for true_label in labels:
        matrix[true_label] = {}
        for pred_label in labels:
            count = sum(1 for t, p in zip(y_true, y_pred) 
                        if t == true_label and p == pred_label)
            matrix[true_label][pred_label] = count
    
    # Print header
    header_label = "True \\ Predicted"
    print(f"\n{title} Confusion Matrix:")
    print("=" * 50)
    print(f"{header_label:<20} {'E':<8} {'C':<8} {'N':<8} {'Total':<8}")
    print("-" * 50)
    
    # Print rows
    for true_label in labels:
        true_short = label_short[true_label]
        row = [f"{true_short} ({true_label[:8]})"]
        total = 0
        for pred_label in labels:
            count = matrix[true_label][pred_label]
            row.append(f"{count:<8}")
            total += count
        row.append(f"{total:<8}")
        print(" ".join(row))
    
    # Print column totals
    print("-" * 50)
    row = ["Total"]
    for pred_label in labels:
        total = sum(matrix[t][pred_label] for t in labels)
        row.append(f"{total:<8}")
    grand_total = len(y_true)
    row.append(f"{grand_total:<8}")
    print(" ".join(row))
    print("=" * 50)
    print()


def main():
    parser = argparse.ArgumentParser(description='Train Livnium NLI v3')
    parser.add_argument('--data-dir', type=str, default='experiments/nli/data',
                       help='Directory containing SNLI data files')
    parser.add_argument('--train', type=int, default=1000,
                       help='Number of training examples')
    parser.add_argument('--test', type=int, default=100,
                       help='Number of test examples')
    parser.add_argument('--dev', type=int, default=100,
                       help='Number of dev examples')
    parser.add_argument('--clean', action='store_true',
                       help='Clear learned state before training')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LIVNIUM NLI v3: COMPLETE SYSTEM")
    print("="*70)
    print()
    print("Architecture:")
    print("  • Chain Encoder (positional encoding, sequential matching)")
    print("  • Basin System (energy wells for each class)")
    print("  • Quantum Collapse (3-way decision making)")
    print("  • Moksha Router (convergence detection)")
    print("  • Geometric Feedback (word polarity learning)")
    print()
    
    # Initialize encoder
    encoder = ChainEncoder(vector_size=27)
    print("✓ Chain Encoder initialized")
    
    # Initialize lexicon
    if args.clean:
        SimpleLexicon().clear()
        LivniumNLIClassifier.reset_shared_state()  # Reset shared components
        print("✓ Lexicon cleared (clean start)")
        print("✓ Shared basin system reset")
        print("✓ Decision head reset")
    else:
        print("✓ Lexicon initialized")
        print("✓ Shared basin system initialized")
        # Try to load decision head if it exists
        head_path = os.path.join(os.path.dirname(__file__), 'peak_clarity.pkl')
        if LivniumNLIClassifier.load_peak_clarity(head_path):
            print("✓ Decision head loaded from disk")
        else:
            print("✓ Decision head initialized (new)")
    
    print()
    
    # Load data
    data_dir = args.data_dir
    train_file = os.path.join(data_dir, 'snli_1.0_train.jsonl')
    test_file = os.path.join(data_dir, 'snli_1.0_test.jsonl')
    dev_file = os.path.join(data_dir, 'snli_1.0_dev.jsonl')
    
    print(f"Loading training data from {train_file}...")
    train_examples = load_snli_data(train_file, max_samples=args.train)
    print(f"Loaded {len(train_examples)} training examples.")
    
    test_examples = []
    if args.test > 0:
        print(f"Loading test data from {test_file}...")
        test_examples = load_snli_data(test_file, max_samples=args.test)
        print(f"Loaded {len(test_examples)} test examples.")
    
    dev_examples = []
    if args.dev > 0:
        print(f"Loading dev data from {dev_file}...")
        dev_examples = load_snli_data(dev_file, max_samples=args.dev)
        print(f"Loaded {len(dev_examples)} dev examples.")
    
    print()
    
    # Training
    print("="*70)
    print("TRAINING")
    print("="*70)
    print()
    
    train_y_true = []
    train_y_pred = []
    correct_count = 0
    total_steps = 0
    moksha_count = 0
    
    try:
        from tqdm import tqdm
        iterator = tqdm(train_examples, desc="Training Livnium v3")
    except ImportError:
        iterator = train_examples
    
    for i, ex in enumerate(iterator):
        premise = ex['sentence1']
        hypothesis = ex['sentence2']
        gold_label = ex['gold_label']
        
        try:
            # Encode using chain structure
            encoded_pair = encoder.encode_pair(premise, hypothesis)
            
            # Classify using full Livnium system
            classifier = LivniumNLIClassifier(encoded_pair)
            result = classifier.classify()
            
            # Track Moksha
            if result.is_moksha:
                moksha_count += 1
            
            # Apply learning feedback (trains decision head + basins + word polarities)
            classifier.apply_learning_feedback(gold_label, learning_strength=1.0, train_head=True)
            
            # Track predictions
            train_y_true.append(gold_label)
            train_y_pred.append(result.label)
            
            if result.label == gold_label:
                correct_count += 1
            total_steps += 1
            
            # Periodic reporting
            if (i + 1) % 500 == 0:
                acc = correct_count / total_steps if total_steps > 0 else 0.0
                moksha_rate = moksha_count / total_steps if total_steps > 0 else 0.0
                if hasattr(iterator, 'write'):
                    iterator.write(f"Step {i+1}: Accuracy={acc:.3f} | Moksha Rate={moksha_rate:.3f}")
        
        except Exception as e:
            if hasattr(iterator, 'write'):
                iterator.write(f"\n⚠️  Error processing example {i+1}: {e}")
            continue
    
    # Training results
    train_acc = correct_count / total_steps if total_steps > 0 else 0.0
    moksha_rate = moksha_count / total_steps if total_steps > 0 else 0.0
    print()
    print(f"Training Accuracy: {train_acc:.4f} ({correct_count}/{total_steps})")
    print(f"Moksha Rate: {moksha_rate:.4f} ({moksha_count}/{total_steps})")
    print()
    
    if train_y_true and train_y_pred:
        print_confusion_matrix(train_y_true, train_y_pred, "Training Set")
    
    # Test evaluation
    if test_examples:
        print("="*70)
        print("TEST EVALUATION")
        print("="*70)
        print()
        
        test_y_true = []
        test_y_pred = []
        test_correct = 0
        test_total = 0
        
        try:
            from tqdm import tqdm
            iterator = tqdm(test_examples, desc="Test Evaluation")
        except ImportError:
            iterator = test_examples
        
        for ex in iterator:
            premise = ex['sentence1']
            hypothesis = ex['sentence2']
            gold_label = ex['gold_label']
            
            try:
                encoded_pair = encoder.encode_pair(premise, hypothesis)
                classifier = LivniumNLIClassifier(encoded_pair)
                result = classifier.classify()
                
                test_y_true.append(gold_label)
                test_y_pred.append(result.label)
                
                if result.label == gold_label:
                    test_correct += 1
                test_total += 1
            except Exception:
                continue
        
        test_acc = test_correct / test_total if test_total > 0 else 0.0
        print(f"Test Accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
        print()
        
        if test_y_true and test_y_pred:
            print_confusion_matrix(test_y_true, test_y_pred, "Test Set")
    
    # Dev evaluation
    if dev_examples:
        print("="*70)
        print("DEV EVALUATION")
        print("="*70)
        print()
        
        dev_y_true = []
        dev_y_pred = []
        dev_correct = 0
        dev_total = 0
        
        try:
            from tqdm import tqdm
            iterator = tqdm(dev_examples, desc="Dev Evaluation")
        except ImportError:
            iterator = dev_examples
        
        for ex in iterator:
            premise = ex['sentence1']
            hypothesis = ex['sentence2']
            gold_label = ex['gold_label']
            
            try:
                encoded_pair = encoder.encode_pair(premise, hypothesis)
                classifier = LivniumNLIClassifier(encoded_pair)
                result = classifier.classify()
                
                dev_y_true.append(gold_label)
                dev_y_pred.append(result.label)
                
                if result.label == gold_label:
                    dev_correct += 1
                dev_total += 1
            except Exception:
                continue
        
        dev_acc = dev_correct / dev_total if dev_total > 0 else 0.0
        print(f"Dev Accuracy: {dev_acc:.4f} ({dev_correct}/{dev_total})")
        print()
        
        if dev_y_true and dev_y_pred:
            print_confusion_matrix(dev_y_true, dev_y_pred, "Dev Set")
    
    # Save brain and decision head
    brain_path = os.path.join(os.path.dirname(__file__), 'brain_state.pkl')
    lexicon = SimpleLexicon()
    lexicon.save_to_file(brain_path)
    print(f"✓ Brain saved to: {brain_path}")
    print(f"  - Words learned: {len(lexicon.polarity_store)}")
    
    head_path = os.path.join(os.path.dirname(__file__), 'peak_clarity.pkl')
    LivniumNLIClassifier.save_peak_clarity(head_path)
    print(f"✓ Peak clarity system saved to: {head_path}")
    
    # Show learned peaks
    peaks = LivniumNLIClassifier._shared_peak_clarity.get_peaks()
    print(f"  - E peak: r={peaks['entailment']['r_mean']:.3f} (count={peaks['entailment']['count']})")
    print(f"  - C peak: r={peaks['contradiction']['r_mean']:.3f} (count={peaks['contradiction']['count']})")
    print(f"  - N: valley (no peak)")
    print()
    
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print()
    print("Expected Performance:")
    print("  • 200-400 examples: 46-49% accuracy")
    print("  • 1000 examples: 50-55% accuracy")
    print("  • 5000+ examples: 55-60%+ accuracy")
    print()


if __name__ == "__main__":
    main()

