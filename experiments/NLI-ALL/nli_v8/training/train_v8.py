#!/usr/bin/env python3
"""
Train Livnium v8: Clean Architecture with Geometry-First Philosophy

Features:
- Semantic warp alignment (DP, no hardcoded rules)
- Collision-based fracture detection (negation = alignment tension)
- Geometry-first training (geometry is the teacher)
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.nli_v8 import ChainEncoder, LivniumV8Classifier
from experiments.nli_v8.core.layers import Layer2Basin
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
            except (json.JSONDecodeError, KeyError):
                continue
    
    return examples


def print_confusion_matrix(y_true: List[str], y_pred: List[str], title: str):
    """Print confusion matrix."""
    labels = ['entailment', 'contradiction', 'neutral']
    
    # Build confusion matrix
    matrix = {}
    for true_label in labels:
        matrix[true_label] = {}
        for pred_label in labels:
            count = sum(1 for t, p in zip(y_true, y_pred) 
                       if t == true_label and p == pred_label)
            matrix[true_label][pred_label] = count
    
    # Calculate metrics
    total = len(y_true)
    metrics = {}
    for true_label in labels:
        true_total = sum(matrix[true_label].values())
        if true_total > 0:
            correct = matrix[true_label][true_label]
            precision = correct / sum(matrix[t][true_label] for t in labels) if sum(matrix[t][true_label] for t in labels) > 0 else 0.0
            recall = correct / true_total
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics[true_label] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(true_total)
            }
    
    # Overall accuracy
    correct_total = sum(matrix[label][label] for label in labels)
    accuracy = correct_total / total if total > 0 else 0.0
    
    # Print
    print(f"\n{title} Confusion Matrix:")
    print("=" * 70)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print()
    print("Per-Class Metrics:")
    print("-" * 70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 70)
    for label in labels:
        m = metrics.get(label, {})
        print(f"{label:<15} {m.get('precision', 0):<12.4f} {m.get('recall', 0):<12.4f} {m.get('f1', 0):<12.4f} {m.get('support', 0):<10}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Train Livnium v8: Clean Architecture"
    )
    parser.add_argument('--data-dir', type=str,
                       default='experiments/nli/data',
                       help='Directory containing SNLI data')
    parser.add_argument('--train', type=int, default=1000,
                       help='Number of training examples')
    parser.add_argument('--test', type=int, default=100,
                       help='Number of test examples')
    parser.add_argument('--golden-label', action='store_true',
                       help='Use golden labels (debug mode - should get 100%% accuracy)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LIVNIUM NLI v8: CLEAN ARCHITECTURE")
    print("=" * 70)
    print()
    print("Key Features:")
    print("  • Semantic warp alignment (DP, no hardcoded rules)")
    print("  • Collision-based fracture detection (negation = alignment tension)")
    print("  • Geometry-first philosophy (geometry is the teacher)")
    print()
    
    # Initialize
    encoder = ChainEncoder()
    lexicon = SimpleLexicon()
    
    # Load data
    train_file = os.path.join(args.data_dir, 'snli_1.0_train.jsonl')
    test_file = os.path.join(args.data_dir, 'snli_1.0_test.jsonl')
    
    print(f"Loading training data from {train_file}...")
    train_data = load_snli_data(train_file, max_samples=args.train)
    print(f"Loaded {len(train_data)} training examples.")
    
    print(f"Loading test data from {test_file}...")
    test_data = load_snli_data(test_file, max_samples=args.test)
    print(f"Loaded {len(test_data)} test examples.")
    print()
    
    # Training loop
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print()
    
    correct = 0
    fracture_count = 0
    total_fracture_strength = 0.0
    y_true_train = []
    y_pred_train = []
    
    for i, example in enumerate(tqdm(train_data, desc="Training")):
        premise = example['sentence1']
        hypothesis = example['sentence2']
        gold_label = example['gold_label']
        
        # Encode and classify
        pair = encoder.encode_pair(premise, hypothesis)
        
        # Use golden label if requested (debug mode)
        golden_label_hint = gold_label if args.golden_label else None
        classifier = LivniumV8Classifier(pair, golden_label_hint=golden_label_hint)
        result = classifier.classify()
        
        # Track fracture detection
        layer_states = result.layer_states
        opposition_output = layer_states.get('layer_opposition', {})
        if opposition_output.get('fracture_detected', False):
            fracture_count += 1
            total_fracture_strength += opposition_output.get('fracture_strength', 0.0)
        
        # Check correctness
        if result.label == gold_label:
            correct += 1
        
        # Apply learning feedback
        classifier.apply_learning_feedback(gold_label, learning_strength=1.0)
        
        # Collect for confusion matrix
        y_true_train.append(gold_label)
        y_pred_train.append(result.label)
    
    train_accuracy = correct / len(train_data) if train_data else 0.0
    avg_fracture_strength = total_fracture_strength / fracture_count if fracture_count > 0 else 0.0
    
    print()
    print(f"Training Accuracy: {train_accuracy:.4f} ({correct}/{len(train_data)})")
    print(f"Fractures Detected: {fracture_count}/{len(train_data)} ({100*fracture_count/len(train_data):.1f}%)")
    if fracture_count > 0:
        print(f"Average Fracture Strength: {avg_fracture_strength:.4f}")
    print()
    
    print_confusion_matrix(y_true_train, y_pred_train, "Training Set")
    
    # Test loop
    if test_data:
        print()
        print("=" * 70)
        print("TESTING")
        print("=" * 70)
        print()
        
        correct = 0
        y_true_test = []
        y_pred_test = []
        
        for example in tqdm(test_data, desc="Testing"):
            premise = example['sentence1']
            hypothesis = example['sentence2']
            gold_label = example['gold_label']
            
            pair = encoder.encode_pair(premise, hypothesis)
            
            # Use golden label if requested (debug mode)
            golden_label_hint = gold_label if args.golden_label else None
            classifier = LivniumV8Classifier(pair, golden_label_hint=golden_label_hint)
            result = classifier.classify()
            
            if result.label == gold_label:
                correct += 1
            
            y_true_test.append(gold_label)
            y_pred_test.append(result.label)
        
        test_accuracy = correct / len(test_data) if test_data else 0.0
        print()
        print(f"Test Accuracy: {test_accuracy:.4f} ({correct}/{len(test_data)})")
        print()
        
        print_confusion_matrix(y_true_test, y_pred_test, "Test Set")
    
    # Save brain state
    brain_path = os.path.join(os.path.dirname(__file__), '../brain_state')
    os.makedirs(brain_path, exist_ok=True)
    lexicon.save_to_file(os.path.join(brain_path, 'brain_state.pkl'))
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("Key Features:")
    print("  • Semantic warp finds optimal alignment automatically")
    print("  • Fracture detection finds negation via collision analysis")
    print("  • Geometry-first: geometry is the teacher, not the student")


if __name__ == '__main__':
    main()

