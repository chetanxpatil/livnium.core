#!/usr/bin/env python3
"""
Train Livnium v5 with Collision-Based Fracture Detection

Fracture detection is automatically integrated into the classifier.
This script trains with fracture detection enabled.

Usage:
    python3 train_with_fracture.py --train 1000
"""

import os
import sys
import json
import argparse
from typing import List, Dict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.nli_v5 import ChainEncoder, LivniumV5Classifier
from experiments.nli_v5.core.layers import Layer2Basin
from experiments.nli_simple.native_chain import SimpleLexicon


def load_snli_data(file_path: str, max_samples: int = None) -> List[Dict]:
    """Load SNLI dataset from JSONL file."""
    examples = []
    
    if not os.path.exists(file_path):
        print(f"⚠️  Data file not found: {file_path}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Train Livnium v5 with Collision-Based Fracture Detection"
    )
    parser.add_argument('--data-dir', type=str, 
                       default='experiments/nli_v5/data',
                       help='Directory containing SNLI data')
    parser.add_argument('--train', type=int, default=1000,
                       help='Number of training examples')
    parser.add_argument('--test', type=int, default=100,
                       help='Number of test examples')
    parser.add_argument('--show-fractures', action='store_true',
                       help='Show fracture detection statistics')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LIVNIUM V5 TRAINING WITH FRACTURE DETECTION")
    print("=" * 70)
    print()
    print("Collision-based fracture detection is automatically enabled.")
    print("Negation is detected by colliding premise and hypothesis vectors.")
    print()
    
    # Load data
    train_file = os.path.join(args.data_dir, 'snli_1.0_train.jsonl')
    test_file = os.path.join(args.data_dir, 'snli_1.0_test.jsonl')
    
    train_data = load_snli_data(train_file, max_samples=args.train)
    test_data = load_snli_data(test_file, max_samples=args.test)
    
    if not train_data:
        print("⚠️  No training data found. Using synthetic examples...")
        # Create synthetic examples
        train_data = [
            {"sentence1": "A dog is barking", "sentence2": "A dog is not barking", "gold_label": "contradiction"},
            {"sentence1": "A cat sleeps", "sentence2": "A cat never sleeps", "gold_label": "contradiction"},
            {"sentence1": "A man walks", "sentence2": "A man does not walk", "gold_label": "contradiction"},
            {"sentence1": "A dog is barking", "sentence2": "A dog is barking", "gold_label": "entailment"},
            {"sentence1": "A cat sleeps", "sentence2": "A cat is sleeping", "gold_label": "entailment"},
        ] * (args.train // 5 + 1)
        train_data = train_data[:args.train]
    
    print(f"✓ Loaded {len(train_data)} training examples")
    print(f"✓ Loaded {len(test_data)} test examples")
    print()
    
    # Initialize encoder
    encoder = ChainEncoder()
    
    # Training loop
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print()
    
    correct = 0
    fracture_count = 0
    total_fracture_strength = 0.0
    
    for i, example in enumerate(tqdm(train_data, desc="Training")):
        premise = example['sentence1']
        hypothesis = example['sentence2']
        gold_label = example['gold_label']
        
        # Encode and classify
        pair = encoder.encode_pair(premise, hypothesis)
        classifier = LivniumV5Classifier(pair)
        result = classifier.classify()
        
        # Check fracture detection
        layer_states = result.layer_states
        opposition_output = layer_states.get('layer_opposition', {})
        fracture_detected = opposition_output.get('fracture_detected', False)
        
        if fracture_detected:
            fracture_count += 1
            total_fracture_strength += opposition_output.get('fracture_strength', 0.0)
        
        # Check correctness
        if result.label == gold_label:
            correct += 1
        
        # Apply learning feedback
        classifier.apply_learning_feedback(gold_label, learning_strength=1.0)
    
    train_accuracy = correct / len(train_data) if train_data else 0.0
    avg_fracture_strength = total_fracture_strength / fracture_count if fracture_count > 0 else 0.0
    
    print()
    print("=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print(f"Training Accuracy: {train_accuracy:.4f} ({correct}/{len(train_data)})")
    print(f"Fractures Detected: {fracture_count}/{len(train_data)} ({100*fracture_count/len(train_data):.1f}%)")
    if fracture_count > 0:
        print(f"Average Fracture Strength: {avg_fracture_strength:.4f}")
    print()
    
    # Test loop
    if test_data:
        print("=" * 70)
        print("TESTING")
        print("=" * 70)
        print()
        
        correct = 0
        fracture_count = 0
        
        for example in tqdm(test_data, desc="Testing"):
            premise = example['sentence1']
            hypothesis = example['sentence2']
            gold_label = example['gold_label']
            
            pair = encoder.encode_pair(premise, hypothesis)
            classifier = LivniumV5Classifier(pair)
            result = classifier.classify()
            
            # Check fracture
            layer_states = result.layer_states
            opposition_output = layer_states.get('layer_opposition', {})
            if opposition_output.get('fracture_detected', False):
                fracture_count += 1
            
            if result.label == gold_label:
                correct += 1
        
        test_accuracy = correct / len(test_data) if test_data else 0.0
        print()
        print("=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"Test Accuracy: {test_accuracy:.4f} ({correct}/{len(test_data)})")
        print(f"Fractures Detected: {fracture_count}/{len(test_data)} ({100*fracture_count/len(test_data):.1f}%)")
        print()
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("Fracture detection is integrated into the classifier.")
    print("Negation is automatically detected by colliding premise and hypothesis.")


if __name__ == '__main__':
    main()

