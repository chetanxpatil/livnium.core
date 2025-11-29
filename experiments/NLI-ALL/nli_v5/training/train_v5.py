"""
Livnium NLI v5 Training: Clean & Simplified

Streamlined training script with clear structure.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root (go up from training/ to nli_v5/ to experiments/ to project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.nli_v5 import ChainEncoder, LivniumV5Classifier
from experiments.nli_v5.core.layers import Layer2Basin
from experiments.nli_v5.core.pattern_learner import PatternLearner
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
    """Print confusion matrix in readable format."""
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
    
    # Print JSON
    print(f"\n{title} Confusion Matrix (JSON):")
    print("=" * 70)
    print(json.dumps({
        'title': title,
        'confusion_matrix': matrix,
        'metrics': metrics,
        'overall_accuracy': float(accuracy),
        'total_samples': int(total),
        'class_totals': {label: sum(matrix[label].values()) for label in labels},
        'prediction_totals': {label: sum(matrix[t][label] for t in labels) for label in labels}
    }, indent=2))
    print("=" * 70)
    
    # Print human-readable
    print(f"\n{title} Confusion Matrix (Human-readable):")
    print("=" * 50)
    header = "True \\ Predicted"
    print(f"{header:<20} {'E':<8} {'C':<8} {'N':<8} {'Total':<8}")
    print("-" * 50)
    
    for true_label in labels:
        true_short = label_short[true_label]
        row = [f"{true_short} ({true_label[:8]})"]
        row_total = 0
        for pred_label in labels:
            count = matrix[true_label][pred_label]
            row.append(f"{count:<8}")
            row_total += count
        row.append(f"{row_total:<8}")
        print(" ".join(row))
    
    print("-" * 50)
    row = ["Total"]
    for pred_label in labels:
        col_total = sum(matrix[t][pred_label] for t in labels)
        row.append(f"{col_total:<8}")
    row.append(f"{total:<8}")
    print(" ".join(row))
    print("=" * 50)
    
    # Print metrics
    print("\nPer-Class Metrics:")
    print("-" * 50)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 50)
    for label in labels:
        if label in metrics:
            m = metrics[label]
            print(f"{label:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<10}")
    print("-" * 50)
    print(f"{'Overall Accuracy':<15} {accuracy:<12.4f}")
    print("=" * 50)
    print()


def main():
    parser = argparse.ArgumentParser(description='Train Livnium NLI v5')
    parser.add_argument('--data-dir', type=str, default='experiments/nli/data',
                        help='Directory containing SNLI data files')
    parser.add_argument('--train', type=int, default=None,
                        help='Maximum number of training examples')
    parser.add_argument('--test', type=int, default=None,
                        help='Maximum number of test examples')
    parser.add_argument('--dev', type=int, default=None,
                        help='Maximum number of dev examples')
    parser.add_argument('--clean', action='store_true',
                        help='Start with clean state (clear lexicon and basins)')
    parser.add_argument('--debug-golden', action='store_true',
                        help='DEBUG: Feed golden labels to decision layer to verify logic')
    parser.add_argument('--invert-labels', action='store_true',
                        help='REVERSE PHYSICS: Force wrong labels to discover invariant geometry (diagnostic only, no training)')
    parser.add_argument('--learn-patterns', action='store_true',
                        help='Learn patterns from golden labels (record geometric signals for analysis)')
    parser.add_argument('--pattern-file', type=str, default=None,
                        help='File to save/load patterns (default: experiments/nli_v5/patterns/patterns.json)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LIVNIUM NLI v5: CLEAN & SIMPLIFIED ARCHITECTURE")
    print("=" * 70)
    print()
    
    if args.invert_labels:
        print("⚠️  REVERSE PHYSICS MODE: Labels will be INVERTED (E↔C)")
        print("   This is a diagnostic experiment to discover invariant geometry.")
        print("   NO TRAINING will occur - geometry will be recorded but not learned.")
        print("   Use this to find what signals refuse to flip when labels are wrong.")
        print()
    
    print("Architecture:")
    print("  • Layer 0: Resonance (raw geometric signal)")
    print("  • Layer 1: Curvature (cold density and distance)")
    print("  • Layer 2: Basins (attraction wells for E and C)")
    print("  • Layer 3: Valley (natural neutral from balance)")
    print("  • Layer 4: Decision (final classification)")
    print()
    
    # Initialize encoder
    encoder = ChainEncoder(vector_size=27)
    print("✓ Chain Encoder initialized")
    
    # Initialize lexicon and basins
    if args.clean:
        SimpleLexicon().clear()
        Layer2Basin.reset_shared_state()
        print("✓ Lexicon cleared (clean start)")
        print("✓ Basins reset")
    else:
        print("✓ Lexicon initialized")
        print("✓ Basins initialized")
    
    print()
    
    # Load data
    train_file = os.path.join(args.data_dir, 'snli_1.0_train.jsonl')
    test_file = os.path.join(args.data_dir, 'snli_1.0_test.jsonl')
    dev_file = os.path.join(args.data_dir, 'snli_1.0_dev.jsonl')
    
    print(f"Loading training data from {train_file}...")
    train_data = load_snli_data(train_file, max_samples=args.train)
    print(f"Loaded {len(train_data)} training examples.")
    
    print(f"Loading test data from {test_file}...")
    test_data = load_snli_data(test_file, max_samples=args.test)
    print(f"Loaded {len(test_data)} test examples.")
    
    print(f"Loading dev data from {dev_file}...")
    dev_data = load_snli_data(dev_file, max_samples=args.dev)
    print(f"Loaded {len(dev_data)} dev examples.")
    
    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print()
    
    # Initialize pattern learner if requested
    pattern_learner = None
    if args.learn_patterns:
        pattern_learner = PatternLearner(debug_mode=args.debug_golden or args.invert_labels)
        if args.invert_labels:
            pattern_learner.invert_mode = True  # Mark as reverse physics mode
            mode_str = "REVERSE PHYSICS MODE"
            print(f"✓ Pattern learner initialized ({mode_str} - will record geometric signals)")
            print("  ⚠️  Labels will be INVERTED (E↔C) to discover invariant geometry")
            print("  ⚠️  NO TRAINING - diagnostic only to find what refuses to flip")
        elif args.debug_golden:
            mode_str = "DEBUG MODE"
            print(f"✓ Pattern learner initialized ({mode_str} - will record geometric signals)")
            print("  ⚠️  In debug mode: Forces will be artificial, but geometric signals are real")
        else:
            mode_str = "NORMAL MODE"
            print(f"✓ Pattern learner initialized ({mode_str} - will record geometric signals)")
        print()
    
    # Training loop
    correct = 0
    y_true_train = []
    y_pred_train = []
    fracture_count = 0
    total_fracture_strength = 0.0
    
    for i, example in enumerate(tqdm(train_data, desc="Training Livnium v5")):
        premise = example['sentence1']
        hypothesis = example['sentence2']
        gold_label = example['gold_label']
        
        # Encode
        pair = encoder.encode_pair(premise, hypothesis)
        
        # REVERSE PHYSICS: Invert labels to discover invariant geometry
        if args.invert_labels:
            # Invert E↔C, keep N as-is (or also invert N - experiment with both)
            label_inverter = {
                'entailment': 'contradiction',
                'contradiction': 'entailment',
                'neutral': 'neutral'  # Keep neutral, or invert to E/C? Experiment!
            }
            inverted_label = label_inverter.get(gold_label, gold_label)
            # REVERSE PHYSICS MODE: Use inverted label but DO NOT set forces
            # Forces compute naturally from geometry - this reveals true invariants
            classifier = LivniumV5Classifier(pair, debug_mode=False, golden_label_hint=inverted_label, reverse_physics_mode=True)
        # DEBUG MODE: Pass golden label to decision layer
        elif args.debug_golden:
            classifier = LivniumV5Classifier(pair, debug_mode=True, golden_label_hint=gold_label)
        else:
            classifier = LivniumV5Classifier(pair)
        
        # Classify
        result = classifier.classify()
        
        # Track fracture detection
        layer_states = result.layer_states
        opposition_output = layer_states.get('layer_opposition', {})
        if opposition_output.get('fracture_detected', False):
            fracture_count += 1
            total_fracture_strength += opposition_output.get('fracture_strength', 0.0)
        
        predicted_label = result.label
        
        # Record patterns if learning patterns
        if pattern_learner is not None:
            if args.invert_labels:
                # Record with INVERTED label - this shows what geometry produces when forced to say wrong thing
                pattern_learner.record(inverted_label, result.layer_states)
            else:
                # Record geometric signals for this golden label
                pattern_learner.record(gold_label, result.layer_states)
        
        # Check correctness (against ORIGINAL gold label, not inverted)
        if predicted_label == gold_label:
            correct += 1
        
        # Apply learning feedback
        # CRITICAL: Do NOT train with inverted labels - that would corrupt the system
        if not args.invert_labels:
            classifier.apply_learning_feedback(gold_label, learning_strength=1.0)
        # In invert mode, we're just diagnosing - no learning!
        
        # Collect for confusion matrix
        y_true_train.append(gold_label)
        y_pred_train.append(predicted_label)
        
        # Log progress
        if (i + 1) % 500 == 0:
            accuracy = correct / (i + 1)
            print(f"Step {i + 1}: Accuracy={accuracy:.3f}")
    
    train_accuracy = correct / len(train_data) if train_data else 0.0
    avg_fracture_strength = total_fracture_strength / fracture_count if fracture_count > 0 else 0.0
    
    print()
    print(f"Training Accuracy: {train_accuracy:.4f} ({correct}/{len(train_data)})")
    print(f"Fractures Detected: {fracture_count}/{len(train_data)} ({100*fracture_count/len(train_data):.1f}%)")
    if fracture_count > 0:
        print(f"Average Fracture Strength: {avg_fracture_strength:.4f}")
    print()
    
    print_confusion_matrix(y_true_train, y_pred_train, "Training Set")
    
    # Analyze patterns if learning
    if pattern_learner is not None:
        print()
        print("=" * 70)
        print("PATTERN ANALYSIS")
        print("=" * 70)
        pattern_learner.analyze()
        pattern_learner.print_analysis()
        
        # Save patterns
        pattern_file = args.pattern_file or os.path.join(
            os.path.dirname(__file__), '..', 'patterns', 'patterns.json'
        )
        pattern_learner.save_patterns(pattern_file)
        print()
    
    # Test evaluation
    print("=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)
    print()
    
    test_correct = 0
    y_true_test = []
    y_pred_test = []
    
    for example in tqdm(test_data, desc="Test Evaluation"):
        premise = example['sentence1']
        hypothesis = example['sentence2']
        gold_label = example['gold_label']
        
        pair = encoder.encode_pair(premise, hypothesis)
        
        # REVERSE PHYSICS: Invert labels for diagnostic
        if args.invert_labels:
            label_inverter = {
                'entailment': 'contradiction',
                'contradiction': 'entailment',
                'neutral': 'neutral'
            }
            inverted_label = label_inverter.get(gold_label, gold_label)
            # REVERSE PHYSICS MODE: Use inverted label but DO NOT set forces
            classifier = LivniumV5Classifier(pair, debug_mode=False, golden_label_hint=inverted_label, reverse_physics_mode=True)
        # Use debug mode if flag is set (for verification)
        elif args.debug_golden:
            classifier = LivniumV5Classifier(pair, debug_mode=True, golden_label_hint=gold_label)
        else:
            classifier = LivniumV5Classifier(pair)
        result = classifier.classify()
        
        # Check against ORIGINAL gold label (not inverted)
        if result.label == gold_label:
            test_correct += 1
        
        y_true_test.append(gold_label)
        y_pred_test.append(result.label)
    
    test_accuracy = test_correct / len(test_data) if test_data else 0.0
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_correct}/{len(test_data)})")
    print()
    
    print_confusion_matrix(y_true_test, y_pred_test, "Test Set")
    
    # Dev evaluation
    print("=" * 70)
    print("DEV EVALUATION")
    print("=" * 70)
    print()
    
    dev_correct = 0
    y_true_dev = []
    y_pred_dev = []
    
    for example in tqdm(dev_data, desc="Dev Evaluation"):
        premise = example['sentence1']
        hypothesis = example['sentence2']
        gold_label = example['gold_label']
        
        pair = encoder.encode_pair(premise, hypothesis)
        
        # REVERSE PHYSICS: Invert labels for diagnostic
        if args.invert_labels:
            label_inverter = {
                'entailment': 'contradiction',
                'contradiction': 'entailment',
                'neutral': 'neutral'
            }
            inverted_label = label_inverter.get(gold_label, gold_label)
            # REVERSE PHYSICS MODE: Use inverted label but DO NOT set forces
            classifier = LivniumV5Classifier(pair, debug_mode=False, golden_label_hint=inverted_label, reverse_physics_mode=True)
        # Use debug mode if flag is set (for verification)
        elif args.debug_golden:
            classifier = LivniumV5Classifier(pair, debug_mode=True, golden_label_hint=gold_label)
        else:
            classifier = LivniumV5Classifier(pair)
        result = classifier.classify()
        
        # Check against ORIGINAL gold label (not inverted)
        if result.label == gold_label:
            dev_correct += 1
        
        y_true_dev.append(gold_label)
        y_pred_dev.append(result.label)
    
    dev_accuracy = dev_correct / len(dev_data) if dev_data else 0.0
    print(f"Dev Accuracy: {dev_accuracy:.4f} ({dev_correct}/{len(dev_data)})")
    print()
    
    print_confusion_matrix(y_true_dev, y_pred_dev, "Dev Set")
    
    # Save brain
    brain_path = os.path.join(os.path.dirname(__file__), '..', 'brain_state', 'brain_state.pkl')
    lexicon = SimpleLexicon()
    lexicon.save_to_file(brain_path)
    print(f"✓ Brain saved to: {brain_path}")
    print(f"  - Words learned: {len(lexicon.polarity_store)}")
    print()
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("Key Features:")
    print("  • Clean 5-layer architecture")
    print("  • Proper 3-class prediction (E/C/N)")
    print("  • Gravity shapes everything")
    print("  • No manual tuning needed")


if __name__ == '__main__':
    main()

