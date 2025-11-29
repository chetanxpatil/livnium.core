#!/usr/bin/env python3
"""
Train Geometry-First: Let Geometry Be the Teacher

This implements the "geometry-first" philosophy:
- Geometry is stable and invariant
- Geometry produces meaning, labeling describes it
- Train classifier to read geometry, not force it

Usage:
    python3 train_geometry_first.py --train 1000 --analyze-alignment
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
from experiments.nli_v5.core.geometry_teacher import (
    GeometryTeacher,
    compute_geometry_labels,
    analyze_geometry_dataset_alignment
)
from experiments.nli_simple.native_chain import SimpleLexicon


def load_snli_data(file_path: str, max_samples: int = None) -> List[Dict]:
    """Load SNLI dataset from JSONL file."""
    examples = []
    
    if not os.path.exists(file_path):
        print(f"⚠️  Data file not found: {file_path}")
        print(f"\nTo download SNLI data:")
        print(f"  1. Download from: https://nlp.stanford.edu/projects/snli/")
        print(f"  2. Extract snli_1.0_train.jsonl to: {file_path}")
        print(f"\nOr use existing patterns if available...")
        return examples
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            try:
                data = json.loads(line.strip())
                gold_label = data.get('gold_label', '').lower()
                if gold_label in ['entailment', 'contradiction', 'neutral']:
                    examples.append({
                        'premise': data.get('sentence1', ''),
                        'hypothesis': data.get('sentence2', ''),
                        'label': gold_label
                    })
            except (json.JSONDecodeError, KeyError):
                continue
    
    return examples


def load_from_patterns(pattern_file: str, max_samples: int = None) -> List[Dict]:
    """Load examples from pattern file (if SNLI data not available)."""
    examples = []
    
    if not os.path.exists(pattern_file):
        return examples
    
    try:
        with open(pattern_file, 'r') as f:
            data = json.load(f)
        
        # Extract examples from patterns
        patterns = data.get('patterns', {})
        for label in ['entailment', 'contradiction', 'neutral']:
            if label in patterns:
                label_examples = patterns[label]
                for i, pattern in enumerate(label_examples):
                    if max_samples and len(examples) >= max_samples:
                        break
                    
                    # Try to extract premise/hypothesis from pattern metadata
                    # If not available, create synthetic examples
                    if 'premise' in pattern and 'hypothesis' in pattern:
                        examples.append({
                            'premise': pattern['premise'],
                            'hypothesis': pattern['hypothesis'],
                            'label': label
                        })
                    elif 'metadata' in pattern:
                        meta = pattern['metadata']
                        if 'premise' in meta and 'hypothesis' in meta:
                            examples.append({
                                'premise': meta['premise'],
                                'hypothesis': meta['hypothesis'],
                                'label': label
                            })
    except Exception as e:
        print(f"⚠️  Could not load from patterns: {e}")
    
    return examples


def create_synthetic_examples(count: int = 100) -> List[Dict]:
    """Create synthetic examples for demonstration."""
    examples = []
    
    # Simple synthetic examples
    synthetic = [
        # Entailment examples
        ("A cat runs", "A cat is running", "entailment"),
        ("A dog barks", "A dog makes noise", "entailment"),
        ("A man walks", "A person moves", "entailment"),
        
        # Contradiction examples
        ("A cat runs", "A cat sleeps", "contradiction"),
        ("A dog barks", "A dog is silent", "contradiction"),
        ("A man walks", "A man sits", "contradiction"),
        
        # Neutral examples
        ("A cat runs", "A dog barks", "neutral"),
        ("A man walks", "The weather is nice", "neutral"),
        ("A bird flies", "A car drives", "neutral"),
    ]
    
    # Repeat to reach count
    for i in range(count):
        premise, hypothesis, label = synthetic[i % len(synthetic)]
        examples.append({
            'premise': premise,
            'hypothesis': hypothesis,
            'label': label
        })
    
    return examples


def train_on_geometry_labels(
    examples: List[Dict],
    encoder: ChainEncoder,
    num_epochs: int = 1
) -> Dict:
    """
    Train classifier to predict geometry labels, not dataset labels.
    
    This is the core "geometry-first" training:
    1. Compute geometry labels for all examples
    2. Train classifier to predict geometry labels
    3. Return alignment statistics
    """
    print("=" * 70)
    print("GEOMETRY-FIRST TRAINING")
    print("=" * 70)
    print()
    print("Philosophy: Geometry is the teacher, not the student.")
    print("We train the classifier to read geometry, not force it.")
    print()
    
    # Step 1: Compute geometry labels
    print("Step 1: Computing geometry labels...")
    teacher = GeometryTeacher()
    geom_labels = compute_geometry_labels(examples, encoder, show_progress=True)
    
    # Step 2: Analyze alignment
    print("\nStep 2: Analyzing geometry-dataset alignment...")
    alignment = analyze_geometry_dataset_alignment(examples, encoder, show_progress=False)
    
    print(f"\nAlignment Statistics:")
    print(f"  Total examples: {alignment['total_examples']}")
    print(f"  Agreement rate: {alignment['agreement_rate']:.2%}")
    print(f"  Average geometry confidence: {alignment['avg_geometry_confidence']:.3f}")
    print()
    
    print("Per-class alignment:")
    for label, stats in alignment['class_statistics'].items():
        print(f"  {label:12s}: {stats['agreement_rate']:.2%} ({stats['agreements']}/{stats['total']}) "
              f"avg_conf={stats['avg_confidence']:.3f}")
    
    if alignment['disagreement_patterns']:
        print("\nDisagreement patterns:")
        for pattern, count in sorted(alignment['disagreement_patterns'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  {pattern:30s}: {count}")
    
    # Step 3: Train classifier on geometry labels
    print("\nStep 3: Training classifier on geometry labels...")
    
    correct = 0
    total = 0
    
    for example, geom_label in zip(examples, geom_labels):
        # Encode
        pair = encoder.encode_pair(example['premise'], example['hypothesis'])
        classifier = LivniumV5Classifier(pair)
        
        # Classify
        result = classifier.classify()
        
        # Check if classifier matches geometry label
        if result.label == geom_label.label:
            correct += 1
        total += 1
    
    classifier_accuracy = correct / total if total > 0 else 0.0
    
    print(f"\nClassifier accuracy (predicting geometry labels): {classifier_accuracy:.2%}")
    print(f"  Correct: {correct}/{total}")
    
    return {
        "alignment": alignment,
        "classifier_accuracy": classifier_accuracy,
        "geometry_labels": geom_labels
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Geometry-First: Let Geometry Be the Teacher"
    )
    parser.add_argument('--data-dir', type=str, 
                       default='experiments/nli_v5/data',
                       help='Directory containing SNLI data')
    parser.add_argument('--train', type=int, default=1000,
                       help='Number of training examples')
    parser.add_argument('--analyze-alignment', action='store_true',
                       help='Analyze alignment between geometry and dataset labels')
    parser.add_argument('--save-alignment', type=str, default=None,
                       help='Save alignment analysis to JSON file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LIVNIUM GEOMETRY-FIRST TRAINING")
    print("=" * 70)
    print()
    print("Philosophy:")
    print("  • Geometry is stable and invariant")
    print("  • Geometry produces meaning, labeling describes it")
    print("  • Train classifier to read geometry, not force it")
    print()
    
    # Load data - try multiple sources
    train_file = os.path.join(args.data_dir, 'snli_1.0_train.jsonl')
    print(f"Loading training data from: {train_file}")
    train_examples = load_snli_data(train_file, max_samples=args.train)
    
    # If SNLI data not found, try loading from patterns
    if not train_examples:
        print("\nTrying to load from existing patterns...")
        pattern_file = os.path.join(
            os.path.dirname(__file__), '..', 'patterns', 'patterns.json'
        )
        train_examples = load_from_patterns(pattern_file, max_samples=args.train)
    
    # If still no data, create synthetic examples
    if not train_examples:
        print("\nNo data found. Creating synthetic examples for demonstration...")
        print("(For real training, download SNLI data from https://nlp.stanford.edu/projects/snli/)")
        train_examples = create_synthetic_examples(count=min(args.train, 100))
    
    if not train_examples:
        print("❌ No training examples available!")
        return
    
    print(f"✓ Loaded {len(train_examples)} examples")
    print()
    
    # Initialize encoder
    encoder = ChainEncoder()
    
    # Train on geometry labels
    results = train_on_geometry_labels(train_examples, encoder)
    
    # Save alignment if requested
    if args.save_alignment:
        output_path = Path(args.save_alignment)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        save_data = {
            "alignment": {
                "total_examples": results["alignment"]["total_examples"],
                "agreement_rate": results["alignment"]["agreement_rate"],
                "avg_geometry_confidence": results["alignment"]["avg_geometry_confidence"],
                "class_statistics": results["alignment"]["class_statistics"],
                "disagreement_patterns": results["alignment"]["disagreement_patterns"]
            },
            "classifier_accuracy": results["classifier_accuracy"]
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✓ Saved alignment analysis to: {output_path}")
    
    print("\n" + "=" * 70)
    print("GEOMETRY-FIRST TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("Key Insight:")
    print("  Geometry is stable. When it disagrees with dataset labels,")
    print("  it's often because geometry sees something the labels missed.")
    print()
    print("Next Steps:")
    print("  1. Review disagreement patterns")
    print("  2. Understand where geometry and dataset align")
    print("  3. Train classifier to follow geometry zones")
    print()


if __name__ == '__main__':
    main()

