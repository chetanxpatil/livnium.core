"""
Train Livnium NLI v6: Corrected 3-Axis Manifold

Uses ONLY invariant signals discovered through reverse physics:
- Opposition = Resonance × Sign(Divergence)
- Cold Attraction (invariant)
- Curvature (perfect invariant)

Expected: 36% → 45-50% accuracy
"""

import os
import sys
import argparse
import json
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.nli_v6.encoder import ChainEncoder
from experiments.nli_v6.classifier import LivniumV6Classifier
from experiments.nli_simple.native_chain import SimpleLexicon


def load_snli_data(file_path: str, max_examples: int = None):
    """Load SNLI data from JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_examples and len(examples) >= max_examples:
                break
            data = json.loads(line.strip())
            if data['gold_label'] in ['entailment', 'contradiction', 'neutral']:
                examples.append({
                    'sentence1': data['sentence1'],
                    'sentence2': data['sentence2'],
                    'gold_label': data['gold_label']
                })
    return examples


def print_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Print confusion matrix."""
    from sklearn.metrics import confusion_matrix, classification_report
    
    labels = ['entailment', 'contradiction', 'neutral']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    print(f"\n{title} (JSON):")
    print("=" * 70)
    cm_dict = {
        "title": title,
        "confusion_matrix": {
            labels[i]: {
                labels[j]: int(cm[i, j]) for j in range(len(labels))
            } for i in range(len(labels))
        },
        "metrics": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    }
    
    # Add overall accuracy
    correct = sum(y_true[i] == y_pred[i] for i in range(len(y_true)))
    cm_dict["overall_accuracy"] = correct / len(y_true) if y_true else 0.0
    cm_dict["total_samples"] = len(y_true)
    cm_dict["class_totals"] = {label: y_true.count(label) for label in labels}
    cm_dict["prediction_totals"] = {label: y_pred.count(label) for label in labels}
    
    print(json.dumps(cm_dict, indent=2))
    print("=" * 70)
    
    # Human-readable
    print(f"\n{title} (Human-readable):")
    print("=" * 50)
    true_predicted = "True \\ Predicted"
    print(f"{true_predicted:<15} {'E':<10} {'C':<10} {'N':<10} {'Total':<10}")
    print("-" * 50)
    for i, label in enumerate(labels):
        row_sum = sum(cm[i, :])
        print(f"{label[:14]:<15} {cm[i,0]:<10} {cm[i,1]:<10} {cm[i,2]:<10} {row_sum:<10}")
    print("-" * 50)
    col_sums = [sum(cm[:, j]) for j in range(len(labels))]
    print(f"{'Total':<15} {col_sums[0]:<10} {col_sums[1]:<10} {col_sums[2]:<10} {sum(col_sums):<10}")
    print("=" * 50)
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    print("\nPer-Class Metrics:")
    print("-" * 50)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print("-" * 50)
    for label in labels:
        if label in report:
            print(f"{label:<15} {report[label]['precision']:<12.4f} {report[label]['recall']:<12.4f} "
                  f"{report[label]['f1-score']:<12.4f} {report[label]['support']:<12.0f}")
    print("-" * 50)
    print(f"{'Overall Accuracy':<15} {cm_dict['overall_accuracy']:<12.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Train Livnium NLI v6')
    parser.add_argument('--data-dir', type=str, default='experiments/nli/data',
                        help='Directory containing SNLI data files')
    parser.add_argument('--train', type=int, default=None,
                        help='Maximum number of training examples')
    parser.add_argument('--test', type=int, default=None,
                        help='Maximum number of test examples')
    parser.add_argument('--dev', type=int, default=None,
                        help='Maximum number of dev examples')
    parser.add_argument('--clean', action='store_true',
                        help='Start with clean state (clear lexicon)')
    parser.add_argument('--debug-golden', action='store_true',
                        help='DEBUG: Feed golden labels to decision layer')
    parser.add_argument('--force-incorrect', action='store_true',
                        help='DEBUG: Force INCORRECT labels (invert E↔C) to see geometry\'s true prediction')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LIVNIUM NLI v6: CORRECTED 3-AXIS MANIFOLD")
    print("=" * 70)
    print()
    
    if args.force_incorrect:
        print("⚠️  FORCE INCORRECT MODE: Labels will be INVERTED (E↔C) in Layer 4")
        print("   Geometry (Layers 0-3) is REAL, but decision uses INCORRECT label")
        print("   This reveals what geometry truly thinks vs what we force it to say")
        print("   Use this to discover invariant signals")
        print()
    elif args.debug_golden:
        print("⚠️  DEBUG MODE: Golden labels will be forced in Layer 4")
        print("   Geometry (Layers 0-3) is REAL, but decision uses golden label")
        print("   Use this to verify decision logic and record patterns")
        print()
    
    print("Architecture:")
    print("  • Layer 0: Resonance (raw geometric signal)")
    print("  • Layer 1: Curvature (cold density and distance)")
    print("  • Layer 2: Opposition (NEW: resonance × sign(divergence))")
    print("  • Layer 3: Attraction (cold/far/city)")
    print("  • Layer 4: Decision (simplified - uses only invariant signals)")
    print()
    print("Key Innovation:")
    print("  • Uses ONLY invariant signals (Resonance, Cold Attraction, Curvature)")
    print("  • Removes noise from divergence magnitude")
    print("  • Expected: 36% → 45-50% accuracy")
    print()
    
    # Initialize encoder
    encoder = ChainEncoder(vector_size=27)
    print("✓ Chain Encoder initialized")
    
    # Initialize lexicon
    if args.clean:
        SimpleLexicon().clear()
        print("✓ Lexicon cleared (clean start)")
    else:
        print("✓ Lexicon initialized")
    
    print()
    
    # Load data
    data_dir = args.data_dir
    train_file = os.path.join(data_dir, 'snli_1.0_train.jsonl')
    test_file = os.path.join(data_dir, 'snli_1.0_test.jsonl')
    dev_file = os.path.join(data_dir, 'snli_1.0_dev.jsonl')
    
    print(f"Loading training data from {train_file}...")
    train_data = load_snli_data(train_file, args.train)
    print(f"Loaded {len(train_data)} training examples.")
    
    print(f"Loading test data from {test_file}...")
    test_data = load_snli_data(test_file, args.test)
    print(f"Loaded {len(test_data)} test examples.")
    
    print(f"Loading dev data from {dev_file}...")
    dev_data = load_snli_data(dev_file, args.dev)
    print(f"Loaded {len(dev_data)} dev examples.")
    
    print()
    
    # Training loop
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print()
    
    correct = 0
    y_true_train = []
    y_pred_train = []
    
    for i, example in enumerate(tqdm(train_data, desc="Training Livnium v6")):
        premise = example['sentence1']
        hypothesis = example['sentence2']
        gold_label = example['gold_label']
        
        # Encode
        pair = encoder.encode_pair(premise, hypothesis)
        
        # Classify
        if args.force_incorrect:
            # FORCE INCORRECT MODE: Invert labels to see geometry's true prediction
            classifier = LivniumV6Classifier(pair, debug_mode=True, golden_label_hint=gold_label, force_incorrect=True)
        elif args.debug_golden:
            # DEBUG MODE: Force golden label in Layer 4 (verify decision logic)
            classifier = LivniumV6Classifier(pair, debug_mode=True, golden_label_hint=gold_label)
        else:
            classifier = LivniumV6Classifier(pair)
        
        result = classifier.classify()
        predicted_label = result.label
        
        # Check correctness
        # In force_incorrect mode, predicted_label will be WRONG (inverted), but geometry_prediction shows truth
        # In debug mode, predicted_label will match gold_label (forced), but geometry is real
        if args.force_incorrect:
            # Check if geometry's prediction matches golden label (geometry is correct!)
            geometry_pred = result.layer_states.get('geometry_prediction', predicted_label)
            if geometry_pred == gold_label:
                correct += 1
        elif predicted_label == gold_label:
            correct += 1
        
        # Apply learning feedback
        classifier.apply_learning_feedback(gold_label, learning_strength=1.0)
        
        # Collect for confusion matrix
        y_true_train.append(gold_label)
        y_pred_train.append(predicted_label)
        
        # Log progress
        if (i + 1) % 500 == 0:
            accuracy = correct / (i + 1)
            print(f"Step {i + 1}: Accuracy={accuracy:.3f}")
    
    train_accuracy = correct / len(train_data) if train_data else 0.0
    print()
    print(f"Training Accuracy: {train_accuracy:.4f} ({correct}/{len(train_data)})")
    print()
    
    print_confusion_matrix(y_true_train, y_pred_train, "Training Set")
    
    # Test evaluation
    print()
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
        if args.force_incorrect:
            classifier = LivniumV6Classifier(pair, debug_mode=True, golden_label_hint=gold_label, force_incorrect=True)
        elif args.debug_golden:
            classifier = LivniumV6Classifier(pair, debug_mode=True, golden_label_hint=gold_label)
        else:
            classifier = LivniumV6Classifier(pair)
        result = classifier.classify()
        
        if args.force_incorrect:
            geometry_pred = result.layer_states.get('geometry_prediction', result.label)
            if geometry_pred == gold_label:
                test_correct += 1
        elif result.label == gold_label:
            test_correct += 1
        
        y_true_test.append(gold_label)
        y_pred_test.append(result.label)
    
    test_accuracy = test_correct / len(test_data) if test_data else 0.0
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_correct}/{len(test_data)})")
    print()
    
    print_confusion_matrix(y_true_test, y_pred_test, "Test Set")
    
    # Dev evaluation
    print()
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
        if args.force_incorrect:
            classifier = LivniumV6Classifier(pair, debug_mode=True, golden_label_hint=gold_label, force_incorrect=True)
        elif args.debug_golden:
            classifier = LivniumV6Classifier(pair, debug_mode=True, golden_label_hint=gold_label)
        else:
            classifier = LivniumV6Classifier(pair)
        result = classifier.classify()
        
        if args.force_incorrect:
            geometry_pred = result.layer_states.get('geometry_prediction', result.label)
            if geometry_pred == gold_label:
                dev_correct += 1
        elif result.label == gold_label:
            dev_correct += 1
        
        y_true_dev.append(gold_label)
        y_pred_dev.append(result.label)
    
    dev_accuracy = dev_correct / len(dev_data) if dev_data else 0.0
    print(f"Dev Accuracy: {dev_accuracy:.4f} ({dev_correct}/{len(dev_data)})")
    print()
    
    print_confusion_matrix(y_true_dev, y_pred_dev, "Dev Set")
    
    # Save brain
    brain_path = os.path.join(os.path.dirname(__file__), 'brain_state.pkl')
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
    print("  • Corrected 3-axis manifold")
    print("  • Uses ONLY invariant signals")
    print("  • Opposition axis: resonance × sign(divergence)")
    print("  • Removes noise from divergence magnitude")


if __name__ == '__main__':
    main()

