"""
Livnium NLI v4 Training: Layered Core Architecture

7-layer geological architecture - each layer builds on the one below.
Gravity shapes everything. No manual tuning.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.nli_v4 import LayeredLivniumClassifier
from experiments.nli_v4.layer2_basin import Layer2Basin
from experiments.nli_v4.cluster_tracker import ClusterTracker
from experiments.nli_v4.frozen_basin_centers import FrozenBasinCenters
from experiments.nli_v4.feature_logger import FeatureLogger
from experiments.nli_v4.auto_rule_updater import AutoRuleUpdater
from experiments.nli_v3.chain_encoder import ChainEncoder
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
    """Print confusion matrix in AI-readable format (JSON + human-readable)."""
    import json
    
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
    
    # Create structured output
    structured_output = {
        'title': title,
        'confusion_matrix': matrix,
        'metrics': metrics,
        'overall_accuracy': float(accuracy),
        'total_samples': int(total),
        'class_totals': {label: sum(matrix[label].values()) for label in labels},
        'prediction_totals': {label: sum(matrix[t][label] for t in labels) for label in labels}
    }
    
    # Print JSON (AI-readable)
    print(f"\n{title} Confusion Matrix (JSON):")
    print("=" * 70)
    print(json.dumps(structured_output, indent=2))
    print("=" * 70)
    
    # Also print human-readable format
    print(f"\n{title} Confusion Matrix (Human-readable):")
    print("=" * 50)
    header_label = "True \\ Predicted"
    print(f"{header_label:<20} {'E':<8} {'C':<8} {'N':<8} {'Total':<8}")
    print("-" * 50)
    
    # Print rows
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
    
    # Print column totals
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
    parser = argparse.ArgumentParser(description='Train Livnium NLI v4 (Layered Architecture)')
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
    parser.add_argument('--unsupervised', action='store_true',
                        help='Unsupervised mode: no labels, just physics (geometry discovers meaning)')
    parser.add_argument('--cluster-output', type=str, default=None,
                        help='Directory to save geometry-discovered clusters')
    parser.add_argument('--log-features', type=str, default=None,
                        help='Path to CSV file for logging geometric features (for rule discovery)')
    parser.add_argument('--auto-rules', action='store_true',
                        help='Enable auto rule discovery and updating during training')
    parser.add_argument('--rule-update-interval', type=int, default=1000,
                        help='Update rules every N training steps (requires --auto-rules)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LIVNIUM NLI v4: LAYERED CORE ARCHITECTURE")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  • Layer 0: Pure Resonance (bedrock)")
    print("  • Layer 1: Curvature (field shape)")
    print("  • Layer 2: Basin (attraction wells)")
    print("  • Layer 3: Valley (natural neutral)")
    print("  • Layer 4: Meta Routing (reads geometry)")
    print("  • Layer 5: Temporal Stability (Moksha)")
    print("  • Layer 6: Semantic Memory (word polarities)")
    print("  • Layer 7: Decision (final classification)")
    print()
    
    # Initialize encoder
    encoder = ChainEncoder(vector_size=27)
    print("✓ Chain Encoder initialized")
    
    # Initialize lexicon
    if args.clean:
        SimpleLexicon().clear()
        Layer2Basin.reset_shared_state()  # Reset shared basins
        print("✓ Lexicon cleared (clean start)")
        print("✓ Shared basins reset")
    else:
        print("✓ Lexicon initialized")
        print("✓ Shared basins initialized")
    
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
    if args.unsupervised:
        print("UNSUPERVISED TRAINING (Geometry Discovers Meaning)")
    else:
        print("TRAINING")
    print("=" * 70)
    print()
    
    # Initialize cluster tracker (for unsupervised mode)
    cluster_tracker = ClusterTracker() if args.unsupervised else None
    
    # Initialize frozen basin centers (for stabilizing the three-phase universe)
    frozen_centers = FrozenBasinCenters(vector_size=27, ema_alpha=0.1) if args.unsupervised else None
    
    # Initialize feature logger (for rule discovery)
    feature_logger = FeatureLogger(args.log_features) if args.log_features else None
    
    # Initialize auto rule updater (for self-evolving rules)
    auto_rule_updater = None
    if args.auto_rules:
        features_file = args.log_features or "experiments/nli_v4/features_auto.csv"
        if not args.log_features:
            # Auto-enable feature logging if auto-rules is enabled
            feature_logger = FeatureLogger(features_file)
        auto_rule_updater = AutoRuleUpdater(
            features_file=features_file,
            update_interval=args.rule_update_interval
        )
        print(f"✓ Auto rule updater enabled (updates every {args.rule_update_interval} steps)")
        print()
    
    # Training loop
    correct = 0
    moksha_count = 0
    y_true_train = []
    y_pred_train = []
    
    for i, example in enumerate(tqdm(train_data, desc="Training Livnium v4")):
        premise = example['sentence1']
        hypothesis = example['sentence2']
        gold_label = example['gold_label']
        
        # Encode
        pair = encoder.encode_pair(premise, hypothesis)
        classifier = LayeredLivniumClassifier(pair)
        
        # Classify
        result = classifier.classify()
        predicted_label = result.label
        basin_index = result.basin_index  # Geometry-discovered cluster
        
        # Extract forces for label prediction (works in both modes)
        basin_forces = result.layer_states.get('basin_forces', {})
        cold_force = basin_forces.get('basin_0_cold', 0.33)
        far_force = basin_forces.get('basin_1_far', 0.33)
        city_force = basin_forces.get('basin_2_city', 0.33)
        
        # Get E/N/C label from force competition
        predicted_enc_label = classifier.layer7.decide(cold_force, far_force, city_force)
        
        # Log geometric features for rule discovery (if enabled)
        if feature_logger is not None:
            # Extract geometric features
            features = classifier.extract_geometric_features(result)
            
            # Map gold label to E/N/C format
            label_map = {'entailment': 'E', 'contradiction': 'C', 'neutral': 'N'}
            true_label_enc = label_map.get(gold_label, '')
            
            # Log features
            feature_logger.log(features, predicted_enc_label, true_label_enc)
        
        # Unsupervised mode: track clusters with predicted labels and frozen centers
        if args.unsupervised:
            # Add sentence vector to frozen centers
            if frozen_centers is not None:
                # Get sentence vector representation (combine premise + hypothesis)
                premise_vec = pair.premise.get_sentence_vector()
                hypothesis_vec = pair.hypothesis.get_sentence_vector()
                # Combine vectors (simple concatenation and normalize)
                combined_vec = np.concatenate([premise_vec, hypothesis_vec])
                # Or use average (simpler)
                combined_vec = (premise_vec + hypothesis_vec) / 2.0
                frozen_centers.add_vector(result.basin_index, combined_vec)
            # Track which basin this sentence fell into (AME may have modified it)
            final_basin = result.basin_index  # Use AME-modified basin
            
            # Track assignment in AME
            classifier.ame.track_assignment(final_basin)
            
            cluster_tracker.add(
                basin_index=final_basin,
                premise=premise,
                hypothesis=hypothesis,
                confidence=result.confidence,
                layer_states=result.layer_states,
                predicted_label=predicted_enc_label  # Add predicted E/N/C label
            )
            
            # STEP 2: Competitive word polarity updates
            # Basins compete for words - winning basin pulls harder
            tokens = set(pair.premise.tokens + pair.hypothesis.tokens)
            basin_names = ['cold', 'far', 'city']
            basin_force_key = f'basin_{final_basin}_{basin_names[final_basin]}'
            basin_force = basin_forces.get(basin_force_key, 0.5)  # Default 0.5 if not found
            
            # Update word polarities competitively
            classifier.layer6.update_competitive(
                tokens=tokens,
                basin_index=final_basin,
                basin_force=basin_force,
                strength=1.0
            )
        
        else:
            # Supervised mode: check correctness and apply label-based feedback
            if predicted_label == gold_label:
                correct += 1
            
            # Apply learning feedback (label-based)
            classifier.apply_learning_feedback(gold_label, learning_strength=1.0)
        
        # Collect for confusion matrix
        y_true_train.append(gold_label)
        y_pred_train.append(predicted_label)
        
        # Track Moksha (works in both modes)
        if result.is_moksha:
            moksha_count += 1
        
        # Auto rule update (if enabled)
        if auto_rule_updater is not None:
            auto_rule_updater.update_loop(i + 1, classifier.layer7)
        
        # Log progress
        if (i + 1) % 500 == 0:
            moksha_rate = moksha_count / (i + 1)
            
            # Update frozen centers periodically (every 500 steps)
            if args.unsupervised and frozen_centers is not None:
                frozen_centers.update_frozen_centers()
            
            # Get physics state from last result
            physics_state = result.layer_states
            entropy = physics_state.get('entropy', 0.0)
            imbalance = physics_state.get('class_imbalance', 0.0)
            temp = physics_state.get('temperature', 0.0)
            
            if args.unsupervised:
                # Unsupervised: show cluster distribution + AME stats + frozen centers
                cluster_stats = cluster_tracker.get_statistics()
                ame_stats = classifier.ame.get_statistics()
                turbulence = physics_state.get('turbulence', 0.0)
                active_basins = physics_state.get('active_basins', 3)
                
                frozen_stats = frozen_centers.get_statistics() if frozen_centers else {}
                frozen_info = ""
                if frozen_stats.get('cold_center_initialized'):
                    frozen_info = f" | Frozen Centers: C={frozen_stats['cold_count']} F={frozen_stats['far_count']} N={frozen_stats['city_count']}"
                
                # Show predicted label from last example
                print(f"Step {i + 1}: Basin 0={cluster_stats['basin_0_cold']['count']} | "
                      f"Basin 1={cluster_stats['basin_1_far']['count']} | "
                      f"Basin 2={cluster_stats['basin_2_city']['count']} | "
                      f"Pred={predicted_enc_label} | "
                      f"Moksha={moksha_rate:.3f} | "
                      f"Turbulence={turbulence:.4f} | Active Basins={active_basins}{frozen_info}")
            else:
                # Supervised: show accuracy
                accuracy = correct / (i + 1)
                print(f"Step {i + 1}: Accuracy={accuracy:.3f} | Moksha={moksha_rate:.3f} | "
                      f"Entropy={entropy:.4f} | Imbalance={imbalance:.3f} | Temp={temp:.3f}")
    
    train_moksha_rate = moksha_count / len(train_data) if train_data else 0.0
    
    # Close feature logger
    if feature_logger is not None:
        feature_logger.close()
        print(f"✓ Geometric features logged to: {args.log_features}")
        print()
    
    # Final update of frozen centers
    if args.unsupervised and frozen_centers is not None:
        frozen_centers.update_frozen_centers()
        frozen_stats = frozen_centers.get_statistics()
        print(f"✓ Frozen basin centers updated:")
        print(f"  - Cold (E): {frozen_stats['cold_count']} vectors, initialized={frozen_stats['cold_center_initialized']}")
        print(f"  - Far (C): {frozen_stats['far_count']} vectors, initialized={frozen_stats['far_center_initialized']}")
        print(f"  - City (N): {frozen_stats['city_count']} vectors, initialized={frozen_stats['city_center_initialized']}")
        print()
    
    print()
    if args.unsupervised:
        print("UNSUPERVISED TRAINING COMPLETE")
        print(f"Moksha Rate: {train_moksha_rate:.4f} ({moksha_count}/{len(train_data)})")
        print()
        
        # Print cluster statistics
        cluster_tracker.print_statistics()
        
        # Export clusters if output directory specified
        if args.cluster_output:
            output_path = Path(args.cluster_output)
            summary = cluster_tracker.export_clusters(output_path)
            print(f"✓ Clusters exported to: {output_path}")
            print(f"  - Basin 0 (Cold): {summary['statistics']['basin_0_cold']['count']} entries")
            print(f"  - Basin 1 (Far): {summary['statistics']['basin_1_far']['count']} entries")
            print(f"  - Basin 2 (City): {summary['statistics']['basin_2_city']['count']} entries")
            print()
    else:
        train_accuracy = correct / len(train_data) if train_data else 0.0
        print(f"Training Accuracy: {train_accuracy:.4f} ({correct}/{len(train_data)})")
        print(f"Moksha Rate: {train_moksha_rate:.4f} ({moksha_count}/{len(train_data)})")
        print()
    
    print_confusion_matrix(y_true_train, y_pred_train, "Training Set")
    
    # Test evaluation (skip in unsupervised mode)
    if not args.unsupervised:
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
            classifier = LayeredLivniumClassifier(pair)
            result = classifier.classify()
            
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
            classifier = LayeredLivniumClassifier(pair)
            result = classifier.classify()
            
            if result.label == gold_label:
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
    print("Layered Architecture Benefits:")
    print("  • Two peaks always stay peaks")
    print("  • Valley forms only where curvature overlaps")
    print("  • Neutral cannot corrupt E/C")
    print("  • No manual tuning needed")
    print("  • Gravity shapes everything")


if __name__ == '__main__':
    main()

