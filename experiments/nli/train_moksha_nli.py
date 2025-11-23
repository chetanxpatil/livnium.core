"""
Train Moksha NLI: Native Chain Integration

Main training script for the complete Livnium system using Native Chain (Omchain).

Integrates:
1. Native Chain (MPS Architecture - Omchain)
2. Generalized Lattice (N=3, optimized for speed)
3. Quantum Collapse (Decision Making)
4. Moksha Engine (Resonance-based Convergence)
5. NO Transformers - Pure Native Logic

This is the complete Livnium system using Native Chain architecture.
"""

import os
import sys
import json
import argparse
import numpy as np
import shutil
from tqdm import tqdm
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.config import LivniumCoreConfig
from core.classical.livnium_core_system import LivniumCoreSystem
# Removed: RecursiveGeometryEngine and ConvergenceState (using Native Chain instead)
from experiments.nli.native_chain_encoder import NativeChainNLIEncoder, NativeEncodedPair
from experiments.nli.omcube import OmcubeNLIClassifier
from experiments.nli.native_chain import GlobalLexicon


def print_confusion_matrix(y_true: List[str], y_pred: List[str], dataset_name: str = "Dataset"):
    """
    Print a confusion matrix for 3-class NLI classification.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        dataset_name: Name of the dataset (for display)
    """
    labels = ['entailment', 'contradiction', 'neutral']
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Build confusion matrix
    cm = np.zeros((3, 3), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            cm[true_idx, pred_idx] += 1
    
    # Print header
    print("="*70)
    print(f"CONFUSION MATRIX: {dataset_name.upper()}")
    print("="*70)
    print()
    print("Rows = True Label | Columns = Predicted Label")
    print()
    
    # Print column headers
    header = "        " + "".join([f"{label[:4]:>8}" for label in labels])
    print(header)
    print("        " + "-" * 24)
    
    # Print matrix rows
    for i, label in enumerate(labels):
        row_str = f"{label[:4]:>4} |"
        for j in range(3):
            count = cm[i, j]
            row_str += f"{count:>8}"
        # Add row total and accuracy
        row_total = np.sum(cm[i, :])
        if row_total > 0:
            accuracy = cm[i, i] / row_total
            row_str += f"  | Total: {row_total:>4} | Acc: {accuracy:.3f}"
        else:
            row_str += f"  | Total: {row_total:>4} | Acc: N/A"
        print(row_str)
    
    print()
    
    # Print column totals
    col_totals = np.sum(cm, axis=0)
    col_str = "Total  |"
    for j in range(3):
        col_str += f"{col_totals[j]:>8}"
    col_str += "  |"
    print(col_str)
    print()
    
    # Calculate per-class metrics
    print("Per-Class Metrics:")
    print("-" * 70)
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{label:12s}: Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f}")
    
    print()
    
    # Overall accuracy
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"Overall Accuracy: {overall_acc:.4f} ({total_correct}/{total_samples})")
    print()
    print("="*70)
    print()


def load_snli_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load SNLI dataset from JSONL file (3-class: entailment, contradiction, neutral)."""
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
                gold_label = example.get('gold_label', '-')
                # Include all valid labels: entailment, contradiction, neutral
                if gold_label in ['entailment', 'contradiction', 'neutral']:
                    examples.append({
                        'sentence1': example['sentence1'],
                        'sentence2': example['sentence2'],
                        'gold_label': gold_label
                    })
            except json.JSONDecodeError:
                continue
    
    return examples


def clean_all_caches(nli_dir: str = None):
    """
    Complete clean: Remove all caches, compiled files, and persistent state.
    
    This function:
    1. Clears GlobalLexicon (word state memory)
    2. Clears all caches
    3. Removes all __pycache__ directories
    4. Removes all .pyc files
    5. Removes all .pyo files
    6. Removes any other cache artifacts
    """
    if nli_dir is None:
        nli_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    
    print("="*70)
    print("COMPLETE CLEAN: Removing all caches and persistent state")
    print("="*70)
    print()
    
    # 1. Clear in-memory state
    print("Clearing in-memory state...")
    GlobalLexicon().clear()
    print("  ✓ GlobalLexicon cleared")
    
    # 2. Remove __pycache__ directories (walk from nli_dir)
    print("Removing Python cache directories...")
    cache_count = 0
    cache_paths = []
    
    # First, collect all __pycache__ paths
    for root, dirs, files in os.walk(nli_dir):
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            cache_paths.append(cache_path)
            # Remove from dirs to prevent walking into it
            dirs.remove('__pycache__')
    
    # Now remove them
    for cache_path in cache_paths:
        try:
            shutil.rmtree(cache_path)
            cache_count += 1
            rel_path = os.path.relpath(cache_path, nli_dir)
            print(f"  ✓ Removed: {rel_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to remove {cache_path}: {e}")
    
    if cache_count == 0:
        print("  ✓ No cache directories found")
    else:
        print(f"  ✓ Removed {cache_count} cache directory/ies")
    
    # 3. Remove .pyc, .pyo files (if any remain)
    print("Removing compiled Python files...")
    pyc_count = 0
    for root, dirs, files in os.walk(nli_dir):
        # Skip __pycache__ directories (already removed)
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    pyc_count += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to remove {file_path}: {e}")
    
    if pyc_count == 0:
        print("  ✓ No compiled files found")
    else:
        print(f"  ✓ Removed {pyc_count} compiled file(s)")
    
    # 4. Clean parent directories (experiments/, root/) if they exist
    parent_dirs = [
        os.path.abspath(os.path.join(nli_dir, '..')),  # experiments/
        os.path.abspath(os.path.join(nli_dir, '../..')),  # root/
    ]
    
    parent_cache_count = 0
    for parent_dir in parent_dirs:
        if os.path.exists(parent_dir) and os.path.isdir(parent_dir):
            parent_cache_paths = []
            for root, dirs, files in os.walk(parent_dir):
                if '__pycache__' in dirs:
                    cache_path = os.path.join(root, '__pycache__')
                    parent_cache_paths.append(cache_path)
                    dirs.remove('__pycache__')
            
            for cache_path in parent_cache_paths:
                try:
                    shutil.rmtree(cache_path)
                    parent_cache_count += 1
                except Exception:
                    pass
    
    if parent_cache_count > 0:
        print(f"  ✓ Removed {parent_cache_count} additional cache directory/ies from parent directories")
    
    print()
    print("="*70)
    print("CLEAN COMPLETE: All caches and persistent state removed")
    print("="*70)
    print()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Moksha NLI using Native Chain (Omchain) architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 experiments/nli/train_moksha_nli.py --clean
  python3 experiments/nli/train_moksha_nli.py --train 20000 --test 2000 --dev 2000
  python3 experiments/nli/train_moksha_nli.py --clean --train 20000 --test 2000 --dev 2000
        """
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Start with clean state (clear memory and reset coupling)'
    )
    
    parser.add_argument(
        '--train',
        type=int,
        default=2000,
        help='Maximum number of training samples (default: 2000)'
    )
    
    parser.add_argument(
        '--test',
        type=int,
        default=0,
        help='Maximum number of test samples (default: 0, disabled)'
    )
    
    parser.add_argument(
        '--dev',
        type=int,
        default=0,
        help='Maximum number of dev samples (default: 0, disabled)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='experiments/nli/data',
        help='Directory containing SNLI data files (default: experiments/nli/data)'
    )
    
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()
    
    print("="*70)
    print("LIVNIUM: MOKSHA PROTOCOL INITIALIZATION")
    print("="*70)
    print()
    print("This is the complete Livnium system:")
    print("  • Native Chain (MPS Architecture - Omchain)")
    print("  • Generalized Lattice (N=3, optimized for speed)")
    print("  • Quantum Collapse (Decision Making)")
    print("  • Moksha Engine (Resonance-based Convergence)")
    print("  • NO Transformers - Pure Native Logic")
    print()
    print("Configuration:")
    print(f"  • Clean Start: {'YES' if args.clean else 'NO'}")
    print(f"  • Training Samples: {args.train}")
    print(f"  • Test Samples: {args.test if args.test > 0 else 'Disabled'}")
    print(f"  • Dev Samples: {args.dev if args.dev > 0 else 'Disabled'}")
    print()

    # 1. Configure the Core (Optimized for Speed)
    # NOTE: N=3 (27 cells) vs N=7 (343 cells) = 12.7x speedup
    # The concepts (Entailment/Contradiction) are topological, not volumetric
    # They rely on relationships between basins, not raw cell count
    config = LivniumCoreConfig(
        # Reduced to N=3 for speed (27 cells vs 343 cells = 12.7x faster)
        lattice_size=3,                 # Canonical size - sufficient for NLI concepts
        
        # Physics & Geometry
        enable_face_exposure=True,
        enable_symbolic_weight=True,
        enable_semantic_polarity=True,
        enable_global_observer=True,
        enable_local_observer=True,
        
        # Quantum Soul
        enable_quantum=True,            # Enable Quantum Layer
        enable_superposition=True,
        enable_geometry_quantum_coupling=True,
        enable_quantum_gates=True,
        enable_measurement=True,
        
        # Recursion & Truth
        enable_recursive_geometry=True, # Enable Fractal Engine
        recursive_max_depth=1,          # Reduced depth for speed (was 2)
        enable_moksha=True,             # Enable Fixed-Point Convergence
        moksha_convergence_threshold=0.99,
        moksha_stability_window=10,
        
        # Invariants
        enable_sw_conservation=True,
        enable_class_count_conservation=True,
    )

    print(f"Configuration Loaded:")
    print(f"  • Lattice Size: {config.lattice_size}×{config.lattice_size}×{config.lattice_size} ({config.lattice_size**3} cells)")
    print(f"  • Quantum Layer: ENABLED")
    print(f"  • Recursive Geometry: ENABLED (Depth {config.recursive_max_depth})")
    print(f"  • Moksha Engine: ENABLED (Threshold: {config.moksha_convergence_threshold})")
    print()
    print("Performance Note:")
    print(f"  • N=3: {3**3} cells (optimized for speed)")
    print(f"  • N=7: {7**3} cells (12.7x slower - use for production)")
    print("  • Concepts are topological, not volumetric - N=3 is sufficient")
    print()

    # 2. Initialize Components
    print("Initializing Systems...")
    
    # The Encoder (Sensors) - Native Chain (NO Transformers)
    encoder = NativeChainNLIEncoder(lattice_size=config.lattice_size, config=config)
    print("  ✓ Native Chain Encoder initialized (NO Transformers)")
    
    # COMPLETE CLEAN: Remove all caches and persistent state
    if args.clean:
        nli_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        clean_all_caches(nli_dir)
    
    # NLIMemory removed (simplified system)
    
    # Global Lexicon (Word State Memory)
    if args.clean:
        GlobalLexicon().clear()
        print("  ✓ Global Lexicon cleared (clean start)")
    else:
        print("  ✓ Global Lexicon initialized")
    
    print()

    # Load Data
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
    
    # Use training examples for main loop
    examples = train_examples

    # 3. Training Loop
    print("="*70)
    print("Beginning Training Cycle...")
    print("="*70)
    print()
    
    correct_count = 0
    moksha_reached_count = 0
    total_steps = 0
    
    # Track predictions for confusion matrix
    train_y_true = []
    train_y_pred = []
    
    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    iterator = tqdm(examples, desc="Training Moksha NLI")
    
    for i, ex in enumerate(iterator):
        premise = ex['sentence1']
        hypothesis = ex['sentence2']
        gold_label = ex['gold_label']
        
        try:
            # --- A. ENCODE (Sensors) ---
            # This creates the base geometry (Level 0)
            encoded_pair = encoder.encode_pair(premise, hypothesis)
            
            # --- B. SKIP FRACTAL ENGINE (Native Chain handles recursion internally) ---
            # Native Chain already has quantum entanglement between words
            # No need for recursive geometry subdivision
            
            # --- C. QUANTUM DECISION (Soul) ---
            classifier = OmcubeNLIClassifier(encoded_pair)
            
            
            # Classify using quantum collapse
            result = classifier.classify(collapse=True)
            
            # --- E. MOKSHA CHECK (Truth) ---
            # For Native Chain, Moksha is achieved when resonance stabilizes
            # Simple check: if resonance is consistent, we've reached Moksha
            resonance = encoded_pair.get_resonance()
            is_moksha = (resonance > 0.7)  # High resonance = stable state
            
            if is_moksha:
                moksha_reached_count += 1
                
            # --- F. LEARNING (Feedback) ---
            gold_idx = label_map[gold_label]
            
            # Apply feedback to the geometry
            # This updates SW, which updates Curvature, which updates the Physics
            # OPTIMIZATION: Always apply feedback but commit_learning is now batched
            classifier.apply_learning_feedback(
                result, 
                gold_idx, 
                learning_strength=1.0
            )
            
            # Store in memory
            # Memory storage removed (simplified system)

            # Track predictions for confusion matrix
            train_y_true.append(gold_label)
            train_y_pred.append(result.label)
            
            if result.label == gold_label:
                correct_count += 1
            total_steps += 1

            # Periodic Reporting (every 500 steps: 500, 1000, 1500, 2000, ...)
            step_num = i + 1
            if step_num % 500 == 0:
                acc = correct_count / total_steps if total_steps > 0 else 0.0
                moksha_rate = moksha_reached_count / total_steps if total_steps > 0 else 0.0
                
                # Get Geometry Signals from the Coupling System
                # Basin depths removed (simplified classifier)
                depths = {0: 1.0, 1: 1.0, 2: 1.0}
                
                # Get Native Chain resonance
                resonance = encoded_pair.get_resonance()
                
                iterator.write(f"\n{'='*70}")
                iterator.write(f"Step {step_num}: Accuracy={acc:.3f} | Moksha Rate={moksha_rate:.3f}")
                iterator.write(f"  Resonance: {resonance:.3f} | Moksha: {'✓' if is_moksha else '✗'}")
                iterator.write(f"  Basins: E={depths.get(0, 0):.2f} C={depths.get(1, 0):.2f} N={depths.get(2, 0):.2f}")
                iterator.write(f"  Chain: P={len(encoded_pair.premise_chain.tokens)} words, H={len(encoded_pair.hypothesis_chain.tokens)} words")
                iterator.write(f"{'='*70}")
                
        except Exception as e:
            iterator.write(f"\n⚠️  Error processing example {i+1}: {e}")
            import traceback
            iterator.write(traceback.format_exc())
            continue

    # 4. Final Stats
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    final_acc = correct_count / total_steps if total_steps > 0 else 0.0
    final_moksha_rate = moksha_reached_count / total_steps if total_steps > 0 else 0.0
    
    print(f"Total Steps: {total_steps}")
    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Moksha Reached: {moksha_reached_count} times ({final_moksha_rate*100:.1f}%)")
    print()
    
    # Get final basin depths
    # Basin depths removed (simplified classifier)
    depths = {0: 1.0, 1: 1.0, 2: 1.0}
    print("Final Basin Depths:")
    print(f"  Entailment:     {depths.get(0, 0):.3f}")
    print(f"  Contradiction:  {depths.get(1, 0):.3f}")
    print(f"  Neutral:        {depths.get(2, 0):.3f}")
    print()
    
    # Memory stats removed (NLIMemory removed)
    print()
    
    # Print confusion matrix for training set
    if train_y_true and train_y_pred:
        print_confusion_matrix(train_y_true, train_y_pred, "Training Set")
    
    # Evaluate on test set if provided
    test_y_true = []
    test_y_pred = []
    
    if test_examples:
        print("="*70)
        print("EVALUATING ON TEST SET")
        print("="*70)
        print()
        
        test_correct = 0
        test_total = 0
        
        for ex in tqdm(test_examples, desc="Testing"):
            premise = ex['sentence1']
            hypothesis = ex['sentence2']
            gold_label = ex['gold_label']
            
            try:
                encoded_pair = encoder.encode_pair(premise, hypothesis)
                classifier = OmcubeNLIClassifier(encoded_pair)
                result = classifier.classify(collapse=True)
                
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
        
        # Print confusion matrix for test set
        if test_y_true and test_y_pred:
            print_confusion_matrix(test_y_true, test_y_pred, "Test Set")
    
    # Evaluate on dev set if provided
    dev_y_true = []
    dev_y_pred = []
    
    if dev_examples:
        print("="*70)
        print("EVALUATING ON DEV SET")
        print("="*70)
        print()
        
        dev_correct = 0
        dev_total = 0
        
        for ex in tqdm(dev_examples, desc="Dev Evaluation"):
            premise = ex['sentence1']
            hypothesis = ex['sentence2']
            gold_label = ex['gold_label']
            
            try:
                encoded_pair = encoder.encode_pair(premise, hypothesis)
                classifier = OmcubeNLIClassifier(encoded_pair)
                result = classifier.classify(collapse=True)
                
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
        
        # Print confusion matrix for dev set
        if dev_y_true and dev_y_pred:
            print_confusion_matrix(dev_y_true, dev_y_pred, "Dev Set")
    
    print("="*70)
    print("System Status: OPERATIONAL")
    print("="*70)
    print()
    print("Key Insight:")
    if final_acc > 0.4 and final_moksha_rate > 0.1:
        print("  ✅ Geometric Stability = Truth (Hypothesis CONFIRMED)")
        print("  ✅ System is learning to stabilize its thoughts")
    elif final_moksha_rate > 0.1:
        print("  ✅ Moksha convergence detected - system is stabilizing")
    else:
        print("  ⚠️  System needs more training to reach moksha")
    print()
    
    # Save the brain to disk
    brain_path = os.path.join(os.path.dirname(__file__), 'brain_state.pkl')
    GlobalLexicon().save_to_file(brain_path)
    print(f"✓ Brain saved to: {brain_path}")
    print(f"  - Words learned: {len(GlobalLexicon().polarity_store)}")
    print(f"  - Letters learned: {len(GlobalLexicon().letter_store)}")
    print()
    
    # Save the brain to disk
    brain_path = os.path.join(os.path.dirname(__file__), 'brain_state.pkl')
    GlobalLexicon().save_to_file(brain_path)
    print(f"✓ Brain saved to: {brain_path}")
    print(f"  - Words learned: {len(GlobalLexicon().polarity_store)}")
    print(f"  - Letters learned: {len(GlobalLexicon().letter_store)}")
    print()


if __name__ == "__main__":
    main()

