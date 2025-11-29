"""
SNLI Phase 1 Training: Teach Nova to say E/C/N

Phase 1 Goal: Discipline and Basin Collapse
- Output ONLY: "entailment", "contradiction", or "neutral"
- Reward: Correct label
- Punish: Wrong label OR any extra words

Uses EXISTING Nova training pipeline:
- TextToGeometry for signature extraction
- GeometricTokenLearner for clustering
- ClusterDecoder for generation (modified to output only E/C/N)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nova.core.text_to_geometry import TextToGeometry
from nova.training.train_text_to_geometry import train_model, extract_signatures


def load_snli_data(jsonl_path: Path, max_samples: int = None) -> List[Dict]:
    """
    Load SNLI data from JSONL file.
    
    Args:
        jsonl_path: Path to SNLI JSONL file
        max_samples: Maximum number of samples to load (None = all)
        
    Returns:
        List of dicts with 'premise', 'hypothesis', 'gold_label'
    """
    print(f"Loading SNLI data from {jsonl_path}...")
    
    samples = []
    label_map = {
        'entailment': 0,
        'contradiction': 1,
        'neutral': 2
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            data = json.loads(line.strip())
            
            # Skip invalid labels
            gold_label = data.get('gold_label', '').strip()
            if gold_label not in label_map or gold_label == '-':
                continue
            
            premise = data.get('sentence1', '').strip()
            hypothesis = data.get('sentence2', '').strip()
            
            # Skip empty sentences
            if not premise or not hypothesis:
                continue
            
            samples.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'gold_label': gold_label,
                'label_idx': label_map[gold_label]
            })
    
    print(f"✓ Loaded {len(samples)} valid samples")
    print(f"  Label distribution:")
    label_counts = {}
    for s in samples:
        label = s['gold_label']
        label_counts[label] = label_counts.get(label, 0) + 1
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")
    
    return samples


def convert_snli_to_dialogues(samples: List[Dict]) -> List[Dict]:
    """
    Convert SNLI samples to dialogue format for existing training pipeline.
    
    Each "dialogue" is a premise+hypothesis pair, with the label added as a token.
    This allows the existing Nova training to learn E/C/N as vocabulary words.
    """
    dialogues = []
    
    for i, sample in enumerate(samples):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        label = sample['gold_label']  # "entailment", "contradiction", or "neutral"
        
        # Create a "dialogue" with two turns:
        # Turn 1: premise + hypothesis (the input)
        # Turn 2: label (the output we want to learn)
        combined_input = f"{premise} {hypothesis}"
        
        dialogues.append({
            'id': i,
            'sentences': [combined_input, label],  # Input sentence + label as output
            'num_turns': 2,
            'gold_label': label,  # Store for evaluation
            'premise': premise,
            'hypothesis': hypothesis
        })
    
    return dialogues


def extract_signatures_with_divergence(samples: List[Dict],
                                       interface: TextToGeometry,
                                       collapse_steps: int = 12,
                                       output_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract signatures WITH divergence primitive for SNLI pairs.
    
    This is the critical fix: adds divergence = 0.38 - alignment to signatures.
    Without this, all laws (L-C1, L-C2, L-C3, L-C4) cannot function.
    
    Returns:
        X: Array of extended signatures (n_samples, n_features + 3)
        y: Array of labels (n_samples,)
    """
    print(f"\nExtracting signatures WITH divergence primitive...")
    print(f"  This adds: alignment, divergence, fracture to each signature")
    print()
    
    X = []
    y = []
    golden_labels = []  # Store golden labels for first 1000 samples
    
    from tqdm import tqdm
    
    label_names = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
    
    for i, sample in enumerate(tqdm(samples, desc="Extracting signatures")):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        label = sample['label_idx']
        gold_label = label_names.get(label, 'unknown')
        
        try:
            # Use the new method that includes divergence
            extended_sig = interface.get_signature_with_divergence(
                premise,
                hypothesis,
                collapse_steps=collapse_steps
            )
            
            X.append(extended_sig)
            y.append(label)
            
            # Save golden label for first 1000 samples
            if i < 1000:
                # Extract alignment, divergence, fracture from signature
                n_features = len(extended_sig)
                alignment = extended_sig[n_features - 3]
                divergence = extended_sig[n_features - 2]
                fracture = extended_sig[n_features - 1]
                
                golden_labels.append({
                    'sample_idx': i,
                    'premise': premise,
                    'hypothesis': hypothesis,
                    'gold_label': gold_label,
                    'label_idx': label,
                    'alignment': float(alignment),
                    'divergence': float(divergence),
                    'fracture': float(fracture)
                })
        
        except Exception as e:
            if i < 5:
                print(f"  Error processing sample {i}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Extracted {len(X)} signatures with divergence primitive")
    print(f"  Feature dimension: {X.shape[1]} (includes alignment, divergence, fracture)")
    print(f"  Label distribution: {np.bincount(y)}")
    
    # Save golden labels for first 1000 samples
    if output_dir and golden_labels:
        import json
        output_dir.mkdir(parents=True, exist_ok=True)
        golden_labels_path = output_dir / "golden_labels_first_1000.json"
        with open(golden_labels_path, 'w') as f:
            json.dump(golden_labels, f, indent=2)
        print(f"✓ Saved golden labels for first 1000 samples to {golden_labels_path}")
    
    # Compute and print divergence statistics
    if output_dir:
        print("\n" + "=" * 70)
        print("Computing Divergence Law Statistics")
        print("=" * 70)
        from nova.core.divergence_classifier import DivergenceClassifier
        
        classifier = DivergenceClassifier()
        stats = classifier.compute_statistics(X, y)
        classifier.print_statistics()
        
        # Auto-tune thresholds based on statistics
        classifier.auto_tune_thresholds()
        
        # Save classifier with tuned thresholds
        import pickle
        output_dir.mkdir(parents=True, exist_ok=True)
        classifier_path = output_dir / "divergence_classifier.pkl"
        with open(classifier_path, 'wb') as f:
            pickle.dump({
                'divergence_threshold': classifier.divergence_threshold,
                'fracture_threshold': classifier.fracture_threshold,
                'stats': classifier.stats
            }, f)
        print(f"✓ Saved divergence classifier to {classifier_path}")
    
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Train SNLI Phase 1 using existing Nova pipeline (E/C/N only)"
    )
    
    parser.add_argument(
        '--snli-train',
        type=str,
        default='nova/data/snli/snli_1.0_train.jsonl',
        help='Path to SNLI training JSONL file'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10000,
        help='Maximum number of training samples (default: 10000)'
    )
    
    parser.add_argument(
        '--lattice-size',
        type=int,
        default=15,
        help='Lattice size (3, 5, 7, 9, 11, 13, 15, ...). Default: 15 (15×15×15 = 3375 cells)'
    )
    
    parser.add_argument(
        '--collapse-steps',
        type=int,
        default=12,
        help='Number of collapse steps (default: 12)'
    )
    
    parser.add_argument(
        '--impulse-scale',
        type=float,
        default=0.1,
        help='Impulse scale for geometry (default: 0.1)'
    )
    
    parser.add_argument(
        '--num-clusters',
        type=int,
        default=2000,
        help='Number of clusters (default: 2000)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='nova/model/snli_phase1',
        help='Output directory for trained model'
    )
    
    args = parser.parse_args()
    
    # Load SNLI data
    snli_path = Path(args.snli_train)
    if not snli_path.exists():
        print(f"❌ SNLI file not found: {snli_path}")
        return
    
    samples = load_snli_data(snli_path, max_samples=args.max_samples)
    
    if not samples:
        print("❌ No samples loaded. Exiting.")
        return
    
    # Convert SNLI samples to dialogue format
    print("\nConverting SNLI samples to dialogue format...")
    dialogues = convert_snli_to_dialogues(samples)
    print(f"✓ Created {len(dialogues)} 'dialogues' (premise+hypothesis pairs)")
    
    # Initialize geometry interface with symmetry breaking for SNLI
    print("\nInitializing geometry interface...")
    print("  ✓ Enabling symmetry breaking for SNLI (angular variation)")
    interface = TextToGeometry(
        lattice_size=args.lattice_size,
        impulse_scale=args.impulse_scale,
        num_clusters=args.num_clusters,
        break_symmetry_for_snli=True  # SNLI ONLY: Break symmetry for angular variation
    )
    
    # Extract signatures WITH divergence primitive
    print("\n" + "=" * 70)
    print("SNLI Phase 1 Training: WITH Divergence Primitive")
    print("=" * 70)
    print()
    print("Phase 1 Discipline:")
    print("  - Output ONLY: 'entailment', 'contradiction', or 'neutral'")
    print("  - Reward: Correct label")
    print("  - Punish: Wrong label OR any extra words")
    print()
    print("CRITICAL: Adding divergence = 0.38 - alignment to signatures")
    print("  This makes all laws (L-C1, L-C2, L-C3, L-C4) functional")
    print()
    
    # Split into supervised (first 1000) and unsupervised (rest)
    supervised_samples = samples[:1000] if len(samples) > 1000 else samples
    unsupervised_samples = samples[1000:] if len(samples) > 1000 else []
    
    print(f"\nTraining Strategy:")
    print(f"  - Supervised (with labels): {len(supervised_samples)} samples")
    print(f"  - Unsupervised (no labels): {len(unsupervised_samples)} samples")
    print()
    
    # Extract signatures for supervised samples (with labels)
    print("=" * 70)
    print("Phase 1: Supervised Learning (First 1000 with Labels)")
    print("=" * 70)
    X_supervised, y_supervised = extract_signatures_with_divergence(
        supervised_samples,
        interface,
        collapse_steps=args.collapse_steps,
        output_dir=Path(args.output_dir)  # Pass output_dir for saving classifier
    )
    
    if len(X_supervised) == 0:
        print("❌ No signatures extracted from supervised samples. Exiting.")
        return
    
    # Extract signatures for unsupervised samples (without labels)
    X_unsupervised = []
    if unsupervised_samples:
        print("\n" + "=" * 70)
        print("Phase 2: Unsupervised Learning (Remaining Samples)")
        print("=" * 70)
        print("Extracting signatures WITHOUT labels...")
        
        from tqdm import tqdm
        for sample in tqdm(unsupervised_samples, desc="Extracting unsupervised signatures"):
            premise = sample['premise']
            hypothesis = sample['hypothesis']
            
            try:
                extended_sig = interface.get_signature_with_divergence(
                    premise,
                    hypothesis,
                    collapse_steps=args.collapse_steps
                )
                X_unsupervised.append(extended_sig)
            except Exception as e:
                continue
        
        X_unsupervised = np.array(X_unsupervised) if X_unsupervised else np.array([])
        print(f"✓ Extracted {len(X_unsupervised)} unsupervised signatures")
    
    # Combine supervised and unsupervised signatures
    X = np.vstack([X_supervised, X_unsupervised]) if len(X_unsupervised) > 0 else X_supervised
    
    # Convert to format expected by training pipeline
    # For supervised samples: use label tokens
    # For unsupervised samples: use empty tokens (will be assigned to clusters)
    signature_arrays = X
    token_lists = []
    
    # Supervised samples: use label tokens
    for sample in supervised_samples:
        label = sample['gold_label']
        tokens = interface.learner.tokenize(label)
        token_lists.append(tokens)
    
    # Unsupervised samples: use empty tokens (will be assigned during clustering)
    for _ in unsupervised_samples:
        token_lists.append([])  # Empty - will be assigned to clusters based on signature only
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Learn clusters from signatures WITH divergence
    print("\n" + "=" * 70)
    print("Learning Geometric Tokens (WITH Divergence Primitive)")
    print("=" * 70)
    print()
    print(f"Training on {len(signature_arrays)} total signatures:")
    print(f"  - {len(X_supervised)} supervised (with labels)")
    print(f"  - {len(X_unsupervised)} unsupervised (no labels)")
    print()
    
    interface.learner.lattice_size = interface.geometry.lattice_size
    interface.learner.num_clusters = args.num_clusters
    
    # Learn clusters
    # Supervised samples will guide cluster formation with their labels
    # Unsupervised samples will be assigned to clusters based on signature similarity
    interface.learner.learn_clusters(signature_arrays, token_lists)
    
    # Save clusters
    cluster_path = output_dir / "geometric_clusters"
    interface.learner.save(cluster_path)
    
    # Save learned patterns (simplified - just label tokens)
    from collections import defaultdict
    patterns = {
        'word_sequences': defaultdict(list),
        'vocabulary': set(['entailment', 'contradiction', 'neutral'])
    }
    
    # Create simple patterns: each label can follow premise+hypothesis
    for sample in samples:
        label = sample['gold_label']
        patterns['word_sequences'][label] = [label]  # Self-transition
    
    import json
    patterns_path = output_dir / "learned_patterns.json"
    with open(patterns_path, 'w') as f:
        json.dump({
            'word_sequences': dict(patterns['word_sequences']),
            'vocabulary': list(patterns['vocabulary'])
        }, f, indent=2)
    
    print(f"\n✓ Model saved to {output_dir}")
    print(f"  Signatures include: alignment, divergence, fracture")
    print(f"  This enables all laws (L-C1, L-C2, L-C3, L-C4) to function")
    
    print("\n" + "=" * 70)
    print("Phase 1 Training Complete!")
    print("=" * 70)
    print("\nThe model has learned 'entailment', 'contradiction', and 'neutral' as vocabulary words.")
    print("Next: Modify ClusterDecoder to output only these three words in Phase 1 mode.")
    print()


if __name__ == "__main__":
    main()

