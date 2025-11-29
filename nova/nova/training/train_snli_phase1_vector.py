"""
SNLI Phase 1 Training - Vector Mode (Cell-less Livnium)

Uses vector-based geometry instead of 3D lattice.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nova.core.vector_text_to_geometry import VectorTextToGeometry, VectorGeometricTokenLearner
from nova.core.divergence_classifier import DivergenceClassifier
import pickle
from tqdm import tqdm


def load_snli_data(jsonl_path: Path, max_samples: int = None) -> List[Dict]:
    """Load SNLI data from JSONL file."""
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
            gold_label = data.get('gold_label', '').strip()
            
            if gold_label not in label_map or gold_label == '-':
                continue
            
            premise = data.get('sentence1', '').strip()
            hypothesis = data.get('sentence2', '').strip()
            
            if not premise or not hypothesis:
                continue
            
            samples.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'gold_label': gold_label,
                'label_idx': label_map[gold_label]
            })
    
    return samples


def extract_signatures_with_divergence(samples: List[Dict],
                                       interface: VectorTextToGeometry,
                                       collapse_steps: int = 12,
                                       output_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract signatures WITH divergence primitive for SNLI pairs.
    
    Returns:
        X: Array of extended signatures
        y: Array of labels
    """
    print(f"\nExtracting signatures WITH divergence primitive (Vector Mode)...")
    print(f"  Vector dimension: {interface.vector_dim}")
    print(f"  Collapse type: {interface.collapse_type}")
    print()
    
    X = []
    y = []
    golden_labels = []
    
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
    
    # Save golden labels
    if output_dir and golden_labels:
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
        
        classifier = DivergenceClassifier()
        stats = classifier.compute_statistics(X, y)
        classifier.print_statistics()
        
        # Auto-tune thresholds
        classifier.auto_tune_thresholds()
        
        # Save classifier
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
        description="Train SNLI Phase 1 - Vector Mode (Cell-less Livnium)"
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
        '--vector-dim',
        type=int,
        default=256,
        help='Vector dimension (128, 256, 512, 1024). Default: 256'
    )
    
    parser.add_argument(
        '--collapse-type',
        type=str,
        default='tanh',
        choices=['tanh', 'power3', 'relu', 'sigmoid'],
        help='Collapse function type (default: tanh)'
    )
    
    parser.add_argument(
        '--collapse-steps',
        type=int,
        default=12,
        help='Number of collapse steps (default: 12)'
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
        default='nova/model/snli_phase1_vector',
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
    
    # Split into supervised (first 1000) and unsupervised (rest)
    supervised_samples = samples[:1000] if len(samples) > 1000 else samples
    unsupervised_samples = samples[1000:] if len(samples) > 1000 else []
    
    print(f"\nTraining Strategy (Vector Mode):")
    print(f"  - Supervised (with labels): {len(supervised_samples)} samples")
    print(f"  - Unsupervised (no labels): {len(unsupervised_samples)} samples")
    print(f"  - Vector dimension: {args.vector_dim}")
    print(f"  - Collapse type: {args.collapse_type}")
    print()
    
    # Initialize vector-based interface
    print("Initializing vector-based geometry interface...")
    print("  ✓ Cell-less Livnium (no 3D lattice)")
    print("  ✓ Zero token collisions")
    print("  ✓ Fast vector operations")
    interface = VectorTextToGeometry(
        vector_dim=args.vector_dim,
        impulse_scale=0.1,
        collapse_type=args.collapse_type,
        break_symmetry_for_snli=True
    )
    
    # Extract signatures
    print("\n" + "=" * 70)
    print("SNLI Phase 1 Training: Vector Mode (Cell-less)")
    print("=" * 70)
    
    X_supervised, y_supervised = extract_signatures_with_divergence(
        supervised_samples,
        interface,
        collapse_steps=args.collapse_steps,
        output_dir=Path(args.output_dir)
    )
    
    if len(X_supervised) == 0:
        print("❌ No signatures extracted. Exiting.")
        return
    
    # Extract unsupervised signatures
    X_unsupervised = []
    if unsupervised_samples:
        print("\nExtracting unsupervised signatures...")
        for sample in tqdm(unsupervised_samples, desc="Unsupervised"):
            try:
                extended_sig = interface.get_signature_with_divergence(
                    sample['premise'],
                    sample['hypothesis'],
                    collapse_steps=args.collapse_steps
                )
                X_unsupervised.append(extended_sig)
            except Exception:
                continue
        
        X_unsupervised = np.array(X_unsupervised) if X_unsupervised else np.array([])
        print(f"✓ Extracted {len(X_unsupervised)} unsupervised signatures")
    
    # Combine
    X = np.vstack([X_supervised, X_unsupervised]) if len(X_unsupervised) > 0 else X_supervised
    
    # Create token lists
    token_learner = VectorGeometricTokenLearner(
        vector_dim=args.vector_dim,
        num_clusters=args.num_clusters
    )
    
    token_lists = []
    for sample in supervised_samples:
        label = sample['gold_label']
        tokens = token_learner.tokenize(label)
        token_lists.append(tokens)
    
    for _ in unsupervised_samples:
        token_lists.append([])  # Empty for unsupervised
    
    # Learn clusters
    print("\n" + "=" * 70)
    print("Learning Geometric Tokens (Vector Mode)")
    print("=" * 70)
    print(f"Training on {len(X)} total signatures:")
    print(f"  - {len(X_supervised)} supervised (with labels)")
    print(f"  - {len(X_unsupervised)} unsupervised (no labels)")
    print()
    
    token_learner.learn_clusters(X, token_lists)
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clusters
    import json
    cluster_data = {
        'vector_dim': args.vector_dim,
        'num_clusters': args.num_clusters,
        'collapse_type': args.collapse_type,
        'cluster_tokens': {str(k): dict(v) for k, v in token_learner.cluster_tokens.items()}
    }
    cluster_path = output_dir / "geometric_clusters.json"
    with open(cluster_path, 'w') as f:
        json.dump(cluster_data, f, indent=2)
    
    # Save KMeans model
    if token_learner.kmeans:
        with open(output_dir / "geometric_clusters.pkl", 'wb') as f:
            pickle.dump(token_learner.kmeans, f)
    
    # Save learned patterns
    from collections import defaultdict
    patterns = {
        'word_sequences': defaultdict(list),
        'vocabulary': set(['entailment', 'contradiction', 'neutral'])
    }
    
    for sample in supervised_samples:
        label = sample['gold_label']
        patterns['word_sequences'][label].append([label])
    
    patterns['word_sequences'] = {k: list(v) for k, v in patterns['word_sequences'].items()}
    patterns['vocabulary'] = list(patterns['vocabulary'])
    
    patterns_path = output_dir / "learned_patterns.json"
    with open(patterns_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"\n✓ Model saved to {output_dir}")
    print(f"  - geometric_clusters.json")
    print(f"  - geometric_clusters.pkl")
    print(f"  - learned_patterns.json")
    print(f"  - divergence_classifier.pkl")
    print(f"  - golden_labels_first_1000.json")


if __name__ == "__main__":
    main()

