"""
SNLI Phase 1 Test - Vector Mode (Cell-less Livnium)

Tests the vector-based SNLI classifier.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nova.core.vector_text_to_geometry import VectorTextToGeometry, VectorGeometricTokenLearner
from nova.core.divergence_classifier import DivergenceClassifier


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
            
            samples.append({
                'premise': data.get('sentence1', '').strip(),
                'hypothesis': data.get('sentence2', '').strip(),
                'gold_label': gold_label
            })
    
    return samples


def test_phase1_vector(model_dir: Path, 
                     test_file: Path = None,
                     max_samples: int = 100,
                     use_physics_mode: bool = False):
    """
    Test Phase 1 model using vector-based geometry.
    """
    mode_name = "Pure Physics (Divergence Law)" if use_physics_mode else "Cluster + Grammar"
    print("=" * 70)
    print(f"SNLI Phase 1 Test: Vector Mode - {mode_name}")
    print("=" * 70)
    print()
    
    # Load model metadata
    cluster_path = model_dir / "geometric_clusters.json"
    if not cluster_path.exists():
        print(f"❌ Model not found: {cluster_path}")
        return
    
    with open(cluster_path, 'r') as f:
        cluster_data = json.load(f)
        vector_dim = cluster_data.get('vector_dim', 256)
        num_clusters = cluster_data.get('num_clusters', 2000)
        collapse_type = cluster_data.get('collapse_type', 'tanh')
    
    # Initialize vector-based interface
    print("Initializing vector-based geometry interface...")
    print(f"  ✓ Vector dimension: {vector_dim}")
    print(f"  ✓ Collapse type: {collapse_type}")
    print(f"  ✓ Cell-less Livnium (no 3D lattice)")
    interface = VectorTextToGeometry(
        vector_dim=vector_dim,
        impulse_scale=0.1,
        collapse_type=collapse_type,
        break_symmetry_for_snli=True
    )
    
    # Load token learner
    print(f"\nLoading model from {model_dir}...")
    token_learner = VectorGeometricTokenLearner(vector_dim=vector_dim, num_clusters=num_clusters)
    
    # Load KMeans if available
    kmeans_path = model_dir / "geometric_clusters.pkl"
    if kmeans_path.exists():
        with open(kmeans_path, 'rb') as f:
            token_learner.kmeans = pickle.load(f)
        token_learner.trained = True
    
    # Load cluster tokens
    if 'cluster_tokens' in cluster_data:
        from collections import defaultdict, Counter
        token_learner.cluster_tokens = defaultdict(Counter, {
            int(k): Counter(v) for k, v in cluster_data['cluster_tokens'].items()
        })
    
    # Load divergence classifier if available
    divergence_classifier = None
    classifier_path = model_dir / "divergence_classifier.pkl"
    if classifier_path.exists() and use_physics_mode:
        print(f"Loading divergence classifier from {classifier_path}...")
        with open(classifier_path, 'rb') as f:
            classifier_data = pickle.load(f)
            divergence_classifier = DivergenceClassifier(
                divergence_threshold=classifier_data['divergence_threshold'],
                fracture_threshold=classifier_data['fracture_threshold']
            )
            divergence_classifier.stats = classifier_data.get('stats')
        print(f"  ✓ Loaded (thresholds: div={divergence_classifier.divergence_threshold:.4f}, "
              f"frac={divergence_classifier.fracture_threshold:.4f})")
    
    # Load decoder if not using physics mode
    decoder = None
    if not use_physics_mode:
        patterns_path = model_dir / "learned_patterns.json"
        if patterns_path.exists():
            from nova.core.cluster_decoder import ClusterDecoder
            # Create a dummy TextToGeometry for compatibility
            from nova.core.text_to_geometry import TextToGeometry
            dummy_interface = TextToGeometry(lattice_size=3, num_clusters=num_clusters)
            decoder = ClusterDecoder(
                token_learner=token_learner,
                interface=dummy_interface,
                patterns_path=patterns_path,
                collapse_steps=12,
                phase1_mode=True
            )
    
    # Load test data
    if test_file:
        samples = load_snli_data(test_file, max_samples=max_samples)
    else:
        # Default to dev set
        default_dev_file = Path('nova/data/snli/snli_1.0_dev.jsonl')
        if default_dev_file.exists():
            print(f"  No test file specified, defaulting to dev set: {default_dev_file}")
            samples = load_snli_data(default_dev_file, max_samples=max_samples)
        else:
            print("  ⚠ No test file specified and dev set not found. Using 3 example pairs.")
            samples = [
                {
                    'premise': 'A man inspects the uniform of a figure in some East Asian country.',
                    'hypothesis': 'The man is sleeping.',
                    'gold_label': 'contradiction'
                },
                {
                    'premise': 'An older and younger man smiling.',
                    'hypothesis': 'Two men are smiling and laughing at the cats playing on the floor.',
                    'gold_label': 'neutral'
                },
                {
                    'premise': 'A soccer game with multiple males playing.',
                    'hypothesis': 'Some men are playing a sport.',
                    'gold_label': 'entailment'
                }
            ]
    
    print(f"\nTesting on {len(samples)} samples...")
    print()
    
    # Test each sample
    correct = 0
    total = 0
    discipline_violations = 0
    valid_labels = {'entailment', 'contradiction', 'neutral'}
    
    for i, sample in enumerate(samples):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        gold_label = sample['gold_label']
        
        try:
            # Get extended signature with divergence
            extended_sig = interface.get_signature_with_divergence(
                premise,
                hypothesis,
                collapse_steps=12
            )
            
            # MODE A: Pure Physics (divergence law directly)
            if use_physics_mode and divergence_classifier:
                n_features = len(extended_sig)
                divergence = extended_sig[n_features - 2]
                fracture = extended_sig[n_features - 1]
                prediction = divergence_classifier.classify(divergence, fracture)
            
            # MODE B: Cluster + Grammar (original method)
            else:
                if decoder is None:
                    # Fallback: use simple divergence-based classification
                    n_features = len(extended_sig)
                    divergence = extended_sig[n_features - 2]
                    if divergence < -0.1:
                        prediction = 'entailment'
                    elif divergence > 0.1:
                        prediction = 'contradiction'
                    else:
                        prediction = 'neutral'
                else:
                    prediction = decoder.generate_from_signature(extended_sig, max_tokens=1)
            
            # Phase 1 Discipline Check
            words = prediction.strip().split()
            
            if len(words) != 1:
                discipline_violations += 1
                continue
            
            prediction_word = words[0].lower()
            
            if prediction_word not in valid_labels:
                discipline_violations += 1
                continue
            
            # Check correctness
            is_correct = (prediction_word == gold_label.lower())
            
            if is_correct:
                print(f"✓ Sample {i+1}: CORRECT")
                print(f"   Premise: {premise[:60]}...")
                print(f"   Hypothesis: {hypothesis[:60]}...")
                print(f"   Output: '{prediction_word}' (gold: '{gold_label}')")
                correct += 1
            else:
                print(f"✗ Sample {i+1}: WRONG")
                print(f"   Premise: {premise[:60]}...")
                print(f"   Hypothesis: {hypothesis[:60]}...")
                print(f"   Output: '{prediction_word}' (gold: '{gold_label}')")
            
            print()
            total += 1
        
        except Exception as e:
            print(f"❌ Sample {i+1}: ERROR - {e}")
            continue
    
    # Summary
    print("=" * 70)
    print("Phase 1 Test Summary (Vector Mode)")
    print("=" * 70)
    print(f"Total samples tested: {total + discipline_violations}")
    print(f"Discipline violations: {discipline_violations}")
    print(f"Valid predictions: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {correct / total if total > 0 else 0:.4f}")
    print()
    
    if discipline_violations == 0:
        print("✓ DISCIPLINE: All outputs are exactly one word")
    else:
        print(f"⚠ DISCIPLINE: {discipline_violations} violations detected")


def main():
    parser = argparse.ArgumentParser(
        description="Test SNLI Phase 1 Classifier - Vector Mode"
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='nova/model/snli_phase1_vector',
        help='Path to trained model directory'
    )
    
    parser.add_argument(
        '--test-file',
        type=str,
        default=None,
        help='Path to SNLI test/dev JSONL file (if not specified, defaults to dev set)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum number of test samples (default: 100)'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Test on dev set (nova/data/snli/snli_1.0_dev.jsonl)'
    )
    
    parser.add_argument(
        '--physics',
        action='store_true',
        help='Use pure physics mode (divergence law directly, bypasses cluster+grammar)'
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    test_file = None
    if args.dev:
        test_file = Path('nova/data/snli/snli_1.0_dev.jsonl')
    elif args.test_file:
        test_file = Path(args.test_file)
    
    test_phase1_vector(
        model_dir=model_dir,
        test_file=test_file,
        max_samples=args.max_samples,
        use_physics_mode=args.physics
    )


if __name__ == "__main__":
    main()

