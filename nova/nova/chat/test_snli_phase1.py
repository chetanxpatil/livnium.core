"""
SNLI Phase 1 Test: Output ONLY E/C/N

Tests the Phase 1 model using existing Nova infrastructure.
Ensures it outputs exactly one word: "entailment", "contradiction", or "neutral"
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nova.core.text_to_geometry import TextToGeometry
from nova.core.geometric_token_learner import GeometricTokenLearner
from nova.core.cluster_decoder import ClusterDecoder
from nova.core.divergence_classifier import DivergenceClassifier
import pickle


def load_snli_data(jsonl_path: Path, max_samples: int = None) -> List[Dict]:
    """Load SNLI data from JSONL file."""
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
                'gold_label': gold_label
            })
    
    print(f"✓ Loaded {len(samples)} samples")
    return samples


def test_phase1(model_dir: Path, 
                test_file: Path = None,
                max_samples: int = 100,
                use_physics_mode: bool = False):
    """
    Test Phase 1 model using existing Nova infrastructure.
    
    Phase 1 Discipline Check:
    - Output must be EXACTLY one word
    - Word must be "entailment", "contradiction", or "neutral"
    - Reward correct, punish wrong
    """
    mode_name = "Pure Physics (Divergence Law)" if use_physics_mode else "Cluster + Grammar"
    print("=" * 70)
    print(f"SNLI Phase 1 Test: {mode_name}")
    print("=" * 70)
    print()
    
    # Load model metadata to get lattice_size
    cluster_path = model_dir / "geometric_clusters.json"
    if not cluster_path.exists():
        print(f"❌ Model not found: {cluster_path}")
        return
    
    import json
    with open(cluster_path, 'r') as f:
        cluster_data = json.load(f)
        lattice_size = cluster_data.get('lattice_size', 5)
        num_clusters = cluster_data.get('num_clusters', 2000)
    
    # Initialize geometry interface with symmetry breaking for SNLI
    print("Initializing geometry interface...")
    print("  ✓ Enabling symmetry breaking for SNLI (angular variation)")
    interface = TextToGeometry(
        lattice_size=lattice_size,
        impulse_scale=0.1,
        num_clusters=num_clusters,
        break_symmetry_for_snli=True  # SNLI ONLY: Break symmetry for angular variation
    )
    
    # Load token learner
    print(f"Loading model from {model_dir}...")
    token_learner = GeometricTokenLearner(lattice_size=lattice_size, num_clusters=num_clusters)
    token_learner.load(model_dir / "geometric_clusters")
    
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
        print()
    
    # Initialize decoder in Phase 1 mode (only if not using pure physics mode)
    decoder = None
    if not use_physics_mode:
        patterns_path = model_dir / "learned_patterns.json"
        decoder = ClusterDecoder(
            token_learner=token_learner,
            interface=interface,
            patterns_path=patterns_path,
            collapse_steps=12,
            phase1_mode=True  # Enable Phase 1 mode
        )
    
    # Load test data
    if test_file:
        samples = load_snli_data(test_file, max_samples=max_samples)
    else:
        # Default to dev set if no file specified
        default_dev_file = Path('nova/data/snli/snli_1.0_dev.jsonl')
        if default_dev_file.exists():
            print(f"  No test file specified, defaulting to dev set: {default_dev_file}")
            samples = load_snli_data(default_dev_file, max_samples=max_samples)
        else:
            # Fallback to example pairs if dev file doesn't exist
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
                prediction = divergence_classifier.classify_from_signature(extended_sig)
            
            # MODE B: Cluster + Grammar (original method)
            else:
                if decoder is None:
                    print("❌ Decoder not initialized. Cannot use cluster+grammar mode.")
                    continue
                prediction = decoder.generate_from_signature(extended_sig, max_tokens=1)
            
            # Phase 1 Discipline Check
            words = prediction.strip().split()
            
            # Check 1: Must be exactly one word
            if len(words) != 1:
                print(f"❌ Sample {i+1}: DISCIPLINE VIOLATION")
                print(f"   Premise: {premise[:60]}...")
                print(f"   Hypothesis: {hypothesis[:60]}...")
                print(f"   Output: '{prediction}' (expected exactly 1 word, got {len(words)})")
                print(f"   ⚠ PUNISH: Extra words detected")
                print()
                discipline_violations += 1
                continue
            
            prediction_word = words[0].lower()
            
            # Check 2: Must be valid label
            if prediction_word not in valid_labels:
                print(f"❌ Sample {i+1}: DISCIPLINE VIOLATION")
                print(f"   Premise: {premise[:60]}...")
                print(f"   Hypothesis: {hypothesis[:60]}...")
                print(f"   Output: '{prediction_word}' (not a valid label)")
                print(f"   ⚠ PUNISH: Invalid word")
                print()
                discipline_violations += 1
                continue
            
            # Check 3: Correctness
            is_correct = (prediction_word == gold_label.lower())
            
            if is_correct:
                print(f"✓ Sample {i+1}: CORRECT")
                print(f"   Premise: {premise[:60]}...")
                print(f"   Hypothesis: {hypothesis[:60]}...")
                print(f"   Output: '{prediction_word}' (gold: '{gold_label}')")
                print(f"   ✓ REWARD: Correct label")
                correct += 1
            else:
                print(f"✗ Sample {i+1}: WRONG")
                print(f"   Premise: {premise[:60]}...")
                print(f"   Hypothesis: {hypothesis[:60]}...")
                print(f"   Output: '{prediction_word}' (gold: '{gold_label}')")
                print(f"   ⚠ PUNISH: Wrong label")
            
            print()
            total += 1
        
        except Exception as e:
            print(f"❌ Sample {i+1}: ERROR - {e}")
            print()
            continue
    
    # Summary
    print("=" * 70)
    print("Phase 1 Test Summary")
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
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test SNLI Phase 1 Classifier"
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='nova/model/snli_phase1',
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
        help='Test on dev set (nova/data/snli/snli_1.0_dev.jsonl) - this is now the default if no file specified'
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
    
    test_phase1(
        model_dir=model_dir,
        test_file=test_file,
        max_samples=args.max_samples,
        use_physics_mode=args.physics
    )


if __name__ == "__main__":
    main()

