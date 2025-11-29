"""
Post-Fix Verification Report: After Increasing Lattice Size

Run this AFTER increasing lattice_size from 5 to 15 (or 21).
This verifies that the fix worked.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nova.core.text_to_geometry import TextToGeometry
from nova.core.geometric_token_learner import GeometricTokenLearner


def load_snli_samples(jsonl_path: Path, max_samples: int = 1000) -> List[Dict]:
    """Load SNLI samples from JSONL file."""
    samples = []
    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data = json.loads(line.strip())
            gold_label = data.get('gold_label', '').strip()
            
            if gold_label not in label_map or gold_label == '-':
                continue
            
            samples.append({
                'premise': data.get('sentence1', '').strip(),
                'hypothesis': data.get('sentence2', '').strip(),
                'gold_label': gold_label,
                'label_idx': label_map[gold_label]
            })
    
    return samples


def print_banner(title: str):
    """Print section banner."""
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)


# ============================================================================
# REPORT 1: Token Collision Rate
# ============================================================================

def report_1_token_collisions(interface: TextToGeometry, samples: List[Dict]):
    """Report token hash collision statistics."""
    print_banner("REPORT 1: Token Collision Rate")
    
    all_tokens = set()
    hash_to_tokens = defaultdict(set)
    
    for sample in samples:
        for text in [sample['premise'], sample['hypothesis']]:
            tokens = interface.learner.tokenize(text)
            for token in tokens:
                if token.strip() and token.isalnum():
                    all_tokens.add(token.lower())
                    coords, _ = interface.learner.token_hash(token)
                    hash_to_tokens[coords].add(token.lower())
    
    total_tokens = len(all_tokens)
    total_hashes = len(hash_to_tokens)
    collisions = sum(len(tokens) - 1 for tokens in hash_to_tokens.values() if len(tokens) > 1)
    collision_rate = (total_tokens - total_hashes) / total_tokens if total_tokens > 0 else 0
    
    print(f"\nToken Hashing Statistics:")
    print(f"  Unique tokens: {total_tokens}")
    print(f"  Unique coordinates: {total_hashes}")
    print(f"  Total collisions: {collisions}")
    print(f"  Collision rate: {collision_rate:.2%}")
    print(f"  Average words per coordinate: {total_tokens / total_hashes:.2f}" if total_hashes > 0 else "  N/A")
    
    if collision_rate < 0.20:
        print("  ✓ GOOD: Collision rate < 20%")
    elif collision_rate < 0.50:
        print("  ⚠ WARNING: Collision rate 20-50% (acceptable but not ideal)")
    else:
        print("  ❌ BAD: Collision rate > 50% (still too high)")


# ============================================================================
# REPORT 2: OM Signature Statistics
# ============================================================================

def report_2_om_statistics(interface: TextToGeometry, samples: List[Dict], n_samples: int = 500):
    """Report OM (premise) signature statistics."""
    print_banner("REPORT 2: OM Signature Statistics")
    
    om_signatures = []
    om_norms = []
    
    for i, sample in enumerate(samples[:n_samples]):
        om_sig = interface.get_meaning_signature(sample['premise'], collapse_steps=12)
        interface.reset_geometry()
        om_signatures.append(om_sig)
        om_norms.append(np.linalg.norm(om_sig))
    
    om_signatures = np.array(om_signatures)
    om_norms = np.array(om_norms)
    
    print(f"\nOM (Premise) Signatures (n={len(om_signatures)}):")
    print(f"  Mean value: {np.mean(om_signatures):.6f}")
    print(f"  Std: {np.std(om_signatures):.6f}")
    print(f"  Min: {np.min(om_signatures):.6f}")
    print(f"  Max: {np.max(om_signatures):.6f}")
    print(f"  Mean L2 norm: {np.mean(om_norms):.6f}")
    print(f"  Std L2 norm: {np.std(om_norms):.6f}")


# ============================================================================
# REPORT 3: LO Signature Statistics
# ============================================================================

def report_3_lo_statistics(interface: TextToGeometry, samples: List[Dict], n_samples: int = 500):
    """Report LO (hypothesis) signature statistics."""
    print_banner("REPORT 3: LO Signature Statistics")
    
    lo_signatures = []
    lo_norms = []
    
    for i, sample in enumerate(samples[:n_samples]):
        lo_sig = interface.get_meaning_signature(sample['hypothesis'], collapse_steps=12)
        interface.reset_geometry()
        lo_signatures.append(lo_sig)
        lo_norms.append(np.linalg.norm(lo_sig))
    
    lo_signatures = np.array(lo_signatures)
    lo_norms = np.array(lo_norms)
    
    print(f"\nLO (Hypothesis) Signatures (n={len(lo_signatures)}):")
    print(f"  Mean value: {np.mean(lo_signatures):.6f}")
    print(f"  Std: {np.std(lo_signatures):.6f}")
    print(f"  Min: {np.min(lo_signatures):.6f}")
    print(f"  Max: {np.max(lo_signatures):.6f}")
    print(f"  Mean L2 norm: {np.mean(lo_norms):.6f}")
    print(f"  Std L2 norm: {np.std(lo_norms):.6f}")


# ============================================================================
# REPORT 4: OM vs LO Cosine Similarity
# ============================================================================

def report_4_om_lo_similarity(interface: TextToGeometry, samples: List[Dict], n_samples: int = 500):
    """Report OM vs LO cosine similarity distribution."""
    print_banner("REPORT 4: OM vs LO Cosine Similarity")
    
    cosine_sims = []
    
    for i, sample in enumerate(samples[:n_samples]):
        om_sig = interface.get_meaning_signature(sample['premise'], collapse_steps=12)
        interface.reset_geometry()
        lo_sig = interface.get_meaning_signature(sample['hypothesis'], collapse_steps=12)
        interface.reset_geometry()
        
        om_norm = om_sig / (np.linalg.norm(om_sig) + 1e-10)
        lo_norm = lo_sig / (np.linalg.norm(lo_sig) + 1e-10)
        cos_sim = np.dot(om_norm, lo_norm)
        cosine_sims.append(cos_sim)
    
    cosine_sims = np.array(cosine_sims)
    
    print(f"\nOM vs LO Cosine Similarity (n={len(cosine_sims)}):")
    print(f"  Mean: {np.mean(cosine_sims):.6f}")
    print(f"  Std: {np.std(cosine_sims):.6f}")
    print(f"  Min: {np.min(cosine_sims):.6f}")
    print(f"  Max: {np.max(cosine_sims):.6f}")
    print(f"  Median: {np.median(cosine_sims):.6f}")
    
    # Histogram
    print(f"\nCosine Similarity Distribution (20 bins):")
    hist, bins = np.histogram(cosine_sims, bins=20, range=(-1, 1))
    for i in range(len(hist)):
        bin_center = (bins[i] + bins[i+1]) / 2
        bar = '█' * int(hist[i] / max(hist) * 50) if max(hist) > 0 else ''
        print(f"  [{bin_center:6.3f}]: {bar} ({hist[i]})")
    
    if np.mean(cosine_sims) < 0.95:
        print("\n  ✓ GOOD: OM ≠ LO (cosine < 0.95)")
    elif np.mean(cosine_sims) < 0.99:
        print("\n  ⚠ WARNING: OM ≈ LO (cosine 0.95-0.99)")
    else:
        print("\n  ❌ BAD: OM == LO (cosine > 0.99)")


# ============================================================================
# REPORT 5: Alignment Distribution Per Label
# ============================================================================

def report_5_alignment_per_label(interface: TextToGeometry, samples: List[Dict], n_samples: int = 500):
    """Report alignment distribution for each label."""
    print_banner("REPORT 5: Alignment Distribution Per Label")
    
    label_alignments = defaultdict(list)
    
    for i, sample in enumerate(samples[:n_samples]):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        label = sample['gold_label']
        
        extended_sig = interface.get_signature_with_divergence(premise, hypothesis, collapse_steps=12)
        interface.reset_geometry()
        
        n_features = len(extended_sig)
        alignment = extended_sig[n_features - 3]
        label_alignments[label].append(alignment)
    
    print(f"\nAlignment Distribution Per Label:")
    for label in ['entailment', 'neutral', 'contradiction']:
        if label in label_alignments:
            align_arr = np.array(label_alignments[label])
            print(f"\n  {label.upper()}:")
            print(f"    Mean: {np.mean(align_arr):.6f}")
            print(f"    Std: {np.std(align_arr):.6f}")
            print(f"    Min: {np.min(align_arr):.6f}")
            print(f"    Max: {np.max(align_arr):.6f}")
            print(f"    Median: {np.median(align_arr):.6f}")


# ============================================================================
# REPORT 6: Divergence Distribution Per Label
# ============================================================================

def report_6_divergence_per_label(interface: TextToGeometry, samples: List[Dict], n_samples: int = 500):
    """Report divergence distribution for each label."""
    print_banner("REPORT 6: Divergence Distribution Per Label")
    
    label_divergences = defaultdict(list)
    
    for i, sample in enumerate(samples[:n_samples]):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        label = sample['gold_label']
        
        extended_sig = interface.get_signature_with_divergence(premise, hypothesis, collapse_steps=12)
        interface.reset_geometry()
        
        n_features = len(extended_sig)
        divergence = extended_sig[n_features - 2]
        label_divergences[label].append(divergence)
    
    print(f"\nDivergence Distribution Per Label:")
    for label in ['entailment', 'neutral', 'contradiction']:
        if label in label_divergences:
            div_arr = np.array(label_divergences[label])
            print(f"\n  {label.upper()}:")
            print(f"    Mean: {np.mean(div_arr):.6f}")
            print(f"    Std: {np.std(div_arr):.6f}")
            print(f"    Min: {np.min(div_arr):.6f}")
            print(f"    Max: {np.max(div_arr):.6f}")
            print(f"    Median: {np.median(div_arr):.6f}")
    
    # Check if divergence separates labels
    if 'entailment' in label_divergences and 'contradiction' in label_divergences:
        ent_mean = np.mean(label_divergences['entailment'])
        con_mean = np.mean(label_divergences['contradiction'])
        if ent_mean < con_mean:
            print("\n  ✓ GOOD: Divergence separates E/C (entailment < contradiction)")
        else:
            print("\n  ❌ BAD: Divergence does NOT separate E/C")


# ============================================================================
# REPORT 7: Fracture Distribution Per Label
# ============================================================================

def report_7_fracture_per_label(interface: TextToGeometry, samples: List[Dict], n_samples: int = 500):
    """Report fracture distribution for each label."""
    print_banner("REPORT 7: Fracture Distribution Per Label")
    
    label_fractures = defaultdict(list)
    
    for i, sample in enumerate(samples[:n_samples]):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        label = sample['gold_label']
        
        extended_sig = interface.get_signature_with_divergence(premise, hypothesis, collapse_steps=12)
        interface.reset_geometry()
        
        n_features = len(extended_sig)
        fracture = extended_sig[n_features - 1]
        label_fractures[label].append(fracture)
    
    print(f"\nFracture Distribution Per Label:")
    for label in ['entailment', 'neutral', 'contradiction']:
        if label in label_fractures:
            frac_arr = np.array(label_fractures[label])
            print(f"\n  {label.upper()}:")
            print(f"    Mean: {np.mean(frac_arr):.6f}")
            print(f"    Std: {np.std(frac_arr):.6f}")
            print(f"    Min: {np.min(frac_arr):.6f}")
            print(f"    Max: {np.max(frac_arr):.6f}")
            print(f"    Median: {np.median(frac_arr):.6f}")


# ============================================================================
# REPORT 8: Collapse Evolution Trace
# ============================================================================

def report_8_collapse_evolution(interface: TextToGeometry, samples: List[Dict], n_samples: int = 10):
    """Trace collapse evolution across steps."""
    print_banner("REPORT 8: Collapse Evolution Trace")
    
    similarities = {
        'step0_vs_step4': [],
        'step4_vs_step8': [],
        'step8_vs_step12': [],
        'step12_vs_step20': []
    }
    
    for i, sample in enumerate(samples[:n_samples]):
        text = sample['premise']
        
        # Get signatures at different collapse steps
        sig_0 = interface.get_meaning_signature(text, collapse_steps=0)
        interface.reset_geometry()
        sig_4 = interface.get_meaning_signature(text, collapse_steps=4)
        interface.reset_geometry()
        sig_8 = interface.get_meaning_signature(text, collapse_steps=8)
        interface.reset_geometry()
        sig_12 = interface.get_meaning_signature(text, collapse_steps=12)
        interface.reset_geometry()
        sig_20 = interface.get_meaning_signature(text, collapse_steps=20)
        interface.reset_geometry()
        
        # Normalize
        def norm(sig):
            return sig / (np.linalg.norm(sig) + 1e-10)
        
        sig_0_norm = norm(sig_0)
        sig_4_norm = norm(sig_4)
        sig_8_norm = norm(sig_8)
        sig_12_norm = norm(sig_12)
        sig_20_norm = norm(sig_20)
        
        # Compute similarities
        similarities['step0_vs_step4'].append(np.dot(sig_0_norm, sig_4_norm))
        similarities['step4_vs_step8'].append(np.dot(sig_4_norm, sig_8_norm))
        similarities['step8_vs_step12'].append(np.dot(sig_8_norm, sig_12_norm))
        similarities['step12_vs_step20'].append(np.dot(sig_12_norm, sig_20_norm))
    
    print(f"\nCollapse Evolution (n={n_samples} sentences):")
    for key, sims in similarities.items():
        sims_arr = np.array(sims)
        print(f"  {key}: mean={np.mean(sims_arr):.6f}, std={np.std(sims_arr):.6f}")
    
    # Check if collapse evolves
    step0_4 = np.mean(similarities['step0_vs_step4'])
    step12_20 = np.mean(similarities['step12_vs_step20'])
    
    if step0_4 < 0.99:
        print("\n  ✓ GOOD: Collapse evolves (step0 ≠ step4)")
    else:
        print("\n  ❌ BAD: Collapse is flat (step0 ≈ step4)")


# ============================================================================
# REPORT 9: Distance Between Signatures
# ============================================================================

def report_9_signature_distances(interface: TextToGeometry, samples: List[Dict], n_samples: int = 200):
    """Compute distance matrix between signatures."""
    print_banner("REPORT 9: Distance Between Signatures")
    
    signatures = []
    
    for i, sample in enumerate(samples[:n_samples]):
        sig = interface.get_meaning_signature(sample['premise'], collapse_steps=12)
        interface.reset_geometry()
        signatures.append(sig)
    
    signatures = np.array(signatures)
    
    # Normalize
    sig_norms = signatures / (np.linalg.norm(signatures, axis=1, keepdims=True) + 1e-10)
    
    # Compute distance matrix (cosine distance)
    distance_matrix = 1 - np.dot(sig_norms, sig_norms.T)
    
    # Get upper triangle (avoid duplicates)
    triu_indices = np.triu_indices(n_samples, k=1)
    distances = distance_matrix[triu_indices]
    
    print(f"\nSignature Distance Matrix (n={n_samples}×{n_samples}):")
    print(f"  Mean distance: {np.mean(distances):.6f}")
    print(f"  Std distance: {np.std(distances):.6f}")
    print(f"  Min distance: {np.min(distances):.6f}")
    print(f"  Max distance: {np.max(distances):.6f}")
    print(f"  Median distance: {np.median(distances):.6f}")
    
    if np.mean(distances) > 0.01:
        print("\n  ✓ GOOD: Signatures are distinct (mean distance > 0.01)")
    else:
        print("\n  ❌ BAD: Signatures are identical (mean distance < 0.01)")


# ============================================================================
# REPORT 10: Physics Accuracy Dry-Run
# ============================================================================

def report_10_physics_accuracy(interface: TextToGeometry, samples: List[Dict], n_samples: int = 300):
    """Test physics-based classification accuracy."""
    print_banner("REPORT 10: Physics Accuracy Dry-Run")
    
    def classify_from_divergence(divergence: float, fracture: float) -> str:
        if divergence < -0.1:
            return 'entailment'
        elif divergence > 0.1:
            return 'contradiction'
        else:
            return 'neutral'
    
    confusion_matrix = {
        'entailment': {'entailment': 0, 'neutral': 0, 'contradiction': 0},
        'neutral': {'entailment': 0, 'neutral': 0, 'contradiction': 0},
        'contradiction': {'entailment': 0, 'neutral': 0, 'contradiction': 0}
    }
    
    for i, sample in enumerate(samples[:n_samples]):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        gold_label = sample['gold_label']
        
        extended_sig = interface.get_signature_with_divergence(premise, hypothesis, collapse_steps=12)
        interface.reset_geometry()
        
        n_features = len(extended_sig)
        divergence = extended_sig[n_features - 2]
        fracture = extended_sig[n_features - 1]
        
        pred_label = classify_from_divergence(divergence, fracture)
        confusion_matrix[gold_label][pred_label] += 1
    
    print(f"\nConfusion Matrix (Gold vs Predicted, n={n_samples}):")
    print(f"{'':15s} {'Pred E':>10s} {'Pred N':>10s} {'Pred C':>10s}")
    for gold_label in ['entailment', 'neutral', 'contradiction']:
        row = confusion_matrix[gold_label]
        print(f"{gold_label:15s} {row['entailment']:10d} {row['neutral']:10d} {row['contradiction']:10d}")
    
    # Compute accuracy
    correct = sum(confusion_matrix[label][label] for label in confusion_matrix)
    accuracy = correct / n_samples if n_samples > 0 else 0
    print(f"\n  Overall Accuracy: {accuracy:.4f} ({correct}/{n_samples})")
    
    if accuracy > 0.45:
        print("  ✓ GOOD: Physics accuracy > 45%")
    elif accuracy > 0.35:
        print("  ⚠ WARNING: Physics accuracy 35-45% (improving but not great)")
    else:
        print("  ❌ BAD: Physics accuracy < 35% (still at baseline)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all verification reports."""
    print("=" * 70)
    print("Post-Fix Verification Report")
    print("Run this AFTER increasing lattice_size from 5 to 15 (or 21)")
    print("=" * 70)
    
    # Get lattice size from user or default
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lattice-size', type=int, default=15, help='Lattice size (should be 15 or 21 after fix)')
    args = parser.parse_args()
    
    print(f"\nUsing lattice_size = {args.lattice_size}")
    print(f"Expected cells: {args.lattice_size ** 3}")
    
    # Load SNLI data
    snli_path = Path("nova/data/snli/snli_1.0_train.jsonl")
    if not snli_path.exists():
        print(f"❌ SNLI file not found: {snli_path}")
        return
    
    print(f"\nLoading SNLI samples from {snli_path}...")
    samples = load_snli_samples(snli_path, max_samples=1000)
    print(f"✓ Loaded {len(samples)} samples")
    
    # Initialize interface with new lattice size
    print(f"\nInitializing geometry interface with lattice_size={args.lattice_size}...")
    interface = TextToGeometry(
        lattice_size=args.lattice_size,
        impulse_scale=0.1,
        num_clusters=2000,
        break_symmetry_for_snli=True
    )
    print("✓ Interface initialized")
    
    # Run all reports
    report_1_token_collisions(interface, samples)
    report_2_om_statistics(interface, samples, n_samples=500)
    report_3_lo_statistics(interface, samples, n_samples=500)
    report_4_om_lo_similarity(interface, samples, n_samples=500)
    report_5_alignment_per_label(interface, samples, n_samples=500)
    report_6_divergence_per_label(interface, samples, n_samples=500)
    report_7_fracture_per_label(interface, samples, n_samples=500)
    report_8_collapse_evolution(interface, samples, n_samples=10)
    report_9_signature_distances(interface, samples, n_samples=200)
    report_10_physics_accuracy(interface, samples, n_samples=300)
    
    print("\n" + "=" * 70)
    print("All verification reports complete!")
    print("=" * 70)
    print("\nCheck the results above:")
    print("  ✓ GOOD = Fix is working")
    print("  ⚠ WARNING = Needs improvement")
    print("  ❌ BAD = Still broken")


if __name__ == "__main__":
    main()

