"""
Vector Mode Verification Report

Run this to verify vector mode is working correctly.
Checks alignment variation, divergence separation, etc.
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

from nova.core.vector_text_to_geometry import VectorTextToGeometry


def load_snli_samples(jsonl_path: Path, max_samples: int = 500) -> List[Dict]:
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


def main():
    """Run vector mode verification."""
    print("=" * 70)
    print("Vector Mode Verification Report")
    print("=" * 70)
    
    # Load SNLI data
    snli_path = Path("nova/data/snli/snli_1.0_train.jsonl")
    if not snli_path.exists():
        print(f"❌ SNLI file not found: {snli_path}")
        return
    
    print(f"\nLoading SNLI samples from {snli_path}...")
    samples = load_snli_samples(snli_path, max_samples=500)
    print(f"✓ Loaded {len(samples)} samples")
    
    # Initialize vector interface
    print("\nInitializing vector-based geometry interface...")
    interface = VectorTextToGeometry(
        vector_dim=256,
        impulse_scale=0.1,
        collapse_type='tanh',
        break_symmetry_for_snli=True
    )
    print("✓ Interface initialized")
    
    # Report 1: OM vs LO Cosine Similarity
    print_banner("REPORT 1: OM vs LO Cosine Similarity")
    
    cosine_sims = []
    alignments = []
    divergences = []
    fractures = []
    
    label_alignments = defaultdict(list)
    label_divergences = defaultdict(list)
    
    for i, sample in enumerate(samples):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        label = sample['gold_label']
        
        extended_sig = interface.get_signature_with_divergence(premise, hypothesis, collapse_steps=12)
        
        n_features = len(extended_sig)
        alignment = extended_sig[n_features - 3]
        divergence = extended_sig[n_features - 2]
        fracture = extended_sig[n_features - 1]
        
        # Extract OM and LO vectors
        vector_dim = interface.vector_dim
        om_vec = extended_sig[:vector_dim]
        lo_vec = extended_sig[vector_dim:2*vector_dim]
        
        om_norm = om_vec / (np.linalg.norm(om_vec) + 1e-10)
        lo_norm = lo_vec / (np.linalg.norm(lo_vec) + 1e-10)
        cos_sim = np.dot(om_norm, lo_norm)
        
        cosine_sims.append(cos_sim)
        alignments.append(alignment)
        divergences.append(divergence)
        fractures.append(fracture)
        
        label_alignments[label].append(alignment)
        label_divergences[label].append(divergence)
    
    cosine_sims = np.array(cosine_sims)
    alignments = np.array(alignments)
    divergences = np.array(divergences)
    fractures = np.array(fractures)
    
    print(f"\nOM vs LO Cosine Similarity (n={len(cosine_sims)}):")
    print(f"  Mean: {np.mean(cosine_sims):.6f}")
    print(f"  Std: {np.std(cosine_sims):.6f}")
    print(f"  Min: {np.min(cosine_sims):.6f}")
    print(f"  Max: {np.max(cosine_sims):.6f}")
    
    if np.mean(cosine_sims) < 0.95:
        print("  ✓ GOOD: OM ≠ LO (cosine < 0.95)")
    else:
        print("  ❌ BAD: OM ≈ LO (cosine > 0.95)")
    
    # Report 2: Alignment/Divergence Distributions
    print_banner("REPORT 2: Alignment/Divergence Distributions")
    
    print(f"\nAlignment Values:")
    print(f"  Mean: {np.mean(alignments):.6f}")
    print(f"  Std: {np.std(alignments):.6f}")
    print(f"  Range: [{np.min(alignments):.6f}, {np.max(alignments):.6f}]")
    
    print(f"\nDivergence Values:")
    print(f"  Mean: {np.mean(divergences):.6f}")
    print(f"  Std: {np.std(divergences):.6f}")
    print(f"  Range: [{np.min(divergences):.6f}, {np.max(divergences):.6f}]")
    
    # Report 3: Label-wise Separation
    print_banner("REPORT 3: Label-wise Separation")
    
    print(f"\nAlignment per Label:")
    for label in ['entailment', 'neutral', 'contradiction']:
        if label in label_alignments:
            align_arr = np.array(label_alignments[label])
            print(f"  {label.upper()}: mean={np.mean(align_arr):.6f}, std={np.std(align_arr):.6f}")
    
    print(f"\nDivergence per Label:")
    for label in ['entailment', 'neutral', 'contradiction']:
        if label in label_divergences:
            div_arr = np.array(label_divergences[label])
            print(f"  {label.upper()}: mean={np.mean(div_arr):.6f}, std={np.std(div_arr):.6f}")
    
    # Check if divergence separates labels
    if 'entailment' in label_divergences and 'contradiction' in label_divergences:
        ent_mean = np.mean(label_divergences['entailment'])
        con_mean = np.mean(label_divergences['contradiction'])
        if ent_mean < con_mean:
            print("\n  ✓ GOOD: Divergence separates E/C (entailment < contradiction)")
        else:
            print("\n  ❌ BAD: Divergence does NOT separate E/C")
    
    # Report 4: Physics Accuracy Dry-Run
    print_banner("REPORT 4: Physics Accuracy Dry-Run")
    
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
    
    for i, sample in enumerate(samples):
        gold_label = sample['gold_label']
        divergence = divergences[i]
        fracture = fractures[i]
        
        pred_label = classify_from_divergence(divergence, fracture)
        confusion_matrix[gold_label][pred_label] += 1
    
    print(f"\nConfusion Matrix (Gold vs Predicted, n={len(samples)}):")
    print(f"{'':15s} {'Pred E':>10s} {'Pred N':>10s} {'Pred C':>10s}")
    for gold_label in ['entailment', 'neutral', 'contradiction']:
        row = confusion_matrix[gold_label]
        print(f"{gold_label:15s} {row['entailment']:10d} {row['neutral']:10d} {row['contradiction']:10d}")
    
    correct = sum(confusion_matrix[label][label] for label in confusion_matrix)
    accuracy = correct / len(samples) if len(samples) > 0 else 0
    print(f"\n  Overall Accuracy: {accuracy:.4f} ({correct}/{len(samples)})")
    
    if accuracy > 0.45:
        print("  ✓ GOOD: Physics accuracy > 45%")
    elif accuracy > 0.35:
        print("  ⚠ WARNING: Physics accuracy 35-45%")
    else:
        print("  ❌ BAD: Physics accuracy < 35%")
    
    print("\n" + "=" * 70)
    print("Vector Mode Verification Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

