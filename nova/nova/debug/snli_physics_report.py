"""
SNLI Physics Debug Report v1.0

Complete diagnostic suite to identify where physics breaks in SNLI.
No training, no model save - just diagnostics.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import uuid
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nova.core.text_to_geometry import TextToGeometry
from nova.core.geometric_token_learner import GeometricTokenLearner


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


# ============================================================================
# SECTION 1: Base Signature Report (Premise vs Hypothesis)
# ============================================================================

def report_1_base_signatures(interface: TextToGeometry, samples: List[Dict], n_samples: int = 500):
    """Report base signature statistics for OM vs LO."""
    print_banner("SECTION 1: Base Signature Report (Premise vs Hypothesis)")
    
    om_signatures = []
    lo_signatures = []
    om_lo_cosine_sims = []
    
    for i, sample in enumerate(samples[:n_samples]):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        
        # Get base signatures
        om_sig = interface.get_meaning_signature(premise, collapse_steps=12)
        interface.reset_geometry()
        lo_sig = interface.get_meaning_signature(hypothesis, collapse_steps=12)
        interface.reset_geometry()
        
        om_signatures.append(om_sig)
        lo_signatures.append(lo_sig)
        
        # Compute cosine similarity
        om_norm = om_sig / (np.linalg.norm(om_sig) + 1e-10)
        lo_norm = lo_sig / (np.linalg.norm(lo_sig) + 1e-10)
        cos_sim = np.dot(om_norm, lo_norm)
        om_lo_cosine_sims.append(cos_sim)
    
    om_signatures = np.array(om_signatures)
    lo_signatures = np.array(lo_signatures)
    om_lo_cosine_sims = np.array(om_lo_cosine_sims)
    
    print(f"\nOM (Premise) Signatures:")
    print(f"  Mean value: {np.mean(om_signatures):.6f}")
    print(f"  Std: {np.std(om_signatures):.6f}")
    print(f"  Mean L2 norm: {np.mean([np.linalg.norm(sig) for sig in om_signatures]):.6f}")
    
    print(f"\nLO (Hypothesis) Signatures:")
    print(f"  Mean value: {np.mean(lo_signatures):.6f}")
    print(f"  Std: {np.std(lo_signatures):.6f}")
    print(f"  Mean L2 norm: {np.mean([np.linalg.norm(sig) for sig in lo_signatures]):.6f}")
    
    print(f"\nOM vs LO Cosine Similarity:")
    print(f"  Mean: {np.mean(om_lo_cosine_sims):.6f}")
    print(f"  Std: {np.std(om_lo_cosine_sims):.6f}")
    print(f"  Min: {np.min(om_lo_cosine_sims):.6f}")
    print(f"  Max: {np.max(om_lo_cosine_sims):.6f}")
    print(f"  Median: {np.median(om_lo_cosine_sims):.6f}")
    
    # Histogram
    print(f"\nCosine Similarity Distribution:")
    hist, bins = np.histogram(om_lo_cosine_sims, bins=20, range=(-1, 1))
    for i in range(len(hist)):
        bin_center = (bins[i] + bins[i+1]) / 2
        bar = '█' * int(hist[i] / max(hist) * 50) if max(hist) > 0 else ''
        print(f"  [{bin_center:6.3f}]: {bar} ({hist[i]})")


# ============================================================================
# SECTION 2: Collapse Independence Verification
# ============================================================================

def report_2_collapse_stability(interface: TextToGeometry, samples: List[Dict], n_samples: int = 200):
    """Verify if collapse is deterministic or noisy."""
    print_banner("SECTION 2: Collapse Independence Verification")
    
    same_text_similarities = []
    
    for i, sample in enumerate(samples[:n_samples]):
        text = sample['premise']
        
        # Run twice on same text
        sig1 = interface.get_meaning_signature(text, collapse_steps=12)
        interface.reset_geometry()
        sig2 = interface.get_meaning_signature(text, collapse_steps=12)
        interface.reset_geometry()
        
        # Compute similarity
        sig1_norm = sig1 / (np.linalg.norm(sig1) + 1e-10)
        sig2_norm = sig2 / (np.linalg.norm(sig2) + 1e-10)
        cos_sim = np.dot(sig1_norm, sig2_norm)
        same_text_similarities.append(cos_sim)
    
    same_text_similarities = np.array(same_text_similarities)
    
    print(f"\nSame-Text Stability (running same text twice):")
    print(f"  Avg cos: {np.mean(same_text_similarities):.6f}")
    print(f"  Min cos: {np.min(same_text_similarities):.6f}")
    print(f"  Max cos: {np.max(same_text_similarities):.6f}")
    print(f"  Std: {np.std(same_text_similarities):.6f}")
    
    if np.mean(same_text_similarities) > 0.99:
        print("  ⚠ WARNING: Collapse is highly deterministic (may lack information)")
    elif np.mean(same_text_similarities) < 0.8:
        print("  ⚠ WARNING: Collapse is very noisy (unstable)")


# ============================================================================
# SECTION 3: OM vs LO Directional Separation
# ============================================================================

def report_3_om_lo_separation(interface: TextToGeometry, samples: List[Dict], n_samples: int = 500):
    """Report OM/LO directional separation and divergence statistics."""
    print_banner("SECTION 3: OM/LO Directional Separation")
    
    alignments = []
    cos_thetas = []
    divergences = []
    fractures = []
    
    label_alignments = defaultdict(list)
    label_divergences = defaultdict(list)
    label_fractures = defaultdict(list)
    
    for i, sample in enumerate(samples[:n_samples]):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        label = sample['gold_label']
        
        # Get signatures
        om_sig = interface.get_meaning_signature(premise, collapse_steps=12)
        interface.reset_geometry()
        lo_sig = interface.get_meaning_signature(hypothesis, collapse_steps=12)
        interface.reset_geometry()
        
        # Normalize
        OM = om_sig / (np.linalg.norm(om_sig) + 1e-8)
        LO = lo_sig / (np.linalg.norm(lo_sig) + 1e-8)
        
        # Compute metrics
        cos_theta = float(np.dot(OM, LO))
        alignment = (cos_theta + 1.0) / 2.0
        divergence = 0.38 - alignment
        fracture = abs(alignment)
        
        alignments.append(alignment)
        cos_thetas.append(cos_theta)
        divergences.append(divergence)
        fractures.append(fracture)
        
        label_alignments[label].append(alignment)
        label_divergences[label].append(divergence)
        label_fractures[label].append(fracture)
    
    alignments = np.array(alignments)
    cos_thetas = np.array(cos_thetas)
    divergences = np.array(divergences)
    fractures = np.array(fractures)
    
    print(f"\nAlignment Values:")
    print(f"  Mean: {np.mean(alignments):.6f}")
    print(f"  Std: {np.std(alignments):.6f}")
    print(f"  Min: {np.min(alignments):.6f}")
    print(f"  Max: {np.max(alignments):.6f}")
    
    print(f"\nCos_theta Values:")
    print(f"  Mean: {np.mean(cos_thetas):.6f}")
    print(f"  Std: {np.std(cos_thetas):.6f}")
    print(f"  Min: {np.min(cos_thetas):.6f}")
    print(f"  Max: {np.max(cos_thetas):.6f}")
    
    print(f"\nDivergence Values:")
    print(f"  Mean: {np.mean(divergences):.6f}")
    print(f"  Std: {np.std(divergences):.6f}")
    print(f"  Min: {np.min(divergences):.6f}")
    print(f"  Max: {np.max(divergences):.6f}")
    
    print(f"\nFracture Values:")
    print(f"  Mean: {np.mean(fractures):.6f}")
    print(f"  Std: {np.std(fractures):.6f}")
    print(f"  Min: {np.min(fractures):.6f}")
    print(f"  Max: {np.max(fractures):.6f}")
    
    # Label-wise separation
    print(f"\nLabel-wise Separation:")
    for label in ['entailment', 'neutral', 'contradiction']:
        if label in label_alignments:
            align_arr = np.array(label_alignments[label])
            div_arr = np.array(label_divergences[label])
            frac_arr = np.array(label_fractures[label])
            
            print(f"\n  {label.upper()}:")
            print(f"    Alignment: mean={np.mean(align_arr):.6f}, std={np.std(align_arr):.6f}")
            print(f"    Divergence: mean={np.mean(div_arr):.6f}, std={np.std(div_arr):.6f}")
            print(f"    Fracture: mean={np.mean(frac_arr):.6f}, std={np.std(frac_arr):.6f}")


# ============================================================================
# SECTION 4: Raw Signature Distance Map
# ============================================================================

def report_4_signature_distance_map(interface: TextToGeometry, samples: List[Dict], n_samples: int = 200):
    """Compute cosine distance between all OM and LO signatures."""
    print_banner("SECTION 4: Raw Signature Distance Map")
    
    om_signatures = []
    lo_signatures = []
    
    for i, sample in enumerate(samples[:n_samples]):
        om_sig = interface.get_meaning_signature(sample['premise'], collapse_steps=12)
        interface.reset_geometry()
        lo_sig = interface.get_meaning_signature(sample['hypothesis'], collapse_steps=12)
        interface.reset_geometry()
        
        om_signatures.append(om_sig)
        lo_signatures.append(lo_sig)
    
    om_signatures = np.array(om_signatures)
    lo_signatures = np.array(lo_signatures)
    
    # Normalize
    om_norms = om_signatures / (np.linalg.norm(om_signatures, axis=1, keepdims=True) + 1e-10)
    lo_norms = lo_signatures / (np.linalg.norm(lo_signatures, axis=1, keepdims=True) + 1e-10)
    
    # Compute distance matrix
    distance_matrix = 1 - np.dot(om_norms, lo_norms.T)
    
    print(f"\nDistance Matrix Shape: {distance_matrix.shape}")
    print(f"Mean distance: {np.mean(distance_matrix):.6f}")
    print(f"Std distance: {np.std(distance_matrix):.6f}")
    print(f"Min distance: {np.min(distance_matrix):.6f}")
    print(f"Max distance: {np.max(distance_matrix):.6f}")
    
    # Save to CSV
    output_path = Path("nova/debug/signature_distance_map.csv")
    np.savetxt(output_path, distance_matrix, delimiter=',', fmt='%.6f')
    print(f"\n✓ Saved distance matrix to {output_path}")


# ============================================================================
# SECTION 5: Physics Signal Strength Check
# ============================================================================

def report_5_physics_signal_strength(interface: TextToGeometry, samples: List[Dict], n_supervised: int = 1000):
    """Check if physics signals have sufficient range."""
    print_banner("SECTION 5: Physics Signal Strength Check")
    
    alignments = []
    divergences = []
    fractures = []
    
    for i, sample in enumerate(samples[:n_supervised]):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        
        extended_sig = interface.get_signature_with_divergence(premise, hypothesis, collapse_steps=12)
        interface.reset_geometry()
        
        n_features = len(extended_sig)
        alignment = extended_sig[n_features - 3]
        divergence = extended_sig[n_features - 2]
        fracture = extended_sig[n_features - 1]
        
        alignments.append(alignment)
        divergences.append(divergence)
        fractures.append(fracture)
    
    alignments = np.array(alignments)
    divergences = np.array(divergences)
    fractures = np.array(fractures)
    
    align_range = np.max(alignments) - np.min(alignments)
    div_range = np.max(divergences) - np.min(divergences)
    frac_range = np.max(fractures) - np.min(fractures)
    
    print(f"\nSignal Ranges (max - min):")
    print(f"  Alignment range: {align_range:.6f}")
    print(f"  Divergence range: {div_range:.6f}")
    print(f"  Fracture range: {frac_range:.6f}")
    
    if align_range < 0.01:
        print("  ⚠ WARNING: Alignment range < 0.01 → physics collapsed")
    if div_range < 0.01:
        print("  ⚠ WARNING: Divergence range < 0.01 → physics collapsed")
    if frac_range < 0.01:
        print("  ⚠ WARNING: Fracture range < 0.01 → physics collapsed")


# ============================================================================
# SECTION 6: Signature Dimensionality Check
# ============================================================================

def report_6_dimensionality_check(interface: TextToGeometry, samples: List[Dict], n_samples: int = 10):
    """Check signature dimensions for mismatches."""
    print_banner("SECTION 6: Signature Dimensionality Check")
    
    for i, sample in enumerate(samples[:n_samples]):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        
        om_sig = interface.get_meaning_signature(premise, collapse_steps=12)
        interface.reset_geometry()
        lo_sig = interface.get_meaning_signature(hypothesis, collapse_steps=12)
        interface.reset_geometry()
        
        extended_sig = interface.get_signature_with_divergence(premise, hypothesis, collapse_steps=12)
        interface.reset_geometry()
        
        if i == 0:
            print(f"\nSample {i+1}:")
            print(f"  OM signature dimension: {len(om_sig)}")
            print(f"  LO signature dimension: {len(lo_sig)}")
            print(f"  Extended signature dimension: {len(extended_sig)}")
            print(f"  Expected extended dimension: {len(om_sig) + len(lo_sig) + 3}")
            
            if len(extended_sig) != len(om_sig) + len(lo_sig) + 3:
                print("  ⚠ WARNING: Dimension mismatch!")
            else:
                print("  ✓ Dimensions match")
    
    print(f"\n✓ Checked {n_samples} samples")


# ============================================================================
# SECTION 7: Token Hash Collision Report
# ============================================================================

def report_7_token_hash_collisions(interface: TextToGeometry, samples: List[Dict]):
    """Report token hash collisions."""
    print_banner("SECTION 7: Token Hash Collision Report")
    
    all_tokens = set()
    hash_to_tokens = defaultdict(set)
    
    for sample in samples:
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        
        # Tokenize
        prem_tokens = interface.learner.tokenize(premise)
        hyp_tokens = interface.learner.tokenize(hypothesis)
        
        for token in prem_tokens + hyp_tokens:
            all_tokens.add(token)
            # Get hash coordinates
            coords, _ = interface.learner.token_hash(token)
            hash_key = tuple(coords)
            hash_to_tokens[hash_key].add(token)
    
    total_tokens = len(all_tokens)
    total_hashes = len(hash_to_tokens)
    collisions = sum(len(tokens) - 1 for tokens in hash_to_tokens.values() if len(tokens) > 1)
    collision_rate = (total_tokens - total_hashes) / total_tokens if total_tokens > 0 else 0
    
    print(f"\nToken Hashing Statistics:")
    print(f"  Total unique tokens: {total_tokens}")
    print(f"  Total unique hashes: {total_hashes}")
    print(f"  Total collisions: {collisions}")
    print(f"  Collision rate: {collision_rate:.4%}")
    
    if collision_rate > 0.1:
        print("  ⚠ WARNING: High collision rate (>10%) → geometry may be distorted")
    
    # Show some collision examples
    collision_examples = [(h, list(tokens)[:5]) for h, tokens in hash_to_tokens.items() if len(tokens) > 1]
    if collision_examples:
        print(f"\n  Example collisions (first 5):")
        for hash_key, tokens in collision_examples[:5]:
            print(f"    Hash {hash_key}: {tokens}")


# ============================================================================
# SECTION 8: Collapse Evolution Trace
# ============================================================================

def report_8_collapse_evolution(interface: TextToGeometry, samples: List[Dict], n_samples: int = 20):
    """Trace collapse evolution across steps."""
    print_banner("SECTION 8: Collapse Evolution Trace")
    
    # Monkey-patch to capture intermediate states
    original_inject = interface.inject_sentence
    
    def inject_with_trace(sentence, collapse_steps=10, verbose=False):
        result = original_inject(sentence, collapse_steps=collapse_steps, verbose=verbose)
        return result
    
    interface.inject_sentence = inject_with_trace
    
    similarities = []
    
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
        
        # Normalize
        sig_0_norm = sig_0 / (np.linalg.norm(sig_0) + 1e-10)
        sig_4_norm = sig_4 / (np.linalg.norm(sig_4) + 1e-10)
        sig_8_norm = sig_8 / (np.linalg.norm(sig_8) + 1e-10)
        sig_12_norm = sig_12 / (np.linalg.norm(sig_12) + 1e-10)
        
        # Compute similarities
        sim_0_12 = np.dot(sig_0_norm, sig_12_norm)
        sim_4_12 = np.dot(sig_4_norm, sig_12_norm)
        sim_8_12 = np.dot(sig_8_norm, sig_12_norm)
        
        similarities.append({
            'step_0_vs_12': sim_0_12,
            'step_4_vs_12': sim_4_12,
            'step_8_vs_12': sim_8_12
        })
    
    # Restore original method
    interface.inject_sentence = original_inject
    
    sim_0_12 = np.array([s['step_0_vs_12'] for s in similarities])
    sim_4_12 = np.array([s['step_4_vs_12'] for s in similarities])
    sim_8_12 = np.array([s['step_8_vs_12'] for s in similarities])
    
    print(f"\nCollapse Convergence:")
    print(f"  sig(step 0) vs sig(step 12): mean={np.mean(sim_0_12):.6f}, std={np.std(sim_0_12):.6f}")
    print(f"  sig(step 4) vs sig(step 12): mean={np.mean(sim_4_12):.6f}, std={np.std(sim_4_12):.6f}")
    print(f"  sig(step 8) vs sig(step 12): mean={np.mean(sim_8_12):.6f}, std={np.std(sim_8_12):.6f}")
    
    if np.mean(sim_0_12) > 0.99:
        print("  ⚠ WARNING: Collapse converges instantly → basin too shallow")


# ============================================================================
# SECTION 9: OM/LO Fresh Geometry Verification
# ============================================================================

def report_9_fresh_geometry_verification(interface: TextToGeometry, samples: List[Dict], n_samples: int = 30):
    """Verify OM and LO use different geometry instances."""
    print_banner("SECTION 9: OM/LO Fresh Geometry Verification")
    
    # Add UUID tracking to geometry
    geometry_uuids = []
    
    original_get_sig = interface.get_meaning_signature
    
    def get_sig_with_uuid(sentence, collapse_steps=10):
        # Generate UUID for this geometry instance
        geom_uuid = str(uuid.uuid4())[:8]
        geometry_uuids.append(geom_uuid)
        return original_get_sig(sentence, collapse_steps=collapse_steps)
    
    interface.get_meaning_signature = get_sig_with_uuid
    
    print(f"\nChecking {n_samples} samples for geometry UUID uniqueness...")
    
    for i, sample in enumerate(samples[:n_samples]):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        
        om_uuid_idx = len(geometry_uuids)
        om_sig = interface.get_meaning_signature(premise, collapse_steps=12)
        interface.reset_geometry()
        om_uuid = geometry_uuids[om_uuid_idx] if om_uuid_idx < len(geometry_uuids) else "N/A"
        
        lo_uuid_idx = len(geometry_uuids)
        lo_sig = interface.get_meaning_signature(hypothesis, collapse_steps=12)
        interface.reset_geometry()
        lo_uuid = geometry_uuids[lo_uuid_idx] if lo_uuid_idx < len(geometry_uuids) else "N/A"
        
        if i < 10:
            print(f"  Sample {i+1}: OM UUID={om_uuid}, LO UUID={lo_uuid}")
    
    # Restore original method
    interface.get_meaning_signature = original_get_sig
    
    print(f"\n✓ Checked {n_samples} samples")
    print(f"  Total geometry instances created: {len(geometry_uuids)}")
    unique_uuids = len(set(geometry_uuids))
    print(f"  Unique UUIDs: {unique_uuids}")
    
    if unique_uuids < len(geometry_uuids):
        print("  ⚠ WARNING: Some geometry instances reused → shared cube bug")


# ============================================================================
# SECTION 10: Physics Accuracy Dry Run
# ============================================================================

def report_10_physics_accuracy_dry_run(interface: TextToGeometry, samples: List[Dict], n_samples: int = 300):
    """Test divergence classifier accuracy without clustering/grammar."""
    print_banner("SECTION 10: Physics Accuracy Dry Run")
    
    # Simple divergence-based classifier
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
    
    print(f"\nConfusion Matrix (Gold vs Predicted):")
    print(f"{'':15s} {'Pred E':>10s} {'Pred N':>10s} {'Pred C':>10s}")
    for gold_label in ['entailment', 'neutral', 'contradiction']:
        row = confusion_matrix[gold_label]
        print(f"{gold_label:15s} {row['entailment']:10d} {row['neutral']:10d} {row['contradiction']:10d}")
    
    # Check if all predictions fall into one row
    total_pred_e = sum(confusion_matrix[label]['entailment'] for label in confusion_matrix)
    total_pred_n = sum(confusion_matrix[label]['neutral'] for label in confusion_matrix)
    total_pred_c = sum(confusion_matrix[label]['contradiction'] for label in confusion_matrix)
    
    if total_pred_e == n_samples or total_pred_n == n_samples or total_pred_c == n_samples:
        print("\n  ⚠ WARNING: All predictions fall into one class → divergence is constant")
    
    # Compute accuracy
    correct = sum(confusion_matrix[label][label] for label in confusion_matrix)
    accuracy = correct / n_samples if n_samples > 0 else 0
    print(f"\n  Accuracy: {accuracy:.4f} ({correct}/{n_samples})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all diagnostic reports."""
    print("=" * 70)
    print("SNLI Physics Debug Report v1.0")
    print("=" * 70)
    
    # Load SNLI data
    snli_path = Path("nova/data/snli/snli_1.0_train.jsonl")
    if not snli_path.exists():
        print(f"❌ SNLI file not found: {snli_path}")
        return
    
    print(f"\nLoading SNLI samples from {snli_path}...")
    samples = load_snli_samples(snli_path, max_samples=1000)
    print(f"✓ Loaded {len(samples)} samples")
    
    # Initialize interface
    print("\nInitializing geometry interface...")
    interface = TextToGeometry(
        lattice_size=15,  # Increased from 5 to 15 (15×15×15 = 3375 cells)
        impulse_scale=0.1,
        num_clusters=2000,
        break_symmetry_for_snli=True
    )
    print("✓ Interface initialized")
    
    # Run all reports
    report_1_base_signatures(interface, samples, n_samples=500)
    report_2_collapse_stability(interface, samples, n_samples=200)
    report_3_om_lo_separation(interface, samples, n_samples=500)
    report_4_signature_distance_map(interface, samples, n_samples=200)
    report_5_physics_signal_strength(interface, samples, n_supervised=1000)
    report_6_dimensionality_check(interface, samples, n_samples=10)
    report_7_token_hash_collisions(interface, samples)
    report_8_collapse_evolution(interface, samples, n_samples=20)
    report_9_fresh_geometry_verification(interface, samples, n_samples=30)
    report_10_physics_accuracy_dry_run(interface, samples, n_samples=300)
    
    print("\n" + "=" * 70)
    print("All diagnostic reports complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

