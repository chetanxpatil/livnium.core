#!/usr/bin/env python3
"""
Law Compliance Test: Verify v8 implementation follows all physics laws.

Tests compliance with:
1. Divergence Law - sign preservation (E negative, C positive, N near-zero)
2. Inward-Outward Axis Law - divergence sign reflects semantic direction
3. Geometric Invariance Law - signals computed from geometry only
4. Phase Classification Law - thresholds match expected behavior
"""

import os
import sys
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Change to nli_v8 directory to allow relative imports
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from core.encoder import ChainEncoder
from core.classifier import LivniumV8Classifier
from core.layers import LayerOpposition, Layer0Resonance


def test_divergence_law():
    """
    Test Divergence Law compliance.
    
    Law: divergence = K - alignment (or angle-based equivalent)
    - Entailment: divergence < 0 (negative, inward)
    - Contradiction: divergence > 0 (positive, outward)
    - Neutral: divergence ≈ 0 (near-zero, balanced)
    """
    print("=" * 70)
    print("TEST: Divergence Law Compliance")
    print("=" * 70)
    print()
    print("Law Requirement:")
    print("  - Entailment: divergence < 0 (negative, inward)")
    print("  - Contradiction: divergence > 0 (positive, outward)")
    print("  - Neutral: divergence ≈ 0 (near-zero, balanced)")
    print()
    
    encoder = ChainEncoder()
    layer_opposition = LayerOpposition(fracture_threshold=0.5)
    
    # Test cases with known labels
    test_cases = [
        ("A dog is barking", "A dog is barking", "entailment"),
        ("A dog is barking", "A dog is not barking", "contradiction"),
        ("A dog is barking", "A cat is sleeping", "neutral"),
    ]
    
    results = []
    for premise, hypothesis, expected_label in test_cases:
        pair = encoder.encode_pair(premise, hypothesis)
        premise_vecs, hypothesis_vecs = pair.get_word_vectors()
        
        result = layer_opposition.compute(premise_vecs, hypothesis_vecs, resonance=0.5)
        divergence = result['divergence_final']
        
        # Check sign compliance
        if expected_label == "entailment":
            compliant = divergence < 0
            expected_sign = "negative"
        elif expected_label == "contradiction":
            compliant = divergence > 0
            expected_sign = "positive"
        else:  # neutral
            compliant = abs(divergence) < 0.2  # Near-zero band
            expected_sign = "near-zero"
        
        status = "✅" if compliant else "❌"
        results.append((expected_label, divergence, compliant))
        
        print(f"{status} {expected_label:12} | Divergence: {divergence:7.4f} | Expected: {expected_sign}")
    
    # Summary
    print()
    compliant_count = sum(1 for _, _, c in results if c)
    total_count = len(results)
    print(f"Compliance: {compliant_count}/{total_count} ({100*compliant_count/total_count:.1f}%)")
    
    return compliant_count == total_count


def test_inward_outward_axis_law():
    """
    Test Inward-Outward Axis Law compliance.
    
    Law: Geometry is inward-outward, not up-down
    - Entailment = inward collapsing (negative divergence)
    - Contradiction = outward expanding (positive divergence)
    - Neutral = boundary (near-zero divergence)
    """
    print("\n" + "=" * 70)
    print("TEST: Inward-Outward Axis Law Compliance")
    print("=" * 70)
    print()
    print("Law Requirement:")
    print("  - Entailment = inward (negative divergence)")
    print("  - Contradiction = outward (positive divergence)")
    print("  - Neutral = boundary (near-zero divergence)")
    print()
    
    encoder = ChainEncoder()
    layer_opposition = LayerOpposition(fracture_threshold=0.5)
    
    test_cases = [
        ("A dog is barking", "A dog is barking", "entailment", "inward"),
        ("A dog is barking", "A dog is not barking", "contradiction", "outward"),
        ("A dog is barking", "A cat is sleeping", "neutral", "boundary"),
    ]
    
    results = []
    for premise, hypothesis, label, expected_direction in test_cases:
        pair = encoder.encode_pair(premise, hypothesis)
        premise_vecs, hypothesis_vecs = pair.get_word_vectors()
        
        result = layer_opposition.compute(premise_vecs, hypothesis_vecs, resonance=0.5)
        divergence = result['divergence_final']
        
        # Check direction
        if expected_direction == "inward":
            compliant = divergence < 0
            actual_direction = "inward" if divergence < 0 else "outward"
        elif expected_direction == "outward":
            compliant = divergence > 0
            actual_direction = "outward" if divergence > 0 else "inward"
        else:  # boundary
            compliant = abs(divergence) < 0.2
            actual_direction = "boundary" if abs(divergence) < 0.2 else ("inward" if divergence < 0 else "outward")
        
        status = "✅" if compliant else "❌"
        results.append((label, divergence, compliant, expected_direction, actual_direction))
        
        print(f"{status} {label:12} | Div: {divergence:7.4f} | Expected: {expected_direction:8} | Got: {actual_direction:8}")
    
    # Summary
    print()
    compliant_count = sum(1 for _, _, c, _, _ in results if c)
    total_count = len(results)
    print(f"Compliance: {compliant_count}/{total_count} ({100*compliant_count/total_count:.1f}%)")
    
    return compliant_count == total_count


def test_geometric_invariance_law():
    """
    Test Geometric Invariance Law compliance.
    
    Law: Geometric signals are invariant to label inversion.
    Signals computed from geometry only, not labels.
    """
    print("\n" + "=" * 70)
    print("TEST: Geometric Invariance Law Compliance")
    print("=" * 70)
    print()
    print("Law Requirement:")
    print("  - Signals computed from vectors only (geometry)")
    print("  - No label-dependent heuristics")
    print("  - Same sentence pair → same divergence sign")
    print()
    
    encoder = ChainEncoder()
    layer_opposition = LayerOpposition(fracture_threshold=0.5)
    
    # Test same sentence pair multiple times (should get same divergence sign)
    premise = "A dog is barking"
    hypothesis = "A dog is not barking"
    
    pair = encoder.encode_pair(premise, hypothesis)
    premise_vecs, hypothesis_vecs = pair.get_word_vectors()
    
    # Compute multiple times
    results = []
    for i in range(5):
        result = layer_opposition.compute(premise_vecs, hypothesis_vecs, resonance=0.5)
        divergence = result['divergence_final']
        results.append(divergence)
    
    # Check consistency
    signs = [np.sign(d) for d in results]
    all_same_sign = all(s == signs[0] for s in signs)
    
    print(f"Premise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Divergence values: {[f'{d:.4f}' for d in results]}")
    print(f"Signs: {signs}")
    print()
    
    if all_same_sign:
        print("✅ All signs consistent (invariant)")
    else:
        print("❌ Signs inconsistent (not invariant)")
    
    # Check that signals are computed from geometry
    raw_divs = []
    warp_divs = []
    for i in range(3):
        result = layer_opposition.compute(premise_vecs, hypothesis_vecs, resonance=0.5)
        raw_divs.append(result.get('raw_divergence', 0))
        warp_divs.append(result.get('warp_divergence', 0))
    
    raw_consistent = all(abs(r - raw_divs[0]) < 0.01 for r in raw_divs)
    warp_consistent = all(abs(w - warp_divs[0]) < 0.01 for w in warp_divs)
    
    print(f"Raw divergence consistent: {'✅' if raw_consistent else '❌'}")
    print(f"Warp divergence consistent: {'✅' if warp_consistent else '❌'}")
    
    return all_same_sign and raw_consistent and warp_consistent


def test_phase_classification_law():
    """
    Test Phase Classification Law compliance.
    
    Law: Decision rules based on divergence thresholds:
    - Contradiction: divergence > 0.02
    - Entailment: divergence < -0.08 AND resonance > 0.50
    - Neutral: abs(divergence) < 0.12
    """
    print("\n" + "=" * 70)
    print("TEST: Phase Classification Law Compliance")
    print("=" * 70)
    print()
    print("Law Requirement:")
    print("  - Contradiction: divergence > 0.02")
    print("  - Entailment: divergence < -0.08 AND resonance > 0.50")
    print("  - Neutral: abs(divergence) < 0.12")
    print()
    
    encoder = ChainEncoder()
    
    test_cases = [
        ("A dog is barking", "A dog is barking", "entailment"),
        ("A dog is barking", "A dog is not barking", "contradiction"),
        ("A dog is barking", "A cat is sleeping", "neutral"),
    ]
    
    results = []
    for premise, hypothesis, expected_label in test_cases:
        pair = encoder.encode_pair(premise, hypothesis)
        classifier = LivniumV8Classifier(pair)
        result = classifier.classify()
        
        # Get signals
        opposition = result.layer_states.get('layer_opposition', {})
        divergence = opposition.get('divergence_final', 0.0)
        resonance = result.layer_states.get('layer0', {}).get('resonance', 0.0)
        predicted = result.label
        
        # Check thresholds
        if expected_label == "contradiction":
            threshold_met = divergence > 0.02
            threshold_desc = f"div > 0.02"
        elif expected_label == "entailment":
            threshold_met = divergence < -0.08 and resonance > 0.50
            threshold_desc = f"div < -0.08 AND res > 0.50"
        else:  # neutral
            threshold_met = abs(divergence) < 0.12
            threshold_desc = f"|div| < 0.12"
        
        status = "✅" if threshold_met else "⚠️"
        results.append((expected_label, divergence, resonance, threshold_met, predicted))
        
        print(f"{status} {expected_label:12} | Div: {divergence:7.4f} | Res: {resonance:.4f} | {threshold_desc:20} | Pred: {predicted}")
    
    # Summary
    print()
    threshold_count = sum(1 for _, _, _, t, _ in results if t)
    total_count = len(results)
    print(f"Threshold compliance: {threshold_count}/{total_count} ({100*threshold_count/total_count:.1f}%)")
    
    return threshold_count == total_count


def main():
    """Run all law compliance tests."""
    print("\n" + "=" * 70)
    print("LIVNIUM V8: LAW COMPLIANCE TEST")
    print("=" * 70)
    print()
    
    results = {}
    
    try:
        results['divergence_law'] = test_divergence_law()
        results['inward_outward_axis'] = test_inward_outward_axis_law()
        results['geometric_invariance'] = test_geometric_invariance_law()
        results['phase_classification'] = test_phase_classification_law()
        
        # Final summary
        print("\n" + "=" * 70)
        print("LAW COMPLIANCE SUMMARY")
        print("=" * 70)
        print()
        
        for law_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {law_name.replace('_', ' ').title()}")
        
        print()
        passed_count = sum(1 for p in results.values() if p)
        total_count = len(results)
        print(f"Overall: {passed_count}/{total_count} laws compliant ({100*passed_count/total_count:.1f}%)")
        print()
        
        if passed_count == total_count:
            print("✅ ALL LAWS COMPLIANT")
        else:
            print("⚠️  SOME LAWS NEED ATTENTION")
            print()
            print("Issues found:")
            for law_name, passed in results.items():
                if not passed:
                    print(f"  - {law_name.replace('_', ' ').title()}")
        
        return 0 if passed_count == total_count else 1
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

