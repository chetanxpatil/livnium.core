"""
Verify Divergence Fix: Test that the corrected physics law works correctly.

This script tests the new divergence formula to ensure:
- Entailment → negative divergence (convergence)
- Contradiction → positive divergence (divergence)
- Neutral → near-zero divergence
"""

import os
import sys
import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.nli_v5 import ChainEncoder, LivniumV5Classifier


def test_divergence_formula():
    """Test the divergence formula directly."""
    print("=" * 80)
    print("TESTING DIVERGENCE FORMULA")
    print("=" * 80)
    print()
    
    # Test cases: (alignment, expected_divergence_sign, description)
    test_cases = [
        (0.9, "negative", "High alignment (entailment)"),
        (0.8, "negative", "High alignment (entailment)"),
        (0.5, "zero", "Neutral alignment"),
        (0.3, "positive", "Low alignment (contradiction)"),
        (0.1, "positive", "Very low alignment (contradiction)"),
        (-0.3, "positive", "Negative alignment (strong contradiction)"),
        (-0.5, "positive", "Strong negative alignment (contradiction)"),
    ]
    
    print(f"{'Alignment':<15} {'Divergence':<15} {'Expected':<20} {'Status':<10}")
    print("-" * 80)
    
    all_passed = True
    for alignment, expected_sign, desc in test_cases:
        # New formula: divergence = 0.5 - alignment
        divergence = 0.5 - alignment
        
        if expected_sign == "negative":
            passed = divergence < 0
            expected_str = "negative (< 0)"
        elif expected_sign == "positive":
            passed = divergence > 0
            expected_str = "positive (> 0)"
        else:  # zero
            passed = abs(divergence) < 0.1
            expected_str = "near zero"
        
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False
        
        print(f"{alignment:>6.2f}         {divergence:>6.3f}         {expected_str:<20} {status:<10}  # {desc}")
    
    print()
    if all_passed:
        print("✓ All divergence formula tests PASSED")
    else:
        print("✗ Some tests FAILED - formula needs adjustment")
    
    return all_passed


def test_real_examples():
    """Test divergence on real SNLI examples."""
    print("\n" + "=" * 80)
    print("TESTING ON REAL EXAMPLES")
    print("=" * 80)
    print()
    
    encoder = ChainEncoder(vector_size=27)
    
    # Test cases: (premise, hypothesis, expected_label, expected_div_sign)
    test_cases = [
        ("A cat runs", "A cat is running", "entailment", "negative"),
        ("A man is cooking", "A man is eating", "neutral", "zero"),
        ("A dog barks", "A cat meows", "contradiction", "positive"),
        ("The sky is blue", "The sky is blue", "entailment", "negative"),
        ("It is sunny", "It is raining", "contradiction", "positive"),
    ]
    
    print(f"{'Premise':<25} {'Hypothesis':<25} {'Label':<12} {'Divergence':<12} {'Expected':<12} {'Status':<10}")
    print("-" * 100)
    
    all_passed = True
    for premise, hypothesis, expected_label, expected_sign in test_cases:
        pair = encoder.encode_pair(premise, hypothesis)
        classifier = LivniumV5Classifier(pair)
        result = classifier.classify()
        
        divergence = result.layer_states.get('divergence', 0.0)
        predicted_label = result.label
        
        # Check divergence sign
        if expected_sign == "negative":
            sign_ok = divergence < 0
            expected_str = "negative"
        elif expected_sign == "positive":
            sign_ok = divergence > 0
            expected_str = "positive"
        else:  # zero
            sign_ok = abs(divergence) < 0.15
            expected_str = "near zero"
        
        label_ok = predicted_label == expected_label
        
        status = "✓" if (sign_ok and label_ok) else "⚠️" if sign_ok else "✗"
        if not sign_ok:
            all_passed = False
        
        premise_short = premise[:24] if len(premise) <= 24 else premise[:21] + "..."
        hyp_short = hypothesis[:24] if len(hypothesis) <= 24 else hypothesis[:21] + "..."
        
        print(f"{premise_short:<25} {hyp_short:<25} {expected_label:<12} "
              f"{divergence:>8.4f}     {expected_str:<12} {status:<10}")
    
    print()
    if all_passed:
        print("✓ All real example tests PASSED")
    else:
        print("⚠️  Some examples show incorrect divergence signs")
        print("   This may be expected if examples are ambiguous")
    
    return all_passed


def analyze_divergence_distribution():
    """Analyze divergence distribution from pattern file if available."""
    pattern_file = os.path.join(os.path.dirname(__file__), '..', 'patterns', 'patterns_normal.json')
    
    if not os.path.exists(pattern_file):
        print("\n" + "=" * 80)
        print("PATTERN FILE NOT FOUND")
        print("=" * 80)
        print(f"\nTo analyze divergence distribution, run:")
        print(f"  python3 experiments/nli_v5/train_v5.py --clean --train 1000 --learn-patterns")
        return
    
    import json
    with open(pattern_file, 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("DIVERGENCE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()
    
    for label in ['entailment', 'contradiction', 'neutral']:
        if label not in data.get('patterns', {}):
            continue
        
        patterns = data['patterns'][label]
        divergences = [p.get('divergence', 0.0) for p in patterns]
        
        if not divergences:
            continue
        
        mean_div = np.mean(divergences)
        std_div = np.std(divergences)
        min_div = np.min(divergences)
        max_div = np.max(divergences)
        median_div = np.median(divergences)
        
        # Count correct signs
        if label == 'entailment':
            correct_sign = sum(1 for d in divergences if d < 0)
            expected = "negative"
        elif label == 'contradiction':
            correct_sign = sum(1 for d in divergences if d > 0)
            expected = "positive"
        else:  # neutral
            correct_sign = sum(1 for d in divergences if abs(d) < 0.15)
            expected = "near zero"
        
        correct_pct = 100.0 * correct_sign / len(divergences)
        
        print(f"{label.upper()}:")
        print(f"  Mean: {mean_div:.4f} ± {std_div:.4f}")
        print(f"  Range: [{min_div:.4f}, {max_div:.4f}]")
        print(f"  Median: {median_div:.4f}")
        print(f"  Expected: {expected}")
        print(f"  Correct sign: {correct_sign}/{len(divergences)} ({correct_pct:.1f}%)")
        
        if label == 'entailment' and mean_div > 0:
            print(f"  ⚠️  PROBLEM: Mean divergence is POSITIVE (should be negative)")
        elif label == 'contradiction' and mean_div < 0:
            print(f"  ⚠️  PROBLEM: Mean divergence is NEGATIVE (should be positive)")
        elif label == 'neutral' and abs(mean_div) > 0.2:
            print(f"  ⚠️  PROBLEM: Mean divergence is too far from zero")
        else:
            print(f"  ✓ Divergence sign is correct")
        print()


def main():
    print("\n" + "=" * 80)
    print("LIVNIUM DIVERGENCE FIX VERIFICATION")
    print("=" * 80)
    print("\nTesting the corrected physics law: divergence = 0.5 - alignment")
    print()
    
    # Test 1: Formula correctness
    formula_ok = test_divergence_formula()
    
    # Test 2: Real examples
    examples_ok = test_real_examples()
    
    # Test 3: Distribution analysis
    analyze_divergence_distribution()
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    
    if formula_ok and examples_ok:
        print("✓ Divergence formula is CORRECT")
        print("✓ Physics law restored: push → contradiction, pull → entailment")
        print()
        print("Next steps:")
        print("  1. Run full training to see accuracy improvement")
        print("  2. Compare patterns again to verify contradiction divergence is positive")
        print("  3. Monitor accuracy: should rise from ~36% → ~44-50%")
    else:
        print("⚠️  Some tests failed - review divergence computation")
    
    print()


if __name__ == '__main__':
    main()

