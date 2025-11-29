"""
Quick test script for nli_v5 to verify everything works.
"""

import sys
import os

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.nli_v5 import ChainEncoder, LivniumV5Classifier

def test_basic_classification():
    """Test basic classification functionality."""
    print("Testing Livnium NLI v5...")
    print("=" * 70)
    
    # Initialize encoder
    encoder = ChainEncoder()
    print("✓ Encoder initialized")
    
    # Test cases
    test_cases = [
        ("A cat runs", "A cat is running", "entailment"),
        ("A dog barks", "A cat meows", "contradiction"),
        ("A man walks", "The weather is nice", "neutral"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for premise, hypothesis, expected_label in test_cases:
        # Encode
        pair = encoder.encode_pair(premise, hypothesis)
        classifier = LivniumV5Classifier(pair)
        
        # Classify
        result = classifier.classify()
        
        # Check
        is_correct = result.label == expected_label
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Premise: '{premise}'")
        print(f"  Hypothesis: '{hypothesis}'")
        print(f"  Expected: {expected_label}, Got: {result.label}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Scores: E={result.scores['entailment']:.3f}, "
              f"C={result.scores['contradiction']:.3f}, "
              f"N={result.scores['neutral']:.3f}")
        print()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%} ({correct}/{total})")
    print("=" * 70)
    
    if accuracy > 0:
        print("✓ Basic classification works!")
    else:
        print("⚠️  Classification needs tuning")
    
    return accuracy > 0


if __name__ == '__main__':
    success = test_basic_classification()
    sys.exit(0 if success else 1)

