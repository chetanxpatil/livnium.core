#!/usr/bin/env python3
"""Test collision-based fracture detection integrated into Livnium classifier."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.nli_v5 import ChainEncoder, LivniumV5Classifier

print("=" * 70)
print("FRACTURE DETECTION INTEGRATION TEST")
print("=" * 70)
print()
print("Testing collision-based negation detection in Livnium classifier.")
print()

# Initialize encoder
encoder = ChainEncoder()

# Test cases with negation
test_cases = [
    # (premise, hypothesis, expected_label)
    ("A dog is barking", "A dog is not barking", "contradiction"),
    ("A cat sleeps", "A cat never sleeps", "contradiction"),
    ("A man walks", "A man does not walk", "contradiction"),
    ("A bird flies", "A bird can not fly", "contradiction"),
    ("A dog is barking", "A dog is barking", "entailment"),  # No negation
    ("A cat sleeps", "A cat is sleeping", "entailment"),  # No negation
    ("A man walks", "A person moves", "entailment"),  # No negation
    ("A dog barks", "A cat meows", "neutral"),  # No negation, different subjects
]

correct = 0
total = len(test_cases)

print("Test Results:")
print("-" * 70)

for premise, hypothesis, expected_label in test_cases:
    # Encode and classify
    pair = encoder.encode_pair(premise, hypothesis)
    classifier = LivniumV5Classifier(pair)
    result = classifier.classify()
    
    # Get fracture info from layer states
    layer_states = result.layer_states
    opposition_output = layer_states.get('layer_opposition', {})
    fracture_detected = opposition_output.get('fracture_detected', False)
    fracture_strength = opposition_output.get('fracture_strength', 0.0)
    divergence = opposition_output.get('divergence_final', 0.0)
    
    # Check if correct
    is_correct = result.label == expected_label
    if is_correct:
        correct += 1
    
    status = "✓" if is_correct else "✗"
    
    print(f"{status} Premise: '{premise}'")
    print(f"  Hypothesis: '{hypothesis}'")
    print(f"  Expected: {expected_label:12s} | Got: {result.label:12s} | Confidence: {result.confidence:.3f}")
    
    if fracture_detected:
        print(f"  🔥 FRACTURE DETECTED (strength: {fracture_strength:.4f})")
        print(f"  Divergence: {divergence:.4f}")
    else:
        print(f"  No fracture detected")
        print(f"  Divergence: {divergence:.4f}")
    
    print()

print("=" * 70)
print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
print("=" * 70)
print()
print("Key Insight:")
print("  Negation is detected by COLLIDING premise and hypothesis.")
print("  The fracture (negation operator) creates maximal alignment tension.")
print("  This boosts the contradiction signal in divergence computation.")

