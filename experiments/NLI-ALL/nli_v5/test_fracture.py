#!/usr/bin/env python3
"""Quick test script for fracture detector."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.nli_v5.core.fracture_detector import detect_negation_fracture
from experiments.nli_simple.native_chain import WordVector

# Test cases
test_cases = [
    "dog is not barking",
    "cat never sleeps",
    "man does not walk",
    "bird can not fly",
    "dog is barking",  # No negation
]

print("=" * 70)
print("FRACTURE DETECTOR TEST")
print("=" * 70)
print()

for sentence in test_cases:
    tokens = sentence.lower().split()
    word_vectors = [WordVector(w, vector_size=27).get_vector() for w in tokens]
    
    fracture_index, strength, diff, ghost = detect_negation_fracture(tokens, word_vectors)
    
    print(f"Sentence: {sentence}")
    print(f"Ghost:    {' '.join(ghost)}")
    
    if fracture_index >= 0:
        print(f"✓ Fracture at index {fracture_index}: '{tokens[fracture_index]}' (strength: {strength:.4f})")
    else:
        print("✗ No negation detected")
    
    print(f"Divergence: {diff}")
    print()

