#!/usr/bin/env python3
"""Test collision-based fracture detection."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.nli_v5.core.fracture_dynamics import FractureDynamics
from experiments.nli_simple.native_chain import WordVector

print("=" * 70)
print("COLLISION-BASED FRACTURE DETECTION TEST")
print("=" * 70)
print()
print("Key Insight: Negation is NOT inside a sentence.")
print("Negation is in the COLLISION between premise and hypothesis.")
print()

dynamics = FractureDynamics(fracture_threshold=0.5)

test_cases = [
    ("dog is barking", "dog is not barking"),
    ("cat sleeps", "cat never sleeps"),
    ("man walks", "man does not walk"),
    ("bird flies", "bird can not fly"),
    ("dog is barking", "dog is barking"),  # No negation
    ("the quick brown fox", "the quick brown fox jumps"),  # No negation
]

for premise, hypothesis in test_cases:
    print(f"Premise:    {premise}")
    print(f"Hypothesis: {hypothesis}")
    
    # Vectorize
    p_tokens = premise.lower().split()
    h_tokens = hypothesis.lower().split()
    p_vecs = [WordVector(w, vector_size=27).get_vector() for w in p_tokens]
    h_vecs = [WordVector(w, vector_size=27).get_vector() for w in h_tokens]
    
    # Collide and detect fracture
    fracture = dynamics.detect_alignment_fracture(p_vecs, h_vecs, p_tokens, h_tokens)
    
    print(f"  Alignment energy per position:")
    for i, energy in enumerate(fracture.alignment_energy):
        marker = " ← FRACTURE" if (fracture.is_fractured and i == fracture.fracture_index) else ""
        p_word = p_tokens[i] if i < len(p_tokens) else "(padding)"
        h_word = h_tokens[i] if i < len(h_tokens) else "(padding)"
        print(f"    [{i}] '{p_word}' ↔ '{h_word}': {energy:.4f}{marker}")
    
    if fracture.is_fractured:
        fracture_word = h_tokens[fracture.fracture_index] if fracture.fracture_index < len(h_tokens) else "N/A"
        print(f"  ✓ FRACTURE detected at index {fracture.fracture_index}: '{fracture_word}'")
        print(f"  Fracture strength: {fracture.fracture_strength:.4f}")
        print(f"  Polarity field: {fracture.polarity_field}")
        print(f"  → Meaning inverts after '{fracture_word}'")
    else:
        print(f"  ✗ No fracture (max energy: {fracture.fracture_strength:.4f})")
    
    print()

print("=" * 70)
print("THE PHYSICS LAW:")
print("=" * 70)
print()
print("Negation = the point of maximal alignment tension between premise & hypothesis.")
print()
print("This is vector elastic collision analysis.")
print("Livnium behaves like solid mechanics, elastic deformation, fracture detection.")
print()
print("alignment → resonance")
print("misalignment → tension")
print("operator → discontinuity")

