#!/usr/bin/env python3
"""Test semantic warp alignment."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.nli_v5.core.semantic_warp import SemanticWarp
from experiments.nli_v5.core.fracture_dynamics import FractureDynamics
from experiments.nli_simple.native_chain import WordVector

print("=" * 70)
print("SEMANTIC WARP TEST")
print("=" * 70)
print()
print("Philosophy: Let geometry choose alignment automatically.")
print("No hardcoded words. No rules. Pure physics + optimization.")
print()

warp = SemanticWarp(use_cosine_distance=True)
fracture_detector = FractureDynamics(fracture_threshold=0.5)

test_cases = [
    ("dog is barking", "dog is not barking"),
    ("cat sleeps", "cat never sleeps"),
    ("man walks", "person moves"),  # Paraphrase - should warp smoothly
    ("bird flies", "bird can fly"),  # Different word order
]

for premise, hypothesis in test_cases:
    print(f"Premise:    {premise}")
    print(f"Hypothesis: {hypothesis}")
    
    # Vectorize
    p_tokens = premise.lower().split()
    h_tokens = hypothesis.lower().split()
    p_vecs = [WordVector(w, vector_size=27).get_vector() for w in p_tokens]
    h_vecs = [WordVector(w, vector_size=27).get_vector() for w in h_tokens]
    
    # Find warp alignment
    alignment = warp.align(p_vecs, h_vecs)
    
    print(f"  Warp path: {alignment.warp_path}")
    print(f"  Total energy: {alignment.total_energy:.4f}")
    print(f"  Aligned pairs:")
    for idx, (i, j) in enumerate(alignment.warp_path):
        p_word = p_tokens[i] if i < len(p_tokens) else "N/A"
        h_word = h_tokens[j] if j < len(h_tokens) else "N/A"
        dist = alignment.distance_matrix[i, j]
        marker = " ← LOW" if dist < 0.3 else " ← HIGH"
        print(f"    [{idx}] '{p_word}' ↔ '{h_word}': {dist:.4f}{marker}")
    
    # Test fracture detection with warp
    print(f"  Fracture detection (with warp):")
    fracture = fracture_detector.detect_alignment_fracture(
        p_vecs, h_vecs, p_tokens, h_tokens, use_warp=True
    )
    
    if fracture.is_fractured:
        fracture_word = h_tokens[fracture.fracture_index] if fracture.fracture_index < len(h_tokens) else "N/A"
        print(f"    ✓ FRACTURE at index {fracture.fracture_index}: '{fracture_word}'")
        print(f"    Fracture strength: {fracture.fracture_strength:.4f}")
    else:
        print(f"    ✗ No fracture (max energy: {fracture.fracture_strength:.4f})")
    
    print()

print("=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print()
print("Semantic warp finds optimal alignment automatically.")
print("No hardcoded rules. Geometry chooses the path.")
print()
print("Then fracture detection runs on the warped alignment,")
print("finding negation where alignment breaks.")

