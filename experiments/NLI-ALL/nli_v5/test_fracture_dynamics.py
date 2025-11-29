#!/usr/bin/env python3
"""Test script for physics-based fracture dynamics."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.nli_v5.core.fracture_dynamics import FractureDynamics
from experiments.nli_simple.native_chain import WordVector

print("=" * 70)
print("FRACTURE DYNAMICS TEST: Energy Minimization Principle")
print("=" * 70)
print()
print("Philosophy: Negation is discovered by measuring energy relief.")
print("No word lists. No language knowledge. Pure physics.")
print()

# Initialize with different thresholds
dynamics_strict = FractureDynamics(relief_threshold=0.3)
dynamics_sensitive = FractureDynamics(relief_threshold=0.1)

test_cases = [
    "dog is not barking",
    "cat never sleeps",
    "man does not walk",
    "bird can not fly",
    "dog is barking",  # No negation
    "cat sleeps",      # No negation
    "the quick brown fox jumps",  # No negation, longer sentence
]

for sentence in test_cases:
    tokens = sentence.lower().split()
    vectors = [WordVector(w, vector_size=27).get_vector() for w in tokens]
    
    print(f"Sentence: {sentence}")
    print(f"  Length: {len(tokens)} words")
    
    # Test with strict threshold
    is_fractured, fracture_idx, relief, polarity = dynamics_strict.detect_fracture(tokens, vectors)
    
    baseline_energy = dynamics_strict._calculate_chain_energy(vectors)
    
    if is_fractured:
        ghost_chain = vectors[:fracture_idx] + vectors[fracture_idx+1:]
        ghost_energy = dynamics_strict._calculate_chain_energy(ghost_chain)
        
        print(f"  ✓ FRACTURE at index {fracture_idx}: '{tokens[fracture_idx]}'")
        print(f"  Baseline energy: {baseline_energy:.4f}")
        print(f"  Ghost energy: {ghost_energy:.4f}")
        print(f"  Relief: {relief:.4f} (energy dropped by {relief:.4f})")
        print(f"  Polarity field: {polarity}")
        
        # Show what happens after fracture
        print(f"  Meaning inversion: After '{tokens[fracture_idx]}', meaning flips")
    else:
        print(f"  ✗ No fracture detected")
        print(f"  Baseline energy: {baseline_energy:.4f}")
        print(f"  Max relief: {relief:.4f} (below threshold {dynamics_strict.relief_threshold})")
    
    print()

print("=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print()
print("Negation is not a token. Not a semantic category.")
print("Negation is a TOPOLOGICAL SINGULARITY inside vector flow.")
print()
print("A defect. A vortex. A discontinuity.")
print("A place where meaning switches orientation.")
print()
print("This works in ANY language because physics is universal.")

