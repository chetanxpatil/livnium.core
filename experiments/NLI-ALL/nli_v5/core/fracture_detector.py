"""
Fracture Detector: Simple 10-line negation detection

Detects negation by comparing original vs ghost sentence vectors.
Pure geometry. No rules. No PCA. No syntax.

Option A: Simple version (this file)
Option B: Physics version (future)
Option C: Hybrid version (future)
"""

import numpy as np
from typing import List, Tuple


def apply_structural_pressure(orig_vecs: List[np.ndarray], negated_vecs: List[np.ndarray]) -> Tuple[int, float, np.ndarray]:
    """
    Detect the fracture point (negation) by comparing
    original vs ghost sentence vectors.
    
    Args:
        orig_vecs: List of word vectors for real sentence (e.g., "dog is not barking")
        negated_vecs: List of word vectors for ghost sentence (e.g., "dog is barking")
    
    Returns:
        fracture_index: Index where maximum divergence occurs (negation position)
        fracture_strength: Magnitude of maximum divergence
        diff: Per-position divergence array
    """
    # 1. Pad to same length
    L = max(len(orig_vecs), len(negated_vecs))
    A = np.vstack([orig_vecs + [np.zeros_like(orig_vecs[0])] * (L - len(orig_vecs))])
    B = np.vstack([negated_vecs + [np.zeros_like(negated_vecs[0])] * (L - len(negated_vecs))])
    
    # 2. Compute per-position divergence
    diff = np.linalg.norm(A - B, axis=1)
    
    # 3. Find fracture point
    fracture_index = int(np.argmax(diff))
    fracture_strength = float(np.max(diff))
    
    return fracture_index, fracture_strength, diff


def detect_negation_fracture(
    sentence_tokens: List[str],
    word_vectors: List[np.ndarray],
    negation_words: List[str] = None
) -> Tuple[int, float, np.ndarray, List[str]]:
    """
    Detect negation fracture in a sentence.
    
    Creates ghost sentence by removing negation words, then compares.
    
    Args:
        sentence_tokens: List of word tokens
        word_vectors: Corresponding word vectors
        negation_words: Optional list of negation words to remove (default: common ones)
    
    Returns:
        fracture_index: Index of negation word (if found)
        fracture_strength: Strength of fracture
        diff: Per-position divergence array
        ghost_tokens: Ghost sentence tokens (negation removed)
    """
    if negation_words is None:
        negation_words = ['not', 'never', 'no', "n't", "doesn't", "don't", "won't", "can't", "isn't", "aren't"]
    
    # Create ghost sentence (remove negation words)
    ghost_tokens = []
    ghost_vecs = []
    negation_indices = []
    
    for i, token in enumerate(sentence_tokens):
        token_lower = token.lower().strip(".,!?;:")
        if token_lower not in negation_words:
            ghost_tokens.append(token)
            ghost_vecs.append(word_vectors[i])
        else:
            negation_indices.append(i)
    
    # If no negation found, return zeros
    if not negation_indices:
        diff = np.zeros(len(word_vectors))
        return -1, 0.0, diff, ghost_tokens
    
    # Pad ghost to same length as original for comparison
    L = len(word_vectors)
    ghost_padded = ghost_vecs + [np.zeros_like(word_vectors[0])] * (L - len(ghost_vecs))
    
    # Compute fracture
    fracture_index, fracture_strength, diff = apply_structural_pressure(
        word_vectors,
        ghost_padded
    )
    
    return fracture_index, fracture_strength, diff, ghost_tokens


def example_usage():
    """Example usage of fracture detector."""
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from experiments.nli_simple.native_chain import WordVector
    
    # Example: "dog is not barking"
    sentence = ["dog", "is", "not", "barking"]
    ghost = ["dog", "is", "barking"]
    
    # Create word vectors
    orig_vecs = [WordVector(w, vector_size=27).get_vector() for w in sentence]
    ghost_vecs = [WordVector(w, vector_size=27).get_vector() for w in ghost]
    
    # Detect fracture
    idx, strength, diff = apply_structural_pressure(orig_vecs, ghost_vecs)
    
    print("=" * 70)
    print("FRACTURE DETECTOR EXAMPLE")
    print("=" * 70)
    print()
    print(f"Sentence: {' '.join(sentence)}")
    print(f"Ghost:    {' '.join(ghost)}")
    print()
    print(f"Fracture at index: {idx} ('{sentence[idx] if idx < len(sentence) else 'N/A'}')")
    print(f"Fracture strength: {strength:.4f}")
    print()
    print("Per-position divergence:")
    for i, (word, div) in enumerate(zip(sentence, diff)):
        marker = " ← FRACTURE" if i == idx else ""
        print(f"  [{i}] {word:10s}: {div:.4f}{marker}")
    print()


if __name__ == '__main__':
    example_usage()

