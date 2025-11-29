"""
Fracture Dynamics: Collision-Based Negation Detection

Negation is NOT a property of a single chain.
Negation is a RELATION between two chains.

When premise and hypothesis COLLIDE, the point of maximal alignment tension = negation.

This is vector elastic collision analysis - exactly like solid mechanics.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class AlignmentFracture:
    """Result of collision-based fracture detection."""
    is_fractured: bool
    fracture_index: int  # Position where alignment breaks
    fracture_strength: float  # Magnitude of alignment discontinuity
    alignment_energy: np.ndarray  # Per-position alignment divergence
    polarity_field: np.ndarray  # Polarity inversion field


class FractureDynamics:
    """
    Collision-based fracture detection.
    
    Philosophy: Negation = the point of maximal alignment tension between premise & hypothesis.
    Not inside the sentence, but in the ALIGNMENT between the two sentences.
    
    This is exactly like matter hitting antimatter - the fracture happens at the collision boundary.
    """
    
    def __init__(self, fracture_threshold: float = 0.5):
        """
        Initialize fracture dynamics.
        
        Args:
            fracture_threshold: Minimum alignment divergence to consider a fracture
        """
        self.fracture_threshold = fracture_threshold
    
    def detect_alignment_fracture(
        self,
        premise_vectors: List[np.ndarray],
        hypothesis_vectors: List[np.ndarray],
        premise_tokens: List[str] = None,
        hypothesis_tokens: List[str] = None,
        use_warp: bool = True
    ) -> AlignmentFracture:
        """
        Detect fracture by colliding premise and hypothesis.
        
        Negation appears as a break in alignment - a point where premise and hypothesis
        cannot align, creating maximal divergence (energy spike).
        
        If use_warp=True, uses semantic warp (DP alignment) before fracture detection.
        This finds optimal alignment automatically - no hardcoded rules.
        
        Args:
            premise_vectors: Word vectors for premise sentence
            hypothesis_vectors: Word vectors for hypothesis sentence
            premise_tokens: Optional tokens for debugging
            hypothesis_tokens: Optional tokens for debugging
            use_warp: If True, use semantic warp alignment before fracture detection
        
        Returns:
            AlignmentFracture with fracture detection results
        """
        # 0. OPTIONAL: Use semantic warp to find optimal alignment first
        if use_warp:
            from .semantic_warp import SemanticWarp
            warp = SemanticWarp(use_cosine_distance=True)
            warp_alignment = warp.align(premise_vectors, hypothesis_vectors)
            
            # Get aligned vectors according to warp path
            aligned_premise, aligned_hypothesis = warp.get_aligned_vectors(
                premise_vectors, hypothesis_vectors, warp_alignment
            )
            
            # Use warped alignment for fracture detection
            premise_vectors = aligned_premise
            hypothesis_vectors = aligned_hypothesis
        
        # 1. Align vectors by position (only compare where both have words)
        min_len = min(len(premise_vectors), len(hypothesis_vectors))
        max_len = max(len(premise_vectors), len(hypothesis_vectors))
        
        # 2. Compute alignment energy for each position where BOTH have words
        alignment_energy = np.zeros(max_len)
        
        for i in range(max_len):
            if i < len(premise_vectors) and i < len(hypothesis_vectors):
                # Both have words at this position - compute divergence
                p_vec = premise_vectors[i]
                h_vec = hypothesis_vectors[i]
                alignment_energy[i] = self._compute_alignment_divergence(p_vec, h_vec)
            else:
                # Padding position - set to NaN (will be ignored)
                alignment_energy[i] = np.nan
        
        # 3. Find the largest discontinuity (energy spike) among valid positions
        # Look for sudden jumps in divergence, not just maximum
        valid_indices = np.where(~np.isnan(alignment_energy))[0]
        valid_energy = alignment_energy[valid_indices]
        
        if len(valid_energy) == 0:
            # No valid positions to compare
            return AlignmentFracture(
                is_fractured=False,
                fracture_index=-1,
                fracture_strength=0.0,
                alignment_energy=alignment_energy,
                polarity_field=np.ones(max_len, dtype=np.float32)
            )
        
        # Find divergence spikes (sudden increases)
        # Negation creates a discontinuity - a sudden jump from low to high divergence
        if len(valid_energy) > 1:
            # Compute divergence changes (derivative)
            divergence_changes = np.diff(valid_energy)
            
            # Find the largest positive jump (sudden increase in divergence)
            # This is where negation appears
            if np.any(divergence_changes > 0):
                max_jump_idx = np.argmax(divergence_changes)
                # Fracture is at the position AFTER the jump
                fracture_index = int(valid_indices[max_jump_idx + 1])
                fracture_strength = float(valid_energy[max_jump_idx + 1])
            else:
                # No sudden jumps - use maximum divergence
                max_valid_idx = np.argmax(valid_energy)
                fracture_index = int(valid_indices[max_valid_idx])
                fracture_strength = float(valid_energy[max_valid_idx])
        else:
            # Only one position - use it if divergence is high
            fracture_index = int(valid_indices[0])
            fracture_strength = float(valid_energy[0])
        
        # 4. Determine if it's a fracture
        # Fracture must be significant AND at a valid position
        # Also check that divergence is high enough (not just a small jump)
        is_fractured = (fracture_strength > self.fracture_threshold and 
                       fracture_index < min_len)
        
        # 5. Generate polarity field
        # After fracture, meaning inverts
        polarity_field = np.ones(max_len, dtype=np.float32)
        if is_fractured:
            # Fracture point itself (negation operator)
            polarity_field[fracture_index] = 0.0
            
            # Everything after fracture is inverted
            if fracture_index + 1 < max_len:
                polarity_field[fracture_index + 1:] = -1.0
        
        return AlignmentFracture(
            is_fractured=is_fractured,
            fracture_index=fracture_index,
            fracture_strength=fracture_strength,
            alignment_energy=alignment_energy,
            polarity_field=polarity_field
        )
    
    def _compute_alignment_divergence(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute alignment divergence between two vectors.
        
        Measures how much two vectors resist alignment.
        Higher divergence = more tension = potential fracture point.
        
        Uses cosine distance (semantic divergence).
        
        Args:
            vec1: First vector (premise word)
            vec2: Second vector (hypothesis word)
        
        Returns:
            Alignment divergence (0 = aligned, 1 = opposite, 2 = orthogonal)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # If either vector is zero (padding), divergence is high
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 2.0  # Maximum divergence (no alignment possible)
        
        # Cosine similarity
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        # Cosine distance (divergence)
        # 0 = perfectly aligned, 1 = opposite, 2 = orthogonal
        divergence = 1.0 - cosine_sim
        
        return float(divergence)
    
    def apply_polarity_inversion(
        self,
        vectors: List[np.ndarray],
        polarity_field: np.ndarray
    ) -> List[np.ndarray]:
        """
        Apply polarity inversion to vectors based on fracture.
        
        After fracture point, meaning flips (vectors inverted).
        
        Args:
            vectors: Original word vectors
            polarity_field: Polarity field from fracture detection
        
        Returns:
            Mutated vectors with polarity applied
        """
        mutated = []
        for i, vec in enumerate(vectors):
            if i < len(polarity_field):
                if polarity_field[i] == -1.0:
                    # Inverted flow - flip meaning
                    mutated.append(-vec)
                else:
                    # Normal flow - keep original
                    mutated.append(vec)
            else:
                mutated.append(vec)
        
        return mutated


def example_usage():
    """Example usage of collision-based fracture detection."""
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from experiments.nli_simple.native_chain import WordVector
    
    print("=" * 70)
    print("COLLISION-BASED FRACTURE DETECTION")
    print("=" * 70)
    print()
    print("Philosophy: Negation = maximal alignment tension between premise & hypothesis.")
    print("Not inside the sentence, but in the COLLISION between sentences.")
    print()
    
    # Initialize
    dynamics = FractureDynamics(fracture_threshold=0.5)
    
    # Test cases: premise vs hypothesis collisions
    test_cases = [
        # (premise, hypothesis, expected_fracture_word)
        ("dog is barking", "dog is not barking", "not"),
        ("cat sleeps", "cat never sleeps", "never"),
        ("man walks", "man does not walk", "not"),
        ("bird flies", "bird can not fly", "not"),
        ("dog is barking", "dog is barking", None),  # No negation
    ]
    
    for premise, hypothesis, expected_word in test_cases:
        print(f"Premise:    {premise}")
        print(f"Hypothesis: {hypothesis}")
        
        # Vectorize
        p_tokens = premise.lower().split()
        h_tokens = hypothesis.lower().split()
        p_vecs = [WordVector(w, vector_size=27).get_vector() for w in p_tokens]
        h_vecs = [WordVector(w, vector_size=27).get_vector() for w in h_tokens]
        
        # Detect fracture
        fracture = dynamics.detect_alignment_fracture(p_vecs, h_vecs, p_tokens, h_tokens)
        
        if fracture.is_fractured:
            # Find which word is at fracture position
            fracture_word = None
            if fracture.fracture_index < len(h_tokens):
                fracture_word = h_tokens[fracture.fracture_index]
            
            status = "✓" if fracture_word == expected_word else "?"
            print(f"  {status} FRACTURE at index {fracture.fracture_index}: '{fracture_word}'")
            print(f"  Fracture strength: {fracture.fracture_strength:.4f}")
            print(f"  Alignment energy: {fracture.alignment_energy}")
            print(f"  Polarity field: {fracture.polarity_field}")
        else:
            status = "✓" if expected_word is None else "✗"
            print(f"  {status} No fracture detected (max energy: {fracture.fracture_strength:.4f})")
        
        print()


if __name__ == '__main__':
    example_usage()
