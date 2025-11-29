"""
Semantic Warp: Dynamic Programming Alignment

Finds optimal alignment between premise and hypothesis vectors using pure geometry.
No hardcoded words. No rules. No heuristics. Pure physics + optimization.

Uses dynamic programming (like DTW) to find minimum-energy path through distance matrix.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WarpAlignment:
    """Result of semantic warp alignment."""
    premise_indices: List[int]  # Which premise positions are aligned
    hypothesis_indices: List[int]  # Which hypothesis positions are aligned
    warp_path: List[Tuple[int, int]]  # (premise_idx, hypothesis_idx) pairs
    total_energy: float  # Total energy of the warp path
    distance_matrix: np.ndarray  # Full distance matrix


class SemanticWarp:
    """
    Semantic warp alignment using dynamic programming.
    
    Philosophy: Let geometry choose the alignment automatically.
    Finds the minimum-energy path through the distance matrix.
    
    This is exactly like:
    - Protein folding
    - Speech alignment
    - Dynamic time warping (DTW)
    
    No rules. Pure geometry.
    """
    
    def __init__(self, use_cosine_distance: bool = True):
        """
        Initialize semantic warp.
        
        Args:
            use_cosine_distance: If True, use cosine distance (semantic similarity)
                                If False, use Euclidean distance
        """
        self.use_cosine_distance = use_cosine_distance
    
    def compute_distance_matrix(
        self,
        premise_vectors: List[np.ndarray],
        hypothesis_vectors: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute distance matrix between all premise and hypothesis vectors.
        
        D[i][j] = distance between premise[i] and hypothesis[j]
        
        Args:
            premise_vectors: List of premise word vectors
            hypothesis_vectors: List of hypothesis word vectors
        
        Returns:
            Distance matrix (m x n) where m=len(premise), n=len(hypothesis)
        """
        m = len(premise_vectors)
        n = len(hypothesis_vectors)
        
        distance_matrix = np.zeros((m, n))
        
        for i, p_vec in enumerate(premise_vectors):
            for j, h_vec in enumerate(hypothesis_vectors):
                if self.use_cosine_distance:
                    # Cosine distance (semantic similarity)
                    norm_p = np.linalg.norm(p_vec)
                    norm_h = np.linalg.norm(h_vec)
                    
                    if norm_p > 1e-6 and norm_h > 1e-6:
                        cosine_sim = np.dot(p_vec, h_vec) / (norm_p * norm_h)
                        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                        distance = 1.0 - cosine_sim  # Cosine distance
                    else:
                        distance = 2.0  # Maximum distance
                else:
                    # Euclidean distance
                    distance = np.linalg.norm(p_vec - h_vec)
                
                distance_matrix[i, j] = distance
        
        return distance_matrix
    
    def find_warp_path(
        self,
        distance_matrix: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], float]:
        """
        Find minimum-energy warp path using dynamic programming.
        
        Uses DTW-like algorithm to find optimal alignment path.
        
        Args:
            distance_matrix: Distance matrix (m x n)
        
        Returns:
            (warp_path, total_energy)
            - warp_path: List of (i, j) tuples representing alignment
            - total_energy: Total energy of the optimal path
        """
        m, n = distance_matrix.shape
        
        if m == 0 or n == 0:
            return [], 0.0
        
        # DP table: dp[i][j] = minimum energy to reach (i, j)
        dp = np.full((m, n), np.inf)
        dp[0, 0] = distance_matrix[0, 0]
        
        # Traceback table: track which cell we came from
        traceback = {}
        
        # Fill DP table
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                # Possible moves: (i-1, j), (i, j-1), (i-1, j-1)
                candidates = []
                
                if i > 0:
                    # Move from above
                    energy = dp[i-1, j] + distance_matrix[i, j]
                    candidates.append((energy, (i-1, j)))
                
                if j > 0:
                    # Move from left
                    energy = dp[i, j-1] + distance_matrix[i, j]
                    candidates.append((energy, (i, j-1)))
                
                if i > 0 and j > 0:
                    # Move diagonally (one-to-one alignment)
                    energy = dp[i-1, j-1] + distance_matrix[i, j]
                    candidates.append((energy, (i-1, j-1)))
                
                # Choose minimum energy path
                if candidates:
                    min_energy, prev_cell = min(candidates, key=lambda x: x[0])
                    dp[i, j] = min_energy
                    traceback[(i, j)] = prev_cell
        
        # Traceback to find optimal path
        warp_path = []
        current = (m - 1, n - 1)
        
        while current is not None:
            warp_path.append(current)
            current = traceback.get(current)
        
        warp_path.reverse()
        total_energy = dp[m - 1, n - 1]
        
        return warp_path, total_energy
    
    def align(
        self,
        premise_vectors: List[np.ndarray],
        hypothesis_vectors: List[np.ndarray]
    ) -> WarpAlignment:
        """
        Find optimal semantic warp alignment.
        
        Args:
            premise_vectors: List of premise word vectors
            hypothesis_vectors: List of hypothesis word vectors
        
        Returns:
            WarpAlignment with optimal alignment path
        """
        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(premise_vectors, hypothesis_vectors)
        
        # Find optimal warp path
        warp_path, total_energy = self.find_warp_path(distance_matrix)
        
        # Extract aligned indices
        premise_indices = [i for i, j in warp_path]
        hypothesis_indices = [j for i, j in warp_path]
        
        return WarpAlignment(
            premise_indices=premise_indices,
            hypothesis_indices=hypothesis_indices,
            warp_path=warp_path,
            total_energy=total_energy,
            distance_matrix=distance_matrix
        )
    
    def get_aligned_vectors(
        self,
        premise_vectors: List[np.ndarray],
        hypothesis_vectors: List[np.ndarray],
        alignment: WarpAlignment
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get aligned vectors based on warp path.
        
        Returns vectors aligned according to the warp path.
        Unaligned positions are skipped (not padded).
        
        Args:
            premise_vectors: Original premise vectors
            hypothesis_vectors: Original hypothesis vectors
            alignment: WarpAlignment result
        
        Returns:
            (aligned_premise_vecs, aligned_hypothesis_vecs)
        """
        aligned_premise = []
        aligned_hypothesis = []
        
        for i, j in alignment.warp_path:
            aligned_premise.append(premise_vectors[i])
            aligned_hypothesis.append(hypothesis_vectors[j])
        
        return aligned_premise, aligned_hypothesis


def example_usage():
    """Example usage of semantic warp."""
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from experiments.nli_simple.native_chain import WordVector
    
    print("=" * 70)
    print("SEMANTIC WARP: DYNAMIC PROGRAMMING ALIGNMENT")
    print("=" * 70)
    print()
    print("Philosophy: Let geometry choose the alignment automatically.")
    print("No hardcoded words. No rules. Pure physics + optimization.")
    print()
    
    # Initialize warp
    warp = SemanticWarp(use_cosine_distance=True)
    
    # Test cases
    test_cases = [
        ("dog is barking", "dog is not barking"),
        ("cat sleeps", "cat never sleeps"),
        ("man walks", "person moves"),
        ("bird flies", "bird can fly"),
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
            print(f"    [{idx}] '{p_word}' ↔ '{h_word}': {dist:.4f}")
        print()


if __name__ == '__main__':
    example_usage()

