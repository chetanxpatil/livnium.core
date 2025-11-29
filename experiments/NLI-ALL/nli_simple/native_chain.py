"""
Simple Native Chain: Pure Vector-Based Text Encoding

No LivniumCoreSystem, no QuantumCell, no 3D lattices.
Just: Letter → Vector → Word → Vector → Sentence → List[Vectors]
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# Simple Lexicon (Learned Word Polarity Only)
# ============================================================================

class SimpleLexicon:
    """
    Stores learned word polarities [E, C, N] for each word.
    No letter geometries - just word semantics learned from data.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SimpleLexicon, cls).__new__(cls)
            # Maps word -> [entailment_score, contradiction_score, neutral_score]
            cls._instance.polarity_store = {}
        return cls._instance
    
    def update_word_polarity(self, word: str, label_idx: int, strength: float = 0.1):
        """
        Learn the semantic polarity of a word from training data.
        
        Args:
            word: The word to update (e.g., "not")
            label_idx: 0=Entailment, 1=Contradiction, 2=Neutral
            strength: Learning rate
        """
        w = word.lower()
        if w not in self.polarity_store:
            # Initialize with weak neutral prior
            self.polarity_store[w] = np.array([0.33, 0.33, 0.33], dtype=float)
        
        # Create target vector (one-hot)
        target = np.zeros(3)
        target[label_idx] = 1.0
        
        # Move current polarity towards target (Exponential Moving Average)
        current = self.polarity_store[w]
        self.polarity_store[w] = current * (1.0 - strength) + target * strength
    
    def get_word_polarity(self, word: str) -> np.ndarray:
        """Get learned polarity vector for a word."""
        return self.polarity_store.get(word.lower(), np.array([0.33, 0.33, 0.33]))
    
    def clear(self):
        """Clear all stored polarities."""
        self.polarity_store = {}
    
    def save_to_file(self, filepath: str = 'experiments/nli_simple/brain_state.pkl'):
        """Save learned polarities to disk."""
        import pickle
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        brain_data = {
            'polarity_store': {}
        }
        
        for word, polarity in self.polarity_store.items():
            brain_data['polarity_store'][word] = (
                polarity.tolist() if isinstance(polarity, np.ndarray) else polarity
            )
        
        with open(filepath, 'wb') as f:
            pickle.dump(brain_data, f)
    
    def load_from_file(self, filepath: str = 'experiments/nli_simple/brain_state.pkl'):
        """Load learned polarities from disk."""
        import pickle
        import os
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                brain_data = pickle.load(f)
            
            self.polarity_store = {}
            for word, polarity in brain_data.get('polarity_store', {}).items():
                self.polarity_store[word] = (
                    np.array(polarity) if isinstance(polarity, list) else polarity
                )
            
            return True
        except Exception as e:
            print(f"⚠️  Error loading brain: {e}")
            return False


# ============================================================================
# Letter to Vector (Hash-Based, No 3D Objects)
# ============================================================================

def letter_to_vector(letter: str, vector_size: int = 27) -> np.ndarray:
    """
    Generate a fixed-size vector for a letter using hash-based seeding.
    
    No LivniumCoreSystem, no 3D lattice - just a deterministic vector.
    """
    letter_hash = hash(letter.lower()) % (2**32)
    np.random.seed(letter_hash)
    
    # Generate vector directly
    vector = np.random.uniform(-1.0, 1.0, vector_size)
    
    return vector


# ============================================================================
# Word Vector (Sequence of Letters)
# ============================================================================

class WordVector:
    """
    A word represented as a single vector (aggregated from letter vectors).
    """
    
    def __init__(self, word: str, vector_size: int = 27):
        self.word = word.lower()
        self.letters = list(word.lower())
        self.vector_size = vector_size
        
        # Create letter vectors and aggregate into word vector
        letter_vectors = [letter_to_vector(letter, vector_size) for letter in self.letters]
        
        # Simple aggregation: sum and normalize
        self.vector = np.sum(letter_vectors, axis=0)
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm
    
    def get_vector(self) -> np.ndarray:
        """Get the word vector."""
        return self.vector.copy()


# ============================================================================
# Sentence Vector (Sequence of Words)
# ============================================================================

class SentenceVector:
    """
    A sentence represented as a CHAIN of word vectors with positional structure.
    
    This adds sequential structure (position matters) while keeping it simple.
    No Livnium physics - just positional encoding + chain structure.
    """
    
    def __init__(self, sentence: str, vector_size: int = 27):
        self.sentence = sentence.lower()
        self.tokens = sentence.lower().split()
        self.vector_size = vector_size
        
        # Create word vectors
        self.word_vectors: List[WordVector] = [
            WordVector(word, vector_size) for word in self.tokens
        ]
        
        # Add positional encoding to each word vector
        # This gives structure: position in sentence matters
        self._add_positional_encoding()
    
    def _add_positional_encoding(self):
        """
        Add positional encoding to word vectors.
        
        This creates sequential structure: words at different positions
        get different encodings, allowing the model to learn order-dependent patterns.
        """
        if not self.word_vectors:
            return
        
        num_words = len(self.word_vectors)
        
        for i, word_vec in enumerate(self.word_vectors):
            # Create positional vector (sinusoidal encoding)
            # This is a simple way to encode position without hardcoding
            pos_vec = np.zeros(self.vector_size)
            
            # Use different frequencies for different dimensions
            for dim in range(self.vector_size):
                # Sinusoidal encoding: position affects different dimensions differently
                freq = 2.0 ** (dim / self.vector_size)  # Different frequency per dimension
                pos_vec[dim] = np.sin(i * freq / num_words) if num_words > 0 else 0.0
            
            # Normalize positional vector
            pos_norm = np.linalg.norm(pos_vec)
            if pos_norm > 0:
                pos_vec = pos_vec / pos_norm
            
            # Combine word vector with positional encoding
            # Weight: 0.7 word + 0.3 position (word meaning dominates, position adds structure)
            word_vec.vector = 0.7 * word_vec.vector + 0.3 * pos_vec
            
            # Renormalize
            norm = np.linalg.norm(word_vec.vector)
            if norm > 0:
                word_vec.vector = word_vec.vector / norm
    
    def get_word_vectors(self) -> List[np.ndarray]:
        """Get list of word vectors (with positional encoding)."""
        return [wv.get_vector() for wv in self.word_vectors]
    
    def get_sentence_vector(self) -> np.ndarray:
        """Get aggregated sentence vector (sum of word vectors, normalized)."""
        if not self.word_vectors:
            return np.zeros(self.vector_size)
        
        combined = np.sum([wv.get_vector() for wv in self.word_vectors], axis=0)
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined
    
    def compare(self, other: 'SentenceVector', use_sequence: bool = True) -> float:
        """
        Compare two sentences using chain-based matching.
        
        Args:
            use_sequence: If True, use sequential matching (position matters).
                         If False, use bag-of-words matching.
        
        Returns similarity score [-1, 1]
        """
        if not self.word_vectors or not other.word_vectors:
            return 0.0
        
        if use_sequence:
            # Sequential matching: compare words at similar positions
            # This captures order-dependent patterns
            return self._compare_sequential(other)
        else:
            # Bag-of-words matching: compare any word to any word
            return self._compare_bag_of_words(other)
    
    def _compare_sequential(self, other: 'SentenceVector') -> float:
        """
        Sequential matching: compare words at similar positions with alignment.
        
        This captures:
        - Word order (position matters)
        - Sequential patterns (negation, quantifiers)
        - Position-dependent relationships (subject-verb-object)
        """
        if not self.word_vectors or not other.word_vectors:
            return 0.0
        
        # Aligned matching: compare words at same positions
        # This is crucial for detecting negation, quantifiers, etc.
        aligned_sims = []
        min_len = min(len(self.word_vectors), len(other.word_vectors))
        
        for i in range(min_len):
            p_vec = self.word_vectors[i].get_vector()
            h_vec = other.word_vectors[i].get_vector()
            
            # Cosine similarity
            dot_prod = np.dot(p_vec, h_vec)
            norm_p = np.linalg.norm(p_vec)
            norm_h = np.linalg.norm(h_vec)
            
            if norm_p > 0 and norm_h > 0:
                similarity = dot_prod / (norm_p * norm_h)
                aligned_sims.append(similarity)
        
        # Sliding window matching: allow small position shifts
        # This handles word reordering while maintaining structure
        window_sims = []
        for offset in range(-1, 2):  # Check -1, 0, +1 positions
            offset_sims = []
            for i in range(min_len):
                p_idx = i
                h_idx = i + offset
                if 0 <= h_idx < len(other.word_vectors):
                    p_vec = self.word_vectors[p_idx].get_vector()
                    h_vec = other.word_vectors[h_idx].get_vector()
                    
                    dot_prod = np.dot(p_vec, h_vec)
                    norm_p = np.linalg.norm(p_vec)
                    norm_h = np.linalg.norm(h_vec)
                    
                    if norm_p > 0 and norm_h > 0:
                        similarity = dot_prod / (norm_p * norm_h)
                        offset_sims.append(similarity)
            
            if offset_sims:
                window_sims.append(np.mean(offset_sims))
        
        # Sequential score: aligned matching (position matters)
        seq_score = np.mean(aligned_sims) if aligned_sims else 0.0
        
        # Window score: best sliding window match
        window_score = max(window_sims) if window_sims else 0.0
        
        # Cross-word matching (for flexibility)
        cross_score = self._compare_bag_of_words(other)
        
        # Combine: 60% aligned + 20% window + 20% cross-word
        # Aligned matching is most important for structure
        return 0.6 * seq_score + 0.2 * window_score + 0.2 * cross_score
    
    def _compare_bag_of_words(self, other: 'SentenceVector') -> float:
        """
        Bag-of-words matching: compare any word to any word.
        
        This captures lexical overlap without position constraints.
        """
        total_similarity = 0.0
        matches = 0
        
        # Compare each word in hypothesis against all words in premise
        for h_word_vec in other.word_vectors:
            best_match = -1.0
            
            for p_word_vec in self.word_vectors:
                # Cosine similarity
                p_vec = p_word_vec.get_vector()
                h_vec = h_word_vec.get_vector()
                
                dot_prod = np.dot(p_vec, h_vec)
                norm_p = np.linalg.norm(p_vec)
                norm_h = np.linalg.norm(h_vec)
                
                if norm_p > 0 and norm_h > 0:
                    similarity = dot_prod / (norm_p * norm_h)
                    best_match = max(best_match, similarity)
            
            total_similarity += best_match
            matches += 1
        
        return total_similarity / max(1, matches)

