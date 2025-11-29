"""
Native Chain Encoder: Letter-by-Letter Chained Omcubes (Livnium Phoneme Layer)

ARCHITECTURE:
  Letter → LetterOmcube (3x3x3 geometry)
  Word → WordChain (chained LetterOmcubes)
  Sentence → SentenceChain (chained WordChains)

This creates natural morphological similarity, better generalization, and stable memory.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.quantum.quantum_cell import QuantumCell


# ============================================================================
# GLOBAL LEXICON (Letter-Level Memory)
# ============================================================================
class GlobalLexicon:
    """
    Persistent storage for:
    1. Letter Geometries (Physics)
    2. Word Polarities (Semantics) - REPLACES HARDCODED NEGATION LISTS
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalLexicon, cls).__new__(cls)
            cls._instance.letter_store = {}  # Maps letter -> (weights, amplitudes)
            # Maps word -> [entailment_score, contradiction_score, neutral_score]
            # E.g., "not" -> [0.1, 0.9, 0.0]
            cls._instance.polarity_store = {}
        return cls._instance
    
    def get_state(self, letter: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve learned state for a letter."""
        return self.letter_store.get(letter.lower())
    
    def update_state(self, letter: str, weights: np.ndarray, amplitudes: np.ndarray):
        """Save learned state for a letter."""
        self.letter_store[letter.lower()] = (weights.copy(), amplitudes.copy())

    def update_word_polarity(self, word: str, label_idx: int, strength: float = 0.1):
        """
        Learn the semantic polarity of a word.
        
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
        # "not" will drift towards [0, 1, 0] over time
        current = self.polarity_store[w]
        self.polarity_store[w] = current * (1.0 - strength) + target * strength

    def get_word_polarity(self, word: str) -> np.ndarray:
        """Get learned polarity vector for a word."""
        return self.polarity_store.get(word.lower(), np.array([0.33, 0.33, 0.33]))

    def clear(self):
        """Clear all stored states (for clean start)."""
        self.letter_store = {}
        self.polarity_store = {}
    
    def save_to_file(self, filepath: str = 'experiments/nli/brain_state.pkl'):
        """
        Save the brain (learned state) to disk.
        
        Args:
            filepath: Path to save the brain state (default: experiments/nli/brain_state.pkl)
        """
        import pickle
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON compatibility, or use pickle
        brain_data = {
            'letter_store': {},
            'polarity_store': {}
        }
        
        # Save letter store (convert numpy arrays to lists)
        for letter, (weights, amplitudes) in self.letter_store.items():
            brain_data['letter_store'][letter] = (
                weights.tolist() if isinstance(weights, np.ndarray) else weights,
                amplitudes.tolist() if isinstance(amplitudes, np.ndarray) else amplitudes
            )
        
        # Save polarity store (convert numpy arrays to lists)
        for word, polarity in self.polarity_store.items():
            brain_data['polarity_store'][word] = (
                polarity.tolist() if isinstance(polarity, np.ndarray) else polarity
            )
        
        with open(filepath, 'wb') as f:
            pickle.dump(brain_data, f)
    
    def load_from_file(self, filepath: str = 'experiments/nli/brain_state.pkl'):
        """
        Load the brain (learned state) from disk.
        
        Args:
            filepath: Path to load the brain state from (default: experiments/nli/brain_state.pkl)
        """
        import pickle
        import os
        
        if not os.path.exists(filepath):
            return False  # File doesn't exist
        
        try:
            with open(filepath, 'rb') as f:
                brain_data = pickle.load(f)
            
            # Restore letter store (convert lists back to numpy arrays)
            self.letter_store = {}
            for letter, (weights, amplitudes) in brain_data.get('letter_store', {}).items():
                self.letter_store[letter] = (
                    np.array(weights) if isinstance(weights, list) else weights,
                    np.array(amplitudes) if isinstance(amplitudes, list) else amplitudes
                )
            
            # Restore polarity store (convert lists back to numpy arrays)
            self.polarity_store = {}
            for word, polarity in brain_data.get('polarity_store', {}).items():
                self.polarity_store[word] = (
                    np.array(polarity) if isinstance(polarity, list) else polarity
                )
            
            return True
        except Exception as e:
            print(f"⚠️  Error loading brain from {filepath}: {e}")
            return False


# ============================================================================
# CORE LOGIC: Letter-Level Geometry (Simplified - Direct Vectors)
# ============================================================================

def letter_to_vector(letter: str, lattice_size: int = 3) -> np.ndarray:
    """
    Generate a 27-element vector directly for a letter (no 3D objects).
    
    Uses hash-based seeding for reproducibility, same as before.
    """
    lexicon = GlobalLexicon()
    stored_data = lexicon.get_state(letter)
    
    if stored_data:
        # LOAD FROM MEMORY (Learned State)
        weights, _ = stored_data
        # If stored as 3D array, flatten it
        if weights.ndim == 3:
            return weights.flatten()
        return weights.copy()
    else:
        # CREATE FROM HASH (Factory Default)
        letter_hash = hash(letter.lower()) % (2**32)
        np.random.seed(letter_hash)
        
        # Generate 27-element vector directly
        vector = np.zeros(lattice_size * lattice_size * lattice_size)
        boundary = (lattice_size - 1) // 2
        
        for x in range(-boundary, boundary + 1):
            for y in range(-boundary, boundary + 1):
                for z in range(-boundary, boundary + 1):
                    position_hash = hash((x, y, z)) % (2**16)
                    combined_hash = (letter_hash + position_hash) % (2**32)
                    np.random.seed(combined_hash)
                    
                    # Zero-Centered Weights (-10.0 to 10.0)
                    # Allows dot products to be 0 (orthogonal) or negative (opposing)
                    idx = (x + boundary) * lattice_size * lattice_size + (y + boundary) * lattice_size + (z + boundary)
                    vector[idx] = np.random.uniform(-10.0, 10.0)
        
        return vector


# ============================================================================
# LAYER 1: LetterOmcube
# ============================================================================

class LetterOmcube:
    """
    A single letter with direct vector representation (simplified).
    
    This is the atomic unit - the "phoneme" of Livnium.
    """
    
    def __init__(self, letter: str, lattice_size: int = 3):
        self.letter = letter.lower()
        self.lattice_size = lattice_size
        
        # 1. Create/Load Vector (direct, no 3D objects)
        self.weights = letter_to_vector(letter, lattice_size)
        
        # 2. Create/Load Quantum State
        lexicon = GlobalLexicon()
        stored_data = lexicon.get_state(letter)
        
        if stored_data:
            _, amplitudes = stored_data
            self.quantum_state = QuantumCell(
                coordinates=(0, 0, 0),
                amplitudes=amplitudes.copy(),
                num_levels=3
            )
        else:
            # Default neutral state
            self.quantum_state = QuantumCell(
                coordinates=(0, 0, 0),
                amplitudes=np.array([1.0, 0.0, 0.0], dtype=complex), 
                num_levels=3
            )
    
    def save_state(self):
        """Commit current state to Global Lexicon."""
        lexicon = GlobalLexicon()
        lexicon.update_state(
            self.letter, 
            self.weights.copy(), 
            self.quantum_state.amplitudes.copy()
        )
    
    def get_mass(self) -> float:
        """Get the mass (total ABSOLUTE symbolic weight) of this letter."""
        return float(np.sum(np.abs(self.weights)))


# ============================================================================
# LAYER 2: WordChain
# ============================================================================

class WordChain:
    """
    A word represented as a chain of entangled LetterOmcubes.
    
    Example: "run" → [r_cube, u_cube, n_cube] (entangled)
    """
    
    def __init__(self, word: str, lattice_size: int = 3):
        self.word = word.lower()
        self.letters = list(word.lower())  # Split into letters
        self.lattice_size = lattice_size
        
        # Create LetterOmcubes for each letter
        self.letter_cubes: List[LetterOmcube] = [
            LetterOmcube(letter, lattice_size) for letter in self.letters
        ]
        
        # Entangle letters in sequence
        self._entangle_letters()
    
    def _entangle_letters(self):
        """
        Physically link letters using mass-driven entanglement.
        
        This creates the emergent word geometry from letter chains.
        """
        for i in range(len(self.letter_cubes) - 1):
            prev = self.letter_cubes[i]
            next_letter = self.letter_cubes[i + 1]
            
            # Mass-driven entanglement (same logic as before)
            mass_factor = prev.get_mass() / (27.0 * self.lattice_size**3)
            amplification = 1.0 + mass_factor * 0.1
            
            # Modify next letter's quantum state
            next_letter.quantum_state.amplitudes *= amplification
            next_letter.quantum_state.normalize()
    
    def commit_learning(self):
        """Save the state of all letters in the word to long-term memory."""
        for cube in self.letter_cubes:
            cube.save_state()
    
    def get_mass(self) -> float:
        """Get total mass of the word (sum of all letter masses)."""
        return sum(cube.get_mass() for cube in self.letter_cubes)
    
    def get_geometry_vector(self) -> np.ndarray:
        """
        Get a combined geometry vector for the word.
        
        This aggregates all letter vectors into a single vector.
        Used for word-level comparison.
        """
        if not self.letter_cubes:
            return np.zeros(27)  # 3x3x3 = 27
        
        # Sum all letter vectors directly (fast!)
        combined_weights = np.zeros(27)
        for cube in self.letter_cubes:
            combined_weights += cube.weights
        
        return combined_weights
    
    def get_quantum_vector(self) -> np.ndarray:
        """Get combined quantum state vector for the word."""
        if not self.letter_cubes:
            return np.array([1.0, 0.0, 0.0], dtype=complex)
        
        # Average quantum amplitudes across letters - Numba-accelerated if available
        num_letters = len(self.letter_cubes)
        
        # OPTIMIZATION: Direct NumPy operations (faster than Numba for small arrays)
        # For small arrays (3 elements), NumPy overhead is minimal
        # Numba shines for larger arrays or many iterations
        combined_amplitudes = np.zeros(3, dtype=complex)
        for cube in self.letter_cubes:
            # Direct access to amplitudes (already NumPy array in QuantumCell)
            amps = np.asarray(cube.quantum_state.amplitudes, dtype=np.complex128)
            combined_amplitudes += amps
        
        combined_amplitudes /= num_letters
        
        return combined_amplitudes
    
    @property
    def geometry(self):
        """Backward compatibility: return geometry-like object."""
        class GeometryProxy:
            def __init__(self, weights):
                self.weights = weights.reshape(3, 3, 3) if len(weights) == 27 else weights
        return GeometryProxy(self.get_geometry_vector())
    
    @property
    def quantum_state(self):
        """Backward compatibility: return quantum state-like object that modifies underlying letter cubes."""
        class QuantumStateProxy:
            def __init__(self, word_chain):
                self.word_chain = word_chain
            
            @property
            def amplitudes(self):
                """Get combined quantum amplitudes that sync back when modified."""
                class MutableAmplitudes:
                    def __init__(self, word_chain):
                        self.word_chain = word_chain
                        self._value = word_chain.get_quantum_vector()
                    
                    def __array__(self, dtype=None):
                        if dtype is not None:
                            return np.array(self._value, dtype=dtype)
                        return self._value
                    
                    def __imul__(self, factor):
                        """In-place multiply: apply to all letter cubes."""
                        if not self.word_chain.letter_cubes:
                            return self
                        # Apply scaling to each letter cube
                        for cube in self.word_chain.letter_cubes:
                            cube.quantum_state.amplitudes *= factor
                            cube.quantum_state.normalize()
                        # Update cached value
                        self._value = self.word_chain.get_quantum_vector()
                        return self
                    
                    def __getitem__(self, key):
                        return self._value[key]
                    
                    def __setitem__(self, key, value):
                        self._value[key] = value
                        # Sync back to letter cubes (distribute proportionally)
                        if self.word_chain.letter_cubes:
                            for cube in self.word_chain.letter_cubes:
                                cube.quantum_state.amplitudes = self._value.copy()
                                cube.quantum_state.normalize()
                
                return MutableAmplitudes(self.word_chain)
            
            @amplitudes.setter
            def amplitudes(self, value):
                """Setter for amplitudes - apply to all letter cubes."""
                if not self.word_chain.letter_cubes:
                    return
                # Distribute the new amplitudes to all letter cubes
                for cube in self.word_chain.letter_cubes:
                    cube.quantum_state.amplitudes = np.array(value, dtype=complex).copy()
                    cube.quantum_state.normalize()
            
            def normalize(self):
                """Normalize all letter cube quantum states."""
                for cube in self.word_chain.letter_cubes:
                    cube.quantum_state.normalize()
        
        return QuantumStateProxy(self)


# ============================================================================
# LAYER 3: SentenceChain (Replaces Omchain)
# ============================================================================

class SentenceChain:
    """
    A sentence represented as a chain of WordChains.
    
    This is the top-level structure that replaces the old Omchain.
    """
    
    def __init__(self, sentence: str, lattice_size: int = 3):
        self.sentence = sentence.lower()
        self.tokens = sentence.lower().split()
        self.lattice_size = lattice_size
        
        # Create WordChains for each word
        self.word_chains: List[WordChain] = [
            WordChain(word, lattice_size) for word in self.tokens
        ]
    
    def commit_learning(self):
        """Save the state of all words (and their letters) to long-term memory."""
        for word_chain in self.word_chains:
            word_chain.commit_learning()
    
    def compare(self, other_chain: 'SentenceChain', use_sequence: bool = False) -> float:
        """
        Compare two SentenceChains using sliding window resonance.
        
        This compares word-chains, which internally compare letter-chains.
        """
        total_resonance = 0.0
        matches = 0
        
        # Calculate overall word overlap for semantic distance check
        p_words = set(self.tokens)
        h_words = set(other_chain.tokens)
        overall_overlap = len(p_words & h_words) / max(len(p_words | h_words), 1) if (p_words | h_words) else 0.0
        
        # Compare each word in hypothesis against all words in premise
        for h_idx, h_word_chain in enumerate(other_chain.word_chains):
            best_match = 0.0
            
            for p_idx, p_word_chain in enumerate(self.word_chains):
                # Get combined geometry vectors for both words
                p_geo_vec = p_word_chain.get_geometry_vector()
                h_geo_vec = h_word_chain.get_geometry_vector()
                
                # Geometric Similarity (cosine similarity)
                dot_prod = np.dot(p_geo_vec, h_geo_vec)
                norm_p = np.linalg.norm(p_geo_vec)
                norm_h = np.linalg.norm(h_geo_vec)
                if norm_p > 0 and norm_h > 0:
                    geo_sim = dot_prod / (norm_p * norm_h)
                else:
                    geo_sim = 0.0
                
                # Quantum Interference (from combined quantum vectors)
                p_q_vec = p_word_chain.get_quantum_vector()
                h_q_vec = h_word_chain.get_quantum_vector()
                q_sim = np.real(np.vdot(p_q_vec, h_q_vec))
                
                # Word Match (exact string match)
                word_match = 1.0 if p_word_chain.word == h_word_chain.word else 0.0
                
                # Letter Overlap (morphological similarity)
                p_letters = set(p_word_chain.letters)
                h_letters = set(h_word_chain.letters)
                letter_overlap = len(p_letters & h_letters) / max(len(p_letters | h_letters), 1) if (p_letters | h_letters) else 0.0
                
                # Sequence Position
                position_sim = 1.0
                if use_sequence and len(self.word_chains) > 1 and len(other_chain.word_chains) > 1:
                    p_pos = p_idx / (len(self.word_chains) - 1)
                    h_pos = h_idx / (len(other_chain.word_chains) - 1)
                    position_sim = 1.0 - abs(p_pos - h_pos)
                
                # Combined Score
                # Weighted combination: geometry (primary), quantum, word match, letter overlap, position
                base_match_score = (
                    geo_sim * 0.4 +           # Geometric similarity (primary)
                    q_sim * 0.2 +              # Quantum interference
                    word_match * 0.15 +        # Exact word match
                    letter_overlap * 0.15 +     # Morphological similarity (NEW!)
                    position_sim * 0.1         # Position similarity
                )
                
                # Calculate semantic distance for unrelated pairs
                is_unrelated = (word_match == 0.0 and abs(geo_sim) < 0.3 and overall_overlap < 0.3 and letter_overlap < 0.2)
                
                if geo_sim < 0.0:
                    # Negative geometric = opposition - boost for contradiction
                    match_score = (geo_sim * 0.5 + q_sim * 0.2 + word_match * 0.1 + letter_overlap * 0.1 + position_sim * 0.05)
                elif is_unrelated:
                    # Unrelated topics - apply penalty
                    match_score = base_match_score * 0.5
                else:
                    # Normal case
                    match_score = base_match_score
                
                # Allow negative matches (Geometric Opposition)
                if abs(match_score) > abs(best_match):
                    best_match = match_score
            
            total_resonance += best_match
            matches += 1
        
        return total_resonance / max(1, matches)
    
    @property
    def chain(self):
        """Backward compatibility: return word_chains as 'chain'."""
        return self.word_chains
    
    @property
    def words(self):
        """Backward compatibility: return list of word strings."""
        return [wc.word for wc in self.word_chains]


# ============================================================================
# BACKWARD COMPATIBILITY: Keep old names for existing code
# ============================================================================

# Alias for backward compatibility
Omchain = SentenceChain
WordOmcube = WordChain  # Old code expects WordOmcube, but now it's WordChain

# Backward compatibility function removed - no longer needed
