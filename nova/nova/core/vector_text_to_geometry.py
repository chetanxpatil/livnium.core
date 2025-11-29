"""
Vector-Based Text-to-Geometry Interface

Cell-less Livnium: Pure vector geometry without 3D lattice.

Architecture:
1. Token → MD5 hash → 128-bit vector (normalized)
2. Sentence = sum of token vectors (normalized)
3. Collapse = nonlinear transform (tanh or power)
4. OM/LO = collapsed premise/hypothesis vectors
5. Physics laws work exactly the same (alignment, divergence, fracture)

Benefits:
- Zero token collisions (128-bit space)
- Fast collapse (vector operations)
- True OM/LO variation
- Scalable (256, 512, 1024 dimensions)
"""

import numpy as np
import hashlib
import re
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class VectorTextToGeometry:
    """
    Vector-based text-to-geometry interface.
    
    No lattice, no cells, just pure vector geometry.
    """
    
    def __init__(self, 
                 vector_dim: int = 128,
                 impulse_scale: float = 0.1,
                 collapse_type: str = 'tanh',
                 break_symmetry_for_snli: bool = False):
        """
        Initialize vector-based geometry interface.
        
        Args:
            vector_dim: Dimension of vector space (128, 256, 512, 1024)
            impulse_scale: Scale factor for token vectors (default: 0.1)
            collapse_type: Collapse function ('tanh', 'power3', 'relu')
            break_symmetry_for_snli: If True, add angular tilt for SNLI
        """
        self.vector_dim = vector_dim
        self.impulse_scale = impulse_scale
        self.collapse_type = collapse_type
        self.break_symmetry_for_snli = break_symmetry_for_snli
        
        # Token cache for reproducibility
        self.token_cache: Dict[str, np.ndarray] = {}
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        pattern = r"(\w+|\s+|[^\w\s])"
        return [t for t in re.split(pattern, text) if t.strip()]
    
    def token_to_vector(self, token: str) -> np.ndarray:
        """
        Convert token to high-dimensional vector using MD5 hash.
        
        Uses MD5 hash to create a deterministic 128-bit vector.
        Zero collisions (practically infinite space).
        
        Returns:
            Normalized vector of dimension vector_dim
        """
        if token in self.token_cache:
            return self.token_cache[token]
        
        # Hash token to get deterministic seed
        h = hashlib.md5(token.encode('utf-8')).hexdigest()
        
        # Use hash to seed random number generator
        seed = int(h[:8], 16)  # Use first 8 hex chars as seed
        
        # Generate vector deterministically
        np.random.seed(seed)
        vector = np.random.normal(0, 1, self.vector_dim)
        
        # Normalize to unit vector
        vector = vector / (np.linalg.norm(vector) + 1e-10)
        
        # Scale by impulse_scale
        vector = vector * self.impulse_scale
        
        # Cache for reuse
        self.token_cache[token] = vector
        
        return vector
    
    def sentence_to_vector(self, sentence: str) -> np.ndarray:
        """
        Convert sentence to vector by summing token vectors.
        
        This is conceptually identical to impulse accumulation,
        just without the grid.
        
        Returns:
            Normalized sentence vector
        """
        tokens = self.tokenize(sentence)
        
        # Sum all token vectors
        sentence_vector = np.zeros(self.vector_dim)
        for token in tokens:
            if token.strip() and token.isalnum():  # Only words
                token_vec = self.token_to_vector(token.lower())
                sentence_vector += token_vec
        
        # Normalize
        sentence_vector = sentence_vector / (np.linalg.norm(sentence_vector) + 1e-10)
        
        return sentence_vector
    
    def collapse_vector(self, vector: np.ndarray, collapse_steps: int = 12) -> np.ndarray:
        """
        Apply nonlinear collapse transform to vector.
        
        This creates basins, stability, and nonlinearity without a 3D grid.
        
        Args:
            vector: Input vector
            steps: Number of collapse steps
            
        Returns:
            Collapsed vector
        """
        v = vector.copy()
        
        for step in range(collapse_steps):
            # Apply nonlinear transform
            if self.collapse_type == 'tanh':
                # Tanh creates smooth basins
                v = np.tanh(v * (1.0 + step * 0.1))
            elif self.collapse_type == 'power3':
                # Element-wise cube creates sharper basins
                v = np.sign(v) * np.power(np.abs(v), 1.0 + step * 0.05)
            elif self.collapse_type == 'relu':
                # ReLU with gentle decay
                v = np.maximum(0, v) * (0.98 + 0.02 * step / steps)
            else:
                # Default: tanh
                v = np.tanh(v * (1.0 + step * 0.1))
            
            # Add thermal jitter (entropy) - prevents immediate basin-locking
            jitter_scale = 0.01 * (1.0 - step / max(collapse_steps, 1))
            if jitter_scale > 0:
                jitter = np.random.normal(0, jitter_scale, self.vector_dim)
                v = v + jitter
            
            # Normalize to maintain unit vector
            v = v / (np.linalg.norm(v) + 1e-10)
        
        return v
    
    def get_meaning_signature(self, sentence: str, collapse_steps: int = 12) -> np.ndarray:
        """
        Get meaning signature (collapsed vector) for a sentence.
        
        This is the "fingerprint" of the sentence's meaning in vector space.
        
        Returns:
            Collapsed sentence vector
        """
        # Convert sentence to vector
        sentence_vec = self.sentence_to_vector(sentence)
        
        # Apply collapse
        collapsed_vec = self.collapse_vector(sentence_vec, collapse_steps=collapse_steps)
        
        return collapsed_vec
    
    def get_signature_with_divergence(self, premise: str, hypothesis: str, collapse_steps: int = 12) -> np.ndarray:
        """
        Get signature for premise+hypothesis pair WITH divergence primitive.
        
        OM = direction of meaning of the premise
        LO = direction of meaning of the hypothesis
        
        This implements the Divergence Law: divergence = 0.38 - alignment
        
        Returns:
            Extended signature array: [premise_vec, hypothesis_vec, alignment, divergence, fracture]
        """
        # 1. Premise signature (OM)
        premise_sig = self.get_meaning_signature(premise, collapse_steps=collapse_steps)
        OM = premise_sig / (np.linalg.norm(premise_sig) + 1e-8)
        
        # 2. Hypothesis signature (LO) - with directional bias to break symmetry
        if self.break_symmetry_for_snli:
            # Apply small angular tilt to LO's vector to break symmetry
            hyp_vec = self.sentence_to_vector(hypothesis)
            
            # Add small random noise (angular tilt)
            epsilon = 0.05
            noise = np.random.normal(0, epsilon, self.vector_dim)
            hyp_vec = hyp_vec + noise
            hyp_vec = hyp_vec / (np.linalg.norm(hyp_vec) + 1e-10)
            
            # Collapse rotated vector
            hyp_sig = self.collapse_vector(hyp_vec, collapse_steps=collapse_steps)
        else:
            hyp_sig = self.get_meaning_signature(hypothesis, collapse_steps=collapse_steps)
        
        LO = hyp_sig / (np.linalg.norm(hyp_sig) + 1e-8)
        
        # 3. Alignment
        cos_theta = float(np.dot(OM, LO))
        alignment = (cos_theta + 1.0) / 2.0
        
        # 4. Divergence law
        divergence = 0.38 - alignment
        
        # 5. Fracture (change in alignment)
        fracture = np.linalg.norm(OM - LO)
        
        # Combine: [premise_vec, hypothesis_vec, alignment, divergence, fracture]
        extended_sig = np.concatenate([
            premise_sig,
            hyp_sig,
            np.array([alignment, divergence, fracture])
        ])
        
        return extended_sig
    
    def reset(self):
        """Reset state (clear cache if needed)."""
        # Token cache can stay (deterministic anyway)
        pass


# Compatibility wrapper for existing code
class VectorGeometricTokenLearner:
    """
    Compatibility wrapper for vector-based system.
    
    Provides same interface as GeometricTokenLearner but uses vectors.
    """
    
    def __init__(self, vector_dim: int = 128, num_clusters: int = 100):
        self.vector_dim = vector_dim
        self.num_clusters = num_clusters
        self.kmeans = None
        self.cluster_tokens = {}
        self.trained = False
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        pattern = r"(\w+|\s+|[^\w\s])"
        return [t for t in re.split(pattern, text) if t.strip()]
    
    def get_cluster_id(self, signature: np.ndarray) -> int:
        """Get cluster ID from signature (for compatibility)."""
        if not self.trained or self.kmeans is None:
            return 0
        if signature.ndim == 1:
            signature = signature.reshape(1, -1)
        return int(self.kmeans.predict(signature)[0])
    
    def learn_clusters(self, signatures: np.ndarray, tokens_list: List[List[str]]):
        """Learn clusters from signatures (for compatibility)."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        
        if len(signatures) < self.num_clusters:
            self.num_clusters = len(signatures)
        
        # Normalize signatures
        signatures_normalized = normalize(signatures, norm='l2', axis=1)
        
        # Fit KMeans
        self.kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=42,
            n_init=10
        )
        cluster_ids = self.kmeans.fit_predict(signatures_normalized)
        
        # Build cluster token mappings
        from collections import defaultdict, Counter
        self.cluster_tokens = defaultdict(Counter)
        for idx, cluster_id in enumerate(cluster_ids):
            self.cluster_tokens[int(cluster_id)].update(tokens_list[idx])
        
        self.trained = True

