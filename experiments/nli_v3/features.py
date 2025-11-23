"""
Feature Extraction: Geometric Features from Chain Structure

Extracts features from chain-encoded pairs for classification.
Uses variance-based geometry + learned word polarity.
"""

import numpy as np
from typing import Dict, Tuple

from .chain_encoder import ChainEncodedPair
from experiments.nli_simple.native_chain import SimpleLexicon
from experiments.nli_simple.inference_detectors import SimpleLogic


class FeatureExtractor:
    """
    Extracts geometric features from chain-encoded pairs.
    
    Features:
    - Chain resonance (sequential matching)
    - Variance of word-pair similarities
    - Learned word polarity signals
    - Lexical overlap
    """
    
    def __init__(self):
        self.lexicon = SimpleLexicon()
    
    def extract(self, encoded_pair: ChainEncodedPair) -> Dict[str, float]:
        """
        Extract features from encoded pair.
        
        Args:
            encoded_pair: Chain-encoded premise-hypothesis pair
            
        Returns:
            Dict of feature names to values
        """
        # 1. Chain resonance (primary signal - uses full chain structure)
        resonance = encoded_pair.get_resonance()
        
        # 2. Word-pair similarities (for variance calculation)
        premise_vecs, hypothesis_vecs = encoded_pair.get_word_vectors()
        
        sims = []
        min_len = min(len(premise_vecs), len(hypothesis_vecs))
        
        for i in range(min_len):
            p_vec = premise_vecs[i]
            h_vec = hypothesis_vecs[i]
            sim = self._cosine(p_vec, h_vec)
            sims.append(sim)
        
        mean_sim = float(np.mean(sims)) if sims else 0.0
        var_sim = float(np.var(sims)) if sims else 0.0
        
        # 3. Learned word polarity
        p_tokens = encoded_pair.premise.tokens
        h_tokens = encoded_pair.hypothesis.tokens
        
        has_contradiction, contradiction_strength = SimpleLogic.detect_learned_polarity(
            h_tokens, target_class=1
        )
        
        # 4. Lexical overlap
        overlap = SimpleLogic.compute_overlap(p_tokens, h_tokens)
        
        return {
            'resonance': resonance,
            'mean_similarity': mean_sim,
            'variance': var_sim,
            'has_contradiction_word': float(has_contradiction),
            'contradiction_strength': contradiction_strength,
            'lexical_overlap': overlap
        }
    
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

