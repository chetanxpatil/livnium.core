"""
Chain Encoder: Same as v5 (unchanged)
"""

import numpy as np
from typing import List
from dataclasses import dataclass
import sys
import os

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.nli_simple.native_chain import (
    SimpleLexicon, WordVector, SentenceVector
)


@dataclass
class ChainEncodedPair:
    """Encoded premise-hypothesis pair with chain structure."""
    premise: SentenceVector
    hypothesis: SentenceVector
    
    def get_resonance(self) -> float:
        """Get chain-based similarity (position matters)."""
        return self.premise.compare(self.hypothesis, use_sequence=True)
    
    def get_word_vectors(self) -> tuple:
        """Get word vectors for both sentences."""
        return (
            self.premise.get_word_vectors(),
            self.hypothesis.get_word_vectors()
        )
    
    @property
    def tokens(self) -> List[str]:
        """Get tokens from both sentences."""
        return self.premise.tokens + self.hypothesis.tokens


class ChainEncoder:
    """
    Chain encoder with positional encoding and sequential matching.
    
    This is the clean frontend that captures order-dependent patterns.
    """
    
    def __init__(self, vector_size: int = 27):
        """
        Initialize encoder.
        
        Args:
            vector_size: Size of word vectors (default 27 for chain structure)
        """
        self.vector_size = vector_size
        self.lexicon = SimpleLexicon()
    
    def encode_pair(self, premise: str, hypothesis: str) -> ChainEncodedPair:
        """
        Encode premise-hypothesis pair into chain structure.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            
        Returns:
            ChainEncodedPair with encoded sentences
        """
        premise_vec = SentenceVector(premise, self.vector_size)
        hypothesis_vec = SentenceVector(hypothesis, self.vector_size)
        
        return ChainEncodedPair(premise=premise_vec, hypothesis=hypothesis_vec)

