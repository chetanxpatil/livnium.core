"""
Simple Encoder: Encodes premise-hypothesis pairs as vectors.

No Livnium dependencies - pure vector operations.
"""

from typing import Optional
from dataclasses import dataclass

from .native_chain import SentenceVector


@dataclass
class SimpleEncodedPair:
    """Encoded premise-hypothesis pair."""
    premise: SentenceVector
    hypothesis: SentenceVector
    
    def get_resonance(self) -> float:
        """Get similarity score between premise and hypothesis using CHAIN matching."""
        # Use sequential chain matching (position matters) - this is the key!
        return self.premise.compare(self.hypothesis, use_sequence=True)


class SimpleEncoder:
    """Simple encoder that converts text to sentence vectors."""
    
    def __init__(self, vector_size: int = 27):
        """
        Initialize simple encoder.
        
        Args:
            vector_size: Size of word vectors (default 27)
        """
        self.vector_size = vector_size
    
    def encode_pair(self, premise: str, hypothesis: str) -> SimpleEncodedPair:
        """
        Encode premise-hypothesis pair.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            
        Returns:
            SimpleEncodedPair
        """
        premise_vec = SentenceVector(premise, self.vector_size)
        hypothesis_vec = SentenceVector(hypothesis, self.vector_size)
        
        return SimpleEncodedPair(
            premise=premise_vec,
            hypothesis=hypothesis_vec
        )

