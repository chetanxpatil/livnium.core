"""
Native Chain Encoder: Replaces Transformer-based NLIEncoder

Uses Omchain (Matrix Product State) architecture instead of embeddings.
Pure native logic - no transformers, no embeddings, no external ML models.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .native_chain import Omchain, WordOmcube


@dataclass
class NativeEncodedPair:
    """
    Result of encoding a premise-hypothesis pair using Native Chain.
    
    Replaces EncodedPair from encoders.py.
    """
    premise_chain: Omchain
    hypothesis_chain: Omchain
    resonance_score: float = 0.0
    
    def get_resonance(self) -> float:
        """Get resonance score between premise and hypothesis."""
        # CHAINING DISABLED: No sequence position weighting
        return self.premise_chain.compare(self.hypothesis_chain, use_sequence=False)


class NativeChainNLIEncoder:
    """
    NLI Encoder using Native Chain (Omchain) architecture.
    
    Replaces NLIEncoder from encoders.py.
    Uses pure native logic - no transformers, no embeddings.
    """
    
    def __init__(self, lattice_size: int = 3, config: Optional[Any] = None):
        """
        Initialize Native Chain NLI Encoder.
        
        Args:
            lattice_size: Size of lattice for each word (default 3)
            config: Optional config (kept for compatibility, not used)
        """
        self.lattice_size = lattice_size
        self.config = config
    
    def encode_pair(self,
                   premise: str,
                   hypothesis: str,
                   premise_parse: Optional[str] = None,
                   hypothesis_parse: Optional[str] = None) -> NativeEncodedPair:
        """
        Encode premise-hypothesis pair using Native Chain.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            premise_parse: Optional parse tree (ignored - native chain doesn't use it)
            hypothesis_parse: Optional parse tree (ignored - native chain doesn't use it)
        
        Returns:
            NativeEncodedPair with premise and hypothesis chains
        """
        # Create chains
        premise_chain = Omchain(premise, lattice_size=self.lattice_size)
        hypothesis_chain = Omchain(hypothesis, lattice_size=self.lattice_size)
        
        # Compute resonance (CHAINING DISABLED: no sequence position)
        resonance_score = premise_chain.compare(hypothesis_chain, use_sequence=False)
        
        return NativeEncodedPair(
            premise_chain=premise_chain,
            hypothesis_chain=hypothesis_chain,
            resonance_score=resonance_score
        )

