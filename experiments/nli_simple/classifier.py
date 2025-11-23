"""
Simple Classifier: Minimal wrapper around SimpleNLIClassifier.

No basins, no collapse, no quantum amplitudes - just scores and argmax.
"""

from typing import Dict, Any
from dataclasses import dataclass

from .encoder import SimpleEncodedPair
from .inference_detectors import SimpleNLIClassifier
from .native_chain import SimpleLexicon


@dataclass
class SimpleClassificationResult:
    """Result of classification."""
    label: str  # 'entailment', 'contradiction', or 'neutral'
    confidence: float
    scores: Dict[str, float]


class SimpleNLIClassifierWrapper:
    """
    Minimal classifier wrapper (replaces OmcubeNLIClassifier).
    
    No basins, no collapse, no quantum physics - just:
    - Encode text → vectors
    - Compute similarity scores
    - Apply learned word polarity
    - Return argmax
    """
    
    def __init__(self, encoded_pair: SimpleEncodedPair):
        """
        Initialize simple classifier.
        
        Args:
            encoded_pair: Encoded premise-hypothesis pair
        """
        self.encoded_pair = encoded_pair
        self.classifier = SimpleNLIClassifier(encoded_pair)
        self.lexicon = SimpleLexicon()
    
    def classify(self, use_sequence: bool = True) -> SimpleClassificationResult:
        """
        Classify the encoded pair using CHAIN STRUCTURE.
        
        Args:
            use_sequence: If True, use sequential chain matching (position matters).
                         Defaults to True - chain structure is ALWAYS used unless explicitly disabled.
                         This is CRITICAL - ensures chain structure drives the final decision.
        
        Returns:
            SimpleClassificationResult with label and scores
        """
        """
        Classify the encoded pair using CHAIN STRUCTURE.
        
        Args:
            use_sequence: If True, use sequential chain matching (position matters).
                         This is CRITICAL - ensures chain structure drives the final decision.
        
        Returns:
            SimpleClassificationResult with label and scores
        """
        # FORCE chain structure in classifier
        # This ensures positional encoding, aligned matching, sliding windows all feed the final decision
        result = self.classifier.classify(use_sequence=use_sequence)
        
        return SimpleClassificationResult(
            label=result['label'],
            confidence=result['confidence'],
            scores=result['scores']
        )
    
    def apply_learning_feedback(self, 
                                correct_label: str,
                                learning_strength: float = 1.0):
        """
        Update learned word polarities based on correct label.
        
        Args:
            correct_label: 'entailment', 'contradiction', or 'neutral'
            learning_strength: Learning rate multiplier
        """
        if learning_strength <= 0.0:
            return
        
        # Map label to index
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        label_idx = label_map.get(correct_label, 2)
        
        # Update word polarities for all words in both sentences
        all_words = set(self.encoded_pair.premise.tokens + 
                       self.encoded_pair.hypothesis.tokens)
        
        for word in all_words:
            # Update polarity towards correct class
            self.lexicon.update_word_polarity(word, label_idx, strength=0.15 * learning_strength)

