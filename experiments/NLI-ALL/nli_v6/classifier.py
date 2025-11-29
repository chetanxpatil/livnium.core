"""
Livnium V6 Classifier: Corrected 3-Axis Manifold

Uses ONLY invariant signals:
- Opposition = Resonance × Sign(Divergence)
- Cold Attraction (invariant)
- Curvature (perfect invariant)

Removes noise from divergence magnitude.
"""

from typing import Dict
from dataclasses import dataclass
import numpy as np

from .encoder import ChainEncodedPair
from .layers import (
    Layer0Resonance, Layer1Curvature, Layer2Opposition,
    Layer3Attraction, Layer4Decision
)
from experiments.nli_simple.native_chain import SimpleLexicon


@dataclass
class ClassificationResult:
    """Classification result with all relevant information."""
    label: str  # 'entailment', 'contradiction', 'neutral'
    basin_index: int  # 0=cold(E), 1=far(C), 2=city(N)
    confidence: float
    scores: Dict[str, float]  # {'entailment': ..., 'contradiction': ..., 'neutral': ...}
    layer_states: Dict  # All layer outputs for debugging


class LivniumV6Classifier:
    """
    Livnium V6 Classifier - Corrected 3-Axis Manifold.
    
    Architecture:
    - Layer 0: Resonance (raw geometric signal)
    - Layer 1: Curvature (cold density and distance)
    - Layer 2: Opposition (NEW: resonance × sign(divergence))
    - Layer 3: Attraction (cold/far/city)
    - Layer 4: Decision (simplified - uses only invariant signals)
    """
    
    def __init__(self, encoded_pair: ChainEncodedPair, debug_mode: bool = False, golden_label_hint: str = None, force_incorrect: bool = False):
        """
        Initialize classifier.
        
        Args:
            encoded_pair: Chain-encoded premise-hypothesis pair
            debug_mode: If True, use golden label hint to verify decision logic
            golden_label_hint: Optional golden label for debugging
            force_incorrect: If True, force INCORRECT label (invert E↔C) to see geometry's true prediction
        """
        self.encoded_pair = encoded_pair
        self.debug_mode = debug_mode
        self.golden_label_hint = golden_label_hint
        self.force_incorrect = force_incorrect
        
        # Initialize layers
        self.layer0 = Layer0Resonance()
        self.layer1 = Layer1Curvature()
        self.layer2 = Layer2Opposition()
        self.layer3 = Layer3Attraction()
        self.layer4 = Layer4Decision(debug_mode=debug_mode, golden_label_hint=golden_label_hint, force_incorrect=force_incorrect)
    
    def classify(self) -> ClassificationResult:
        """
        Classify the premise-hypothesis pair.
        
        Returns:
            ClassificationResult with label, confidence, and all layer states
        """
        # Layer 0: Resonance + Divergence
        layer0_output = self.layer0.compute(self.encoded_pair)
        
        # Layer 1: Curvature
        layer1_output = self.layer1.compute(layer0_output, self.encoded_pair)
        
        # Layer 2: Opposition (NEW)
        layer2_output = self.layer2.compute(layer1_output)
        
        # Layer 3: Attractions
        layer3_output = self.layer3.compute(layer2_output, self.encoded_pair)
        
        # Layer 4: Decision (simplified)
        layer4_output = self.layer4.compute(layer3_output)
        
        # Build result
        result = ClassificationResult(
            label=layer4_output['label'],
            basin_index=layer4_output['basin_index'],
            confidence=layer4_output['confidence'],
            scores={
                'entailment': layer4_output.get('e_score', 0.33),
                'contradiction': layer4_output.get('c_score', 0.33),
                'neutral': layer4_output.get('n_score', 0.33),
            },
            layer_states=layer4_output
        )
        
        return result
    
    def apply_learning_feedback(self, gold_label: str, learning_strength: float = 1.0):
        """
        Apply learning feedback (simplified from v5 - no basins in v6).
        
        Args:
            gold_label: Golden label ('entailment', 'contradiction', 'neutral')
            learning_strength: Learning strength (0.0 to 1.0)
        """
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        label_idx = label_map.get(gold_label, 2)
        
        # Update word polarities (semantic memory)
        tokens = set(self.encoded_pair.tokens)
        lexicon = SimpleLexicon()
        
        # Update word polarities based on correct label
        for token in tokens:
            lexicon.update_word_polarity(token, label_idx, strength=0.1 * learning_strength)

