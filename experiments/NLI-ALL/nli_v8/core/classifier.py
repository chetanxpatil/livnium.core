"""
Livnium v8 Classifier: Clean Architecture with Geometry-First Philosophy

Integrates:
- Semantic warp alignment (DP, no hardcoded rules)
- Collision-based fracture detection (negation = alignment tension)
- Geometry-first classification (geometry is the teacher)
"""

from typing import Dict
from dataclasses import dataclass
import numpy as np

from .encoder import ChainEncodedPair
from .layers import (
    Layer0Resonance, LayerOpposition, Layer1Curvature,
    Layer2Basin, Layer3Valley, Layer4Decision
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


class LivniumV8Classifier:
    """
    Livnium V8 Classifier - Clean Architecture.
    
    Architecture:
    - Layer 0: Resonance (raw geometric signal)
    - Layer 1.5: Opposition + Fracture Detection (semantic warp + collision analysis)
    - Layer 1: Curvature (cold density and divergence force)
    - Layer 2: Basins (attraction wells for E and C)
    - Layer 3: Valley (natural neutral from balance)
    - Layer 4: Decision (final classification)
    """
    
    def __init__(self, encoded_pair: ChainEncodedPair, golden_label_hint: str = None):
        """
        Initialize classifier.
        
        Args:
            encoded_pair: Chain-encoded premise-hypothesis pair
            golden_label_hint: Optional golden label for debugging ('entailment', 'contradiction', 'neutral')
        """
        self.encoded_pair = encoded_pair
        self.golden_label_hint = golden_label_hint
        
        # Initialize layers
        self.layer0 = Layer0Resonance()
        self.layer_opposition = LayerOpposition(fracture_threshold=0.5)
        self.layer1 = Layer1Curvature()
        self.layer2 = Layer2Basin()
        self.layer3 = Layer3Valley()
        self.layer4 = Layer4Decision()
    
    def classify(self) -> ClassificationResult:
        """
        Classify using clean layer stack.
        
        Returns:
            ClassificationResult with final decision
        """
        # Layer 0: Resonance
        l0_output = self.layer0.compute(self.encoded_pair)
        
        # Layer 1.5: Opposition + Fracture Detection (with semantic warp)
        premise_vecs, hypothesis_vecs = self.encoded_pair.get_word_vectors()
        opposition_output = self.layer_opposition.compute(
            premise_vecs,
            hypothesis_vecs,
            resonance=l0_output.get('resonance', 0.0)
        )
        
        # Layer 1: Curvature
        l1_output = self.layer1.compute(l0_output, opposition_output)
        
        # Layer 2: Basins
        l2_output = self.layer2.compute(l1_output)
        
        # Layer 3: Valley
        l3_output = self.layer3.compute(l2_output)
        
        # Layer 4: Decision
        l4_output = self.layer4.compute(l3_output, golden_label_hint=self.golden_label_hint)
        
        # Build layer states
        layer_states = {
            **l4_output,
            'layer0': l0_output,
            'layer_opposition': opposition_output,
            'layer1': l1_output,
            'layer2': l2_output,
            'layer3': l3_output,
        }
        
        # Build result
        result = ClassificationResult(
            label=l4_output['label'],
            basin_index=l4_output['basin_index'],
            confidence=l4_output['confidence'],
            scores={
                'entailment': l4_output['e_score'],
                'contradiction': l4_output['c_score'],
                'neutral': l4_output['n_score']
            },
            layer_states=layer_states
        )
        
        return result
    
    def apply_learning_feedback(self, correct_label: str, learning_strength: float = 1.0):
        """
        Apply learning feedback to layers.
        
        Args:
            correct_label: Correct label ('entailment', 'contradiction', or 'neutral')
            learning_strength: Learning strength multiplier
        """
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        correct_idx = label_map.get(correct_label, 2)
        
        # Layer 2: Reinforce basin (E or C only, not N)
        if correct_idx in [0, 1]:
            self.layer2.reinforce(correct_idx, strength=learning_strength)
        
        # Update word polarities (semantic memory)
        tokens = set(self.encoded_pair.tokens)
        lexicon = SimpleLexicon()
        
        for token in tokens:
            lexicon.update_word_polarity(token, correct_idx, strength=0.1 * learning_strength)

