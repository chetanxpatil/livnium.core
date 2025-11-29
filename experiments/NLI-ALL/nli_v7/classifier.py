"""
Livnium V7 Classifier: Geometry Shaping

Key principle: Train ONLY geometry (Layers 0-3), never Layer 4.

Layer 4 is passive - it only observes.
Layers 0-3 shape the manifold through physics reinforcement.
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


class LivniumV7Classifier:
    """
    Livnium V7 Classifier - Geometry Shaping.
    
    Architecture:
    - Layer 0: Resonance (WITH physics reinforcement)
    - Layer 1: Curvature (WITH physics reinforcement)
    - Layer 2: Opposition (WITH physics reinforcement)
    - Layer 3: Attraction (WITH physics reinforcement)
    - Layer 4: Decision (PASSIVE - no learning)
    """
    
    def __init__(self, encoded_pair: ChainEncodedPair):
        """
        Initialize classifier.
        
        Args:
            encoded_pair: Chain-encoded premise-hypothesis pair
        """
        self.encoded_pair = encoded_pair
        
        # Initialize layers (geometry layers learn, Layer 4 is passive)
        self.layer0 = Layer0Resonance()
        self.layer1 = Layer1Curvature()
        self.layer2 = Layer2Opposition()
        self.layer3 = Layer3Attraction()
        self.layer4 = Layer4Decision()  # Passive - no learning
    
    def classify(self) -> ClassificationResult:
        """
        Classify the premise-hypothesis pair.
        
        Returns:
            ClassificationResult with label, confidence, and all layer states
        """
        # Layer 0: Resonance + Divergence (WITH learning)
        layer0_output = self.layer0.compute(self.encoded_pair)
        
        # Layer 1: Curvature (WITH learning)
        layer1_output = self.layer1.compute(layer0_output, self.encoded_pair)
        
        # Layer 2: Opposition (WITH learning)
        layer2_output = self.layer2.compute(layer1_output)
        
        # Layer 3: Attractions (WITH learning)
        layer3_output = self.layer3.compute(layer2_output, self.encoded_pair)
        
        # Layer 4: Decision (PASSIVE - no learning)
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
    
    def reinforce_geometry(self, gold_label: str, learning_strength: float = 0.01):
        """
        Physics reinforcement: Shape the geometry based on correct examples.
        
        This is NOT gradient descent - it's energy tuning.
        Small, continuous updates to the field per sample.
        
        Args:
            gold_label: Golden label ('entailment', 'contradiction', 'neutral')
            learning_strength: Reinforcement strength (default 0.01 - small, continuous)
        """
        # Reinforce each geometry layer
        self.layer0.reinforce(gold_label, strength=learning_strength)
        self.layer1.reinforce(gold_label, strength=learning_strength)
        self.layer2.reinforce(gold_label, strength=learning_strength)
        self.layer3.reinforce(gold_label, strength=learning_strength)
        
        # Layer 4 does NOT learn (passive observer)
        
        # Also update word polarities (semantic memory)
        tokens = set(self.encoded_pair.tokens)
        lexicon = SimpleLexicon()
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        label_idx = label_map.get(gold_label, 2)
        
        for token in tokens:
            lexicon.update_word_polarity(token, label_idx, strength=0.1 * learning_strength)
    
    def get_geometry_state(self) -> Dict:
        """Get current geometry state (for debugging/monitoring)."""
        return {
            'layer0': {
                'equilibrium_threshold': self.layer0.equilibrium_threshold,
                'resonance_scale': self.layer0.resonance_scale,
            },
            'layer1': {
                'cold_density_scale': self.layer1.cold_density_scale,
                'distance_scale': self.layer1.distance_scale,
            },
            'layer2': {
                'opposition_scale': self.layer2.opposition_scale,
            },
            'layer3': {
                'cold_attraction_scale': self.layer3.cold_attraction_scale,
                'far_attraction_scale': self.layer3.far_attraction_scale,
            },
        }

