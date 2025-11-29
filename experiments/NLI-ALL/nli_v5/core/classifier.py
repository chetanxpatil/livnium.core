"""
Livnium V5 Classifier: Clean 5-Layer Architecture

Simplified from v4's 7 layers while keeping the core ideas.
Each layer builds on the one below - gravity shapes everything.
"""

from typing import Dict, Set
from dataclasses import dataclass
import numpy as np

from .encoder import ChainEncodedPair
from .layers import (
    Layer0Resonance, LayerOpposition, Layer1Curvature, Layer2Basin,
    Layer3Valley, Layer4Decision
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


class LivniumV5Classifier:
    """
    Livnium V5 Classifier - Clean 5-layer architecture.
    
    Architecture:
    - Layer 0: Resonance (raw geometric signal)
    - Layer 1: Curvature (cold density and distance)
    - Layer 2: Basins (attraction wells for E and C)
    - Layer 3: Valley (natural neutral from balance)
    - Layer 4: Decision (final classification)
    """
    
    def __init__(self, encoded_pair: ChainEncodedPair, debug_mode: bool = False, golden_label_hint: str = None, reverse_physics_mode: bool = False):
        """
        Initialize classifier.
        
        Args:
            encoded_pair: Chain-encoded premise-hypothesis pair
            debug_mode: If True, use golden label hint to verify decision logic
            golden_label_hint: Optional golden label for debugging ('entailment', 'contradiction', 'neutral')
            reverse_physics_mode: If True, disable force setting (for reverse physics experiments)
        """
        self.encoded_pair = encoded_pair
        self.debug_mode = debug_mode
        self.golden_label_hint = golden_label_hint
        self.reverse_physics_mode = reverse_physics_mode
        
        # Initialize layers
        self.layer0 = Layer0Resonance()
        self.layer_opposition = LayerOpposition(opposition_strength=1.0)  # NEW: Opposition field (Layer 1.5)
        self.layer1 = Layer1Curvature()
        self.layer2 = Layer2Basin()
        self.layer3 = Layer3Valley()
        self.layer4 = Layer4Decision(debug_mode=debug_mode, golden_label_hint=golden_label_hint, reverse_physics_mode=reverse_physics_mode)
    
    def classify(self) -> ClassificationResult:
        """
        Classify using 5-layer stack.
        
        Returns:
            ClassificationResult with final decision
        """
        # Layer 0: Pure Resonance (ONLY similarity signals, no divergence)
        l0_output = self.layer0.compute(self.encoded_pair)
        
        # Layer 1: Curvature (computes without divergence - will be injected next)
        l1_output = self.layer1.compute(l0_output, encoded_pair=self.encoded_pair)
        
        # Layer 1.5: Opposition Field (NEW - fixes Inward-Outward Axis)
        # Compute opposition field and corrected divergence
        premise_vecs, hypothesis_vecs = self.encoded_pair.get_word_vectors()
        opposition_output = self.layer_opposition.compute(
            premise_vecs, 
            hypothesis_vecs,
            resonance=l0_output.get('resonance', 0.0),
            curvature=l1_output.get('curvature', 0.0)
        )
        
        # CRITICAL: Inject opposition-corrected divergence into layer1_output
        # This divergence_final flows through all subsequent layers
        divergence_final = opposition_output['divergence_final']
        l1_output['divergence'] = divergence_final
        l1_output['convergence'] = -divergence_final  # Update convergence based on final divergence
        l1_output['opposition_raw'] = opposition_output['opposition_raw']
        l1_output['opposition_norm'] = opposition_output['opposition_norm']
        
        # Update cold_density and divergence_force based on final divergence
        # Convergence (E): Cold density from negative divergence
        convergence = -divergence_final
        l1_output['cold_density'] = max(0.0, convergence) + max(0.0, l0_output.get('resonance', 0.0) * 0.5)
        
        # Divergence (C): Real repulsive force from positive divergence
        divergence_force = max(0.0, divergence_final)
        if l0_output.get('resonance', 0.0) < 0:
            divergence_force += abs(l0_output.get('resonance', 0.0)) * 0.5
        l1_output['divergence_force'] = divergence_force
        l1_output['distance'] = divergence_force  # For backward compatibility
        
        # Layer 2: Basins
        l2_output = self.layer2.compute(l1_output)
        
        # Layer 3: Valley (City)
        l3_output = self.layer3.compute(l2_output)
        
        # Layer 4: Decision
        l4_output = self.layer4.compute(l3_output)
        
        # Build layer states (include all intermediate outputs)
        layer_states = {
            **l4_output,
            'layer_opposition': opposition_output,  # Include fracture detection info
            'layer0': l0_output,
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
        
        # Update word polarities based on correct label
        # SimpleLexicon.update_word_polarity takes: word, label_idx, strength
        for token in tokens:
            lexicon.update_word_polarity(token, correct_idx, strength=0.1 * learning_strength)
    
    def get_all_states(self) -> Dict:
        """Get states from all layers (for debugging)."""
        return {
            'layer0': self.layer0.__dict__ if hasattr(self.layer0, '__dict__') else {},
            'layer1': self.layer1.__dict__ if hasattr(self.layer1, '__dict__') else {},
            'layer2': {
                'cold_depth': Layer2Basin._shared_cold_depth,
                'far_depth': Layer2Basin._shared_far_depth
            },
            'layer3': self.layer3.__dict__ if hasattr(self.layer3, '__dict__') else {},
            'layer4': self.layer4.__dict__ if hasattr(self.layer4, '__dict__') else {}
        }

