"""
Layer 6: Semantic Memory

Stores polarity shaped by lower layers.
Word polarities learn from geometry, not hardcoded rules.
"""

from typing import Dict, Set
import numpy as np

from experiments.nli_simple.native_chain import SimpleLexicon


class Layer6SemanticMemory:
    """
    Layer 6: Semantic memory (polarity shaped by lower layers).
    
    Builds on Layer 5 stability.
    Stores word polarities that are shaped by geometry.
    """
    
    def __init__(self):
        """Initialize Layer 6 (semantic memory)."""
        self.lexicon = SimpleLexicon()
    
    def compute(self, layer5_output: Dict[str, float], tokens: Set[str]) -> Dict[str, float]:
        """
        Compute semantic features from memory.
        
        Args:
            layer5_output: Output from Layer 5 (contains forces, scores - NO label)
            tokens: Set of word tokens in the pair
            
        Returns:
            Dict with semantic features
        """
        # Layer 6 doesn't need label - it just computes word polarities
        # Label will be decided by Layer 7
        
        # Get word polarities (learned from geometry)
        polarity_vecs = [self.lexicon.get_word_polarity(token) for token in tokens]
        if polarity_vecs:
            avg_polarity = np.mean(polarity_vecs, axis=0)
            polarity_E = float(avg_polarity[0])
            polarity_C = float(avg_polarity[1])
            polarity_N = float(avg_polarity[2])
        else:
            polarity_E = polarity_C = polarity_N = 0.33
        
        return {
            **layer5_output,  # Pass through all
            'polarity_E': polarity_E,
            'polarity_C': polarity_C,
            'polarity_N': polarity_N,
            'avg_polarity': [float(polarity_E), float(polarity_C), float(polarity_N)]
        }
    
    def update(self, tokens: Set[str], correct_label: str, strength: float = 1.0):
        """
        Update semantic memory (word polarities).
        
        Args:
            tokens: Set of word tokens
            correct_label: Correct label ('entailment', 'contradiction', or 'neutral')
            strength: Learning strength
        """
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        correct_idx = label_map.get(correct_label, 2)
        
        for token in tokens:
            self.lexicon.update_word_polarity(
                token,
                correct_idx,
                strength=0.15 * strength
            )
    
    def update_competitive(self, tokens: Set[str], basin_index: int, 
                          basin_force: float, strength: float = 1.0):
        """
        STEP 2: Competitive word polarity updates.
        
        Basins compete for words - winning basin pulls harder.
        Other basins decay.
        
        Args:
            tokens: Set of word tokens
            basin_index: Which basin won (0, 1, or 2)
            basin_force: How strong the basin's pull was
            strength: Learning strength multiplier
        """
        polarity_update_strength = 0.15 * strength * basin_force  # Scales with force
        polarity_decay = 0.05 * strength  # Other basins decay
        
        for token in tokens:
            # Update winning basin
            self.lexicon.update_word_polarity(
                token,
                basin_index,
                strength=polarity_update_strength
            )
            
            # Decay other basins (competition)
            for other_basin in [0, 1, 2]:
                if other_basin != basin_index:
                    # Get current polarity
                    current_polarity = self.lexicon.get_word_polarity(token)
                    # Decay this dimension
                    decayed_value = current_polarity[other_basin] * (1.0 - polarity_decay)
                    # Update (this is a bit hacky - SimpleLexicon doesn't have direct set)
                    # For now, we'll use negative update to decay
                    self.lexicon.update_word_polarity(
                        token,
                        other_basin,
                        strength=-polarity_decay * 0.5  # Small negative update
                    )
    
    def get_state(self) -> Dict:
        """Get current state."""
        return {
            'lexicon_size': len(self.lexicon.polarity_store)
        }

