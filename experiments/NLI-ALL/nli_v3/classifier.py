"""
Livnium NLI Classifier: Complete System

Combines:
- Chain encoder (frontend)
- Feature extraction
- Basin system (energy wells)
- Quantum collapse (decision making)
- Moksha router (convergence)
- Geometric feedback (learning)
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass

from .chain_encoder import ChainEncodedPair
from .features import FeatureExtractor
from .basins import BasinSystem
from .collapse import QuantumCollapse
from .moksha import MokshaRouter
from .peak_clarity import PeakClarity
from experiments.nli_simple.native_chain import SimpleLexicon


@dataclass
class ClassificationResult:
    """Result of classification."""
    label: str  # 'entailment', 'contradiction', or 'neutral'
    class_idx: int  # 0, 1, or 2
    confidence: float
    scores: Dict[str, float]
    is_moksha: bool  # Has convergence been reached?


class LivniumNLIClassifier:
    """
    Complete Livnium NLI Classifier v3.
    
    Architecture:
    1. Chain encoder (positional encoding, sequential matching)
    2. Feature extraction (resonance, variance, word polarity)
    3. Basin system (energy wells for each class) - SHARED across all examples
    4. Quantum collapse (3-way decision)
    5. Moksha router (convergence detection)
    6. Geometric feedback (word polarity learning)
    """
    
    # Shared components (learn from all training examples)
    _shared_basin_system = None
    _shared_moksha_router = None
    _shared_peak_clarity = None
    
    def __init__(self, encoded_pair: ChainEncodedPair):
        """
        Initialize Livnium classifier.
        
        Args:
            encoded_pair: Chain-encoded premise-hypothesis pair
        """
        self.encoded_pair = encoded_pair
        
        # Components (shared across all instances)
        self.feature_extractor = FeatureExtractor()
        
        # Shared basin system (CRITICAL: must be shared to learn from all examples)
        if LivniumNLIClassifier._shared_basin_system is None:
            LivniumNLIClassifier._shared_basin_system = BasinSystem()
        self.basin_system = LivniumNLIClassifier._shared_basin_system
        
        # Shared Moksha router
        if LivniumNLIClassifier._shared_moksha_router is None:
            LivniumNLIClassifier._shared_moksha_router = MokshaRouter(convergence_threshold=0.7)
        self.moksha_router = LivniumNLIClassifier._shared_moksha_router
        
        # Shared peak clarity system (two-peak geometry: E and C are peaks, N is valley)
        if LivniumNLIClassifier._shared_peak_clarity is None:
            LivniumNLIClassifier._shared_peak_clarity = PeakClarity(band_width=0.3)
        self.peak_clarity = LivniumNLIClassifier._shared_peak_clarity
        
        # Per-instance components
        self.collapse = QuantumCollapse(temperature=3.0)
        self.lexicon = SimpleLexicon()
    
    @classmethod
    def reset_shared_state(cls):
        """Reset shared state (for new training run)."""
        cls._shared_basin_system = None
        cls._shared_moksha_router = None
        cls._shared_peak_clarity = None
    
    @classmethod
    def save_peak_clarity(cls, filepath: str):
        """Save peak clarity system."""
        if cls._shared_peak_clarity:
            import pickle
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'e_peak': cls._shared_peak_clarity.e_peak,
                    'c_peak': cls._shared_peak_clarity.c_peak,
                    'band_width': cls._shared_peak_clarity.band_width
                }, f)
    
    @classmethod
    def load_peak_clarity(cls, filepath: str) -> bool:
        """Load peak clarity system."""
        if cls._shared_peak_clarity is None:
            cls._shared_peak_clarity = PeakClarity()
        import pickle
        import os
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            cls._shared_peak_clarity.e_peak = data['e_peak']
            cls._shared_peak_clarity.c_peak = data['c_peak']
            cls._shared_peak_clarity.band_width = data.get('band_width', 0.3)
            return True
        except Exception:
            return False
    
    def classify(self, use_learned_head: bool = True) -> ClassificationResult:
        """
        Classify NLI pair using full Livnium system.
        
        Args:
            use_learned_head: If True, use learned decision head (recommended).
                             If False, use hand-constructed logic (for comparison).
        
        Returns:
            ClassificationResult with label, confidence, and Moksha status
        """
        # 1. Extract features
        features = self.feature_extractor.extract(self.encoded_pair)
        resonance = features['resonance']
        
        if use_learned_head:
            # Use two-peak geometry system (E and C are peaks, N is valley)
            # Works WITH geometry, not against it
            
            # Classify using peak clarity
            variance = features['variance']
            label, confidence, probs = self.peak_clarity.classify(resonance, variance)
            
            # Map label to class index
            label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
            final_class_idx = label_map[label]
            final_label = label
            final_confidence = confidence
            
            # Get basin weights (E and C only - no Neutral basin)
            basin_weights = self.basin_system.get_weights()
            
            # Apply basin weights (geometry reinforces peaks)
            combined_probs = {
                'entailment': probs['entailment'] * (1.0 + basin_weights[0] * 0.15),
                'contradiction': probs['contradiction'] * (1.0 + basin_weights[1] * 0.15),
                'neutral': probs['neutral']  # Neutral has no basin weight
            }
            
            # Renormalize
            total = sum(combined_probs.values())
            if total > 0:
                for key in combined_probs:
                    combined_probs[key] /= total
            
            # Use combined probabilities
            final_probs = np.array([
                combined_probs['entailment'],
                combined_probs['contradiction'],
                combined_probs['neutral']
            ])
            final_class_idx = int(np.argmax(final_probs))
            final_label = ['entailment', 'contradiction', 'neutral'][final_class_idx]
            final_confidence = float(final_probs[final_class_idx])
            
        else:
            # Fallback: Use hand-constructed logic (for comparison or before training)
            variance = features['variance']
            
            # Check for contradiction word
            if features['has_contradiction_word'] and features['contradiction_strength'] > 0.6:
                entailment_score = 0.0
                contradiction_score = 0.5 + (0.5 * features['contradiction_strength'])
                neutral_score = 1.0 - contradiction_score
            else:
                # Use variance-based classification
                variance_threshold = abs(resonance) ** 2
                
                if variance < variance_threshold:
                    if resonance > 0:
                        entailment_score = float(np.clip(resonance, 0.0, 1.0))
                        contradiction_score = 0.0
                        neutral_score = 1.0 - entailment_score
                    else:
                        entailment_score = 0.0
                        contradiction_score = float(np.clip(-resonance, 0.0, 1.0))
                        neutral_score = 1.0 - contradiction_score
                else:
                    entailment_score = (1.0 - variance) * max(0.0, resonance) if resonance > 0 else 0.0
                    contradiction_score = (1.0 - variance) * max(0.0, -resonance) if resonance < 0 else 0.0
                    neutral_score = variance
            
            scores = np.array([entailment_score, contradiction_score, neutral_score])
            basin_weights = self.basin_system.get_weights()
            collapse_result = self.collapse.collapse(scores, basin_weights)
            
            final_label = collapse_result.label
            final_class_idx = collapse_result.class_idx
            final_confidence = collapse_result.confidence
            combined_probs = collapse_result.probabilities
        
        # Check Moksha (convergence)
        moksha_state = self.moksha_router.check_moksha(resonance)
        
        # If using learned head and locked (inside clarity band), force Moksha
        if use_learned_head:
            # Check if we have lock status from decision head
            try:
                if 'is_locked' in locals() and is_locked:
                    # Override with lock status (inside clarity band = converged)
                    moksha_state.is_moksha = True
            except:
                pass
        
        return ClassificationResult(
            label=final_label,
            class_idx=final_class_idx,
            confidence=final_confidence,
            scores=combined_probs,
            is_moksha=moksha_state.is_moksha
        )
    
    def apply_learning_feedback(self, 
                                correct_label: str,
                                learning_strength: float = 1.0,
                                train_head: bool = True):
        """
        Apply learning feedback to the system.
        
        Updates:
        - Basin depths (reinforce correct class)
        - Word polarities (learn semantic structure)
        - Decision head (learn non-linear boundaries) - THE KEY TO 48-55%
        
        Args:
            correct_label: 'entailment', 'contradiction', or 'neutral'
            learning_strength: Learning rate multiplier
            train_head: If True, train the decision head (recommended)
        """
        # Map label to index
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        correct_idx = label_map.get(correct_label, 2)
        
        # Meta head: sees 3 classes but only modifies 2 peaks
        # Neutral examples sculpt the valley instead of reinforcing peaks
        
        class_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
        correct_label = class_map[correct_idx]
        
        # Extract features
        features = self.feature_extractor.extract(self.encoded_pair)
        resonance = features['resonance']
        variance = features['variance']
        
        if correct_label == 'neutral':
            # Neutral = valley creator
            # Do NOT reinforce either basin
            # Instead: sculpt the valley (flatten gradients, add noise)
            
            # 1. Sculpt valley in peak clarity (flatten peaks near this resonance)
            if train_head:
                self.peak_clarity.sculpt_valley(resonance, variance, strength=learning_strength)
            
            # 2. Sculpt valley in basin system (flatten both basins)
            self.basin_system.sculpt_valley(resonance, variance, strength=learning_strength)
            
            # 3. Update word polarities (still learn semantic structure)
            all_words = set(self.encoded_pair.premise.tokens + 
                           self.encoded_pair.hypothesis.tokens)
            for word in all_words:
                self.lexicon.update_word_polarity(
                    word, 
                    correct_idx, 
                    strength=0.15 * learning_strength
                )
        
        else:
            # E or C = reinforce peak
            
            # 1. Reinforce basin (deepen gravity well for correct class)
            self.basin_system.reinforce(correct_idx, strength=learning_strength)
            
            # 2. Update word polarities (learn semantic structure)
            all_words = set(self.encoded_pair.premise.tokens + 
                           self.encoded_pair.hypothesis.tokens)
            for word in all_words:
                self.lexicon.update_word_polarity(
                    word, 
                    correct_idx, 
                    strength=0.15 * learning_strength
                )
            
            # 3. Update peak (deepen E or C peak)
            if train_head:
                self.peak_clarity.update_peak(correct_label, resonance, variance)

