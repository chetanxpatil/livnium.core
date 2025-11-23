"""
Omcube System: Simplified Classification

Minimal wrapper around NLIClassifier with simple class bias tracking.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .native_chain_encoder import NativeEncodedPair
from .inference_detectors import NLIClassifier
from .native_chain import GlobalLexicon



# ============================================================================
# Simple Class Bias Tracker (Replaces Basin Dynamics)
# ============================================================================

class SimpleClassBias:
    """
    Simple class prior tracking (replaces complex basin dynamics).
    
    Tracks how often each class is correct to create a simple bias.
    """
    
    def __init__(self):
        """Initialize with equal priors."""
        self.class_counts = np.array([1.0, 1.0, 1.0], dtype=float)  # [E, C, N]
    
    def update(self, correct_class: int, strength: float = 1.0):
        """Update class count when correct."""
        self.class_counts[correct_class] += strength
    
    def get_weights(self) -> np.ndarray:
        """Get normalized class weights (simple prior)."""
        total = np.sum(self.class_counts)
        if total > 0:
            return self.class_counts / total
        return np.array([1.0/3, 1.0/3, 1.0/3])


# ============================================================================
# OLD CODE (kept for reference, but not used)
# ============================================================================

@dataclass
class OmcubeBasin:
    """Represents a basin (attractor) for one omcube."""
    omcube_index: int  # 0=entailment, 1=contradiction, 2=neutral
    depth: float = 1.0  # Basin depth (deeper = stronger attractor)
    curvature: float = 1.0  # Basin curvature (sharper = more confident)
    energy: float = 0.0  # Current energy level
    
    def __init__(self, omcube_index: int, initial_depth: float = 1.0):
        self.omcube_index = omcube_index
        self.depth = initial_depth
        self.curvature = 1.0
        self.energy = 0.0


class CrossOmcubeCoupling:
    """
    Couples the 3 omcubes (3-class classification) so they influence each other.
    
    When one omcube is reinforced, it affects the others:
    - Correct omcube → deepens its basin
    - Wrong omcube → gets zero reinforcement (no punishment)
    """
    
    def __init__(self, initial_depth: float = 1.0):
        """Initialize coupling system."""
        self.basins = {
            0: OmcubeBasin(0, initial_depth),  # Entailment
            1: OmcubeBasin(1, initial_depth),  # Contradiction
            2: OmcubeBasin(2, initial_depth),  # Neutral
        }
        
        # Coupling strength between omcubes (3x3 matrix)
        self.coupling_matrix = np.array([
            [1.0, -0.3, -0.3],  # Entailment affects others
            [-0.3, 1.0, -0.3],  # Contradiction affects others
            [-0.3, -0.3, 1.0],  # Neutral affects others
        ])
        
        # Learning rates (reward-only)
        # FIX: Increased reinforcement rate for faster basin learning (was 0.15, now 0.3)
        self.reinforcement_rate = 0.3  # Faster reinforcement rate
        self.coupling_strength = 0.5  # Cross-omcube influence
        
        # NATURAL DECAY RATES:
        # FIX: Set all decay rates EQUAL so Neutral doesn't bleed out
        # Equal decay prevents "rich get richer" - all basins decay at same rate
        self.decay_rate_entailment = 0.015
        self.decay_rate_contradiction = 0.015
        self.decay_rate_neutral = 0.015  # Was 0.025 - set to same as others
        
        # Lower the suppression threshold so Neutral stays in the game longer
        self.suppression_threshold_multiplier = 0.5  # Was 0.7 - more lenient
    
    def reinforce_basin(self, correct_omcube: int, strength: float = 1.0):
        """
        Reinforce correct omcube basin with Proportional Growth/Decay (Logistic Curve).
        
        FIX: Proportional decay creates contrast - winners stabilize, losers shrink.
        This prevents infinite growth and creates clear separation between basins.
        
        Args:
            correct_omcube: Index of correct omcube (0, 1, or 2)
            strength: Reinforcement strength (0-1)
        """
        # 1. Proportional Reinforcement (Logistic Growth)
        # Growth slows as basin approaches capacity: dH/dt = r * (1 - H/K)
        current_depth = self.basins[correct_omcube].depth
        capacity = 200.0  # Carrying capacity (stabilization point)
        growth = self.reinforcement_rate * strength * (1.0 - current_depth / capacity)
        self.basins[correct_omcube].depth += max(0.01, growth)  # Always grow a little
        
        # Update energy (lower energy = stronger attractor)
        self.basins[correct_omcube].energy = max(-5.0,
            self.basins[correct_omcube].energy - 0.1 * strength)
        
        # 2. Aggressive Proportional Decay for Losers
        # Losers lose 1% of their depth every time they lose
        # This creates exponential decay: depth * 0.99^n → shrinks to near zero
        decay_factor = 0.01  # 1% decay per loss
        for i in range(3):
            if i != correct_omcube:
                self.basins[i].depth *= (1.0 - decay_factor)
                self.basins[i].depth = max(0.1, self.basins[i].depth)
        
        # 3. DISABLE NORMALIZATION (Let the winner take all)
        # Commented out to allow basins to grow freely and diverge
        # self._normalize_basins()  # <-- DISABLED: Let winner grow!
    
    def _apply_natural_decay(self):
        """
        Apply natural decay to all basins (physics-based).
        
        FIX: Equal decay rates prevent "rich get richer" loop.
        All basins decay at the same rate, so no class bleeds out faster.
        """
        # All basins decay at the same rate (equal treatment)
        decay_rate = self.decay_rate_entailment  # Same for all now
        
        self.basins[0].depth = max(0.1, self.basins[0].depth - decay_rate)
        self.basins[1].depth = max(0.1, self.basins[1].depth - decay_rate)
        self.basins[2].depth = max(0.1, self.basins[2].depth - decay_rate)
    
    def get_basin_weights(self) -> np.ndarray:
        """
        Get weights for each omcube based on basin depths.
        
        FIX: Dampen exponential explosion to prevent "rich get richer" loop.
        Uses temperature scaling (softmax with temperature) to soften competition.
        
        Returns:
            Array of 3 weights (one per omcube)
        """
        depths = np.array([self.basins[i].depth for i in [0, 1, 2]], dtype=np.float64)
        
        # Calculate basin weights using softmax with temperature
        temperature = 3.0
        threshold = np.mean(depths) * self.suppression_threshold_multiplier
        suppressed_depths = depths.copy()
        for i in range(3):
            if depths[i] < threshold:
                suppressed_depths[i] = depths[i] * 0.2
        exp_depths = np.exp(suppressed_depths / temperature)
        weights = exp_depths / np.sum(exp_depths)
        
        return weights
    
    def apply_collapse_feedback(self, 
                                   correct_omcube: int,
                                   correct_omcube_unused: int,  # Keep for compatibility
                                   learning_strength: float):
        """
        Apply feedback (clean routing law).
        
        Always reinforce ground-truth basin (correct_omcube).
        All basins decay naturally (Neutral decays faster).
        
        Args:
            correct_omcube: Ground-truth omcube (0, 1, or 2) - THIS is what gets reinforced
            correct_omcube_unused: Unused (kept for compatibility)
            learning_strength: Learning strength (always 1.0)
        """
        if learning_strength > 0.0:
            # Reinforce ground-truth basin (decay applied inside)
            self.reinforce_basin(correct_omcube, learning_strength)
        else:
            # Just decay (shouldn't happen with routing law, but handle it)
            self._apply_natural_decay()
    
    def get_energy_landscape(self) -> Dict[int, float]:
        """Get energy landscape for visualization."""
        return {i: self.basins[i].energy for i in [0, 1, 2]}
    
    def get_basin_depths(self) -> Dict[int, float]:
        """Get basin depths."""
        return {i: self.basins[i].depth for i in [0, 1, 2]}
    
    def _normalize_basins(self):
        """
        Normalize basin depths to maintain stability without enforcing perfect equality.
        
        SOFT CONSTRAINT:
        Only normalize if total depth exceeds a safety ceiling (e.g. 60.0).
        This allows one basin to grow significantly larger than others (breaking symmetry)
        while preventing infinite explosion.
        """
        depths = np.array([self.basins[i].depth for i in [0, 1, 2]])
        
        # Ensure no zeros
        depths = np.maximum(depths, 0.1)
        
        current_total = np.sum(depths)
        target_ceiling = 60.0  # Increased from 30.0 to allow growth
        
        # Only scale down if we hit the ceiling
        if current_total > target_ceiling:
            scale_factor = target_ceiling / current_total
            
            for i in [0, 1, 2]:
                self.basins[i].depth = depths[i] * scale_factor
                # Update curvature
                self.basins[i].curvature = 1.0 + (self.basins[i].depth / 5.0)
        else:
            # No normalization needed - update curvature only
            for i in [0, 1, 2]:
                self.basins[i].curvature = 1.0 + (self.basins[i].depth / 5.0)


class GeometricFeedback:
    """
    Propagates collapse feedback into geometry and LEARNS WORD POLARITY.
    
    Updates:
    - Quantum amplitudes (entanglement strength)
    - Word semantic polarities (learned from training)
    - Letter vectors (direct updates to GlobalLexicon)
    """
    
    def __init__(self, encoded_pair: NativeEncodedPair):
        """
        Initialize geometric feedback.
        
        Args:
            encoded_pair: The encoded premise-hypothesis pair
        """
        self.encoded_pair = encoded_pair
        self.lexicon = GlobalLexicon()  # Access to global memory
    
    def reinforce_geometry(self, 
                          correct_omcube: int,
                          strength: float = 1.0):
        """
        Reinforce Native Chain using Relational Physics.
        
        Instead of updating symbolic weights, we update:
        - Chain entanglement strength
        - Quantum state amplitudes
        - WordOmcube masses
        
        - Entailment (0): Strengthen resonance (increase entanglement)
        - Contradiction (1): Weaken resonance (decrease entanglement)
        - Neutral (2): Weak reinforcement
        """
        premise_chain = self.encoded_pair.premise_chain
        hypothesis_chain = self.encoded_pair.hypothesis_chain
        
        # FIX: Increased learning rate for faster convergence (was 0.5, now 1.0)
        learning_rate = 1.0 * strength
        
        # OPTIMIZATION: Build word->cube maps for O(1) lookup (was O(P*H) nested loops)
        premise_word_map = {cube.word: cube for cube in premise_chain.chain}
        hypothesis_word_map = {cube.word: cube for cube in hypothesis_chain.chain}
        
        # Find matching words (O(P+H) instead of O(P*H))
        matching_words = set(premise_word_map.keys()) & set(hypothesis_word_map.keys())
        
        if correct_omcube == 0:  # ENTAILMENT -> Strengthen resonance
            # Increase entanglement between matching words
            # FIX: Increased from 0.1 to 0.2 for faster learning (was 5%, now 20%)
            factor = 1.0 + learning_rate * 0.2
            for word in matching_words:
                p_cube = premise_word_map[word]
                h_cube = hypothesis_word_map[word]
                # Strengthen quantum coupling
                # Update amplitudes directly
                p_cube.quantum_state.amplitudes *= factor
                h_cube.quantum_state.amplitudes *= factor
                        
        elif correct_omcube == 1:  # CONTRADICTION -> Weaken resonance
            # Decrease entanglement (polarize)
            # FIX: Increased from 0.1 to 0.2 for faster learning (was 5%, now 20%)
            factor = 1.0 - learning_rate * 0.2
            for word in matching_words:
                p_cube = premise_word_map[word]
                h_cube = hypothesis_word_map[word]
                # Weaken quantum coupling
                # Update amplitudes directly
                p_cube.quantum_state.amplitudes *= factor
                h_cube.quantum_state.amplitudes *= factor
                        
        elif correct_omcube == 2:  # NEUTRAL -> Weak reinforcement
            # FIX: Add reinforcement for neutral (was completely skipped!)
            # Use smaller factor than entailment/contradiction (neutral is ambiguous)
            factor = 1.0 + learning_rate * 0.05  # 5% increase (small but non-zero)
            for word in matching_words:
                p_cube = premise_word_map[word]
                h_cube = hypothesis_word_map[word]
                # Apply small reinforcement for neutral pairs
                p_cube.quantum_state.amplitudes *= factor
                h_cube.quantum_state.amplitudes *= factor
        
        # Batch normalize all cubes at once
        all_cubes = premise_chain.chain + hypothesis_chain.chain
        for cube in all_cubes:
            cube.quantum_state.normalize()
        
        # UPDATE LEARNED WORD POLARITIES (Semantic Learning)
        # Iterate through all words and nudge them towards the correct class label
        all_words = premise_chain.words + hypothesis_chain.words
        unique_words = set(all_words)
        
        for word in unique_words:
            # Faster semantic learning - "not" will learn contradiction faster
            # "cat" will see label=0,1,2 equally and stay Neutral
            self.lexicon.update_word_polarity(word, correct_omcube, strength=0.15)
        
        # DIRECT MEMORY UPDATES (No consolidation delays)
        # Save all letter states immediately
        for word_chain in [premise_chain, hypothesis_chain]:
            if hasattr(word_chain, 'letter_cubes'):
                for letter_cube in word_chain.letter_cubes:
                    letter_cube.save_state()
    
    def apply_collapse_feedback(self,
                                collapsed_omcube: int,
                                correct_omcube: int,
                                learning_strength: float):
        """
        Apply reward-only collapse feedback to geometry (basin physics).
        
        Args:
            collapsed_omcube: Which omcube collapsed
            correct_omcube: Which omcube should have collapsed
            learning_strength: Learning strength (1.0 if correct, 0.0 if wrong)
        """
        if learning_strength > 0.0:
            # Correct collapse: reinforce geometry
            self.reinforce_geometry(correct_omcube, learning_strength)


# ============================================================================
# Classifier: Quantum Collapse
# ============================================================================

@dataclass
class OmcubeCollapseResult:
    """Result of 3-omcube collapse (3-class classification)."""
    label: str  # 'entailment', 'contradiction', or 'neutral'
    collapsed_omcube: int  # 0=entailment, 1=contradiction, 2=neutral
    probabilities: Dict[str, float]  # Probabilities before collapse
    confidence: float  # Probability of collapsed state
    amplitudes: np.ndarray  # Quantum amplitudes before collapse


class OmcubeNLIClassifier:
    """
    Simplified NLI Classifier - Minimal wrapper around NLIClassifier.
    
    Uses NLIClassifier for core classification with simple class bias tracking.
    """
    
    def __init__(self, encoded_pair: NativeEncodedPair):
        """
        Initialize simplified classifier.
        
        Args:
            encoded_pair: Native Chain encoded premise-hypothesis pair
        """
        self.encoded_pair = encoded_pair
        self.nli_classifier = NLIClassifier(encoded_pair)
        
        # Simple class bias tracker (replaces basin dynamics)
        self.class_bias = SimpleClassBias()
        
        # Geometric feedback (word-level learning)
        self.geometric_feedback = GeometricFeedback(encoded_pair)
    
    def classify(self, 
                 collapse: bool = True,
                 deterministic_threshold: float = 0.1) -> OmcubeCollapseResult:
        """
        Classify using NLIClassifier with simple class bias.
        
        Args:
            collapse: Whether to collapse after measurement (unused, kept for compatibility)
            deterministic_threshold: Threshold for deterministic selection (unused, kept for compatibility)
            
        Returns:
            OmcubeCollapseResult with collapsed label
        """
        # Get scores from NLIClassifier (core classification)
        nli_result = self.nli_classifier.classify()
        entailment_score = nli_result['scores']['entailment']
        contradiction_score = nli_result['scores']['contradiction']
        neutral_score = nli_result['scores']['neutral']
        
        # Apply simple class bias (learned prior)
        bias_weights = self.class_bias.get_weights()
        scores = np.array([
            entailment_score * (1.0 + bias_weights[0] * 0.2),  # Small bias boost
            contradiction_score * (1.0 + bias_weights[1] * 0.2),
            neutral_score * (1.0 + bias_weights[2] * 0.2)
        ])
        
        # Get predicted class (argmax)
        measured_omcube = int(np.argmax(scores))
        
        # Convert scores to probabilities (softmax)
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probs = exp_scores / np.sum(exp_scores)
        
        # Map to label
        label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
        label = label_map[measured_omcube]
        
        result = OmcubeCollapseResult(
            label=label,
            collapsed_omcube=measured_omcube,
            probabilities={
                'entailment': float(probs[0]),
                'contradiction': float(probs[1]),
                'neutral': float(probs[2])
            },
            confidence=float(probs[measured_omcube]),
            amplitudes=np.sqrt(probs).astype(complex)  # Simple amplitude representation
        )
        
        # Store feedback reference for learning
        result._geometric_feedback = self.geometric_feedback
        
        return result
    
    def apply_learning_feedback(self,
                                   result: OmcubeCollapseResult,
                                   correct_omcube: int,
                                   learning_strength: float):
        """
        Apply learning feedback - update class bias and geometry.
        
        Args:
            result: Collapse result (used for statistics)
            correct_omcube: Ground-truth omcube (0, 1, or 2)
            learning_strength: Learning strength (always 1.0)
        """
        if learning_strength > 0.0:
            # Update simple class bias
            self.class_bias.update(correct_omcube, learning_strength)
            
            # Update geometry (word-level learning)
            self.geometric_feedback.apply_collapse_feedback(
                correct_omcube, correct_omcube, learning_strength
            )
    
    def get_superposition_state(self) -> Dict[str, Any]:
        """
        Get current classification state.
        
        Returns:
            Dict with probabilities and bias weights
        """
        nli_result = self.nli_classifier.classify()
        bias_weights = self.class_bias.get_weights()
        
        scores = np.array([
            nli_result['scores']['entailment'] * (1.0 + bias_weights[0] * 0.2),
            nli_result['scores']['contradiction'] * (1.0 + bias_weights[1] * 0.2),
            nli_result['scores']['neutral'] * (1.0 + bias_weights[2] * 0.2)
        ])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        
        return {
            'probabilities': {
                'entailment': float(probs[0]),
                'contradiction': float(probs[1]),
                'neutral': float(probs[2])
            },
            'bias_weights': {
                'entailment': float(bias_weights[0]),
                'contradiction': float(bias_weights[1]),
                'neutral': float(bias_weights[2])
            },
            'entropy': float(-np.sum(probs * np.log(probs + 1e-10)))  # Shannon entropy
        }

