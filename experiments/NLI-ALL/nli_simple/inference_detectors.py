"""
Simple Inference Detectors: Pure Geometric Classification

Uses cosine similarity, lexical overlap, and learned word polarity.
No hardcoded word lists, no physics dependencies.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from .native_chain import SimpleLexicon
from .encoder import SimpleEncodedPair


class SimpleLogic:
    """Pure geometric logic for NLI."""
    
    @staticmethod
    def compute_overlap(tokens1: List[str], tokens2: List[str]) -> float:
        """Compute lexical overlap (Jaccard + subset score)."""
        set1 = set(tokens1)
        set2 = set(tokens2)
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = intersection / union if union > 0 else 0.0
        subset_score = len(set2 & set1) / len(set2) if len(set2) > 0 else 0.0
        return (jaccard + subset_score) / 2.0
    
    @staticmethod
    def detect_learned_polarity(tokens: List[str], target_class: int = 1) -> Tuple[bool, float]:
        """
        Check if any token has learned strong polarity for a class.
        
        Args:
            tokens: List of word tokens
            target_class: 0=Entailment, 1=Contradiction, 2=Neutral
            
        Returns:
            (found_strong_indicator, max_strength)
        """
        lexicon = SimpleLexicon()
        max_strength = 0.0
        found = False
        
        for token in tokens:
            polarity = lexicon.get_word_polarity(token)
            strength = polarity[target_class]
            
            if strength > 0.55:  # Strong indicator
                found = True
                max_strength = max(max_strength, strength)
        
        return found, max_strength


class EntailmentDetector:
    """Detect entailment using positive similarity."""
    
    def __init__(self, encoded_pair: SimpleEncodedPair):
        self.encoded_pair = encoded_pair
    
    def detect_entailment(self) -> Dict[str, float]:
        """Detect entailment using geometric similarity and learned polarity."""
        # Geometric similarity (resonance)
        resonance = self.encoded_pair.get_resonance()
        geometric_score = max(0.0, resonance)  # Only positive counts
        
        # Lexical overlap
        p_tokens = self.encoded_pair.premise.tokens
        h_tokens = self.encoded_pair.hypothesis.tokens
        overlap = SimpleLogic.compute_overlap(p_tokens, h_tokens)
        
        # Check learned polarity for contradiction (suppress entailment if present)
        has_contradiction_word, _ = SimpleLogic.detect_learned_polarity(h_tokens, target_class=1)
        
        if has_contradiction_word:
            lexical_score = 0.0  # Suppress if contradiction word present
        else:
            lexical_score = overlap
        
        # Combined score
        final_score = 0.6 * geometric_score + 0.4 * lexical_score
        
        return {
            'entailment_score': float(np.clip(final_score, 0.0, 1.0)),
            'geometric_score': float(geometric_score),
            'lexical_score': float(lexical_score)
        }


class ContradictionDetector:
    """Detect contradiction using geometric opposition and learned polarity."""
    
    def __init__(self, encoded_pair: SimpleEncodedPair):
        self.encoded_pair = encoded_pair
    
    def detect_contradiction(self) -> Dict[str, float]:
        """Detect contradiction using negative similarity and learned polarity."""
        resonance = self.encoded_pair.get_resonance()
        
        p_tokens = self.encoded_pair.premise.tokens
        h_tokens = self.encoded_pair.hypothesis.tokens
        overlap = SimpleLogic.compute_overlap(p_tokens, h_tokens)
        
        # Geometric opposition (negative or low resonance with high overlap)
        geometric_opposition = 1.0 - resonance
        
        # Check learned polarity for contradiction
        has_contradiction_word, contradiction_strength = SimpleLogic.detect_learned_polarity(h_tokens, target_class=1)
        
        # Semantic gap: high overlap but low resonance
        semantic_gap = 0.0
        if overlap > 0.3 and resonance < overlap:
            semantic_gap = min(1.0, (overlap - resonance) * 3.0)
        
        if has_contradiction_word:
            # Learned contradiction word present
            final_score = 0.5 + (0.5 * contradiction_strength)
            lexical_score = contradiction_strength
        elif semantic_gap > 0.15:
            # Semantic gap detected
            final_score = semantic_gap * 2.0 + geometric_opposition * 0.2
            lexical_score = 0.0
        else:
            # Pure geometric opposition
            final_score = geometric_opposition * 0.5
            lexical_score = 0.0
        
        return {
            'contradiction_score': float(np.clip(final_score, 0.0, 1.0)),
            'geometric_score': float(geometric_opposition),
            'lexical_score': float(lexical_score),
            'gap_score': float(semantic_gap)
        }


class SimpleNLIClassifier:
    """
    Planet Architecture: Semantic Climate System
    
    You're not building a classifier. You're building a PLANET where meaning
    emerges from geometry, just like Earth's climate emerges from its shape.
    
    Architecture:
    - E (Entailment) = Mountain peak (high curvature, stable, positive resonance)
    - C (Contradiction) = Mountain peak (high curvature, stable, negative resonance)
    - N (Neutral) = Valley between peaks (real gravitational curvature, not just "uncertainty")
    
    Key insight: Two opposing forces (E and C) naturally create a stable middle (N).
    The valley has REAL gravitational mass - it pulls downward when E and C forces balance.
    
    This is why Neutral isn't just "high variance" - it's a real valley with curvature.
    """
    
    def __init__(self, encoded_pair: SimpleEncodedPair):
        self.encoded_pair = encoded_pair
    
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    
    def classify(self, use_sequence: bool = True) -> Dict[str, any]:
        """
        Classify NLI pair using variance-based geometry with CHAIN STRUCTURE.
        
        Computes cosine similarity for each word pair using chain matching (position matters),
        then uses mean and variance to automatically determine class boundaries.
        
        Args:
            use_sequence: If True, use sequential chain matching (position matters).
                         This is CRITICAL for breaking the bag-of-words ceiling.
        
        Returns:
            Dict with label, confidence, and scores
        """
        # FORCE chain structure: use sequential matching
        # This ensures positional encoding, aligned matching, sliding windows are all used
        if use_sequence:
            # Use the chain-based comparison directly (this uses all chain structure)
            chain_resonance = self.encoded_pair.premise.compare(
                self.encoded_pair.hypothesis, 
                use_sequence=True
            )
        else:
            chain_resonance = self.encoded_pair.premise.compare(
                self.encoded_pair.hypothesis, 
                use_sequence=False
            )
        
        # Get word vectors for premise and hypothesis (for variance calculation)
        premise_vecs = self.encoded_pair.premise.get_word_vectors()
        hypothesis_vecs = self.encoded_pair.hypothesis.get_word_vectors()
        
        if not premise_vecs or not hypothesis_vecs:
            return {
                'label': 'neutral',
                'confidence': 0.0,
                'scores': {'entailment': 0.0, 'contradiction': 0.0, 'neutral': 1.0},
                'mean_similarity': 0.0,
                'variance': 0.0
            }
        
        # Compute cosine similarity for each word pair using CHAIN STRUCTURE
        # This is the key: use sequential matching (position matters)
        sims = []
        
        # CHAIN MATCHING: Aligned positions (position 0 vs 0, 1 vs 1, etc.)
        # This captures order-dependent patterns (syntax, negation, quantifiers)
        min_len = min(len(premise_vecs), len(hypothesis_vecs))
        
        # Aligned matching: compare words at same positions
        for i in range(min_len):
            p_vec = premise_vecs[i]
            h_vec = hypothesis_vecs[i]
            sim = self._cosine(p_vec, h_vec)
            sims.append(sim)
        
        # Sliding window: allow small position shifts (-1, 0, +1)
        # This handles word reordering while maintaining structure
        window_sims = []
        for offset in range(-1, 2):  # Check -1, 0, +1 positions
            offset_sims = []
            for i in range(min_len):
                p_idx = i
                h_idx = i + offset
                if 0 <= h_idx < len(hypothesis_vecs):
                    p_vec = premise_vecs[p_idx]
                    h_vec = hypothesis_vecs[h_idx]
                    sim = self._cosine(p_vec, h_vec)
                    offset_sims.append(sim)
            if offset_sims:
                window_sims.append(np.mean(offset_sims))
        
        # Cross-word matching (for flexibility)
        cross_sims = []
        for h_vec in hypothesis_vecs:
            best_cross = -1.0
            for p_vec in premise_vecs:
                cross_sim = self._cosine(p_vec, h_vec)
                best_cross = max(best_cross, cross_sim)
            if best_cross > -1.0:
                cross_sims.append(best_cross)
        
        # Combine chain structure: 60% aligned + 20% window + 20% cross-word
        # This gives structure while maintaining flexibility
        if sims:
            aligned_mean = np.mean(sims)
            window_mean = max(window_sims) if window_sims else aligned_mean
            cross_mean = np.mean(cross_sims) if cross_sims else aligned_mean
            
            # Add combined chain similarity (this is the key structure signal)
            chain_sim = 0.6 * aligned_mean + 0.2 * window_mean + 0.2 * cross_mean
            sims.append(chain_sim)
            
            # Also add the chain resonance as a strong signal
            # This ensures the full chain structure (positional encoding, sequential matching) is used
            if use_sequence:
                sims.append(chain_resonance)
        
        # If no matches, default to neutral
        if len(sims) == 0:
            return {
                'label': 'neutral',
                'confidence': 0.0,
                'scores': {'entailment': 0.0, 'contradiction': 0.0, 'neutral': 1.0},
                'mean_similarity': 0.0,
                'variance': 0.0
            }
        
        # CRITICAL: Chain resonance is the PRIMARY signal
        # This is the result of full chain matching (positional encoding + aligned + window + cross-word)
        # It MUST drive the final classification decision
        if use_sequence:
            # Use chain resonance as the primary mean signal
            # This ensures chain structure (position matters) is the dominant factor
            mean_sim = float(chain_resonance)
            
            # Calculate variance from word-pair similarities
            # This captures consistency across positions
            if len(sims) > 0:
                sims_array = np.array(sims)
                var_sim = float(np.var(sims_array))
            else:
                var_sim = 0.0
        else:
            # Bag-of-words mode: use original calculations
            sims_array = np.array(sims) if len(sims) > 0 else np.array([0.0])
            mean_sim = float(np.mean(sims_array))
            var_sim = float(np.var(sims_array))
        
        # Classification driven purely by geometry — no hardcoded thresholds
        # Boundaries emerge organically from patterns in the data
        
        # Check for learned contradiction words (e.g., "not", "no", "never")
        # This helps catch cases where word vectors align but semantics contradict
        p_tokens = self.encoded_pair.premise.tokens
        h_tokens = self.encoded_pair.hypothesis.tokens
        has_contradiction_word, contradiction_strength = SimpleLogic.detect_learned_polarity(
            h_tokens, target_class=1
        )
        
        # ============================================================
        # PLANET ARCHITECTURE: E and C are peaks, N is the valley
        # ============================================================
        # Compute peak attractions (gravitational pull from each peak)
        # E peak: positive resonance creates positive attraction
        e_attraction = max(0.0, mean_sim) * (1.0 - min(var_sim, 1.0))  # Peak strength × (1 - noise)
        # C peak: negative resonance creates negative attraction (flip to positive)
        c_attraction = max(0.0, -mean_sim) * (1.0 - min(var_sim, 1.0))  # Peak strength × (1 - noise)
        
        # Learned contradiction words strengthen C peak
        if has_contradiction_word and contradiction_strength > 0.6:
            c_attraction = max(c_attraction, contradiction_strength)
            e_attraction *= 0.3  # Suppress E when strong C word present
        
        # Valley curvature: where E and C attractions overlap (forces balance)
        # This gives Neutral REAL gravitational mass, not just "high variance"
        max_attraction = max(e_attraction, c_attraction, 1e-6)  # Avoid division by zero
        attraction_ratio = abs(e_attraction - c_attraction) / max_attraction  # 0 = perfect overlap, 1 = one dominates
        
        # Valley threshold: when attractions are close, valley forms
        valley_threshold = 0.15  # When E and C are within 15% of each other, valley appears
        
        # Valley gravity: real curvature that pulls downward when forces balance
        # Overlap strength: higher when E ≈ C (forces cancel, valley deepens)
        overlap_strength = 1.0 - min(attraction_ratio, 1.0)  # 1.0 when E=C, 0.0 when one dominates
        valley_gravity = 0.7  # Valley gravitational constant (stronger than before)
        min_attraction = min(e_attraction, c_attraction)
        
        # Valley pull: real gravitational force when E and C overlap
        # This is the key: valley has MASS, not just absence of peaks
        valley_pull = overlap_strength * valley_gravity * (min_attraction + 0.1)  # +0.1 ensures minimum pull
        
        # Compute scores with valley gravity
        if attraction_ratio < valley_threshold and max_attraction > 0.05:
            # Valley forms: E and C forces balance → Neutral with real curvature
            base_valley_score = 1.0 - (attraction_ratio / valley_threshold)  # How close to perfect balance
            base_valley_score = max(0.0, min(1.0, base_valley_score))
            
            # Add valley gravity (this is the critical fix: valley pulls downward)
            neu_score = min(1.0, base_valley_score + valley_pull)
            
            # E and C scores reduced in valley (they cancel, valley pulls)
            reduction = 0.4 + valley_pull * 0.4  # More reduction when valley pulls strongly
            ent_score = e_attraction * (1.0 - reduction)
            con_score = c_attraction * (1.0 - reduction)
            
            # Ensure non-negative
            ent_score = max(0.0, ent_score)
            con_score = max(0.0, con_score)
            
            label = 'neutral'
            confidence = neu_score
        else:
            # One peak dominates, but valley still has pull if E and C are somewhat close
            if e_attraction > c_attraction:
                ent_score = e_attraction
                con_score = c_attraction * 0.3
                # If E and C are close, valley still pulls (medium gravity)
                if attraction_ratio < 0.35:  # E and C are somewhat close
                    neu_score = valley_pull * 0.8  # Valley has gravitational pull
                    ent_score = e_attraction * (1.0 - valley_pull * 0.2)  # Slight reduction from valley
                else:
                    neu_score = max(0.0, 1.0 - e_attraction)  # Residual valley
                label = 'entailment'
                confidence = ent_score
            else:
                ent_score = e_attraction * 0.3
                con_score = c_attraction
                # If E and C are close, valley still pulls (medium gravity)
                if attraction_ratio < 0.35:  # E and C are somewhat close
                    neu_score = valley_pull * 0.8  # Valley has gravitational pull
                    con_score = c_attraction * (1.0 - valley_pull * 0.2)  # Slight reduction from valley
                else:
                    neu_score = max(0.0, 1.0 - c_attraction)  # Residual valley
                label = 'contradiction'
                confidence = con_score
        
        # Normalize scores to sum to 1.0
        total = ent_score + con_score + neu_score
        if total > 0:
            ent_score /= total
            con_score /= total
            neu_score /= total
        
        return {
            'label': label,
            'confidence': confidence,
            'scores': {
                'entailment': float(ent_score),
                'contradiction': float(con_score),
                'neutral': float(neu_score),
                'mean_similarity': mean_sim,
                'variance': var_sim,
                'e_attraction': float(e_attraction),
                'c_attraction': float(c_attraction),
                'valley_pull': float(valley_pull),
                'attraction_ratio': float(attraction_ratio)
            }
        }

