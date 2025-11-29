"""
Inference Detectors: Native Logic

FIXED: Detects Contradiction via Geometric Opposition (Negative Resonance).

"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

from .native_chain_encoder import NativeEncodedPair
from .native_chain import GlobalLexicon



class NativeLogic:
    """Pure Python logic for extracting semantic signals from tokens."""
    
    @staticmethod
    def get_tokens(token_map: Dict[str, Any]) -> List[str]:
        return [t.lower() for t in token_map.keys()]

    @staticmethod
    def detect_learned_polarity(tokens: List[str], target_class: int = 1) -> Tuple[bool, float]:
        """
        Check if any token has learned to be a strong indicator for a class.
        target_class: 0=Entailment, 1=Contradiction, 2=Neutral
        
        Returns: (found_strong_indicator, max_strength)
        """
        lexicon = GlobalLexicon()
        max_strength = 0.0
        found = False
        
        for token in tokens:
            # Get learned polarity vector [E, C, N]
            polarity = lexicon.get_word_polarity(token)
            
            # Check strength of target class
            strength = polarity[target_class]
            
            # If word strongly points to target class (e.g. > 0.55)
            # "not" will eventually reach ~0.9 for Contradiction
            if strength > 0.55:
                found = True
                max_strength = max(max_strength, strength)
                
        return found, max_strength

    @staticmethod
    def detect_double_negative(premise_tokens: List[str], hypothesis_tokens: List[str]) -> bool:
        """
        Detect if premise and hypothesis have same polarity using learned word polarities.
        
        Uses learned polarity vectors to determine if both sentences have similar semantic polarity.
        No hard-coded word lists - only geometry + learned data.
        
        Returns True if same polarity (both positive OR both negative), False if opposite.
        """
        lexicon = GlobalLexicon()
        
        # Get average polarity for premise and hypothesis
        premise_polarities = [lexicon.get_word_polarity(token) for token in premise_tokens]
        hypothesis_polarities = [lexicon.get_word_polarity(token) for token in hypothesis_tokens]
        
        if not premise_polarities or not hypothesis_polarities:
            return False
        
        # Average polarity vectors
        avg_premise = np.mean(premise_polarities, axis=0)
        avg_hypothesis = np.mean(hypothesis_polarities, axis=0)
        
        # Check if both lean toward same class (entailment vs contradiction)
        # If both have high contradiction polarity OR both have low contradiction polarity
        pre_contradiction = avg_premise[1]  # Contradiction class
        hyp_contradiction = avg_hypothesis[1]
        
        # Same polarity if both are high contradiction OR both are low contradiction
        both_high = (pre_contradiction > 0.5) and (hyp_contradiction > 0.5)
        both_low = (pre_contradiction < 0.4) and (hyp_contradiction < 0.4)
        
        return both_high or both_low

    @staticmethod
    def compute_overlap(tokens1: List[str], tokens2: List[str]) -> float:
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
    def detect_negation_geometric(encoded_pair) -> Tuple[bool, float]:
        """
        Detect negation using geometric opposition (METHOD 1).
        
        Returns:
            (has_negation, opposition_strength)
            - has_negation: True if geometric opposition detected
            - opposition_strength: How strong the opposition is [0, 1]
        """
        resonance = encoded_pair.get_resonance()
        
        # Negative resonance = geometric opposition = likely negation
        if resonance < -0.2:
            return (True, abs(resonance))
        
        # Low resonance with high overlap = semantic gap = possible negation
        p_tokens = encoded_pair.premise_chain.tokens
        h_tokens = encoded_pair.hypothesis_chain.tokens
        overlap = NativeLogic.compute_overlap(p_tokens, h_tokens)
        
        if overlap > 0.4 and resonance < 0.3:
            # High overlap but low resonance = contradiction/negation
            opposition = (0.3 - resonance) / 0.3  # Normalize to [0, 1]
            return (True, opposition)
        
        return (False, 0.0)

    @staticmethod
    def detect_negation_word_level(encoded_pair) -> Tuple[bool, float]:
        """
        Detect negation using word-level geometric opposition (pure geometry, no word lists).
        
        Checks if specific word pairs have negative geometric similarity.
        Uses learned polarity to identify words that might indicate negation.
        
        Returns:
            (has_negation, opposition_strength)
        """
        premise_chain = encoded_pair.premise_chain
        hypothesis_chain = encoded_pair.hypothesis_chain
        lexicon = GlobalLexicon()
        
        max_opposition = 0.0
        has_learned_negation_word = False
        
        # Check each word pair for geometric opposition
        for h_word_chain in hypothesis_chain.word_chains:
            h_geo = h_word_chain.get_geometry_vector()
            h_word = h_word_chain.word
            
            # Check learned polarity (no hard-coded list)
            h_polarity = lexicon.get_word_polarity(h_word)
            if h_polarity[1] > 0.55:  # Strong contradiction polarity
                has_learned_negation_word = True
            
            for p_word_chain in premise_chain.word_chains:
                p_geo = p_word_chain.get_geometry_vector()
                
                # Compute geometric similarity
                dot_prod = np.dot(p_geo, h_geo)
                norm_p = np.linalg.norm(p_geo)
                norm_h = np.linalg.norm(h_geo)
                if norm_p > 0 and norm_h > 0:
                    geo_sim = dot_prod / (norm_p * norm_h)
                else:
                    geo_sim = 0.0
                    
                # Negative similarity = opposition
                if geo_sim < 0:
                    opposition = abs(geo_sim)
                    max_opposition = max(max_opposition, opposition)
        
        # If we found geometric opposition, or learned negation word with any opposition
        if max_opposition > 0.2 or (has_learned_negation_word and max_opposition > 0.1):
            return (True, min(1.0, max_opposition))
        
        return (False, 0.0)

    @staticmethod
    def detect_negation_resonance_pattern(encoded_pair) -> Tuple[bool, float]:
        """
        Detect negation using resonance patterns (METHOD 3).
        
        Pattern: High overlap + negative/low resonance = negation
        Pattern: Negative resonance = direct geometric opposition
        
        Returns:
            (has_negation, confidence)
        """
        resonance = encoded_pair.get_resonance()
        p_tokens = encoded_pair.premise_chain.tokens
        h_tokens = encoded_pair.hypothesis_chain.tokens
        overlap = NativeLogic.compute_overlap(p_tokens, h_tokens)
        
        # Pattern 1: Direct negative resonance
        if resonance < -0.15:
            return (True, min(1.0, abs(resonance) * 2.0))
        
        # Pattern 2: High overlap but very low resonance (semantic gap)
        if overlap > 0.5 and resonance < 0.2:
            gap = (0.5 - resonance) / 0.5  # Normalize
            return (True, gap * 0.8)  # Slightly lower confidence
        
        # Pattern 3: Moderate overlap but negative resonance
        if overlap > 0.3 and resonance < 0.0:
            return (True, abs(resonance) * 1.5)
        
        return (False, 0.0)


class EntailmentDetector:
    """Detect entailment using positive resonance."""
    
    def __init__(self, encoded_pair: NativeEncodedPair):
        self.encoded_pair = encoded_pair
    
    def detect_entailment(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        # FIX: Only positive resonance counts for entailment
        resonance = self.encoded_pair.get_resonance()
        geometric_score = max(0.0, resonance)
        
        p_tokens = self.encoded_pair.premise_chain.tokens
        h_tokens = self.encoded_pair.hypothesis_chain.tokens
        
        overlap = NativeLogic.compute_overlap(p_tokens, h_tokens)
        
        # Check learned polarity for Contradiction (Class 1)
        has_learned_negation, neg_strength = NativeLogic.detect_learned_polarity(h_tokens, target_class=1)
        
        # Check learned polarity for Entailment (Class 0)
        # Note: Words don't usually learn strong Entailment polarity because they are context-dependent,
        # but double-negatives or strong confirmers might.
        
        if has_learned_negation:
            # If "not" is present (high contradiction polarity), suppress entailment
            lexical_score = 0.0
        else:
            lexical_score = overlap
        
        # Combined: 60% Geometry, 40% Lexical
        final_score = 0.6 * geometric_score + 0.4 * lexical_score
        
        return {
            'entailment_score': float(np.clip(final_score, 0.0, 1.0)),
            'lexical_score': float(lexical_score),
            'geometric_score': float(geometric_score),
            'entails': final_score > 0.5
        }
    

class ContradictionDetector:
    """Detect contradiction using geometric opposition (negative resonance)."""
    
    def __init__(self, encoded_pair: NativeEncodedPair):
        self.encoded_pair = encoded_pair
    
    def detect_contradiction(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        resonance = self.encoded_pair.get_resonance()
        
        p_tokens = self.encoded_pair.premise_chain.tokens
        h_tokens = self.encoded_pair.hypothesis_chain.tokens
        
        overlap = NativeLogic.compute_overlap(p_tokens, h_tokens)
        
        # 1. CHECK LEARNED POLARITY (Replaces hardcoded list)
        has_learned_negation, neg_strength = NativeLogic.detect_learned_polarity(h_tokens, target_class=1)
        
        geometric_opposition = 1.0 - resonance
        
        semantic_gap = 0.0
        if overlap > 0.3:
            if resonance < overlap:
                semantic_gap = min(1.0, (overlap - resonance) * 3.5)
            elif overlap > 0.5:
                resonance_gap = 1.0 - resonance
                semantic_gap = min(1.0, overlap * resonance_gap * 6.0)
            
        if has_learned_negation:
            # Learned that this word (e.g. "not") implies contradiction
            # Strength depends on how certain the lexicon is
            final_score = 0.5 + (0.5 * neg_strength)
            lexical_score = neg_strength  # Use learned negation strength
        elif semantic_gap > 0.15:
            final_score = semantic_gap * 2.5 + geometric_opposition * 0.2
            lexical_score = 0.0  # No lexical negation detected
        else:
            final_score = geometric_opposition * 0.5
            lexical_score = 0.0  # No lexical negation detected
        
        return {
            'contradiction_score': float(np.clip(final_score, 0.0, 1.0)),
            'lexical_score': float(lexical_score),
            'geometric_score': float(geometric_opposition),
            'gap_score': float(semantic_gap),
            'contradicts': final_score > 0.5
        }


class NLIClassifier:
    """Complete NLI classifier with explicit Neutral detection."""
    
    def __init__(self, encoded_pair: NativeEncodedPair):
        self.encoded_pair = encoded_pair
        self.entailment_detector = EntailmentDetector(encoded_pair)
        self.contradiction_detector = ContradictionDetector(encoded_pair)
    
    def classify(self, entailment_threshold: float = 0.8, contradiction_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Classify NLI pair into Entailment, Contradiction, or Neutral.
        
        FIXED:
        - Raised thresholds from 0.4 to 0.6 (require higher confidence)
        - Added logic: If both E and C are moderate (0.3-0.6), choose Neutral
        """
        entailment_result = self.entailment_detector.detect_entailment()
        contradiction_result = self.contradiction_detector.detect_contradiction()
        
        ent_score = entailment_result['entailment_score']
        con_score = contradiction_result['contradiction_score']
        
        # FIX: Neutral is the absence of strong signals OR when both are moderate
        # Boost neutral when both scores are moderate (uncertainty)
        resonance = self.encoded_pair.get_resonance()
        
        # Check if both scores are moderate (uncertainty case)
        both_moderate = (0.35 < ent_score < 0.6 and 0.35 < con_score < 0.6)
        
        # Also check if resonance is moderate for unrelated topics
        moderate_resonance = (resonance < 0.6)
        moderate_entailment = (ent_score < 0.8)
        
        # For unrelated topics: moderate resonance + moderate entailment = Neutral
        # Check lexical overlap - filter out common words (is, the, a, etc.)
        p_tokens = self.encoded_pair.premise_chain.tokens
        h_tokens = self.encoded_pair.hypothesis_chain.tokens
        lexical_overlap = NativeLogic.compute_overlap(p_tokens, h_tokens)
        
        # Filter out words with neutral polarity (learned, not hard-coded)
        # Words that appear equally in all classes stay neutral
        lexicon = GlobalLexicon()
        p_content = []
        h_content = []
        
        for token in p_tokens:
            polarity = lexicon.get_word_polarity(token)
            # Keep words that have some class preference (not perfectly neutral)
            max_polarity = np.max(polarity)
            if max_polarity > 0.4:  # Has some class preference
                p_content.append(token)
        
        for token in h_tokens:
            polarity = lexicon.get_word_polarity(token)
            max_polarity = np.max(polarity)
            if max_polarity > 0.4:  # Has some class preference
                h_content.append(token)
        
        semantic_overlap = NativeLogic.compute_overlap(p_content, h_content) if (p_content and h_content) else 0.0
        
        # Low overlap: either lexical < 0.4 OR semantic < 0.2 (after filtering common words)
        low_overlap = (lexical_overlap < 0.4) or (semantic_overlap < 0.2)
        
        # Unrelated neutral: moderate resonance + moderate entailment + low overlap + low contradiction
        is_unrelated = (resonance < 0.6 and lexical_overlap < 0.4)
        
        if both_moderate or is_unrelated:
            # Both moderate OR unrelated topics → Neutral
            neutral_score = 0.75
        else:
            # Standard calculation: absence of strong signals
            neutral_score = 1.0 - max(ent_score, con_score)
        
        scores = {
            'entailment': float(ent_score),
            'contradiction': float(con_score),
            'neutral': float(neutral_score)
        }
        
        # Hard classification logic using learned polarities
        if con_score > contradiction_threshold and con_score > (ent_score + 0.15):
            label = 'contradiction'
            confidence = con_score
            
        elif ent_score >= entailment_threshold and ent_score > (con_score + 0.2):
            label = 'entailment'
            confidence = ent_score
            
        elif neutral_score > 0.7 and (both_moderate or is_unrelated):
            label = 'neutral'
            confidence = neutral_score
            
        else:
            if ent_score < 0.6 and con_score < 0.6:
                label = 'neutral'
                confidence = neutral_score
            elif ent_score > con_score:
                label = 'entailment'
                confidence = ent_score
            else:
                label = 'contradiction'
                confidence = con_score
        
        return {
            'label': label,
            'confidence': float(confidence),
            'scores': scores,
            'entailment_details': entailment_result,
            'contradiction_details': contradiction_result
        }
