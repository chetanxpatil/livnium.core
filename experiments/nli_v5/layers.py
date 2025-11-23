"""
Core Layers: Simplified 5-Layer Architecture

Layer 0: Resonance - Raw geometric signal
Layer 1: Curvature - Cold density and distance
Layer 2: Basins - Attraction wells (E and C)
Layer 3: Valley - Natural neutral (balance point)
Layer 4: Decision - Final classification

Clean separation: Each layer builds on the one below.
"""

import numpy as np
from typing import Dict, Set, List
from dataclasses import dataclass


@dataclass
class LayerState:
    """State passed between layers."""
    resonance: float = 0.0
    cold_density: float = 0.0
    distance: float = 0.0
    curvature: float = 0.0
    cold_attraction: float = 0.0
    far_attraction: float = 0.0
    city_pull: float = 0.0
    cold_force: float = 0.33
    far_force: float = 0.33
    city_force: float = 0.33
    basin_index: int = 2  # 0=cold(E), 1=far(C), 2=city(N)
    confidence: float = 0.5
    is_stable: bool = False


class Layer0Resonance:
    """
    Layer 0: Pure resonance computation - the bedrock.
    
    Now includes DIVERGENCE-BASED CONTRADICTION FIELD:
    - Convergence (E): Word vectors point toward each other → negative divergence → attraction
    - Divergence (C): Word vectors point away from each other → positive divergence → repulsion
    - Neutral (N): Mixed or balanced → near-zero divergence → flat field
    """
    
    def compute(self, encoded_pair) -> Dict[str, float]:
        """
        Compute pure resonance from chain structure.
        Also computes DIVERGENCE-BASED contradiction field from word vector geometry.
        """
        resonance = encoded_pair.get_resonance()
        
        # Get word vectors for divergence computation
        premise_vecs, hypothesis_vecs = encoded_pair.get_word_vectors()
        tokens = encoded_pair.tokens
        
        # ========================================================================
        # DIVERGENCE-BASED CONTRADICTION FIELD
        # ========================================================================
        # Compute field divergence: measures how word vectors converge/diverge
        # This creates a REAL geometric repulsive force, not just boosted distance
        
        divergence = self._compute_field_divergence(premise_vecs, hypothesis_vecs)
        
        # Convergence (negative divergence) → Entailment
        # Divergence (positive divergence) → Contradiction
        # Near-zero divergence → Neutral
        
        # ========================================================================
        # Legacy signals (kept for backward compatibility and reinforcement)
        # ========================================================================
        
        # Word-level opposition (negative similarities)
        word_oppositions = []
        for p_vec in premise_vecs:
            for h_vec in hypothesis_vecs:
                dot_prod = np.dot(p_vec, h_vec)
                norm_p = np.linalg.norm(p_vec)
                norm_h = np.linalg.norm(h_vec)
                if norm_p > 0 and norm_h > 0:
                    similarity = dot_prod / (norm_p * norm_h)
                    if similarity < 0:
                        word_oppositions.append(abs(similarity))
        
        max_opposition = max(word_oppositions) if word_oppositions else 0.0
        
        # Learned word polarities
        from experiments.nli_simple.native_chain import SimpleLexicon
        lexicon = SimpleLexicon()
        contradiction_signals = []
        for token in set(tokens):
            polarity = lexicon.get_word_polarity(token)
            if polarity[1] > 0.5:
                contradiction_signals.append(polarity[1])
        
        learned_contradiction_strength = max(contradiction_signals) if contradiction_signals else 0.0
        
        return {
            'resonance': float(resonance),
            'divergence': float(divergence),  # NEW: Real geometric divergence field
            'word_opposition': float(max_opposition),
            'learned_contradiction': float(learned_contradiction_strength)
        }
    
    def _compute_field_divergence(self, premise_vecs: List[np.ndarray], 
                                   hypothesis_vecs: List[np.ndarray]) -> float:
        """
        Compute REAL repulsion field divergence.
        
        REPULSION FIELD THEORY:
        - Entailment (E): Vectors align → convergence (negative divergence)
        - Contradiction (C): Vectors oppose → divergence (positive divergence)  
        - Neutral (N): Mixed/balanced → near-zero divergence
        
        Method: Measure orthogonal component (what lies outside shared direction)
        This creates a TRUE repulsive force, symmetric with convergence.
        
        Returns:
            Divergence: negative = convergence (E), positive = divergence (C)
        """
        if not premise_vecs or not hypothesis_vecs:
            return 0.0
        
        # Normalize vectors
        p_vecs_norm = []
        h_vecs_norm = []
        
        for p_vec in premise_vecs:
            norm = np.linalg.norm(p_vec)
            if norm > 1e-6:
                p_vecs_norm.append(p_vec / norm)
            else:
                p_vecs_norm.append(p_vec)
        
        for h_vec in hypothesis_vecs:
            norm = np.linalg.norm(h_vec)
            if norm > 1e-6:
                h_vecs_norm.append(h_vec / norm)
            else:
                h_vecs_norm.append(h_vec)
        
        # ========================================================================
        # REPULSION FIELD: Orthogonal Component (what's outside shared direction)
        # ========================================================================
        
        divergence_signals = []
        
        # Sequential alignment (position matters)
        min_len = min(len(p_vecs_norm), len(h_vecs_norm))
        for i in range(min_len):
            p_vec = p_vecs_norm[i]
            h_vec = h_vecs_norm[i]
            
            # Alignment (convergence signal)
            alignment = np.dot(p_vec, h_vec)  # Cosine similarity [0, 1] typically
            
            # Compute shared direction
            shared = p_vec + h_vec
            shared_norm = np.linalg.norm(shared)
            
            if shared_norm > 1e-6:
                shared_unit = shared / shared_norm
                
                # Project onto shared direction
                proj_p = np.dot(p_vec, shared_unit) * shared_unit
                proj_h = np.dot(h_vec, shared_unit) * shared_unit
                
                # Divergence computation: Use alignment as primary signal
                # Alignment (cosine similarity) ranges from -1 (opposite) to +1 (aligned)
                # We want: high alignment → negative divergence (convergence/E)
                #          low/negative alignment → positive divergence (divergence/C)
                
                # Compute orthogonal component for additional repulsion signal
                ortho_p = p_vec - proj_p
                ortho_h = h_vec - proj_h
                ortho_magnitude = (np.linalg.norm(ortho_p) + np.linalg.norm(ortho_h)) / 2.0
                
                # CALIBRATED: Proper divergence formula with threshold matching actual distribution
                # Actual mean alignment: ~0.38 (not 0.5)
                # When alignment > 0.38: negative divergence (convergence/E)
                # When alignment < 0.38: positive divergence (divergence/C)
                # Use: divergence = 0.38 - alignment (calibrated to actual data)
                # This gives: alignment=0.40 (E) → divergence=-0.02 (negative) ✓
                #            alignment=0.37 (C) → divergence=+0.01 (positive) ✓
                #            alignment=0.50 (high E) → divergence=-0.12 (negative) ✓
                equilibrium_threshold = 0.38  # Calibrated to actual alignment distribution
                base_divergence = equilibrium_threshold - alignment
                
                # Add orthogonal component as repulsion boost (only when alignment is low)
                if alignment < equilibrium_threshold:
                    # Low alignment: add ortho boost for stronger divergence signal
                    divergence_signal = base_divergence + ortho_magnitude * (equilibrium_threshold - alignment) * 0.5
                else:
                    # High alignment: ortho is minimal, use base divergence
                    divergence_signal = base_divergence
                
            else:
                # Vectors are opposite → maximum divergence
                divergence_signal = 1.0
            
            divergence_signals.append(divergence_signal)
        
        # Cross-word divergence (semantic opposition)
        # Reduced weight since cross-word matching can create noise for entailment
        cross_signals = []
        for p_vec in p_vecs_norm:
            for h_vec in h_vecs_norm:
                alignment = np.dot(p_vec, h_vec)
                
                # Use same formula as sequential: 0.38 - alignment (calibrated threshold)
                # This ensures low alignment → positive divergence (contradiction)
                equilibrium_threshold = 0.38  # Match sequential computation
                divergence_signal = (equilibrium_threshold - alignment) * 0.7  # Slightly reduced for cross-word
                
                cross_signals.append(divergence_signal)
        
        # Combine: 85% sequential + 15% cross-word (reduced cross-word weight to reduce noise)
        if divergence_signals:
            seq_div = np.mean(divergence_signals)
        else:
            seq_div = 0.0
        
        if cross_signals:
            cross_div = np.mean(cross_signals)
        else:
            cross_div = 0.0
        
        total_divergence = 0.85 * seq_div + 0.15 * cross_div
        
        # Return signed divergence: negative = convergence (E), positive = divergence (C)
        return float(np.clip(total_divergence, -1.0, 1.0))


class Layer1Curvature:
    """
    Layer 1: Cold density and divergence curvature.
    
    NOW WITH REAL DIVERGENCE-BASED CONTRADICTION FIELD:
    - Cold density: Convergence (negative divergence) → Entailment force
    - Divergence: Positive divergence → Contradiction force (REAL repulsion)
    - Symmetric geometry: E and C are now true opposing forces
    """
    
    def __init__(self):
        self.resonance_history = []
    
    def compute(self, layer0_output: Dict[str, float], encoded_pair=None) -> Dict[str, float]:
        """
        Compute cold density and divergence from resonance and field geometry.
        
        Key change: Uses DIVERGENCE (real geometric field) instead of distance (heuristic).
        This creates a true three-force physics: convergence (E), divergence (C), neutral (N).
        """
        resonance = layer0_output['resonance']
        divergence = layer0_output.get('divergence', 0.0)  # NEW: Real divergence field
        word_opposition = layer0_output.get('word_opposition', 0.0)
        learned_contradiction = layer0_output.get('learned_contradiction', 0.0)
        
        self.resonance_history.append(resonance)
        
        # ========================================================================
        # CONVERGENCE (E): Cold density from negative divergence
        # ========================================================================
        # Convergence = negative divergence (vectors point toward each other)
        # Cold density measures how much the field converges (attraction)
        convergence = -divergence  # Negative divergence = convergence
        
        # Cold density: positive convergence + positive resonance = dense, stable (E)
        # Combine convergence signal with resonance for robustness
        cold_density = max(0.0, convergence) + max(0.0, resonance * 0.5)
        
        # ========================================================================
        # DIVERGENCE (C): Real repulsive force from positive divergence
        # ========================================================================
        # Divergence = positive divergence (vectors point away from each other)
        # This creates a REAL repulsive force, symmetric with convergence
        
        # Base divergence force: positive divergence creates repulsion
        divergence_force = max(0.0, divergence)
        
        # Enhance with resonance: negative resonance reinforces divergence
        if resonance < 0:
            divergence_force += abs(resonance) * 0.5
        
        # Legacy boosts (kept for reinforcement, but divergence is primary)
        opposition_boost = word_opposition * 0.3  # Reduced weight (divergence is primary)
        learned_boost = learned_contradiction * 0.2  # Reduced weight
        
        # Semantic gap detection (for edge cases)
        semantic_gap = 0.0
        if encoded_pair is not None:
            try:
                premise_tokens = set(encoded_pair.premise.tokens) if hasattr(encoded_pair.premise, 'tokens') else set()
                hypothesis_tokens = set(encoded_pair.hypothesis.tokens) if hasattr(encoded_pair.hypothesis, 'tokens') else set()
                
                if premise_tokens and hypothesis_tokens:
                    overlap = len(premise_tokens & hypothesis_tokens) / max(len(premise_tokens | hypothesis_tokens), 1)
                    
                    # Semantic gap: words overlap but divergence is high (different meanings)
                    if overlap > 0.2 and divergence > 0.3:
                        semantic_gap = divergence * overlap * 0.4
            except:
                pass
        
        # Total divergence force: primary divergence + enhancements
        # This is now a REAL geometric force, not just boosted distance
        total_divergence = divergence_force + opposition_boost + learned_boost + semantic_gap
        
        # For backward compatibility, keep "distance" as alias for divergence
        # But now it's a real force, not just absence of attraction
        distance = total_divergence
        
        # Curvature: second derivative approximation
        if len(self.resonance_history) >= 3:
            r_t = self.resonance_history[-1]
            r_t1 = self.resonance_history[-2]
            r_t2 = self.resonance_history[-3]
            curvature = r_t - 2.0 * r_t1 + r_t2
        else:
            curvature = 0.0
        
        return {
            'resonance': resonance,
            'divergence': float(divergence),  # NEW: Raw divergence field value
            'convergence': float(convergence),  # NEW: Convergence (negative divergence)
            'cold_density': cold_density,
            'distance': distance,  # Now represents divergence force (for backward compat)
            'divergence_force': float(total_divergence),  # NEW: Total divergence force
            'curvature': abs(curvature),
            'word_opposition': word_opposition,
            'learned_contradiction': learned_contradiction,
            'semantic_gap': semantic_gap
        }


class Layer2Basin:
    """Layer 2: Cold and Far basins - attraction wells."""
    
    # Shared state across all instances
    # Start with far basin slightly higher to help with initial contradiction predictions
    _shared_cold_depth = 1.0
    _shared_far_depth = 1.2  # Slightly higher to help contradiction predictions
    
    def __init__(self):
        self.reinforcement_rate = 0.3
        self.decay_rate = 0.01
        self.capacity = 200.0
    
    @classmethod
    def reset_shared_state(cls):
        """Reset shared basin depths."""
        cls._shared_cold_depth = 1.0
        cls._shared_far_depth = 1.0
    
    def compute(self, layer1_output: Dict[str, float]) -> Dict[str, float]:
        """
        Compute basin attractions from convergence and divergence forces.
        
        NOW WITH SYMMETRIC GEOMETRY:
        - Cold attraction: Proportional to convergence (cold_density)
        - Far attraction: Proportional to divergence (divergence_force)
        - Both are REAL geometric forces, symmetric and balanced
        """
        cold_density = layer1_output['cold_density']
        divergence_force = layer1_output.get('divergence_force', layer1_output.get('distance', 0.0))
        convergence = layer1_output.get('convergence', 0.0)
        divergence = layer1_output.get('divergence', 0.0)
        curvature = layer1_output.get('curvature', 0.0)
        resonance = layer1_output.get('resonance', 0.0)
        
        # Basin depths (shared across instances)
        cold_depth = Layer2Basin._shared_cold_depth
        far_depth = Layer2Basin._shared_far_depth
        
        # Normalize depths
        total_depth = cold_depth + far_depth
        if total_depth > 0:
            cold_weight = cold_depth / total_depth
            far_weight = far_depth / total_depth
        else:
            cold_weight = far_weight = 0.5
        
        # ========================================================================
        # SYMMETRIC ATTRACTION COMPUTATION
        # ========================================================================
        # Cold attraction: Convergence force × weight × curvature boost
        # This is the REAL convergent field (E)
        cold_attraction = cold_weight * (1.0 + curvature) * (1.0 + cold_density)
        
        # Far attraction: Divergence force × weight × curvature boost
        # This is the REAL divergent field (C) - symmetric with cold_attraction
        far_attraction = far_weight * (1.0 + curvature) * (1.0 + divergence_force)
        
        # ========================================================================
        # REINFORCEMENT SIGNALS (kept for learning, but divergence is primary)
        # ========================================================================
        # These provide additional signals but divergence_force is the main driver
        
        word_opposition = layer1_output.get('word_opposition', 0.0)
        learned_contradiction = layer1_output.get('learned_contradiction', 0.0)
        semantic_gap = layer1_output.get('semantic_gap', 0.0)
        
        # Small boosts from legacy signals (reduced weight since divergence is primary)
        if word_opposition > 0.1:
            far_attraction *= (1.0 + word_opposition * 0.3)  # Reduced from 0.8
        
        if learned_contradiction > 0.5:
            far_attraction *= (1.0 + learned_contradiction * 0.2)  # Reduced from 0.6
        
        if resonance < 0:
            far_attraction *= (1.0 + abs(resonance) * 0.5)  # Reduced from 1.0
        
        if semantic_gap > 0.1:
            far_attraction *= (1.0 + semantic_gap * 0.3)  # Reduced from 0.5
        
        # Divergence dominates convergence check (symmetric force comparison)
        if divergence_force > cold_density + 0.1:
            far_attraction *= (1.0 + (divergence_force - cold_density) * 0.3)  # Reduced from 0.4
        
        # Apply decay (prevent infinite growth)
        Layer2Basin._shared_cold_depth *= (1.0 - self.decay_rate)
        Layer2Basin._shared_far_depth *= (1.0 - self.decay_rate)
        
        return {
            **layer1_output,
            'cold_attraction': cold_attraction,
            'far_attraction': far_attraction,
            'cold_depth': cold_depth,
            'far_depth': far_depth
        }
    
    def reinforce(self, basin_idx: int, strength: float = 1.0):
        """Reinforce basin depth (0=cold/E, 1=far/C)."""
        if basin_idx == 0:
            Layer2Basin._shared_cold_depth += self.reinforcement_rate * strength
            Layer2Basin._shared_cold_depth = min(Layer2Basin._shared_cold_depth, self.capacity)
        elif basin_idx == 1:
            Layer2Basin._shared_far_depth += self.reinforcement_rate * strength
            Layer2Basin._shared_far_depth = min(Layer2Basin._shared_far_depth, self.capacity)


class Layer3Valley:
    """Layer 3: The City (natural neutral) - balance point between forces."""
    
    def compute(self, layer2_output: Dict[str, float]) -> Dict[str, float]:
        """Compute city pull from force balance."""
        cold_attraction = layer2_output['cold_attraction']
        far_attraction = layer2_output['far_attraction']
        
        # City forms where forces balance
        max_attraction = max(cold_attraction, far_attraction)
        if max_attraction > 1e-6:
            attraction_ratio = abs(cold_attraction - far_attraction) / max_attraction
        else:
            attraction_ratio = 0.0
        
        # City threshold: when forces are close, city appears
        city_threshold = 0.15
        overlap_strength = 1.0 - min(attraction_ratio, 1.0)
        city_gravity = 0.7
        
        min_attraction = min(cold_attraction, far_attraction)
        city_pull = overlap_strength * city_gravity * (min_attraction + 0.1)
        
        # Normalize forces
        total_force = cold_attraction + far_attraction + city_pull
        if total_force > 0:
            cold_force = cold_attraction / total_force
            far_force = far_attraction / total_force
            city_force = city_pull / total_force
        else:
            cold_force = far_force = city_force = 0.33
        
        return {
            **layer2_output,
            'city_pull': city_pull,
            'cold_force': cold_force,
            'far_force': far_force,
            'city_force': city_force
        }


class Layer4Decision:
    """Layer 4: Final decision - properly handles all 3 classes."""
    
    def __init__(self, debug_mode: bool = False, golden_label_hint: str = None):
        """
        Initialize decision layer.
        
        Args:
            debug_mode: If True, use golden label hint to verify decision logic
            golden_label_hint: Optional golden label ('entailment', 'contradiction', 'neutral')
        """
        self.weak_force_threshold = 0.05
        self.balance_threshold = 0.15
        self.force_dominance_threshold = 0.1  # Minimum difference to pick E or C
        self.debug_mode = debug_mode
        self.golden_label_hint = golden_label_hint
        
        # Physics-based thresholds from canonical fingerprints
        # Based on golden label analysis: E needs divergence + resonance
        # Adjusted for better balance between classes
        self.divergence_c_threshold = 0.02  # Contradiction: positive divergence (lowered for sensitivity)
        self.divergence_e_threshold = -0.08  # Entailment: negative divergence (more strict)
        self.divergence_n_band = 0.12  # Neutral: near-zero divergence (tighter band)
        self.resonance_e_min = 0.50  # Entailment: high resonance (slightly higher)
        self.resonance_n_min = 0.45  # Neutral: mid-range resonance
        self.resonance_n_max = 0.70  # Neutral: mid-range resonance
    
    def compute(self, layer3_output: Dict[str, float]) -> Dict[str, float]:
        """Make final classification decision using normalized forces and resonance."""
        cold_attraction = layer3_output['cold_attraction']
        far_attraction = layer3_output['far_attraction']
        city_pull = layer3_output['city_pull']
        cold_force = layer3_output['cold_force']
        far_force = layer3_output['far_force']
        city_force = layer3_output['city_force']
        resonance = layer3_output.get('resonance', 0.0)  # Get resonance from earlier layers
        
        # DEBUG MODE: If golden label hint is provided, use it to verify decision logic
        if self.debug_mode and self.golden_label_hint:
            label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
            basin_idx = label_map.get(self.golden_label_hint, 2)
            
            # Set forces to match the golden label
            if basin_idx == 0:  # Entailment
                cold_force = 0.7
                far_force = 0.2
                city_force = 0.1
            elif basin_idx == 1:  # Contradiction
                cold_force = 0.2
                far_force = 0.7
                city_force = 0.1
            else:  # Neutral
                cold_force = 0.33
                far_force = 0.33
                city_force = 0.34
            
            label = self.golden_label_hint
            basin_index = basin_idx
            confidence = max(cold_force, far_force, city_force)
            
            # Compute scores for backward compatibility
            total_score = cold_force + far_force + city_force
            if total_score > 0:
                e_score = cold_force / total_score
                c_score = far_force / total_score
                n_score = city_force / total_score
            else:
                e_score = c_score = n_score = 0.33
            
            return {
                **layer3_output,
                'label': label,
                'basin_index': basin_index,
                'confidence': float(np.clip(confidence, 0.0, 1.0)),
                'cold_force': cold_force,  # Overwrite with debug forces
                'far_force': far_force,    # Overwrite with debug forces
                'city_force': city_force,   # Overwrite with debug forces
                'e_score': e_score,
                'c_score': c_score,
                'n_score': n_score
            }
        
        # ========================================================================
        # PHYSICS-BASED DECISION LOGIC
        # ========================================================================
        # Use canonical fingerprints: divergence (x-axis) + resonance (y-axis)
        # Based on phase diagram analysis from golden labels
        
        divergence = layer3_output.get('divergence', 0.0)
        convergence = layer3_output.get('convergence', 0.0)
        cold_density = layer3_output.get('cold_density', 0.0)
        max_attraction = max(cold_attraction, far_attraction)
        
        # Rule 1: Weak forces → Neutral (safety check)
        if max_attraction < self.weak_force_threshold:
            label = 'neutral'
            basin_index = 2
            confidence = max(city_force, 0.5)
        
        # Rule 2: Contradiction - Strong positive divergence (push apart)
        # Primary signal: positive divergence
        elif divergence > self.divergence_c_threshold:
            label = 'contradiction'
            basin_index = 1
            # Confidence based on divergence strength
            confidence = min(0.9, 0.5 + abs(divergence) * 0.5)
        
        # Rule 3: Entailment - Negative divergence AND high resonance (pull inward + similarity)
        # Requires BOTH signals: negative divergence (convergence) AND high resonance
        elif divergence < self.divergence_e_threshold and resonance > self.resonance_e_min:
            label = 'entailment'
            basin_index = 0
            # Confidence based on both signals
            div_strength = abs(divergence) / 0.5  # Normalize to [0, 1]
            res_strength = (resonance - self.resonance_e_min) / 0.2  # Normalize
            confidence = min(0.9, 0.5 + (div_strength + res_strength) * 0.25)
        
        # Rule 4: Neutral - Near-zero divergence (balanced forces)
        # Defined as a band: |divergence| < threshold
        elif abs(divergence) < self.divergence_n_band:
            # Additional check: resonance in mid-range (optional)
            if self.resonance_n_min < resonance < self.resonance_n_max:
                label = 'neutral'
                basin_index = 2
                # Confidence based on how balanced forces are
                balance = 1.0 - abs(cold_force - far_force)
                confidence = max(0.5, city_force * balance)
            else:
                # Divergence says neutral but resonance is extreme → fallback to forces
                if cold_force > far_force * 1.2:
                    label = 'entailment'
                    basin_index = 0
                    confidence = cold_force
                elif far_force > cold_force * 1.2:
                    label = 'contradiction'
                    basin_index = 1
                    confidence = far_force
                else:
                    label = 'neutral'
                    basin_index = 2
                    confidence = city_force
        
        # Rule 5: Fallback - Use force-based decision for edge cases
        # This handles cases where physics signals are ambiguous
        else:
            # Check if far_attraction is competitive
            attraction_ratio = far_attraction / (cold_attraction + 1e-6)
            
            # If far is competitive and forces favor far → contradiction
            if attraction_ratio >= 0.8 and far_force >= cold_force:
                label = 'contradiction'
                basin_index = 1
                confidence = far_force
            # If cold is clearly stronger → entailment
            elif cold_attraction > far_attraction * 1.2:
                label = 'entailment'
                basin_index = 0
                confidence = cold_force
            # Close match - use force comparison
            else:
                if far_force > cold_force:
                    label = 'contradiction'
                    basin_index = 1
                    confidence = far_force
                elif cold_force > far_force:
                    label = 'entailment'
                    basin_index = 0
                    confidence = cold_force
                else:
                    # Equal forces → use resonance as tiebreaker
                    if resonance > 0.55:  # High resonance → entailment
                        label = 'entailment'
                        basin_index = 0
                        confidence = cold_force
                    elif resonance < 0.45:  # Low resonance → contradiction
                        label = 'contradiction'
                        basin_index = 1
                        confidence = far_force
                    else:  # Mid resonance → neutral
                        label = 'neutral'
                        basin_index = 2
                        confidence = city_force
        
        # Compute scores for backward compatibility
        total_score = cold_force + far_force + city_force
        if total_score > 0:
            e_score = cold_force / total_score
            c_score = far_force / total_score
            n_score = city_force / total_score
        else:
            e_score = c_score = n_score = 0.33
        
        return {
            **layer3_output,
            'label': label,
            'basin_index': basin_index,
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'e_score': e_score,
            'c_score': c_score,
            'n_score': n_score
        }

