"""
Livnium V6 Layers: Corrected 3-Axis Manifold

Uses ONLY invariant signals discovered through reverse physics:
- Resonance (invariant)
- Cold Attraction (invariant)
- Curvature (perfect invariant)
- Divergence Sign (preserved)

New Layer 2: Opposition = Resonance × Sign(Divergence)
- Removes noise from divergence magnitude
- Uses only invariant signals
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class LayerState:
    """State passed between layers."""
    resonance: float = 0.0
    divergence: float = 0.0
    divergence_sign: float = 0.0  # NEW: Sign only (-1, 0, +1)
    opposition: float = 0.0  # NEW: Resonance × Sign(Divergence)
    cold_density: float = 0.0
    distance: float = 0.0
    curvature: float = 0.0
    cold_attraction: float = 0.0
    far_attraction: float = 0.0
    city_pull: float = 0.0
    label: str = 'neutral'
    basin_index: int = 2
    confidence: float = 0.5


class Layer0Resonance:
    """
    Layer 0: Resonance + Divergence (unchanged from v5)
    
    Computes:
    - Resonance: Raw geometric similarity
    - Divergence: Field divergence (sign is preserved, magnitude is noisy)
    """
    
    def compute(self, encoded_pair) -> Dict[str, float]:
        """Compute resonance and divergence."""
        resonance = encoded_pair.get_resonance()
        
        # Get word vectors for divergence computation
        premise_vecs, hypothesis_vecs = encoded_pair.get_word_vectors()
        
        # Compute divergence (from v5)
        divergence = self._compute_field_divergence(premise_vecs, hypothesis_vecs)
        
        return {
            'resonance': float(resonance),
            'divergence': float(divergence),
        }
    
    def _compute_field_divergence(self, premise_vecs, hypothesis_vecs):
        """Compute field divergence (same as v5 - full implementation)."""
        if not premise_vecs or not hypothesis_vecs:
            return 0.0
        
        # Normalize vectors
        p_vecs_norm = []
        h_vecs_norm = []
        for vec in premise_vecs:
            norm = np.linalg.norm(vec)
            if norm > 0:
                p_vecs_norm.append(vec / norm)
            else:
                p_vecs_norm.append(vec)
        for vec in hypothesis_vecs:
            norm = np.linalg.norm(vec)
            if norm > 0:
                h_vecs_norm.append(vec / norm)
            else:
                h_vecs_norm.append(vec)
        
        # Sequential divergence (main signal)
        divergence_signals = []
        min_len = min(len(p_vecs_norm), len(h_vecs_norm))
        for i in range(min_len):
            p_vec = p_vecs_norm[i]
            h_vec = h_vecs_norm[i]
            alignment = np.dot(p_vec, h_vec)
            
            # CALIBRATED: Proper divergence formula (from v5)
            equilibrium_threshold = 0.38
            base_divergence = equilibrium_threshold - alignment
            
            # Orthogonal component (repulsion)
            dot_prod = np.dot(p_vec, h_vec)
            norm_p_sq = np.dot(p_vec, p_vec)
            if norm_p_sq > 0:
                proj = (dot_prod / norm_p_sq) * p_vec
                ortho = h_vec - proj
                ortho_magnitude = np.linalg.norm(ortho)
            else:
                ortho_magnitude = 0.0
            
            # Add orthogonal component as repulsion boost
            if alignment < equilibrium_threshold:
                divergence_signal = base_divergence + ortho_magnitude * (equilibrium_threshold - alignment) * 0.5
            else:
                divergence_signal = base_divergence
            
            divergence_signals.append(divergence_signal)
        
        # Cross-word divergence (reduced weight)
        cross_signals = []
        for p_vec in p_vecs_norm:
            for h_vec in h_vecs_norm:
                alignment = np.dot(p_vec, h_vec)
                equilibrium_threshold = 0.38
                divergence_signal = (equilibrium_threshold - alignment) * 0.7
                cross_signals.append(divergence_signal)
        
        # Combine: 85% sequential + 15% cross-word
        if divergence_signals:
            seq_div = np.mean(divergence_signals)
        else:
            seq_div = 0.0
        
        if cross_signals:
            cross_div = np.mean(cross_signals)
        else:
            cross_div = 0.0
        
        total_divergence = 0.85 * seq_div + 0.15 * cross_div
        
        return float(np.clip(total_divergence, -1.0, 1.0))


class Layer1Curvature:
    """
    Layer 1: Curvature (unchanged from v5)
    
    Computes:
    - Cold density: How tightly words cluster
    - Distance: Geometric distance
    - Curvature: How meaning bends (perfect invariant)
    """
    
    def compute(self, layer0_output: Dict[str, float], encoded_pair) -> Dict[str, float]:
        """Compute curvature signals."""
        resonance = layer0_output['resonance']
        
        # Cold density (from v5)
        premise_vecs, hypothesis_vecs = encoded_pair.get_word_vectors()
        cold_density = self._compute_cold_density(premise_vecs, hypothesis_vecs, resonance)
        
        # Distance (from v5)
        distance = self._compute_distance(premise_vecs, hypothesis_vecs)
        
        # Curvature (perfect invariant - always 0.0 in current implementation)
        curvature = 0.0
        
        return {
            **layer0_output,
            'cold_density': float(cold_density),
            'distance': float(distance),
            'curvature': float(curvature),
        }
    
    def _compute_cold_density(self, premise_vecs, hypothesis_vecs, resonance):
        """Compute cold density (from v5)."""
        if not premise_vecs or not hypothesis_vecs:
            return 0.0
        
        densities = []
        for p_vec in premise_vecs:
            for h_vec in hypothesis_vecs:
                dot_prod = np.dot(p_vec, h_vec)
                norm_p = np.linalg.norm(p_vec)
                norm_h = np.linalg.norm(h_vec)
                if norm_p > 0 and norm_h > 0:
                    similarity = dot_prod / (norm_p * norm_h)
                    density = similarity * resonance
                    densities.append(density)
        
        return np.mean(densities) if densities else 0.0
    
    def _compute_distance(self, premise_vecs, hypothesis_vecs):
        """Compute geometric distance (from v5)."""
        if not premise_vecs or not hypothesis_vecs:
            return 1.0
        
        distances = []
        for p_vec in premise_vecs:
            for h_vec in hypothesis_vecs:
                diff = p_vec - h_vec
                dist = np.linalg.norm(diff)
                distances.append(dist)
        
        return np.mean(distances) if distances else 1.0


class Layer2Opposition:
    """
    Layer 2: Opposition Axis (NEW in v6)
    
    Computes: opposition = resonance * sign(divergence)
    
    This combines two invariant signals:
    - Resonance (stable)
    - Divergence sign (preserved)
    
    Removes noise from divergence magnitude.
    """
    
    def compute(self, layer1_output: Dict[str, float]) -> Dict[str, float]:
        """Compute opposition axis."""
        resonance = layer1_output['resonance']
        divergence = layer1_output['divergence']
        
        # Extract sign only (ignore noisy magnitude)
        divergence_sign = np.sign(divergence)
        
        # Opposition: resonance weighted by divergence direction
        opposition = resonance * divergence_sign
        
        return {
            **layer1_output,
            'divergence_sign': float(divergence_sign),
            'opposition': float(opposition),
        }


class Layer3Attraction:
    """
    Layer 3: Attractions (unchanged from v5)
    
    Computes:
    - Cold attraction: Semantic gravity (invariant)
    - Far attraction: Repulsion force
    - City pull: Neutral balance
    """
    
    def compute(self, layer2_output: Dict[str, float], encoded_pair) -> Dict[str, float]:
        """Compute attractions."""
        resonance = layer2_output['resonance']
        cold_density = layer2_output['cold_density']
        distance = layer2_output['distance']
        
        # Cold attraction (semantic gravity - invariant)
        cold_attraction = resonance * cold_density / (distance + 0.1)
        
        # Far attraction (repulsion)
        far_attraction = (1.0 - resonance) * distance
        
        # City pull (neutral balance)
        city_pull = 1.0 - abs(cold_attraction - far_attraction)
        
        return {
            **layer2_output,
            'cold_attraction': float(cold_attraction),
            'far_attraction': float(far_attraction),
            'city_pull': float(city_pull),
        }


class Layer4Decision:
    """
    Layer 4: Decision (SIMPLIFIED in v6)
    
    Uses ONLY invariant signals:
    - Opposition (resonance × divergence_sign)
    - Cold attraction (invariant)
    
    Ignores noisy signals:
    - Divergence magnitude
    - Force ratios
    """
    
    def __init__(self, debug_mode: bool = False, golden_label_hint: str = None, force_incorrect: bool = False):
        self.debug_mode = debug_mode
        self.golden_label_hint = golden_label_hint
        self.force_incorrect = force_incorrect  # NEW: Force incorrect label to see geometry's true prediction
        
        # Thresholds from canonical fingerprints (calibrated)
        # Opposition thresholds (from invariant analysis)
        self.opposition_c_threshold = 0.02  # Contradiction: positive opposition
        self.opposition_e_threshold = -0.02  # Entailment: negative opposition
        self.opposition_n_band = 0.05  # Neutral: near-zero opposition
        
        # Resonance thresholds (from invariant analysis)
        self.resonance_e_min = 0.50  # Entailment: high resonance
        self.resonance_n_min = 0.45  # Neutral: mid-range
        self.resonance_n_max = 0.70  # Neutral: mid-range
    
    def compute(self, layer3_output: Dict[str, float]) -> Dict[str, float]:
        """Make final classification using ONLY invariant signals."""
        opposition = layer3_output['opposition']
        resonance = layer3_output['resonance']
        cold_attraction = layer3_output['cold_attraction']
        far_attraction = layer3_output['far_attraction']
        
        # DEBUG MODE: Use golden label hint (or force incorrect)
        if self.debug_mode and self.golden_label_hint:
            label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
            gold_idx = label_map.get(self.golden_label_hint, 2)
            
            # FORCE INCORRECT MODE: Invert the label to see what geometry really thinks
            if self.force_incorrect:
                # Invert E↔C, keep N as-is
                incorrect_map = {0: 1, 1: 0, 2: 2}  # E→C, C→E, N→N
                basin_idx = incorrect_map[gold_idx]
                label = ['entailment', 'contradiction', 'neutral'][basin_idx]
            else:
                # Normal debug mode: use golden label
                basin_idx = gold_idx
                label = self.golden_label_hint
            
            # Set forces to match the label (correct or incorrect)
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
            
            confidence = max(cold_force, far_force, city_force)
            
            total_score = cold_force + far_force + city_force
            if total_score > 0:
                e_score = cold_force / total_score
                c_score = far_force / total_score
                n_score = city_force / total_score
            else:
                e_score = c_score = n_score = 0.33
            
            # Compute what geometry would have predicted (before forcing)
            geometry_prediction = self._compute_geometry_prediction(opposition, resonance, cold_attraction, far_attraction)
            
            return {
                **layer3_output,
                'label': label,  # Forced label (correct or incorrect)
                'basin_index': basin_idx,
                'confidence': float(np.clip(confidence, 0.0, 1.0)),
                'cold_force': cold_force,
                'far_force': far_force,
                'city_force': city_force,
                'e_score': e_score,
                'c_score': c_score,
                'n_score': n_score,
                'geometry_prediction': geometry_prediction,  # What geometry actually thinks
                'forced_label': label,  # What we forced
                'golden_label': self.golden_label_hint,  # Original golden label
            }
        
        # ========================================================================
        # PHYSICS-BASED DECISION USING ONLY INVARIANT SIGNALS
        # ========================================================================
        
        # Rule 1: Contradiction - Positive opposition (high resonance + positive div)
        if opposition > self.opposition_c_threshold:
            label = 'contradiction'
            basin_index = 1
            confidence = min(0.9, 0.5 + abs(opposition) * 0.5)
        
        # Rule 2: Entailment - Negative opposition AND high resonance
        elif opposition < self.opposition_e_threshold and resonance > self.resonance_e_min:
            label = 'entailment'
            basin_index = 0
            # Confidence based on opposition strength and resonance
            opp_strength = abs(opposition) / 0.5
            res_strength = (resonance - self.resonance_e_min) / 0.2
            confidence = min(0.9, 0.5 + (opp_strength + res_strength) * 0.25)
        
        # Rule 3: Neutral - Near-zero opposition (balanced)
        elif abs(opposition) < self.opposition_n_band:
            label = 'neutral'
            basin_index = 2
            # Additional check: resonance in mid-range
            if self.resonance_n_min < resonance < self.resonance_n_max:
                confidence = max(0.5, cold_attraction * 0.5)
            else:
                confidence = 0.5
        
        # Rule 4: Fallback - Use cold attraction (invariant)
        else:
            if cold_attraction > far_attraction * 1.2:
                label = 'entailment'
                basin_index = 0
                confidence = cold_attraction
            elif far_attraction > cold_attraction * 1.2:
                label = 'contradiction'
                basin_index = 1
                confidence = far_attraction
            else:
                label = 'neutral'
                basin_index = 2
                confidence = 0.5
        
        # Compute scores
        total_score = cold_attraction + far_attraction + layer3_output.get('city_pull', 0.33)
        if total_score > 0:
            e_score = cold_attraction / total_score
            c_score = far_attraction / total_score
            n_score = layer3_output.get('city_pull', 0.33) / total_score
        else:
            e_score = c_score = n_score = 0.33
        
        return {
            **layer3_output,
            'label': label,
            'basin_index': basin_index,
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'e_score': e_score,
            'c_score': c_score,
            'n_score': n_score,
        }
    
    def _compute_geometry_prediction(self, opposition, resonance, cold_attraction, far_attraction):
        """Compute what geometry would predict without forced labels."""
        # Rule 1: Contradiction - Positive opposition
        if opposition > self.opposition_c_threshold:
            return 'contradiction'
        
        # Rule 2: Entailment - Negative opposition AND high resonance
        elif opposition < self.opposition_e_threshold and resonance > self.resonance_e_min:
            return 'entailment'
        
        # Rule 3: Neutral - Near-zero opposition
        elif abs(opposition) < self.opposition_n_band:
            return 'neutral'
        
        # Rule 4: Fallback - Use cold attraction
        else:
            if cold_attraction > far_attraction * 1.2:
                return 'entailment'
            elif far_attraction > cold_attraction * 1.2:
                return 'contradiction'
            else:
                return 'neutral'

