"""
Livnium V7 Layers: Geometry Shaping

Key principle: Train ONLY geometry, never Layer 4.

Layer 4 is passive - it only observes.
Layers 0-3 shape the manifold through physics reinforcement.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class GeometryState:
    """State that gets reinforced through physics."""
    resonance: float = 0.0
    divergence: float = 0.0
    divergence_sign: float = 0.0
    opposition: float = 0.0
    cold_density: float = 0.0
    distance: float = 0.0
    curvature: float = 0.0
    cold_attraction: float = 0.0
    far_attraction: float = 0.0
    city_pull: float = 0.0


class Layer0Resonance:
    """
    Layer 0: Resonance + Divergence (WITH physics reinforcement)
    
    Learns:
    - Divergence threshold (equilibrium point)
    - Resonance scaling
    """
    
    def __init__(self):
        # Learnable parameters (shaped by physics)
        self.equilibrium_threshold = 0.38  # Starting point (from v5 calibration)
        self.resonance_scale = 1.0  # Resonance amplification factor
        
        # Physics reinforcement history
        self.divergence_history = []
        self.resonance_history = []
    
    def compute(self, encoded_pair) -> Dict[str, float]:
        """Compute resonance and divergence."""
        resonance = encoded_pair.get_resonance()
        resonance_scaled = resonance * self.resonance_scale
        
        premise_vecs, hypothesis_vecs = encoded_pair.get_word_vectors()
        divergence = self._compute_field_divergence(premise_vecs, hypothesis_vecs)
        
        # Store for reinforcement
        self.resonance_history.append(resonance_scaled)
        self.divergence_history.append(divergence)
        
        return {
            'resonance': float(resonance_scaled),
            'divergence': float(divergence),
        }
    
    def reinforce(self, label: str, strength: float = 0.01):
        """
        Physics reinforcement: Shape the manifold based on correct examples.
        
        Args:
            label: 'entailment', 'contradiction', or 'neutral'
            strength: Reinforcement strength (small, continuous updates)
        """
        if not self.divergence_history:
            return
        
        last_divergence = self.divergence_history[-1]
        last_resonance = self.resonance_history[-1] if self.resonance_history else 0.0
        
        if label == 'entailment':
            # Deepen inward basin: make negative divergence more negative
            if last_divergence < 0:
                # Basin is correct - deepen it slightly
                self.equilibrium_threshold += strength * 0.01  # Shift threshold up slightly
                self.resonance_scale += strength * 0.02  # Amplify resonance
        elif label == 'contradiction':
            # Amplify outward push: make positive divergence more positive
            if last_divergence > 0:
                # Push is correct - amplify it
                self.equilibrium_threshold -= strength * 0.01  # Shift threshold down slightly
        elif label == 'neutral':
            # Enforce equilibrium: push divergence toward zero
            if abs(last_divergence) > 0.05:
                # Too far from equilibrium - nudge back
                if last_divergence > 0:
                    self.equilibrium_threshold += strength * 0.01
                else:
                    self.equilibrium_threshold -= strength * 0.01
        
        # Clip thresholds to reasonable ranges
        self.equilibrium_threshold = np.clip(self.equilibrium_threshold, 0.2, 0.6)
        self.resonance_scale = np.clip(self.resonance_scale, 0.5, 2.0)
    
    def _compute_field_divergence(self, premise_vecs, hypothesis_vecs):
        """Compute field divergence (same as v6, but uses learnable threshold)."""
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
        
        # Sequential divergence
        divergence_signals = []
        min_len = min(len(p_vecs_norm), len(h_vecs_norm))
        for i in range(min_len):
            p_vec = p_vecs_norm[i]
            h_vec = h_vecs_norm[i]
            alignment = np.dot(p_vec, h_vec)
            
            # Use LEARNABLE threshold (shaped by physics)
            base_divergence = self.equilibrium_threshold - alignment
            
            # Orthogonal component
            dot_prod = np.dot(p_vec, h_vec)
            norm_p_sq = np.dot(p_vec, p_vec)
            if norm_p_sq > 0:
                proj = (dot_prod / norm_p_sq) * p_vec
                ortho = h_vec - proj
                ortho_magnitude = np.linalg.norm(ortho)
            else:
                ortho_magnitude = 0.0
            
            if alignment < self.equilibrium_threshold:
                divergence_signal = base_divergence + ortho_magnitude * (self.equilibrium_threshold - alignment) * 0.5
            else:
                divergence_signal = base_divergence
            
            divergence_signals.append(divergence_signal)
        
        # Cross-word divergence
        cross_signals = []
        for p_vec in p_vecs_norm:
            for h_vec in h_vecs_norm:
                alignment = np.dot(p_vec, h_vec)
                divergence_signal = (self.equilibrium_threshold - alignment) * 0.7
                cross_signals.append(divergence_signal)
        
        # Combine
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
    Layer 1: Curvature (WITH physics reinforcement)
    
    Learns:
    - Cold density formula
    - Distance scaling
    """
    
    def __init__(self):
        self.cold_density_scale = 1.0
        self.distance_scale = 1.0
    
    def compute(self, layer0_output: Dict[str, float], encoded_pair) -> Dict[str, float]:
        """Compute curvature signals."""
        resonance = layer0_output['resonance']
        
        premise_vecs, hypothesis_vecs = encoded_pair.get_word_vectors()
        cold_density = self._compute_cold_density(premise_vecs, hypothesis_vecs, resonance)
        distance = self._compute_distance(premise_vecs, hypothesis_vecs)
        
        return {
            **layer0_output,
            'cold_density': float(cold_density * self.cold_density_scale),
            'distance': float(distance * self.distance_scale),
            'curvature': 0.0,  # Perfect invariant
        }
    
    def reinforce(self, label: str, strength: float = 0.01):
        """Physics reinforcement: Shape cold density and distance."""
        if label == 'entailment':
            # Deepen cold density (inward basin)
            self.cold_density_scale += strength * 0.02
        elif label == 'contradiction':
            # Amplify distance (outward push)
            self.distance_scale += strength * 0.02
        
        # Clip to reasonable ranges
        self.cold_density_scale = np.clip(self.cold_density_scale, 0.5, 2.0)
        self.distance_scale = np.clip(self.distance_scale, 0.5, 2.0)
    
    def _compute_cold_density(self, premise_vecs, hypothesis_vecs, resonance):
        """Compute cold density."""
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
        """Compute geometric distance."""
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
    Layer 2: Opposition (WITH physics reinforcement)
    
    Learns:
    - Opposition scaling
    """
    
    def __init__(self):
        self.opposition_scale = 1.0
    
    def compute(self, layer1_output: Dict[str, float]) -> Dict[str, float]:
        """Compute opposition axis."""
        resonance = layer1_output['resonance']
        divergence = layer1_output['divergence']
        
        divergence_sign = np.sign(divergence)
        opposition = resonance * divergence_sign * self.opposition_scale
        
        return {
            **layer1_output,
            'divergence_sign': float(divergence_sign),
            'opposition': float(opposition),
        }
    
    def reinforce(self, label: str, strength: float = 0.01):
        """Physics reinforcement: Amplify opposition signal."""
        if label in ['entailment', 'contradiction']:
            # Amplify opposition for E/C
            self.opposition_scale += strength * 0.02
        elif label == 'neutral':
            # Reduce opposition for N (enforce equilibrium)
            self.opposition_scale -= strength * 0.01
        
        self.opposition_scale = np.clip(self.opposition_scale, 0.5, 2.0)


class Layer3Attraction:
    """
    Layer 3: Attractions (WITH physics reinforcement)
    
    Learns:
    - Cold attraction formula
    - Far attraction decay
    """
    
    def __init__(self):
        self.cold_attraction_scale = 1.0
        self.far_attraction_scale = 1.0
    
    def compute(self, layer2_output: Dict[str, float], encoded_pair) -> Dict[str, float]:
        """Compute attractions."""
        resonance = layer2_output['resonance']
        cold_density = layer2_output['cold_density']
        distance = layer2_output['distance']
        
        # Cold attraction (semantic gravity - invariant)
        cold_attraction = resonance * cold_density / (distance + 0.1) * self.cold_attraction_scale
        
        # Far attraction (repulsion)
        far_attraction = (1.0 - resonance) * distance * self.far_attraction_scale
        
        # City pull (neutral balance)
        city_pull = 1.0 - abs(cold_attraction - far_attraction)
        
        return {
            **layer2_output,
            'cold_attraction': float(cold_attraction),
            'far_attraction': float(far_attraction),
            'city_pull': float(city_pull),
        }
    
    def reinforce(self, label: str, strength: float = 0.01):
        """Physics reinforcement: Shape attractions."""
        if label == 'entailment':
            # Deepen cold attraction (inward basin)
            self.cold_attraction_scale += strength * 0.02
        elif label == 'contradiction':
            # Amplify far attraction (outward push)
            self.far_attraction_scale += strength * 0.02
        elif label == 'neutral':
            # Enforce balance
            # Nudge both toward equilibrium
            if self.cold_attraction_scale > self.far_attraction_scale:
                self.cold_attraction_scale -= strength * 0.01
            else:
                self.far_attraction_scale -= strength * 0.01
        
        self.cold_attraction_scale = np.clip(self.cold_attraction_scale, 0.5, 2.0)
        self.far_attraction_scale = np.clip(self.far_attraction_scale, 0.5, 2.0)


class Layer4Decision:
    """
    Layer 4: Decision (PASSIVE - NO LEARNING)
    
    Only observes geometry.
    Never learns.
    Pure physics-based rules.
    """
    
    def __init__(self):
        # Fixed thresholds (from canonical fingerprints)
        self.opposition_c_threshold = 0.02
        self.opposition_e_threshold = -0.02
        self.opposition_n_band = 0.05
        self.resonance_e_min = 0.50
        self.resonance_n_min = 0.45
        self.resonance_n_max = 0.70
    
    def compute(self, layer3_output: Dict[str, float]) -> Dict[str, float]:
        """Make final classification using ONLY invariant signals."""
        opposition = layer3_output['opposition']
        resonance = layer3_output['resonance']
        cold_attraction = layer3_output['cold_attraction']
        far_attraction = layer3_output['far_attraction']
        
        # Rule 1: Contradiction - Positive opposition
        if opposition > self.opposition_c_threshold:
            label = 'contradiction'
            basin_index = 1
            confidence = min(0.9, 0.5 + abs(opposition) * 0.5)
        
        # Rule 2: Entailment - Negative opposition AND high resonance
        elif opposition < self.opposition_e_threshold and resonance > self.resonance_e_min:
            label = 'entailment'
            basin_index = 0
            opp_strength = abs(opposition) / 0.5
            res_strength = (resonance - self.resonance_e_min) / 0.2
            confidence = min(0.9, 0.5 + (opp_strength + res_strength) * 0.25)
        
        # Rule 3: Neutral - Near-zero opposition
        elif abs(opposition) < self.opposition_n_band:
            label = 'neutral'
            basin_index = 2
            if self.resonance_n_min < resonance < self.resonance_n_max:
                confidence = max(0.5, cold_attraction * 0.5)
            else:
                confidence = 0.5
        
        # Rule 4: Fallback - Use cold attraction
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

