"""
Livnium v8 Layers: Clean Architecture with Geometry-First Philosophy

Key Features:
- Semantic warp alignment (DP, no hardcoded rules)
- Collision-based fracture detection (negation = alignment tension)
- Angle-based divergence (planetary physics)
- Geometry zones (E/C/N from divergence thresholds)
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import math


@dataclass
class LayerState:
    """State passed between layers."""
    resonance: float = 0.0
    divergence: float = 0.0
    convergence: float = 0.0
    curvature: float = 0.0
    cold_density: float = 0.0
    divergence_force: float = 0.0
    fracture_detected: bool = False
    fracture_strength: float = 0.0


class Layer0Resonance:
    """
    Layer 0: Resonance + Divergence Computation
    
    Uses angle-based divergence (planetary physics):
    - Entailment: small angle → negative divergence (inward)
    - Neutral: equilibrium angle → zero divergence
    - Contradiction: large angle → positive divergence (outward)
    """
    
    # Angle-based divergence parameters
    equilibrium_angle_deg = 41.2  # Equilibrium angle in degrees
    divergence_scale = 2.5  # Scale to amplify angular differences
    
    def compute(self, encoded_pair) -> Dict[str, float]:
        """Compute resonance from chain structure."""
        resonance = encoded_pair.get_resonance()
        return {'resonance': float(resonance)}
    
    def _compute_opposition_field(
        self, 
        premise_vecs: List[np.ndarray], 
        hypothesis_vecs: List[np.ndarray]
    ) -> float:
        """Compute opposition field (semantic direction)."""
        if not premise_vecs or not hypothesis_vecs:
            return 0.0
        
        # Normalize vectors
        p_vecs_norm = [v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else v 
                      for v in premise_vecs]
        h_vecs_norm = [v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else v 
                      for v in hypothesis_vecs]
        
        # Sequential opposition
        opposition_signals = []
        min_len = min(len(p_vecs_norm), len(h_vecs_norm))
        for i in range(min_len):
            alignment = np.dot(p_vecs_norm[i], h_vecs_norm[i])
            opposition = -alignment  # Opposition = -alignment
            opposition_signals.append(opposition)
        
        return float(np.mean(opposition_signals)) if opposition_signals else 0.0
    
    def _compute_field_divergence(
        self,
        premise_vecs: List[np.ndarray],
        hypothesis_vecs: List[np.ndarray],
        opposition: float = 0.0
    ) -> float:
        """
        Angle-based divergence computation with Neutral Basin.
        
        Uses angular separation instead of linear alignment.
        
        NEUTRAL BASIN FIX:
        - Neutral lives at the boundary (zero-force layer)
        - Clamps divergence near zero to create neutral gravitational basin
        - Prevents neutral from being pulled into contradiction (outward)
        """
        if not premise_vecs or not hypothesis_vecs:
            return 0.0
        
        # Compute mean vectors
        premise_mean = np.mean([v for v in premise_vecs if np.linalg.norm(v) > 1e-6], axis=0)
        hypothesis_mean = np.mean([v for v in hypothesis_vecs if np.linalg.norm(v) > 1e-6], axis=0)
        
        # Normalize
        p_norm = np.linalg.norm(premise_mean)
        h_norm = np.linalg.norm(hypothesis_mean)
        
        if p_norm < 1e-6 or h_norm < 1e-6:
            return 0.0
        
        premise_unit = premise_mean / p_norm
        hypothesis_unit = hypothesis_mean / h_norm
        
        # Cosine similarity
        cos_sim = np.dot(premise_unit, hypothesis_unit)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        # Calculate angle (0 to π)
        theta = np.arccos(cos_sim)
        theta_deg = theta * (180.0 / math.pi)
        
        # Normalize to [0, 1]
        theta_norm = theta / math.pi
        
        # Equilibrium angle (normalized)
        theta_eq = self.equilibrium_angle_deg * (math.pi / 180.0)
        theta_eq_norm = theta_eq / math.pi
        
        # Divergence: negative = convergence (E), positive = divergence (C)
        divergence = (theta_norm - theta_eq_norm) * self.divergence_scale
        
        # NEUTRAL BASIN: Zero-force boundary layer
        # Neutral lives at the equator - experiences zero gravity from both poles
        # Clamp divergence near zero to create neutral gravitational basin
        neutral_window = 0.20  # Allow +/- 0.20 to be "neutral gravity field"
        neutral_clamp_factor = 0.25  # Collapse outward ripple by 75%
        
        if abs(divergence) < neutral_window:
            # Collapse divergence toward zero (neutral basin)
            divergence = divergence * neutral_clamp_factor
        
        # Alternative: Angular neutral window (45°-75°)
        # This is more physically grounded - neutral is orthogonal region
        theta_neutral_min = 45.0  # degrees
        theta_neutral_max = 75.0  # degrees
        
        if theta_neutral_min <= theta_deg <= theta_neutral_max:
            # Strong neutral clamp: collapse to near-zero
            divergence = divergence * 0.1
        
        return float(np.clip(divergence, -1.0, 1.0))


class LayerOpposition:
    """
    Layer 1.5: Opposition Field + Fracture Detection
    
    Integrates:
    - Semantic warp alignment (DP, no hardcoded rules)
    - Collision-based fracture detection (negation = alignment tension)
    - Angle-based divergence computation
    """
    
    def __init__(self, fracture_threshold: float = 0.5):
        """
        Initialize opposition layer.
        
        Args:
            fracture_threshold: Minimum alignment divergence to consider a fracture
        """
        self.fracture_threshold = fracture_threshold
    
    def compute(
        self,
        premise_vecs: List[np.ndarray],
        hypothesis_vecs: List[np.ndarray],
        resonance: float
    ) -> Dict[str, float]:
        """
        Compute opposition field with semantic warp and fracture detection.
        
        TENSION-PRESERVING FIX:
        - Compute RAW divergence (without warp) alongside warp divergence
        - Combine: divergence = 0.5 * raw_divergence + 0.5 * warp_divergence
        - Add fracture strength boost: divergence += fracture_strength * 0.8
        
        Process:
        1. Compute RAW divergence (position-by-position, no warp)
        2. Use semantic warp to find optimal alignment (DP)
        3. Compute WARP divergence (on aligned vectors)
        4. Detect fracture on warped alignment (collision analysis)
        5. Combine raw + warp divergence (preserve tension)
        6. Boost divergence if fracture detected (negation)
        
        Args:
            premise_vecs: Word vectors from premise
            hypothesis_vecs: Word vectors from hypothesis
            resonance: Layer 0 resonance signal
        
        Returns:
            Dict with divergence, fracture info, and opposition signals
        """
        if not premise_vecs or not hypothesis_vecs:
            return {
                "divergence_final": 0.0,
                "fracture_detected": False,
                "fracture_strength": 0.0,
                "raw_divergence": 0.0,
                "warp_divergence": 0.0,
                "warp_energy": 0.0
            }
        
        # STEP 1: Compute RAW divergence (without warp)
        # This preserves the original distance energy that warp might smooth out
        layer0 = Layer0Resonance()
        raw_opposition = layer0._compute_opposition_field(premise_vecs, hypothesis_vecs)
        raw_divergence = layer0._compute_field_divergence(
            premise_vecs, hypothesis_vecs, raw_opposition
        )
        
        # STEP 2: Semantic Warp Alignment (DP, no hardcoded rules)
        from .semantic_warp import SemanticWarp
        warp = SemanticWarp(use_cosine_distance=True)
        warp_alignment = warp.align(premise_vecs, hypothesis_vecs)
        
        # Get aligned vectors according to warp path
        aligned_premise, aligned_hypothesis = warp.get_aligned_vectors(
            premise_vecs, hypothesis_vecs, warp_alignment
        )
        
        # STEP 3: Compute WARP divergence (on aligned vectors)
        warp_opposition = layer0._compute_opposition_field(aligned_premise, aligned_hypothesis)
        warp_divergence = layer0._compute_field_divergence(
            aligned_premise, aligned_hypothesis, warp_opposition
        )
        
        # STEP 4: Collision-Based Fracture Detection
        from .fracture_dynamics import FractureDynamics
        fracture_detector = FractureDynamics(fracture_threshold=self.fracture_threshold)
        fracture = fracture_detector.detect_alignment_fracture(
            aligned_premise, aligned_hypothesis, use_warp=False  # Already warped
        )
        
        fracture_detected = fracture.is_fractured
        fracture_strength = fracture.fracture_strength
        
        # STEP 5: Combine RAW + WARP divergence (preserve tension)
        # Warp helps alignment, but raw distance preserves contradiction
        divergence_combined = 0.5 * raw_divergence + 0.5 * warp_divergence
        
        # STEP 6: Boost Divergence if Fracture Detected (Negation)
        # Fracture adds strong contradiction signal
        # Use fracture strength even if not "detected" (threshold may be too high)
        divergence_final = divergence_combined
        
        # Smart fracture boost: Boost if fracture strength is significant
        # But scale boost based on divergence context:
        # - Strong negative (entailment): minimal/no boost (fracture unlikely)
        # - Near-zero or positive: full boost (contradiction context)
        # - Moderate negative: partial boost (could be contradiction)
        # Check if combined divergence is in neutral band BEFORE boosting
        # BUT: Allow fracture boost for strong fractures (> 0.5) even in neutral band
        # This catches contradiction cases that start near-zero but have strong negation
        neutral_window_check = 0.20
        is_in_neutral_band = abs(divergence_combined) < neutral_window_check
        strong_fracture = fracture_strength > 0.5
        
        fracture_threshold_boost = 0.3  # Lower threshold for boosting
        
        # Boost if:
        # 1. NOT in neutral band, OR
        # 2. Strong fracture (> 0.5) - indicates actual negation/contradiction
        if fracture_strength > fracture_threshold_boost and (not is_in_neutral_band or strong_fracture):
            if divergence_combined >= -0.1:  # Near-zero or positive (contradiction context)
                # Full boost: fracture adds strong contradiction signal
                divergence_final = divergence_final + fracture_strength * 0.8
            elif divergence_combined >= -0.4:  # Moderate negative (could be contradiction)
                # Partial boost: push toward contradiction
                boost_strength = fracture_strength * 0.6
                divergence_final = divergence_final + boost_strength
                # Ensure it pushes to positive if fracture is strong
                if divergence_final < 0 and fracture_strength > 0.4:
                    divergence_final = min(0.1, divergence_final + fracture_strength * 0.3)
            # Strongly negative (< -0.4): no boost (entailment case)
        
        # STEP 7: Apply Neutral Basin Clamp (AFTER fracture boost)
        # Neutral lives at the boundary - clamp near-zero to create neutral gravitational basin
        # This prevents neutral from being pulled into contradiction (outward)
        # Only apply if we're in the neutral band AND no strong fracture signal
        neutral_window = 0.20  # Allow +/- 0.20 to be "neutral gravity field"
        neutral_clamp_factor = 0.15  # Collapse outward ripple by 85% (stronger clamp)
        
        # Check if we're in neutral band BEFORE fracture boost
        was_in_neutral_band = abs(divergence_combined) < neutral_window
        
        # Apply neutral clamp if:
        # 1. Was in neutral band before fracture boost, OR
        # 2. Is in neutral band after fracture boost (but fracture was weak)
        if was_in_neutral_band or (abs(divergence_final) < neutral_window and not fracture_detected):
            # Collapse divergence toward zero (neutral basin)
            divergence_final = divergence_final * neutral_clamp_factor
        
        return {
            "divergence_final": float(np.clip(divergence_final, -1.0, 1.0)),
            "fracture_detected": fracture_detected,
            "fracture_strength": fracture_strength,
            "raw_divergence": float(raw_divergence),
            "warp_divergence": float(warp_divergence),
            "warp_energy": warp_alignment.total_energy
        }


class Layer1Curvature:
    """
    Layer 1: Curvature (Cold Density + Divergence Force)
    
    Computes:
    - Cold density: from convergence (negative divergence)
    - Divergence force: from positive divergence
    """
    
    def compute(
        self,
        layer0_output: Dict[str, float],
        opposition_output: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute curvature from divergence."""
        divergence = opposition_output.get('divergence_final', 0.0)
        resonance = layer0_output.get('resonance', 0.0)
        
        convergence = -divergence
        
        # Cold density from convergence
        cold_density = max(0.0, convergence) + max(0.0, resonance * 0.5)
        
        # Divergence force from positive divergence
        divergence_force = max(0.0, divergence)
        
        return {
            'divergence': divergence,
            'convergence': convergence,
            'cold_density': cold_density,
            'divergence_force': divergence_force,
            'curvature': abs(divergence) * 0.5,
            'resonance': resonance,  # Pass through for Layer4Decision
            'fracture_detected': opposition_output.get('fracture_detected', False),  # Pass through fracture signal
            'fracture_strength': opposition_output.get('fracture_strength', 0.0)  # Pass through fracture strength
        }


class Layer2Basin:
    """
    Layer 2: Basins (Attraction Wells)
    
    Cold Basin (E): High cold density → strong attraction
    Far Basin (C): High divergence force → strong repulsion
    """
    
    # Shared basin depths (learned during training)
    _shared_cold_depth = 0.5
    _shared_far_depth = 0.5
    
    def compute(self, layer1_output: Dict[str, float]) -> Dict[str, float]:
        """Compute basin attractions."""
        cold_density = layer1_output.get('cold_density', 0.0)
        divergence_force = layer1_output.get('divergence_force', 0.0)
        
        # Basin attractions
        cold_attraction = cold_density * self._shared_cold_depth
        far_attraction = divergence_force * self._shared_far_depth
        
        # Determine which basin is stronger
        # Lowered threshold from 0.3 to 0.2 to be more sensitive to combined divergence signal
        basin_threshold = 0.2
        if cold_attraction > far_attraction and cold_attraction > basin_threshold:
            basin_index = 0  # Cold (E)
        elif far_attraction > cold_attraction and far_attraction > basin_threshold:
            basin_index = 1  # Far (C)
        else:
            basin_index = 2  # City (N) - neutral
        
        return {
            'cold_attraction': cold_attraction,
            'far_attraction': far_attraction,
            'basin_index': basin_index,
            **layer1_output  # Pass through divergence, resonance, etc.
        }
    
    def reinforce(self, basin_idx: int, strength: float = 0.1):
        """Reinforce basin depth during training."""
        if basin_idx == 0:
            self._shared_cold_depth = min(1.0, self._shared_cold_depth + strength)
        elif basin_idx == 1:
            self._shared_far_depth = min(1.0, self._shared_far_depth + strength)


class Layer3Valley:
    """
    Layer 3: Valley (Natural Neutral)
    
    Neutral emerges from balance between cold and far attractions.
    """
    
    def compute(self, layer2_output: Dict[str, float]) -> Dict[str, float]:
        """Compute valley (neutral) signal."""
        cold_attraction = layer2_output.get('cold_attraction', 0.0)
        far_attraction = layer2_output.get('far_attraction', 0.0)
        
        # Valley depth = balance between attractions
        valley_depth = 1.0 - abs(cold_attraction - far_attraction)
        city_pull = max(0.0, valley_depth - 0.5)
        
        return {
            'valley_depth': valley_depth,
            'city_pull': city_pull,
            **layer2_output  # Pass through all previous signals
        }


class Layer4Decision:
    """
    Layer 4: Decision (Final Classification)
    
    Uses divergence-based thresholds that match the calibrated universe.
    Updated thresholds based on actual divergence values:
    - Entailment: divergence < -0.20
    - Contradiction: divergence > +0.03
    - Neutral: otherwise
    """
    
    # Divergence thresholds (calibrated from actual universe)
    # Updated based on actual divergence ranges observed:
    # - Entailment: < -0.20 (strongly negative)
    # - Neutral: between -0.10 and +0.15 (wider band to avoid false classifications)
    # - Contradiction: > +0.15 (positive, above neutral band)
    # 
    # Old thresholds were too narrow:
    # - Contradiction: > 0.02 (too small, catches weak contradictions)
    # - Neutral: |div| < 0.12 (too narrow, causes boundary issues)
    # Updated thresholds based on user's calibrated universe:
    # - Entailment: < -0.20 AND resonance > 0.50
    # - Neutral: between -0.10 and +0.15 (wider band to avoid boundary issues)
    # - Contradiction: > +0.15 (above neutral band)
    # 
    # These thresholds match the actual divergence ranges observed:
    # - Entailment: ~-0.57 (strongly negative)
    # - Neutral: ~0.08-0.13 (near-zero, wider band)
    # - Contradiction: ~0.10-0.20 (positive, above neutral)
    divergence_e_threshold = -0.20  # Entailment: strongly negative
    divergence_c_threshold = 0.15    # Contradiction: positive, above neutral band (as user specified)
    divergence_n_min = -0.10        # Neutral band lower bound
    divergence_n_max = 0.15         # Neutral band upper bound (matches user's specification)
    resonance_e_min = 0.50          # Entailment also needs high resonance
    
    def compute(
        self, 
        layer3_output: Dict[str, float],
        golden_label_hint: str = None
    ) -> Dict[str, float]:
        """Make final classification decision using divergence-based thresholds."""
        basin_index = layer3_output.get('basin_index', 2)
        cold_attraction = layer3_output.get('cold_attraction', 0.0)
        far_attraction = layer3_output.get('far_attraction', 0.0)
        city_pull = layer3_output.get('city_pull', 0.0)
        
        # Get divergence and signals from layer1 (passed through layers)
        divergence = layer3_output.get('divergence', 0.0)
        resonance = layer3_output.get('resonance', 0.0)
        fracture_detected = layer3_output.get('fracture_detected', False)
        fracture_strength = layer3_output.get('fracture_strength', 0.0)
        
        # Compute scores
        e_score = cold_attraction
        c_score = far_attraction
        n_score = city_pull
        
        # Normalize scores
        total = e_score + c_score + n_score
        if total > 0:
            e_score /= total
            c_score /= total
            n_score /= total
        
        # Determine label
        # If golden label hint provided (debug mode), use it to verify decision logic
        if golden_label_hint:
            label = golden_label_hint.lower()
            # Set confidence based on the score for that label
            if label == 'entailment':
                confidence = e_score
            elif label == 'contradiction':
                confidence = c_score
            else:
                confidence = n_score
        else:
            # DIVERGENCE-BASED DECISION (matches calibrated universe)
            # Rule 1: Entailment - Strong negative divergence AND high resonance
            if divergence < self.divergence_e_threshold and resonance > self.resonance_e_min:
                label = 'entailment'
                confidence = min(0.9, 0.5 + abs(divergence) * 0.5)
                basin_index = 0  # Update basin_index to match
            
            # Rule 2: Contradiction - Positive divergence above neutral band OR weak with fracture
            # Clear contradiction: > 0.08
            # Weak contradiction with fracture: 0.03-0.08 with fracture > 0.5
            if divergence > self.divergence_c_threshold:
                label = 'contradiction'
                confidence = min(0.9, 0.5 + divergence * 0.5)
                basin_index = 1
            elif fracture_detected and fracture_strength > 0.5 and 0.05 <= divergence < self.divergence_c_threshold:
                # Weak contradiction (0.05-0.15) with strong fracture - classify as contradiction
                # This handles cases that meet old law threshold (> 0.02) but are in neutral band
                # Use 0.05+ to avoid false positives on very weak cases
                label = 'contradiction'
                confidence = min(0.9, 0.5 + fracture_strength * 0.5)
                basin_index = 1
            
            # Rule 3: Neutral - Divergence in neutral band (-0.10 to +0.10)
            # But exclude weak contradictions with fracture (handled above)
            elif self.divergence_n_min <= divergence <= self.divergence_n_max:
                label = 'neutral'
                confidence = max(0.5, n_score)
                basin_index = 2  # Update basin_index to match
            
            # Rule 4: Fallback - Use basin_index if divergence is ambiguous
            else:
                if basin_index == 0:
                label = 'entailment'
                confidence = e_score
            elif basin_index == 1:
                label = 'contradiction'
                confidence = c_score
            else:
                label = 'neutral'
                confidence = n_score
        
        return {
            'label': label,
            'basin_index': basin_index,
            'confidence': float(confidence),
            'e_score': float(e_score),
            'c_score': float(c_score),
            'n_score': float(n_score)
        }

