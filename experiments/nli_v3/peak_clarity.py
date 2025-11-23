"""
Peak Clarity: Two-Peak Geometry System

Entailment and Contradiction are stable peaks.
Neutral is the valley between them - NOT a peak.

Architecture:
  E_peak = +α (positive resonance)
  C_peak = -α (negative resonance)
  Neutral = near 0, high variance (valley)

Rules:
  resonance > +band  → Entailment
  resonance < -band  → Contradiction
  else               → Neutral (valley)
"""

import numpy as np
from typing import Dict, Tuple


class PeakClarity:
    """
    Two-peak geometry system: E and C are peaks, N is the valley.
    
    Neutral is NOT a stable attractor - it's the region between peaks.
    """
    
    def __init__(self, band_width: float = 0.3):
        """
        Initialize two-peak system.
        
        Args:
            band_width: Resonance threshold for peak assignment (default 0.3)
        """
        self.band_width = band_width
        
        # Two peaks only: E and C
        # Neutral has no peak - it's the valley
        self.e_peak = {
            'r_mean': 0.5,   # Positive resonance
            'r_std': 0.2,
            'count': 0
        }
        
        self.c_peak = {
            'r_mean': -0.5,  # Negative resonance
            'r_std': 0.2,
            'count': 0
        }
    
    def update_peak(self, class_name: str, resonance: float, variance: float):
        """
        Update peak statistics (E or C only - no Neutral peak).
        
        Args:
            class_name: 'entailment' or 'contradiction' (NOT 'neutral')
            resonance: Resonance value
            variance: Variance value (for noise tracking)
        """
        if class_name == 'entailment':
            peak = self.e_peak
        elif class_name == 'contradiction':
            peak = self.c_peak
        else:
            # Neutral - do nothing (no peak to update)
            return
        
        count = peak['count']
        
        if count == 0:
            peak['r_mean'] = resonance
            peak['count'] = 1
        else:
            # Online mean update
            old_mean = peak['r_mean']
            peak['r_mean'] = (old_mean * count + resonance) / (count + 1)
            
            # Update std (exponential moving average)
            r_diff = resonance - old_mean
            decay = 0.95
            peak['r_std'] = np.sqrt(decay * peak['r_std']**2 + (1 - decay) * r_diff**2)
            peak['r_std'] = max(peak['r_std'], 0.1)  # Minimum std
            
            peak['count'] += 1
    
    def sculpt_valley(self, resonance: float, variance: float, strength: float = 1.0):
        """
        Sculpt the valley for Neutral examples.
        
        Valley creator: doesn't reinforce peaks, but:
        - Adds local noise around this resonance for both peaks
        - Slightly flattens gradients trying to pull into E or C
        - Pushes peaks away from this region
        
        Args:
            resonance: Resonance value where Neutral lives
            variance: Variance value (high variance = noisy region)
            strength: Valley sculpting strength
        """
        # Option: Add noise to both peaks (flatten gradients)
        # This makes the region around this resonance less attractive
        
        # Increase std of both peaks slightly (flatten them near this resonance)
        noise_factor = 0.1 * strength * variance  # More variance = more flattening
        
        # Flatten E peak if it's close to this resonance
        if abs(resonance - self.e_peak['r_mean']) < 0.5:
            self.e_peak['r_std'] = min(self.e_peak['r_std'] + noise_factor, 1.0)
        
        # Flatten C peak if it's close to this resonance
        if abs(resonance - self.c_peak['r_mean']) < 0.5:
            self.c_peak['r_std'] = min(self.c_peak['r_std'] + noise_factor, 1.0)
        
        # Track valley regions (for visualization/debugging)
        if not hasattr(self, 'valley_regions'):
            self.valley_regions = []
        self.valley_regions.append({
            'r': resonance,
            'v': variance,
            'strength': strength
        })
        # Keep only recent valley regions (last 100)
        if len(self.valley_regions) > 100:
            self.valley_regions.pop(0)
    
    def classify(self, resonance: float, variance: float) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify using two-peak geometry with gradient-overlap rule.
        
        Neutral = saddle point where E and C gradients cancel.
        
        Args:
            resonance: Chain resonance value
            variance: Variance value
            
        Returns:
            (label, confidence, scores_dict)
        """
        # Compute distance from each peak
        d_e = abs(resonance - self.e_peak['r_mean']) / (self.e_peak['r_std'] + 1e-6)
        d_c = abs(resonance - self.c_peak['r_mean']) / (self.c_peak['r_std'] + 1e-6)
        
        # Peak scores (Gaussian) - these represent gradient strength
        score_e = np.exp(-0.5 * d_e * d_e)
        score_c = np.exp(-0.5 * d_c * d_c)
        
        # Band-pass: kill if too far from peak
        if d_e > 3.0:  # ~3σ
            score_e = 0.0
        if d_c > 3.0:
            score_c = 0.0
        
        # Gradient strengths (normalized peak scores)
        g_E = score_e  # Entailment gradient strength
        g_C = score_c  # Contradiction gradient strength
        
        # Noise suppression (high variance reduces gradient strength)
        noise_factor = 1.0 / (1.0 + variance)
        g_E *= noise_factor
        g_C *= noise_factor
        
        # Gradient-overlap rule: Neutral = saddle where gradients cancel
        # Use RATIO-BASED threshold (scale-invariant, bulletproof)
        # ratio = |g_E - g_C| / max(g_E, g_C)
        # If ratio < 0.12 → gradients cancel → Neutral (saddle point)
        max_gradient = max(g_E, g_C)
        if max_gradient > 1e-6:  # Avoid division by zero
            gradient_ratio = abs(g_E - g_C) / max_gradient
        else:
            # Both gradients are zero → Neutral
            gradient_ratio = 0.0
        
        saddle_threshold = 0.12  # Relative overlap threshold (scale-invariant)
        
        if gradient_ratio < saddle_threshold and max_gradient > 0.1:
            # Gradients cancel → Neutral (saddle point)
            # Score is proportional to how close to perfect cancellation
            score_n = 1.0 - (gradient_ratio / saddle_threshold)
            score_n = max(0.0, min(1.0, score_n))  # Clip to [0, 1]
            
            # E and C scores are reduced in the saddle (they cancel)
            score_e = g_E * (1.0 - score_n * 0.5)
            score_c = g_C * (1.0 - score_n * 0.5)
            
            label = 'neutral'
            confidence = score_n
        else:
            # Gradients don't cancel → one peak wins
            if g_E > g_C:
                label = 'entailment'
                confidence = g_E
                score_e = g_E
                score_c = g_C * 0.3  # Reduced when E wins
                score_n = 1.0 - g_E  # Neutral = absence of strong peak
            else:
                label = 'contradiction'
                confidence = g_C
                score_e = g_E * 0.3  # Reduced when C wins
                score_c = g_C
                score_n = 1.0 - g_C  # Neutral = absence of strong peak
        
        # Normalize scores
        total = score_e + score_c + score_n
        if total > 0:
            score_e /= total
            score_c /= total
            score_n /= total
        
        return label, confidence, {
            'entailment': float(score_e),
            'contradiction': float(score_c),
            'neutral': float(score_n)
        }
    
    def get_peaks(self) -> Dict[str, Dict[str, float]]:
        """Get peak statistics (for interpretability)."""
        return {
            'entailment': {
                'r_mean': float(self.e_peak['r_mean']),
                'r_std': float(self.e_peak['r_std']),
                'count': int(self.e_peak['count'])
            },
            'contradiction': {
                'r_mean': float(self.c_peak['r_mean']),
                'r_std': float(self.c_peak['r_std']),
                'count': int(self.c_peak['count'])
            },
            'neutral': {
                'r_mean': 0.0,  # No peak - valley
                'r_std': 0.0,
                'count': 0
            }
        }

