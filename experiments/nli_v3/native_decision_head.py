"""
Native Decision Head V2: Band-Pass Geometry Engine

A peak-based, band-pass classifier that learns class peaks in feature space
and uses Gaussian distance scoring with sharp fall-off outside bands.

Key concepts:
1. Learn peaks (μ, σ) per class during training
2. At inference: compute distance from peaks → Gaussian score
3. Band-pass: kill scores outside band (sharp fall-off)
4. Noise suppression: down-weight high variance cases
5. Lock mechanism: high-confidence cases trigger Moksha

This is NOT a neural network - it's pure geometric peak detection.
"""

import numpy as np
from typing import Dict, Tuple
import pickle
import os


class NativeDecisionHead:
    """
    Band-pass geometry engine: peak-based classification with sharp fall-off.
    
    Learns class peaks in feature space:
    - resonance (r) and variance (v) define the primary peak
    - polarity and overlap provide secondary signals
    
    At inference:
    - Compute distance from each class peak
    - Apply Gaussian scoring with band-pass filter
    - Suppress noise (high variance)
    - Lock high-confidence cases
    """
    
    def __init__(self, band_k: float = 4.0, lock_threshold: float = 0.8, v_lock: float = 0.3):
        """
        Initialize band-pass geometry engine.
        
        Args:
            band_k: Band width (squared distance threshold, default 4.0 = 2σ)
            lock_threshold: Peak score threshold for locking (default 0.8)
            v_lock: Variance threshold for locking (default 0.3, low noise)
        """
        self.band_k = band_k
        self.lock_threshold = lock_threshold
        self.v_lock = v_lock
        
        # Per-class peak statistics (learned during training)
        # For each class: [mean_r, mean_v, std_r, std_v, count]
        self.peak_stats = {
            'entailment': {'r': 0.0, 'v': 0.0, 'r_std': 0.5, 'v_std': 0.3, 'count': 0},
            'contradiction': {'r': 0.0, 'v': 0.0, 'r_std': 0.5, 'v_std': 0.3, 'count': 0},
            'neutral': {'r': 0.0, 'v': 0.0, 'r_std': 0.5, 'v_std': 0.3, 'count': 0}
        }
        
        # Secondary feature weights (learned)
        self.alpha = 0.5  # Weight for polarity
        self.beta = 0.3   # Weight for overlap (entailment)
        self.gamma = 0.4  # Weight for variance (neutral)
        self.delta = 0.3  # Weight for (1-overlap) (neutral)
        
        # Training state
        self.trained = False
    
    def _update_peak(self, class_name: str, r: float, v: float):
        """
        Update peak statistics for a class (online mean/std update).
        
        Args:
            class_name: 'entailment', 'contradiction', or 'neutral'
            r: Resonance value
            v: Variance value
        """
        stats = self.peak_stats[class_name]
        count = stats['count']
        
        if count == 0:
            # First sample: initialize
            stats['r'] = r
            stats['v'] = v
            stats['count'] = 1
        else:
            # Online update (Welford's algorithm for variance)
            old_mean_r = stats['r']
            old_mean_v = stats['v']
            
            # Update mean
            stats['r'] = (old_mean_r * count + r) / (count + 1)
            stats['v'] = (old_mean_v * count + v) / (count + 1)
            
            # Update std (simplified: running variance)
            # For simplicity, use exponential moving average of squared differences
            r_diff = r - old_mean_r
            v_diff = v - old_mean_v
            
            # Update std with exponential moving average
            decay = 0.95  # Decay factor
            stats['r_std'] = np.sqrt(decay * stats['r_std']**2 + (1 - decay) * r_diff**2)
            stats['v_std'] = np.sqrt(decay * stats['v_std']**2 + (1 - decay) * v_diff**2)
            
            # Ensure minimum std to avoid division by zero
            stats['r_std'] = max(stats['r_std'], 0.1)
            stats['v_std'] = max(stats['v_std'], 0.05)
            
            stats['count'] += 1
    
    def _compute_peak_score(self, class_name: str, r: float, v: float) -> Tuple[float, float]:
        """
        Compute peak score and squared distance for a class.
        
        Args:
            class_name: Class to score
            r: Resonance value
            v: Variance value
            
        Returns:
            (peak_score, squared_distance)
        """
        stats = self.peak_stats[class_name]
        
        # Distance from peak (normalized by std)
        dr = r - stats['r']
        dv = v - stats['v']
        
        # Normalized squared distance
        eps = 1e-6
        d2 = (dr * dr) / (stats['r_std']**2 + eps) + (dv * dv) / (stats['v_std']**2 + eps)
        
        # Gaussian peak score
        peak_score = np.exp(-0.5 * d2)
        
        # Band-pass: kill if outside band
        if d2 > self.band_k:
            peak_score = 0.0
        
        return peak_score, d2
    
    def forward(self, features: Dict[str, float]) -> Tuple[np.ndarray, bool]:
        """
        Forward pass: peak-based scoring with band-pass and noise suppression.
        
        Args:
            features: Dict with 'resonance', 'variance', 'polarity_E', 'polarity_C', 'lexical_overlap'
            
        Returns:
            (probabilities [E, C, N], is_locked)
        """
        # Extract features
        r = features.get('resonance', 0.0)
        v = features.get('variance', 0.0)
        pE = features.get('polarity_E', 0.33)
        pC = features.get('polarity_C', 0.33)
        o = features.get('lexical_overlap', 0.0)
        
        # Compute peak scores for each class
        peak_E, d2_E = self._compute_peak_score('entailment', r, v)
        peak_C, d2_C = self._compute_peak_score('contradiction', r, v)
        peak_N, d2_N = self._compute_peak_score('neutral', r, v)
        
        # Noise suppression: down-weight high variance
        noise_factor = 1.0 / (1.0 + v)
        
        # Apply noise suppression to peak scores
        peak_E *= noise_factor
        peak_C *= noise_factor
        peak_N *= noise_factor
        
        # Combine with secondary features
        # Entailment: positive resonance, high overlap, high polarity_E
        score_E = peak_E * (1.0 + self.alpha * pE + self.beta * o)
        
        # Contradiction: negative resonance OR high polarity_C
        score_C = peak_C * (1.0 + self.alpha * pC)
        
        # Neutral: high variance OR low overlap
        score_N = peak_N * (1.0 + self.gamma * v + self.delta * (1.0 - o))
        
        scores = np.array([score_E, score_C, score_N])
        
        # Check for lock (clarity band)
        max_score = np.max(scores)
        max_idx = int(np.argmax(scores))
        is_locked = (max_score > self.lock_threshold) and (v < self.v_lock)
        
        # Softmax for probabilities
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probs = exp_scores / np.sum(exp_scores)
        
        return probs, is_locked
    
    def train_step(self,
                   features: Dict[str, float],
                   target: int,
                   learning_rate: float = 0.01) -> Tuple[float, bool]:
        """
        Single training step: update peaks and secondary weights.
        
        Args:
            features: Feature dict
            target: Target class index (0, 1, or 2)
            learning_rate: Learning rate for secondary weights
            
        Returns:
            (loss, is_locked)
        """
        # Forward pass
        probs, is_locked = self.forward(features)
        loss = -np.log(probs[target] + 1e-10)  # Cross-entropy
        
        # Extract features
        r = features.get('resonance', 0.0)
        v = features.get('variance', 0.0)
        pE = features.get('polarity_E', 0.33)
        pC = features.get('polarity_C', 0.33)
        o = features.get('lexical_overlap', 0.0)
        
        # Update peak statistics for correct class
        class_names = ['entailment', 'contradiction', 'neutral']
        self._update_peak(class_names[target], r, v)
        
        # Update secondary weights (simple gradient descent)
        target_onehot = np.zeros(3)
        target_onehot[target] = 1.0
        
        grad_probs = probs - target_onehot
        
        # Gradients for secondary weights
        peak_E, _ = self._compute_peak_score('entailment', r, v)
        peak_C, _ = self._compute_peak_score('contradiction', r, v)
        peak_N, _ = self._compute_peak_score('neutral', r, v)
        
        noise_factor = 1.0 / (1.0 + v)
        peak_E *= noise_factor
        peak_C *= noise_factor
        peak_N *= noise_factor
        
        # Gradient for alpha (polarity weight)
        grad_alpha = (
            grad_probs[0] * peak_E * pE +
            grad_probs[1] * peak_C * pC
        )
        
        # Gradient for beta (overlap weight for entailment)
        grad_beta = grad_probs[0] * peak_E * o
        
        # Gradient for gamma (variance weight for neutral)
        grad_gamma = grad_probs[2] * peak_N * v
        
        # Gradient for delta ((1-overlap) weight for neutral)
        grad_delta = grad_probs[2] * peak_N * (1.0 - o)
        
        # Update weights
        self.alpha -= learning_rate * grad_alpha
        self.beta -= learning_rate * grad_beta
        self.gamma -= learning_rate * grad_gamma
        self.delta -= learning_rate * grad_delta
        
        # Clip weights to reasonable range
        self.alpha = np.clip(self.alpha, 0.0, 2.0)
        self.beta = np.clip(self.beta, 0.0, 2.0)
        self.gamma = np.clip(self.gamma, 0.0, 2.0)
        self.delta = np.clip(self.delta, 0.0, 2.0)
        
        self.trained = True
        return float(loss), is_locked
    
    def predict(self, features: Dict[str, float]) -> Tuple[int, Dict[str, float], bool]:
        """
        Predict class from features.
        
        Args:
            features: Feature dict
            
        Returns:
            (class_idx, probabilities_dict, is_locked)
        """
        probs, is_locked = self.forward(features)
        class_idx = int(np.argmax(probs))
        
        return class_idx, {
            'entailment': float(probs[0]),
            'contradiction': float(probs[1]),
            'neutral': float(probs[2])
        }, is_locked
    
    def get_peaks(self) -> Dict[str, Dict[str, float]]:
        """Get learned peak statistics (for interpretability)."""
        return {
            'entailment': {
                'r_mean': float(self.peak_stats['entailment']['r']),
                'v_mean': float(self.peak_stats['entailment']['v']),
                'r_std': float(self.peak_stats['entailment']['r_std']),
                'v_std': float(self.peak_stats['entailment']['v_std']),
                'count': int(self.peak_stats['entailment']['count'])
            },
            'contradiction': {
                'r_mean': float(self.peak_stats['contradiction']['r']),
                'v_mean': float(self.peak_stats['contradiction']['v']),
                'r_std': float(self.peak_stats['contradiction']['r_std']),
                'v_std': float(self.peak_stats['contradiction']['v_std']),
                'count': int(self.peak_stats['contradiction']['count'])
            },
            'neutral': {
                'r_mean': float(self.peak_stats['neutral']['r']),
                'v_mean': float(self.peak_stats['neutral']['v']),
                'r_std': float(self.peak_stats['neutral']['r_std']),
                'v_std': float(self.peak_stats['neutral']['v_std']),
                'count': int(self.peak_stats['neutral']['count'])
            }
        }
    
    def get_weights(self) -> Dict[str, float]:
        """Get learned secondary weights (for interpretability)."""
        return {
            'alpha': float(self.alpha),  # Polarity weight
            'beta': float(self.beta),    # Overlap weight (entailment)
            'gamma': float(self.gamma),  # Variance weight (neutral)
            'delta': float(self.delta),  # (1-overlap) weight (neutral)
            'band_k': float(self.band_k),
            'lock_threshold': float(self.lock_threshold),
            'v_lock': float(self.v_lock)
        }
    
    def save(self, filepath: str):
        """Save model weights and peak statistics."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'peak_stats': self.peak_stats,
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'delta': self.delta,
                'band_k': self.band_k,
                'lock_threshold': self.lock_threshold,
                'v_lock': self.v_lock,
                'trained': self.trained
            }, f)
    
    def load(self, filepath: str) -> bool:
        """Load model weights and peak statistics."""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.peak_stats = data['peak_stats']
            self.alpha = data['alpha']
            self.beta = data['beta']
            self.gamma = data['gamma']
            self.delta = data['delta']
            self.band_k = data.get('band_k', 4.0)
            self.lock_threshold = data.get('lock_threshold', 0.8)
            self.v_lock = data.get('v_lock', 0.3)
            self.trained = data.get('trained', False)
            return True
        except Exception:
            return False
