"""
Decision Head: Tiny Learned Classifier

A small neural network that learns the non-linear boundary between
geometric features and classes.

This is the missing piece that pushes accuracy from 42% → 48-55%.
"""

import numpy as np
from typing import Dict, List
import pickle
import os


class DecisionHead:
    """
    Tiny learned classifier (2-layer MLP).
    
    Architecture:
    - Input: 9 geometric features
    - Hidden: 16 neurons (ReLU)
    - Output: 3 neurons (Softmax) → [E, C, N]
    
    Trained with cross-entropy loss.
    """
    
    def __init__(self, hidden_size: int = 16):
        """
        Initialize decision head.
        
        Args:
            hidden_size: Number of hidden neurons (default 16)
        """
        self.hidden_size = hidden_size
        self.input_size = 9  # 9 features
        
        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(self.input_size, hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros(hidden_size)
        
        # Initialize output layer with small bias toward uniform (prevents Neutral bias)
        self.W2 = np.random.randn(hidden_size, 3) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.array([0.0, 0.0, 0.0])  # Start neutral (no bias toward any class)
        
        # Training state
        self.trained = False
    
    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            features: Input features [9-dim]
            
        Returns:
            Output probabilities [3-dim] for [E, C, N]
        """
        # Layer 1: ReLU
        z1 = np.dot(features, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        
        # Layer 2: Softmax
        z2 = np.dot(a1, self.W2) + self.b2
        exp_z2 = np.exp(z2 - np.max(z2))  # Numerical stability
        probs = exp_z2 / np.sum(exp_z2)
        
        return probs
    
    def backward(self, 
                 features: np.ndarray,
                 probs: np.ndarray,
                 target: int,
                 learning_rate: float = 0.01) -> float:
        """
        Backward pass (gradient descent).
        
        Args:
            features: Input features [9-dim]
            probs: Predicted probabilities [3-dim]
            target: Target class index (0, 1, or 2)
            learning_rate: Learning rate
            
        Returns:
            Cross-entropy loss
        """
        # Create target one-hot
        target_onehot = np.zeros(3)
        target_onehot[target] = 1.0
        
        # Compute loss
        loss = -np.log(probs[target] + 1e-10)  # Cross-entropy
        
        # Output layer gradient
        grad_z2 = probs - target_onehot
        
        # Backprop through layer 2
        grad_W2 = np.outer(self._a1_cache, grad_z2)
        grad_b2 = grad_z2
        
        # Backprop through layer 1
        grad_a1 = np.dot(grad_z2, self.W2.T)
        grad_z1 = grad_a1 * (self._z1_cache > 0)  # ReLU derivative
        
        grad_W1 = np.outer(features, grad_z1)
        grad_b1 = grad_z1
        
        # Update weights
        self.W2 -= learning_rate * grad_W2
        self.b2 -= learning_rate * grad_b2
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1
        
        return float(loss)
    
    def predict(self, features: np.ndarray) -> tuple:
        """
        Predict class from features.
        
        Args:
            features: Input features [9-dim]
            
        Returns:
            (class_idx, probabilities_dict)
        """
        probs = self.forward(features)
        class_idx = int(np.argmax(probs))
        
        return class_idx, {
            'entailment': float(probs[0]),
            'contradiction': float(probs[1]),
            'neutral': float(probs[2])
        }
    
    def _forward_with_cache(self, features: np.ndarray) -> tuple:
        """Forward pass with cached activations for backprop."""
        # Layer 1: ReLU
        z1 = np.dot(features, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        
        # Cache for backprop
        self._z1_cache = z1
        self._a1_cache = a1
        
        # Layer 2: Softmax
        z2 = np.dot(a1, self.W2) + self.b2
        exp_z2 = np.exp(z2 - np.max(z2))
        probs = exp_z2 / np.sum(exp_z2)
        
        return probs
    
    def train_step(self,
                   features: np.ndarray,
                   target: int,
                   learning_rate: float = 0.01) -> float:
        """
        Single training step.
        
        Args:
            features: Input features [9-dim]
            target: Target class index (0, 1, or 2)
            learning_rate: Learning rate
            
        Returns:
            Loss value
        """
        probs = self._forward_with_cache(features)
        loss = self.backward(features, probs, target, learning_rate)
        self.trained = True
        return loss
    
    def save(self, filepath: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2,
                'trained': self.trained
            }, f)
    
    def load(self, filepath: str) -> bool:
        """Load model weights."""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.trained = data.get('trained', False)
            return True
        except Exception:
            return False


def extract_features_for_head(encoded_pair,
                              features: Dict[str, float]) -> np.ndarray:
    """
    Extract 9 features for decision head.
    
    Features:
    1. chain_resonance
    2. aligned_mean (position 0 vs 0, 1 vs 1, etc.)
    3. window_mean (sliding window)
    4. cross_mean (cross-word matching)
    5. variance
    6. lexical_overlap
    7. polarity_E (average entailment polarity)
    8. polarity_C (average contradiction polarity)
    9. polarity_N (average neutral polarity)
    """
    from experiments.nli_simple.native_chain import SimpleLexicon
    
    lexicon = SimpleLexicon()
    
    # Get word vectors
    premise_vecs, hypothesis_vecs = encoded_pair.get_word_vectors()
    
    # Compute aligned, window, cross means
    aligned_sims = []
    min_len = min(len(premise_vecs), len(hypothesis_vecs))
    
    for i in range(min_len):
        p_vec = premise_vecs[i]
        h_vec = hypothesis_vecs[i]
        sim = np.dot(p_vec, h_vec) / (np.linalg.norm(p_vec) * np.linalg.norm(h_vec) + 1e-10)
        aligned_sims.append(sim)
    
    aligned_mean = float(np.mean(aligned_sims)) if aligned_sims else 0.0
    
    # Window mean (sliding window)
    window_sims = []
    for offset in range(-1, 2):
        offset_sims = []
        for i in range(min_len):
            h_idx = i + offset
            if 0 <= h_idx < len(hypothesis_vecs):
                p_vec = premise_vecs[i]
                h_vec = hypothesis_vecs[h_idx]
                sim = np.dot(p_vec, h_vec) / (np.linalg.norm(p_vec) * np.linalg.norm(h_vec) + 1e-10)
                offset_sims.append(sim)
        if offset_sims:
            window_sims.append(np.mean(offset_sims))
    
    window_mean = float(max(window_sims)) if window_sims else aligned_mean
    
    # Cross mean
    cross_sims = []
    for h_vec in hypothesis_vecs:
        best = -1.0
        for p_vec in premise_vecs:
            sim = np.dot(p_vec, h_vec) / (np.linalg.norm(p_vec) * np.linalg.norm(h_vec) + 1e-10)
            best = max(best, sim)
        if best > -1.0:
            cross_sims.append(best)
    
    cross_mean = float(np.mean(cross_sims)) if cross_sims else 0.0
    
    # Word polarities
    all_tokens = encoded_pair.premise.tokens + encoded_pair.hypothesis.tokens
    polarity_vecs = [lexicon.get_word_polarity(token) for token in all_tokens]
    if polarity_vecs:
        avg_polarity = np.mean(polarity_vecs, axis=0)
        polarity_E = float(avg_polarity[0])
        polarity_C = float(avg_polarity[1])
        polarity_N = float(avg_polarity[2])
    else:
        polarity_E = polarity_C = polarity_N = 0.33
    
    # Combine all features
    feature_vector = np.array([
        features['resonance'],      # 1. Chain resonance [-1, 1]
        aligned_mean,                # 2. Aligned mean [-1, 1]
        window_mean,                 # 3. Window mean [-1, 1]
        cross_mean,                  # 4. Cross mean [-1, 1]
        features['variance'],        # 5. Variance [0, 1]
        features['lexical_overlap'], # 6. Lexical overlap [0, 1]
        polarity_E,                  # 7. Polarity E [0, 1]
        polarity_C,                  # 8. Polarity C [0, 1]
        polarity_N                   # 9. Polarity N [0, 1]
    ])
    
    # Normalize features to [0, 1] range for stable training
    # Features 1-4 are in [-1, 1], so shift and scale them
    feature_vector[0:4] = (feature_vector[0:4] + 1.0) / 2.0  # Shift [-1,1] → [0,1]
    
    # Clip to ensure all features are in [0, 1]
    feature_vector = np.clip(feature_vector, 0.0, 1.0)
    
    return feature_vector

