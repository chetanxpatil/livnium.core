"""
SNLI Phase 1 Classifier: Geometric Signature → E/C/N

Uses existing Nova infrastructure:
- TextToGeometry to get signatures
- Simple classifier on top of signatures
- Outputs ONLY: "entailment", "contradiction", or "neutral"
- Enforces discipline: no extra words allowed
"""

import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path
import json
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from nova.core.text_to_geometry import TextToGeometry


class SNLIClassifier:
    """
    Phase 1 SNLI Classifier: Geometric Signatures → E/C/N
    
    Architecture:
    1. Premise + Hypothesis → Geometric Signatures (using TextToGeometry)
    2. Combine signatures (concatenate or merge)
    3. Classify → "entailment", "contradiction", or "neutral"
    """
    
    def __init__(self, 
                 interface: TextToGeometry,
                 collapse_steps: int = 12):
        """
        Initialize SNLI classifier.
        
        Args:
            interface: TextToGeometry interface for signature extraction
            collapse_steps: Number of collapse steps for signature generation
        """
        self.interface = interface
        self.collapse_steps = collapse_steps
        
        # Classifier (trained on geometric signatures)
        self.classifier: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Label mapping
        self.label_to_word = {
            0: "entailment",
            1: "contradiction", 
            2: "neutral"
        }
        self.word_to_label = {v: k for k, v in self.label_to_word.items()}
        
    def get_signature_pair(self, premise: str, hypothesis: str) -> np.ndarray:
        """
        Get combined geometric signature for premise+hypothesis pair.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            
        Returns:
            Combined signature vector
        """
        # Get signatures for both sentences
        premise_sig = self.interface.get_meaning_signature(
            premise, 
            collapse_steps=self.collapse_steps
        )
        self.interface.reset_geometry()
        
        hypothesis_sig = self.interface.get_meaning_signature(
            hypothesis,
            collapse_steps=self.collapse_steps
        )
        self.interface.reset_geometry()
        
        # Combine signatures: concatenate (preserves both geometric patterns)
        combined = np.concatenate([premise_sig, hypothesis_sig])
        
        return combined
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              normalize: bool = True):
        """
        Train classifier on geometric signatures.
        
        Args:
            X: Array of combined signatures (n_samples, n_features)
            y: Array of labels (0=entailment, 1=contradiction, 2=neutral)
            normalize: Whether to normalize features
        """
        print(f"Training SNLI classifier on {len(X)} samples...")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Label distribution: {np.bincount(y)}")
        
        # Normalize features
        if normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            print("  ✓ Features normalized")
        
        # Train logistic regression classifier
        self.classifier = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        self.classifier.fit(X, y)
        
        # Evaluate training accuracy
        train_acc = self.classifier.score(X, y)
        print(f"  ✓ Training accuracy: {train_acc:.4f}")
        
    def predict(self, premise: str, hypothesis: str) -> str:
        """
        Predict label for premise+hypothesis pair.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            
        Returns:
            One word: "entailment", "contradiction", or "neutral"
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Get combined signature
        combined_sig = self.get_signature_pair(premise, hypothesis)
        
        # Normalize if scaler exists
        if self.scaler is not None:
            combined_sig = self.scaler.transform(combined_sig.reshape(1, -1))
        else:
            combined_sig = combined_sig.reshape(1, -1)
        
        # Predict label
        label_idx = self.classifier.predict(combined_sig)[0]
        
        # Return word (ONLY one word, no extra words)
        return self.label_to_word[label_idx]
    
    def predict_proba(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Get probability distribution over labels.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            
        Returns:
            Dict with probabilities for each label
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Get combined signature
        combined_sig = self.get_signature_pair(premise, hypothesis)
        
        # Normalize if scaler exists
        if self.scaler is not None:
            combined_sig = self.scaler.transform(combined_sig.reshape(1, -1))
        else:
            combined_sig = combined_sig.reshape(1, -1)
        
        # Get probabilities
        probs = self.classifier.predict_proba(combined_sig)[0]
        
        return {
            self.label_to_word[i]: float(probs[i])
            for i in range(3)
        }
    
    def save(self, model_dir: Path):
        """Save classifier to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        classifier_path = model_dir / "snli_classifier.pkl"
        with open(classifier_path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'scaler': self.scaler,
                'collapse_steps': self.collapse_steps,
                'lattice_size': self.interface.geometry.lattice_size
            }, f)
        
        # Save metadata
        metadata = {
            'label_to_word': self.label_to_word,
            'word_to_label': self.word_to_label,
            'feature_dim': self.interface.geometry.lattice_size ** 3 * 2  # premise + hypothesis
        }
        metadata_path = model_dir / "snli_classifier_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved SNLI classifier to {model_dir}")
    
    def load(self, model_dir: Path):
        """Load classifier from disk."""
        model_dir = Path(model_dir)
        
        # Load classifier
        classifier_path = model_dir / "snli_classifier.pkl"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier not found: {classifier_path}")
        
        with open(classifier_path, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.scaler = data['scaler']
            self.collapse_steps = data.get('collapse_steps', 12)
        
        # Load metadata
        metadata_path = model_dir / "snli_classifier_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.label_to_word = metadata['label_to_word']
                self.word_to_label = metadata['word_to_label']
        
        print(f"✓ Loaded SNLI classifier from {model_dir}")

