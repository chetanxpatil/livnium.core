"""
Quantum Classifier for Livnium

Implements quantum-mechanical classification using feature qubits and quantum measurement.
Supports 3-class NLI classification (Entailment, Neutral, Contradiction).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from quantum.features import QuantumFeatureSet
from quantum.features_v2 import QuantumFeatureSetV2
from quantum.gates import (
    hadamard_gate,
    rotate_y_gate,
    rotate_z_gate,
    get_probabilities,
    normalize_state,
    create_superposition_from_value
)


class QuantumClassifier:
    """
    Quantum classifier that uses quantum measurement for classification decisions.
    
    Creates a class qubit in superposition and applies feature-conditioned gates
    based on quantum feature measurements. Final measurement determines class.
    """
    
    def __init__(
        self,
        n_classes: int = 3,
        use_interference: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize quantum classifier.
        
        Args:
            n_classes: Number of classes (default 3 for NLI: Entailment, Neutral, Contradiction)
            use_interference: Whether to use quantum interference effects
            random_seed: Optional seed for reproducibility
        """
        self.n_classes = n_classes
        self.use_interference = use_interference
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            self.random_seed = random_seed
        else:
            self.random_seed = None
        
        # Class qubit state (will be initialized during prediction)
        self.class_qubit_state: Optional[np.ndarray] = None
        
        # Feature importance learned during training
        self.feature_importance: Dict[str, float] = {}
        
        # Gate configuration (can be customized)
        self.gate_config = {
            'rotation_scale': np.pi / 2,  # Maximum rotation angle
            'interference_strength': 1.0,  # Strength of interference effects
        }
    
    def fit(
        self,
        quantum_feature_sets: List[Union[QuantumFeatureSet, QuantumFeatureSetV2]],
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Learn feature importance from training data.
        
        Args:
            quantum_feature_sets: List of QuantumFeatureSet objects
            labels: Array of class labels (0, 1, 2 for 3-class)
            feature_names: Optional list of feature names
        """
        if len(quantum_feature_sets) != len(labels):
            raise ValueError("Number of feature sets must match number of labels")
        
        # Learn feature importance from training data
        # Compute correlation between feature contribution probability and correct class
        if feature_names is None and quantum_feature_sets:
            feature_names = list(quantum_feature_sets[0].features.keys())
        
        self.feature_importance = {}
        
        for feature_name in feature_names:
            # Get feature contribution probabilities for each sample
            contributions = []
            for qfs in quantum_feature_sets:
                if feature_name in qfs.features:
                    prob = qfs.features[feature_name].get_probability_contributing()
                    contributions.append(prob)
                else:
                    contributions.append(0.0)
            
            # Compute correlation with labels (simplified: use variance as importance)
            if contributions:
                importance = np.std(contributions)  # Features with more variance are more important
                self.feature_importance[feature_name] = float(importance)
            else:
                self.feature_importance[feature_name] = 0.0
        
        # Normalize importance
        total_importance = sum(self.feature_importance.values())
        if total_importance > 0:
            self.feature_importance = {
                name: imp / total_importance
                for name, imp in self.feature_importance.items()
            }
    
    def predict_proba(
        self,
        quantum_feature_set: Union[QuantumFeatureSet, QuantumFeatureSetV2],
        return_uncertainty: bool = False
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Predict class probabilities using quantum measurement.
        
        Args:
            quantum_feature_set: QuantumFeatureSet to classify
            return_uncertainty: Whether to return uncertainty estimate
            
        Returns:
            (probabilities, uncertainty) where probabilities is array of length n_classes
        """
        # Initialize class qubit in equal superposition
        # For 3 classes, we use a 2-qubit system conceptually
        # Simplified: use single qubit with 3 regions on Bloch sphere
        class_state = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        
        # Apply Hadamard to create superposition
        class_state = hadamard_gate(class_state)
        
        # Apply feature-conditioned rotations
        for feature_name, feature in quantum_feature_set.features.items():
            # Get feature contribution probability
            p_contrib = feature.get_probability_contributing()
            
            # Get feature importance (learned during training or default)
            importance = self.feature_importance.get(feature_name, 0.0)
            
            if importance > 0 and p_contrib > 0:
                # Rotate class qubit based on feature strength
                # Strong features rotate more
                rotation_angle = self.gate_config['rotation_scale'] * importance * p_contrib
                
                # Apply rotation (Y-axis for amplitude, Z-axis for phase)
                class_state = rotate_y_gate(class_state, rotation_angle)
                
                # Apply phase shift based on feature value
                if hasattr(feature, 'value'):
                    phase_shift = feature.value * np.pi / 2
                    from quantum.gates import phase_shift_gate
                    class_state = phase_shift_gate(class_state, phase_shift)
        
        # Apply interference effects if enabled
        if self.use_interference:
            # Additional Hadamard for interference
            class_state = hadamard_gate(class_state)
        
        # Store class qubit state
        self.class_qubit_state = class_state
        
        # Convert to class probabilities
        # For 3-class: map qubit probabilities to 3 classes
        p0, p1 = get_probabilities(class_state)
        
        # Map 2-qubit probabilities to 3 classes
        # Entailment (0): high p0
        # Neutral (1): balanced
        # Contradiction (2): high p1
        probabilities = self._qubit_to_class_probs(p0, p1)
        
        # Compute uncertainty if requested
        uncertainty = None
        if return_uncertainty:
            uncertainty = self._compute_uncertainty(class_state, probabilities)
        
        return probabilities, uncertainty
    
    def _qubit_to_class_probs(self, p0: float, p1: float) -> np.ndarray:
        """
        Map 2-qubit probabilities to 3-class probabilities.
        
        Args:
            p0: Probability of |0>
            p1: Probability of |1>
            
        Returns:
            Array of 3 class probabilities
        """
        # Strategy: Use p0 for Entailment, p1 for Contradiction, balance for Neutral
        entailment_prob = p0
        contradiction_prob = p1
        neutral_prob = 1.0 - p0 - p1
        
        # Ensure non-negative and normalize
        probs = np.array([
            max(0.0, entailment_prob),
            max(0.0, neutral_prob),
            max(0.0, contradiction_prob)
        ])
        
        # Normalize
        total = np.sum(probs)
        if total > 1e-10:
            probs /= total
        else:
            # Uniform if all zero
            probs = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        
        return probs
    
    def _compute_uncertainty(
        self,
        class_state: np.ndarray,
        probabilities: np.ndarray
    ) -> float:
        """
        Compute uncertainty in prediction.
        
        Uses quantum uncertainty principle: higher entropy = more uncertainty.
        
        Args:
            class_state: Class qubit state
            probabilities: Class probabilities
            
        Returns:
            Uncertainty value in [0, 1]
        """
        # Compute entropy of probability distribution
        entropy = 0.0
        for p in probabilities:
            if p > 1e-10:
                entropy -= p * np.log2(p)
        
        # Normalize to [0, 1] (max entropy for uniform distribution)
        max_entropy = np.log2(self.n_classes)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Also consider qubit state uncertainty
        p0, p1 = get_probabilities(class_state)
        qubit_entropy = 0.0
        if p0 > 1e-10:
            qubit_entropy -= p0 * np.log2(p0)
        if p1 > 1e-10:
            qubit_entropy -= p1 * np.log2(p1)
        qubit_entropy /= np.log2(2)  # Normalize
        
        # Combine uncertainties
        uncertainty = (normalized_entropy + qubit_entropy) / 2.0
        
        return float(uncertainty)
    
    def predict(
        self,
        quantum_feature_set: Union[QuantumFeatureSet, QuantumFeatureSetV2]
    ) -> int:
        """
        Predict class label.
        
        Args:
            quantum_feature_set: QuantumFeatureSet to classify
            
        Returns:
            Predicted class label (0, 1, or 2)
        """
        probabilities, _ = self.predict_proba(quantum_feature_set, return_uncertainty=False)
        return int(np.argmax(probabilities))
    
    def get_uncertainty(self, quantum_feature_set: Union[QuantumFeatureSet, QuantumFeatureSetV2]) -> float:
        """
        Get prediction uncertainty.
        
        Args:
            quantum_feature_set: QuantumFeatureSet to evaluate
            
        Returns:
            Uncertainty value in [0, 1]
        """
        _, uncertainty = self.predict_proba(quantum_feature_set, return_uncertainty=True)
        return uncertainty if uncertainty is not None else 0.5
    
    def set_gate_config(self, config: Dict[str, float]):
        """
        Update gate configuration.
        
        Args:
            config: Dictionary with gate parameters
                - 'rotation_scale': Maximum rotation angle
                - 'interference_strength': Strength of interference effects
        """
        self.gate_config.update(config)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get learned feature importance."""
        return self.feature_importance.copy()

