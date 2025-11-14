"""
Quantum Feature Representation for Livnium

Converts deterministic features to quantum qubit states, enabling:
- Uncertainty modeling
- Feature entanglement
- Probabilistic feature importance
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from quantum.gates import (
    create_superposition_from_value,
    get_probabilities,
    normalize_state,
    cnot_gate
)


class QuantumFeature:
    """
    Represents a single feature as a quantum qubit.
    
    Each feature exists in superposition: |ψ> = α|0> + β|1>
    where |0> = feature not contributing, |1> = feature contributing
    """
    
    def __init__(self, name: str, value: float, 
                 min_val: float = 0.0, max_val: float = 1.0,
                 initial_state: Optional[np.ndarray] = None):
        """
        Initialize a quantum feature.
        
        Args:
            name: Feature name
            value: Deterministic feature value
            min_val: Minimum possible value (for normalization)
            max_val: Maximum possible value (for normalization)
            initial_state: Optional [α, β] complex amplitudes. If None, converts value to superposition.
        """
        self.name = name
        self.value = value
        
        if initial_state is None:
            # Convert deterministic value to quantum superposition
            self.state_vector = create_superposition_from_value(value, min_val, max_val)
        else:
            self.state_vector = np.array(initial_state, dtype=np.complex128)
            self.state_vector = normalize_state(self.state_vector)
        
        self.entangled_with: List[str] = []  # Names of entangled features
        self.measurement_history: List[int] = []
    
    def get_probability_contributing(self) -> float:
        """Get probability that this feature contributes (P(|1>))"""
        _, p1 = get_probabilities(self.state_vector)
        return p1
    
    def measure(self) -> int:
        """
        Measure the feature, collapsing to |0> (not contributing) or |1> (contributing).
        
        Returns:
            0 if not contributing, 1 if contributing
        """
        p0, p1 = get_probabilities(self.state_vector)
        result = 1 if np.random.rand() < p1 else 0
        
        # Collapse state
        if result == 0:
            self.state_vector = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        else:
            self.state_vector = np.array([0.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        
        self.measurement_history.append(result)
        return result
    
    def get_deterministic_value(self) -> float:
        """Get deterministic value from quantum state (expectation value)"""
        p0, p1 = get_probabilities(self.state_vector)
        # Map probability back to value range
        # This is a simplified mapping - assumes value is proportional to probability
        return p1  # For now, return probability as value
    
    def get_uncertainty(self) -> float:
        """
        Get uncertainty in feature value.
        
        Uses quantum uncertainty: entropy of the probability distribution.
        Higher entropy = more uncertainty.
        
        Returns:
            Uncertainty value in [0, 1]
        """
        p0, p1 = get_probabilities(self.state_vector)
        
        # Compute entropy
        entropy = 0.0
        if p0 > 1e-10:
            entropy -= p0 * np.log2(p0)
        if p1 > 1e-10:
            entropy -= p1 * np.log2(p1)
        
        # Normalize to [0, 1] (max entropy = 1 for 50/50 superposition)
        max_entropy = np.log2(2)  # 1 bit for binary
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(normalized_entropy)
    
    def __repr__(self) -> str:
        p0, p1 = get_probabilities(self.state_vector)
        alpha, beta = self.state_vector[0], self.state_vector[1]
        return f"QuantumFeature({self.name}: {alpha.real:.3f}|0> + {beta.real:.3f}|1>, P(contributing)={p1:.3f})"


class QuantumFeatureSet:
    """
    Collection of quantum features with entanglement support.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize quantum feature set.
        
        Args:
            random_seed: Optional seed for reproducibility
        """
        self.features: Dict[str, QuantumFeature] = {}
        self.entanglement_pairs: List[Tuple[str, str]] = []
        self.random_seed = random_seed
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def add_feature(self, name: str, value: float, 
                   min_val: float = 0.0, max_val: float = 1.0,
                   initial_state: Optional[np.ndarray] = None) -> QuantumFeature:
        """Add a feature to the set."""
        feature = QuantumFeature(name, value, min_val, max_val, initial_state)
        self.features[name] = feature
        return feature
    
    def entangle_features(self, control_name: str, target_name: str):
        """
        Entangle two features using CNOT gate.
        
        Args:
            control_name: Name of control feature
            target_name: Name of target feature
        """
        if control_name not in self.features or target_name not in self.features:
            raise ValueError(f"Features {control_name} or {target_name} not found")
        
        control = self.features[control_name]
        target = self.features[target_name]
        
        # Apply CNOT
        control.state_vector, target.state_vector = cnot_gate(
            control.state_vector, target.state_vector
        )
        
        # Track entanglement
        control.entangled_with.append(target_name)
        target.entangled_with.append(control_name)
        self.entanglement_pairs.append((control_name, target_name))
    
    def measure_all(self) -> Dict[str, int]:
        """
        Measure all features, collapsing their states.
        
        Returns:
            Dictionary mapping feature names to measurement results (0 or 1)
        """
        results = {}
        for name, feature in self.features.items():
            results[name] = feature.measure()
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance as probabilities.
        
        Returns:
            Dictionary mapping feature names to contribution probabilities
        """
        importance = {}
        for name, feature in self.features.items():
            importance[name] = feature.get_probability_contributing()
        return importance
    
    def get_uncertainty(self) -> Dict[str, float]:
        """
        Get uncertainty for all features.
        
        Returns:
            Dictionary mapping feature names to uncertainty values [0, 1]
        """
        uncertainty = {}
        for name, feature in self.features.items():
            uncertainty[name] = feature.get_uncertainty()
        return uncertainty
    
    def to_deterministic_vector(self, feature_names: List[str]) -> np.ndarray:
        """
        Convert quantum features to deterministic feature vector.
        
        Args:
            feature_names: Ordered list of feature names
            
        Returns:
            NumPy array of deterministic feature values
        """
        vector = []
        for name in feature_names:
            if name in self.features:
                vector.append(self.features[name].get_deterministic_value())
            else:
                vector.append(0.0)
        return np.array(vector, dtype=np.float64)
    
    def __repr__(self) -> str:
        return f"QuantumFeatureSet({len(self.features)} features, {len(self.entanglement_pairs)} entanglements)"


def convert_features_to_quantum(
    feature_dict: Dict[str, float],
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    entanglement_pairs: Optional[List[Tuple[str, str]]] = None,
    random_seed: Optional[int] = None
) -> QuantumFeatureSet:
    """
    Convert deterministic features to quantum feature set.
    
    Args:
        feature_dict: Dictionary mapping feature names to values
        feature_ranges: Optional dictionary mapping feature names to (min, max) ranges
        entanglement_pairs: Optional list of (control, target) feature pairs to entangle
        random_seed: Optional seed for reproducibility
        
    Returns:
        QuantumFeatureSet with all features as qubits
    """
    quantum_set = QuantumFeatureSet(random_seed=random_seed)
    
    # Default ranges if not provided
    if feature_ranges is None:
        feature_ranges = {}
    
    # Add all features
    for name, value in feature_dict.items():
        min_val, max_val = feature_ranges.get(name, (0.0, 1.0))
        quantum_set.add_feature(name, value, min_val, max_val)
    
    # Entangle correlated features
    if entanglement_pairs:
        for control_name, target_name in entanglement_pairs:
            if control_name in quantum_set.features and target_name in quantum_set.features:
                quantum_set.entangle_features(control_name, target_name)
    
    return quantum_set

