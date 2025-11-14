"""
Quantum Feature Representation v2 (Upgraded with True Entanglement)

Uses the upgraded Livnium quantum kernel with true multi-qubit entanglement.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from quantum.kernel import LivniumQubit, EntangledPair


class QuantumFeatureV2:
    """
    Represents a single feature as a quantum qubit (upgraded version).
    
    Uses LivniumQubit for true quantum operations.
    """
    
    def __init__(self, name: str, value: float, 
                 min_val: float = 0.0, max_val: float = 1.0,
                 position: Optional[Tuple[int, int, int]] = None,
                 f: int = 1,
                 initial_state: Optional[np.ndarray] = None):
        """
        Initialize a quantum feature.
        
        Args:
            name: Feature name
            value: Deterministic feature value
            min_val: Minimum possible value (for normalization)
            max_val: Maximum possible value (for normalization)
            position: 3D position for qubit (defaults to (0,0,0))
            f: Face exposure (0-3)
            initial_state: Optional [α, β] complex amplitudes
        """
        self.name = name
        self.value = value
        
        if position is None:
            position = (0, 0, 0)
        
        # Convert value to quantum state if needed
        if initial_state is None:
            normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0
            normalized = np.clip(normalized, 0.0, 1.0)
            alpha = np.sqrt(1.0 - normalized)
            beta = np.sqrt(normalized)
            initial_state = np.array([alpha + 0j, beta + 0j], dtype=np.complex128)
        
        # Create LivniumQubit
        self.qubit = LivniumQubit(position=position, f=f, initial_state=initial_state)
        self.entangled_with: List[str] = []
    
    def get_probability_contributing(self) -> float:
        """Get probability that this feature contributes (P(|1>))"""
        _, p1 = self.qubit.get_probabilities()
        return p1
    
    def measure(self) -> int:
        """Measure the feature, collapsing to |0> or |1>."""
        return self.qubit.measure()
    
    def get_uncertainty(self) -> float:
        """Get uncertainty in feature value (quantum entropy)."""
        p0, p1 = self.qubit.get_probabilities()
        
        entropy = 0.0
        if p0 > 1e-10:
            entropy -= p0 * np.log2(p0)
        if p1 > 1e-10:
            entropy -= p1 * np.log2(p1)
        
        max_entropy = np.log2(2)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(normalized_entropy)
    
    def __repr__(self) -> str:
        p0, p1 = self.qubit.get_probabilities()
        return f"QuantumFeatureV2({self.name}: {self.qubit.state_string()}, P(contributing)={p1:.3f})"


class QuantumFeatureSetV2:
    """
    Collection of quantum features with TRUE entanglement support.
    
    Uses EntangledPair for proper 2-qubit entanglement.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize quantum feature set.
        
        Args:
            random_seed: Optional seed for reproducibility
        """
        self.features: Dict[str, QuantumFeatureV2] = {}
        self.entangled_pairs: List[Tuple[str, str, EntangledPair]] = []  # (name1, name2, pair)
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def add_feature(self, name: str, value: float, 
                   min_val: float = 0.0, max_val: float = 1.0,
                   position: Optional[Tuple[int, int, int]] = None,
                   f: int = 1,
                   initial_state: Optional[np.ndarray] = None) -> QuantumFeatureV2:
        """Add a feature to the set."""
        feature = QuantumFeatureV2(name, value, min_val, max_val, position, f, initial_state)
        self.features[name] = feature
        return feature
    
    def entangle_features(self, control_name: str, target_name: str, silent: bool = False):
        """
        Entangle two features using TRUE CNOT gate (4D state space).
        
        Args:
            control_name: Name of control feature
            target_name: Name of target feature
            silent: If True, silently skip if already entangled (default: False)
        
        Returns:
            True if entanglement succeeded, False if skipped
        """
        if control_name not in self.features or target_name not in self.features:
            if not silent:
                raise ValueError(f"Features {control_name} or {target_name} not found")
            return False
        
        control = self.features[control_name]
        target = self.features[target_name]
        
        # Check if already entangled
        if control.qubit.entangled or target.qubit.entangled:
            if not silent:
                raise ValueError(f"Features {control_name} or {target_name} already entangled")
            return False
        
        # Create true entangled pair
        pair = EntangledPair.create_from_qubits(control.qubit, target.qubit)
        
        # Track entanglement
        control.entangled_with.append(target_name)
        target.entangled_with.append(control_name)
        self.entangled_pairs.append((control_name, target_name, pair))
        return True
    
    def measure_all(self) -> Dict[str, int]:
        """
        Measure all features, collapsing their states.
        
        For entangled pairs, measures both simultaneously.
        
        Returns:
            Dictionary mapping feature names to measurement results (0 or 1)
        """
        results = {}
        measured = set()
        
        # First, measure all entangled pairs
        for name1, name2, pair in self.entangled_pairs:
            if name1 not in measured and name2 not in measured:
                r1, r2 = pair.measure()
                results[name1] = r1
                results[name2] = r2
                measured.add(name1)
                measured.add(name2)
        
        # Then measure independent qubits
        for name, feature in self.features.items():
            if name not in measured:
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
            if feature.qubit.entangled:
                # For entangled qubits, get probability from pair
                pair = feature.qubit.entangled_state
                p00, p01, p10, p11 = pair.get_probabilities()
                # Probability that this qubit is |1>
                if feature == self.features[name].qubit == pair.q1:
                    prob = p10 + p11  # q1 is |1> in |10> or |11>
                else:
                    prob = p01 + p11  # q2 is |1> in |01> or |11>
                importance[name] = prob
            else:
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
                # Use expectation value (probability of |1>)
                vector.append(self.features[name].get_probability_contributing())
            else:
                vector.append(0.0)
        return np.array(vector, dtype=np.float64)
    
    def __repr__(self) -> str:
        return f"QuantumFeatureSetV2({len(self.features)} features, {len(self.entangled_pairs)} true entanglements)"


def convert_features_to_quantum_v2(
    feature_dict: Dict[str, float],
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    entanglement_pairs: Optional[List[Tuple[str, str]]] = None,
    random_seed: Optional[int] = None
) -> QuantumFeatureSetV2:
    """
    Convert deterministic features to quantum feature set (v2 with true entanglement).
    
    Args:
        feature_dict: Dictionary mapping feature names to values
        feature_ranges: Optional dictionary mapping feature names to (min, max) ranges
        entanglement_pairs: Optional list of (control, target) feature pairs to entangle
        random_seed: Optional seed for reproducibility
        
    Returns:
        QuantumFeatureSetV2 with all features as qubits
    """
    quantum_set = QuantumFeatureSetV2(random_seed=random_seed)
    
    if feature_ranges is None:
        feature_ranges = {}
    
    # Add all features
    for i, (name, value) in enumerate(feature_dict.items()):
        min_val, max_val = feature_ranges.get(name, (0.0, 1.0))
        # Assign positions in a grid
        position = (i % 5, (i // 5) % 5, i // 25)
        quantum_set.add_feature(name, value, min_val, max_val, position=position)
    
    # Entangle correlated features using TRUE entanglement
    # Only entangle pairs where both qubits are not already entangled
    if entanglement_pairs:
        for control_name, target_name in entanglement_pairs:
            if control_name in quantum_set.features and target_name in quantum_set.features:
                # Silently skip if already entangled (no warning spam)
                quantum_set.entangle_features(control_name, target_name, silent=True)
    
    return quantum_set

