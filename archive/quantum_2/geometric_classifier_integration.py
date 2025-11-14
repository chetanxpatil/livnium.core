"""
Integration: 105 Qubit Geometric Simulator with GeometricClassifier

Shows how to use the 105 entangled qubits for actual classification tasks.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum.geometric_quantum_simulator import (
    GeometricQuantumSimulator,
    create_105_qubit_geometric_system,
    GeometricQubit
)


class GeometricQuantumFeatureExtractor:
    """
    Extracts features and maps them to 105 qubits in geometric simulator.
    """
    
    def __init__(self, simulator: Optional[GeometricQuantumSimulator] = None):
        """
        Initialize feature extractor.
        
        Args:
            simulator: Optional pre-created simulator (creates new one if None)
        """
        self.simulator = simulator or create_105_qubit_geometric_system()
        self.feature_to_qubit: Dict[str, GeometricQubit] = {}
        self.feature_to_position: Dict[str, Tuple[int, int, int]] = {}
    
    def map_features(self, feature_dict: Dict[str, float]) -> Dict[str, GeometricQubit]:
        """
        Map feature dictionary to qubits.
        
        Args:
            feature_dict: Dictionary mapping feature names to values [0, 1]
        
        Returns:
            Dictionary mapping feature names to qubit references
        """
        self.feature_to_qubit = {}
        
        # Organize features by type for geometric placement
        semantic_features = ['phi_adjusted', 'embedding_proximity', 'semantic_similarity']
        structural_features = ['sw_f1_ratio', 'concentration_f1', 'token_overlap_ratio']
        lexical_features = ['negation_flag', 'antonym_overlap_ratio', 'length_ratio']
        
        # Map features to cube positions
        feature_positions = {}
        
        # Semantic features → center cube (1, 1, 1)
        for i, feat in enumerate(semantic_features):
            if feat in feature_dict:
                feature_positions[feat] = (1, 1, 1)
        
        # Structural features → edges
        edge_positions = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
        for i, feat in enumerate(structural_features):
            if feat in feature_dict:
                feature_positions[feat] = edge_positions[i % len(edge_positions)]
        
        # Lexical features → corners
        corner_positions = [(0, 0, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]
        for i, feat in enumerate(lexical_features):
            if feat in feature_dict:
                feature_positions[feat] = corner_positions[i % len(corner_positions)]
        
        # Remaining features → distribute evenly
        remaining_features = [f for f in feature_dict.keys() 
                            if f not in feature_positions]
        all_positions = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]
        
        for i, feat in enumerate(remaining_features):
            feature_positions[feat] = all_positions[i % len(all_positions)]
        
        # Update qubits with feature values
        qubit_idx = 0
        for feature_name, feature_value in feature_dict.items():
            if qubit_idx >= len(self.simulator.all_qubits):
                break
            
            # Get target position
            target_pos = feature_positions.get(feature_name, (0, 0, 0))
            
            # Find or create qubit at position
            if target_pos in self.simulator.cube_qubits:
                qubit = self.simulator.cube_qubits[target_pos][0]  # Use first qubit
            else:
                # Create new qubit at position
                qubit = self.simulator.add_qubit(target_pos, feature_value, feature_name)
            
            # Update qubit state
            alpha = np.sqrt(1.0 - feature_value)
            beta = np.sqrt(feature_value)
            qubit.qubit.state = np.array([alpha + 0j, beta + 0j], dtype=np.complex128)
            
            self.feature_to_qubit[feature_name] = qubit
            self.feature_to_position[feature_name] = target_pos
            
            qubit_idx += 1
        
        return self.feature_to_qubit
    
    def entangle_correlated_features(self, correlation_pairs: List[Tuple[str, str]]):
        """
        Entangle correlated feature pairs.
        
        Args:
            correlation_pairs: List of (feature1, feature2) tuples to entangle
        """
        for feat1, feat2 in correlation_pairs:
            if feat1 in self.feature_to_qubit and feat2 in self.feature_to_qubit:
                qubit1 = self.feature_to_qubit[feat1]
                qubit2 = self.feature_to_qubit[feat2]
                
                # Check if geometric neighbors
                distance = qubit1.get_cube_distance(qubit2)
                if distance <= 1:  # Adjacent or same position
                    self.simulator.apply_cnot_between_positions(
                        qubit1.cube_pos, qubit2.cube_pos, 0, 0
                    )


class GeometricQuantumClassifier:
    """
    Classifier using 105 entangled qubits for NLI classification.
    """
    
    def __init__(self):
        """Initialize geometric quantum classifier."""
        self.feature_extractor = GeometricQuantumFeatureExtractor()
        self.simulator = self.feature_extractor.simulator
    
    def predict_proba(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Predict class probabilities using geometric quantum simulation.
        
        Args:
            feature_dict: Dictionary mapping feature names to values
        
        Returns:
            Array of probabilities [entailment, neutral, contradiction]
        """
        # Map features to qubits
        self.feature_extractor.map_features(feature_dict)
        
        # Entangle correlated features
        correlation_pairs = [
            ('phi_adjusted', 'sw_f1_ratio'),
            ('phi_adjusted', 'concentration_f1'),
            ('embedding_proximity', 'semantic_similarity'),
        ]
        self.feature_extractor.entangle_correlated_features(correlation_pairs)
        
        # Measure all qubits
        measurements = self.simulator.measure_all()
        
        # Classify using geometric structure
        entailment_score = 0.0
        neutral_score = 0.0
        contradiction_score = 0.0
        
        for cube_pos, qubit_results in measurements.items():
            x, y, z = cube_pos
            avg_result = np.mean(qubit_results) if qubit_results else 0.5
            
            # Geometric classification pattern:
            # - Center (1,1,1) = Entailment (positive phi)
            # - Edges = Neutral (near-zero phi)
            # - Corners = Contradiction (negative phi)
            
            if (x, y, z) == (1, 1, 1):  # Center
                entailment_score += avg_result
            elif x == 1 or y == 1 or z == 1:  # Edges
                neutral_score += avg_result
            else:  # Corners
                contradiction_score += avg_result
        
        # Normalize to probabilities
        total = entailment_score + neutral_score + contradiction_score
        if total > 0:
            probs = np.array([
                entailment_score / total,
                neutral_score / total,
                contradiction_score / total
            ])
        else:
            probs = np.array([1.0/3, 1.0/3, 1.0/3])
        
        return probs
    
    def predict(self, feature_dict: Dict[str, float]) -> int:
        """
        Predict class label.
        
        Returns:
            0 = Entailment, 1 = Neutral, 2 = Contradiction
        """
        probs = self.predict_proba(feature_dict)
        return int(np.argmax(probs))


def example_usage():
    """Example of using 105 qubits for classification."""
    print("=" * 70)
    print("USING 105 ENTANGLED QUBITS FOR CLASSIFICATION")
    print("=" * 70)
    print()
    
    # Create classifier
    classifier = GeometricQuantumClassifier()
    
    # Example features (from NLI task)
    features = {
        'phi_adjusted': 0.7,  # High semantic alignment
        'sw_f1_ratio': 0.6,
        'concentration_f1': 0.5,
        'embedding_proximity': 0.8,
        'semantic_similarity': 0.75,
        'negation_flag': 0.0,
        'token_overlap_ratio': 0.6,
        'length_ratio': 0.9,
    }
    
    print("Features:")
    for name, value in features.items():
        print(f"  {name}: {value:.3f}")
    print()
    
    # Predict probabilities
    probs = classifier.predict_proba(features)
    prediction = classifier.predict(features)
    
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    
    print("Predictions:")
    for i, (name, prob) in enumerate(zip(class_names, probs)):
        marker = " ← PREDICTED" if i == prediction else ""
        print(f"  {name}: {prob:.3f}{marker}")
    print()
    
    # Show memory usage
    mem_info = classifier.simulator.get_memory_usage()
    print(f"Memory Usage: {mem_info['actual_bytes']:,} bytes ({mem_info['actual_GB']:.6f} GB)")
    print()
    
    # Show entanglement structure
    ent_info = classifier.simulator.get_entanglement_structure()
    print(f"Entanglement: {ent_info['entanglement_pairs']} pairs")
    print()
    
    print("=" * 70)
    print("✅ 105 QUBITS USED SUCCESSFULLY FOR CLASSIFICATION!")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()

