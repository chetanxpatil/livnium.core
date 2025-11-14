"""
Quantum Configuration Optimizer

Uses quantum machine to calculate optimal configuration parameters for the system.
Applies quantum search to find best gate configurations, feature weights, and system parameters.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
from typing import Dict, List, Tuple, Optional
from quantum.classifier import QuantumClassifier
from quantum.features import QuantumFeatureSet, convert_features_to_quantum
from quantum.gates import get_probabilities, normalize_state
import json
from datetime import datetime


class QuantumConfigOptimizer:
    """
    Uses quantum superposition and interference to search for optimal configurations.
    
    Treats configuration parameters as qubits and uses quantum measurement
    to find optimal parameter combinations.
    """
    
    def __init__(self, random_seed: Optional[int] = 42):
        """Initialize optimizer with seed for reproducibility."""
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Configuration parameter ranges to search
        self.param_ranges = {
            'rotation_scale': (np.pi / 8, np.pi / 2, np.pi),  # min, default, max
            'interference_strength': (0.0, 1.0, 2.0),
            'feature_importance_boost': (0.5, 1.0, 2.0),  # Multiplier for important features
            'uncertainty_threshold': (0.1, 0.5, 0.9),  # When to reject predictions
        }
        
        # System parameters to optimize
        self.system_params = {
            'phi_variance_target': (0.05, 0.1, 0.2),  # Target phi variance
            'temperature_min': (0.5, 0.8, 1.2),  # Minimum temperature
            'metahead_confidence_threshold': (0.3, 0.5, 0.7),  # When to use MetaHead
        }
        
        self.results = []
        self.best_config = None
        self.best_score = -np.inf
    
    def create_config_qubit(self, param_name: str, param_type: str = 'quantum') -> np.ndarray:
        """
        Create a quantum qubit representing a configuration parameter.
        
        Args:
            param_name: Name of parameter
            param_type: 'quantum' for gate params, 'system' for system params
            
        Returns:
            Quantum state vector [α, β] representing parameter value
        """
        if param_type == 'quantum':
            ranges = self.param_ranges.get(param_name, (0.0, 0.5, 1.0))
        else:
            ranges = self.system_params.get(param_name, (0.0, 0.5, 1.0))
        
        min_val, default_val, max_val = ranges
        
        # Create superposition centered around default value
        # Higher probability amplitude for values near default
        normalized_default = (default_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        
        # Create superposition: |β|² represents normalized value
        beta = np.sqrt(normalized_default)
        alpha = np.sqrt(1.0 - normalized_default)
        
        return np.array([alpha + 0j, beta + 0j], dtype=np.complex128)
    
    def measure_config_value(self, qubit_state: np.ndarray, param_name: str, param_type: str = 'quantum') -> float:
        """
        Measure configuration parameter from qubit state.
        
        Args:
            qubit_state: Quantum state [α, β]
            param_name: Parameter name
            param_type: 'quantum' or 'system'
            
        Returns:
            Measured parameter value in its range
        """
        _, p1 = get_probabilities(qubit_state)
        
        if param_type == 'quantum':
            ranges = self.param_ranges.get(param_name, (0.0, 0.5, 1.0))
        else:
            ranges = self.system_params.get(param_name, (0.0, 0.5, 1.0))
        
        min_val, _, max_val = ranges
        
        # Map probability to parameter range
        value = min_val + p1 * (max_val - min_val)
        return float(np.clip(value, min_val, max_val))
    
    def evaluate_config(
        self,
        quantum_config: Dict[str, float],
        system_config: Dict[str, float],
        test_features: List[Dict[str, float]],
        test_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a configuration by testing it on sample data.
        
        Args:
            quantum_config: Quantum gate configuration
            system_config: System parameter configuration
            test_features: List of feature dictionaries
            test_labels: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create quantum classifier with configuration
        classifier = QuantumClassifier(
            n_classes=3,
            use_interference=quantum_config.get('interference_strength', 1.0) > 0.0,
            random_seed=self.random_seed
        )
        
        # Set gate configuration
        classifier.set_gate_config({
            'rotation_scale': quantum_config.get('rotation_scale', np.pi / 2),
            'interference_strength': quantum_config.get('interference_strength', 1.0),
        })
        
        # Convert features to quantum
        quantum_feature_sets = []
        feature_names = list(test_features[0].keys()) if test_features else []
        
        for feat_dict in test_features:
            qfs = convert_features_to_quantum(
                feat_dict,
                random_seed=self.random_seed
            )
            quantum_feature_sets.append(qfs)
        
        # Train classifier
        classifier.fit(quantum_feature_sets, test_labels, feature_names)
        
        # Evaluate
        predictions = []
        uncertainties = []
        correct = 0
        
        for qfs in quantum_feature_sets:
            probs, uncertainty = classifier.predict_proba(qfs, return_uncertainty=True)
            pred = np.argmax(probs)
            predictions.append(pred)
            uncertainties.append(uncertainty if uncertainty is not None else 0.5)
        
        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == test_labels)
        avg_uncertainty = np.mean(uncertainties)
        
        # Feature importance quality (higher variance = better discrimination)
        feature_importance = classifier.get_feature_importance()
        importance_variance = np.var(list(feature_importance.values())) if feature_importance else 0.0
        
        # Score: weighted combination of accuracy, low uncertainty, high importance variance
        score = (
            0.6 * accuracy +  # Primary: accuracy
            0.2 * (1.0 - avg_uncertainty) +  # Lower uncertainty is better
            0.2 * min(importance_variance * 10, 1.0)  # Higher variance is better (capped)
        )
        
        return {
            'accuracy': float(accuracy),
            'uncertainty': float(avg_uncertainty),
            'importance_variance': float(importance_variance),
            'score': float(score),
            'predictions': predictions,
        }
    
    def quantum_search(
        self,
        test_features: List[Dict[str, float]],
        test_labels: np.ndarray,
        n_trials: int = 100,
        n_iterations: int = 10
    ) -> Dict[str, any]:
        """
        Use quantum search to find optimal configuration.
        
        Creates superposition of all configuration parameters and uses quantum
        interference to amplify good configurations.
        
        Args:
            test_features: Test feature dictionaries
            test_labels: True labels
            n_trials: Number of quantum measurements per iteration
            n_iterations: Number of search iterations
            
        Returns:
            Best configuration found
        """
        print("=" * 70)
        print("QUANTUM CONFIGURATION OPTIMIZER")
        print("=" * 70)
        print(f"Searching for optimal configuration...")
        print(f"Parameters to optimize: {list(self.param_ranges.keys())}")
        print(f"System parameters: {list(self.system_params.keys())}")
        print(f"Trials per iteration: {n_trials}")
        print(f"Iterations: {n_iterations}")
        print("=" * 70)
        
        # Initialize configuration qubits
        config_qubits = {}
        for param_name in self.param_ranges.keys():
            config_qubits[param_name] = self.create_config_qubit(param_name, 'quantum')
        
        for param_name in self.system_params.keys():
            config_qubits[param_name] = self.create_config_qubit(param_name, 'system')
        
        best_configs = []
        
        for iteration in range(n_iterations):
            print(f"\n[Iteration {iteration + 1}/{n_iterations}]")
            print("-" * 70)
            
            # Measure configurations multiple times (quantum sampling)
            trial_results = []
            
            for trial in range(n_trials):
                # Measure all configuration parameters
                quantum_config = {}
                system_config = {}
                
                for param_name, qubit_state in config_qubits.items():
                    if param_name in self.param_ranges:
                        value = self.measure_config_value(qubit_state, param_name, 'quantum')
                        quantum_config[param_name] = value
                    else:
                        value = self.measure_config_value(qubit_state, param_name, 'system')
                        system_config[param_name] = value
                
                # Evaluate this configuration
                try:
                    metrics = self.evaluate_config(quantum_config, system_config, test_features, test_labels)
                    metrics['quantum_config'] = quantum_config.copy()
                    metrics['system_config'] = system_config.copy()
                    trial_results.append(metrics)
                except Exception as e:
                    print(f"  ⚠️  Trial {trial + 1} failed: {e}")
                    continue
                
                if (trial + 1) % 20 == 0:
                    print(f"  Completed {trial + 1}/{n_trials} trials...")
            
            # Find best configuration in this iteration
            if trial_results:
                best_trial = max(trial_results, key=lambda x: x['score'])
                best_configs.append(best_trial)
                
                print(f"\n  Best in iteration {iteration + 1}:")
                print(f"    Accuracy: {best_trial['accuracy']:.3f}")
                print(f"    Uncertainty: {best_trial['uncertainty']:.3f}")
                print(f"    Score: {best_trial['score']:.3f}")
                print(f"    Rotation scale: {best_trial['quantum_config']['rotation_scale']:.4f}")
                print(f"    Interference strength: {best_trial['quantum_config']['interference_strength']:.4f}")
                
                # Update global best
                if best_trial['score'] > self.best_score:
                    self.best_score = best_trial['score']
                    self.best_config = {
                        'quantum_config': best_trial['quantum_config'].copy(),
                        'system_config': best_trial['system_config'].copy(),
                        'metrics': {
                            'accuracy': best_trial['accuracy'],
                            'uncertainty': best_trial['uncertainty'],
                            'importance_variance': best_trial['importance_variance'],
                            'score': best_trial['score'],
                        }
                    }
                
                # Use quantum interference to amplify good configurations
                # Rotate qubits toward better values
                for param_name, qubit_state in config_qubits.items():
                    best_value = None
                    if param_name in best_trial['quantum_config']:
                        best_value = best_trial['quantum_config'][param_name]
                    elif param_name in best_trial['system_config']:
                        best_value = best_trial['system_config'][param_name]
                    
                    if best_value is not None:
                        # Get ranges
                        if param_name in self.param_ranges:
                            ranges = self.param_ranges[param_name]
                        else:
                            ranges = self.system_params[param_name]
                        
                        min_val, _, max_val = ranges
                        normalized_best = (best_value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                        
                        # Rotate qubit toward better value (quantum interference)
                        from quantum.gates import rotate_y_gate
                        rotation_angle = (normalized_best - 0.5) * np.pi / 4  # Small rotation
                        config_qubits[param_name] = rotate_y_gate(qubit_state, rotation_angle)
                        config_qubits[param_name] = normalize_state(config_qubits[param_name])
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        
        if self.best_config:
            print(f"\n✅ Best Configuration Found:")
            print(f"   Score: {self.best_config['metrics']['score']:.4f}")
            print(f"   Accuracy: {self.best_config['metrics']['accuracy']:.4f}")
            print(f"   Uncertainty: {self.best_config['metrics']['uncertainty']:.4f}")
            print(f"\n   Quantum Gate Configuration:")
            for key, value in self.best_config['quantum_config'].items():
                print(f"     {key}: {value:.6f}")
            print(f"\n   System Configuration:")
            for key, value in self.best_config['system_config'].items():
                print(f"     {key}: {value:.6f}")
        
        return self.best_config
    
    def save_config(self, filepath: str):
        """Save best configuration to JSON file."""
        if self.best_config is None:
            print("⚠️  No configuration to save. Run optimization first.")
            return
        
        config_data = {
            'timestamp': datetime.now().isoformat(),
            'best_score': float(self.best_score),
            'quantum_config': {k: float(v) for k, v in self.best_config['quantum_config'].items()},
            'system_config': {k: float(v) for k, v in self.best_config['system_config'].items()},
            'metrics': {k: float(v) for k, v in self.best_config['metrics'].items()},
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\n💾 Configuration saved to: {filepath}")
    
    def generate_application_code(self) -> str:
        """
        Generate Python code to apply the optimal configuration.
        
        Returns:
            Python code string that can be executed to apply configuration
        """
        if self.best_config is None:
            return "# No configuration available. Run optimization first."
        
        code_lines = [
            "# Apply Optimal Quantum Configuration",
            "# Generated by QuantumConfigOptimizer",
            "",
            "from quantum.classifier import QuantumClassifier",
            "",
            "# Quantum Gate Configuration",
            "quantum_config = {"
        ]
        
        for key, value in self.best_config['quantum_config'].items():
            code_lines.append(f"    '{key}': {value:.6f},")
        
        code_lines.extend([
            "}",
            "",
            "# System Configuration",
            "system_config = {"
        ])
        
        for key, value in self.best_config['system_config'].items():
            code_lines.append(f"    '{key}': {value:.6f},")
        
        code_lines.extend([
            "}",
            "",
            "# Apply to QuantumClassifier",
            "classifier.set_gate_config(quantum_config)",
            "",
            "# Apply system parameters (update config.py or relevant files)",
            "# phi_variance_target = system_config['phi_variance_target']",
            "# temperature_min = system_config['temperature_min']",
            "# metahead_confidence_threshold = system_config['metahead_confidence_threshold']",
        ])
        
        return "\n".join(code_lines)


def create_test_data(n_samples: int = 50) -> Tuple[List[Dict[str, float]], np.ndarray]:
    """
    Create test data with realistic feature distributions.
    
    Args:
        n_samples: Number of samples
        
    Returns:
        (feature_dicts, labels)
    """
    np.random.seed(42)
    
    features_list = []
    labels = []
    
    # Simulate realistic feature distributions based on system analysis
    for i in range(n_samples):
        # Simulate phi_adjusted with varying variance
        phi_adjusted = np.random.normal(0.0, 0.15)  # Increased variance
        
        # Simulate other features
        sw_distribution = np.random.uniform(0.0, 1.0)
        concentration = np.random.uniform(0.0, 1.0)
        embedding_proximity = np.random.uniform(0.0, 1.0)
        negation_flag = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
        
        # Create label based on features (simplified model)
        if phi_adjusted > 0.1 and embedding_proximity > 0.6:
            label = 0  # Entailment
        elif phi_adjusted < -0.1 or negation_flag > 0.5:
            label = 2  # Contradiction
        else:
            label = 1  # Neutral
        
        features = {
            'phi_adjusted': float(phi_adjusted),
            'sw_distribution': float(sw_distribution),
            'concentration': float(concentration),
            'embedding_proximity': float(embedding_proximity),
            'negation_flag': float(negation_flag),
        }
        
        features_list.append(features)
        labels.append(label)
    
    return features_list, np.array(labels)


if __name__ == "__main__":
    print("🚀 Starting Quantum Configuration Optimization...")
    print()
    
    # Create test data
    print("📊 Creating test data...")
    test_features, test_labels = create_test_data(n_samples=100)
    print(f"   Created {len(test_features)} samples with {len(test_features[0])} features")
    print()
    
    # Initialize optimizer
    optimizer = QuantumConfigOptimizer(random_seed=42)
    
    # Run quantum search
    best_config = optimizer.quantum_search(
        test_features=test_features,
        test_labels=test_labels,
        n_trials=50,  # Reduced for faster testing
        n_iterations=5
    )
    
    # Save configuration
    config_path = "quantum/optimal_config.json"
    optimizer.save_config(config_path)
    
    # Generate application code
    print("\n" + "=" * 70)
    print("APPLICATION CODE")
    print("=" * 70)
    application_code = optimizer.generate_application_code()
    print(application_code)
    
    # Save application code
    code_path = "quantum/apply_optimal_config.py"
    with open(code_path, 'w') as f:
        f.write(application_code)
    print(f"\n💾 Application code saved to: {code_path}")
    
    print("\n" + "=" * 70)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review optimal_config.json for the best configuration")
    print("2. Run apply_optimal_config.py to apply the configuration")
    print("3. Test the system with the new configuration")
    print("4. Monitor accuracy and uncertainty metrics")

