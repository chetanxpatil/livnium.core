
# Apply optimal quantum configuration to GeometricClassifier
# Add this to your training script or main.py

from layers.layer3.meta.geometric_classifier import GeometricClassifier

# Update classifier configuration
if hasattr(classifier, 'quantum_classifier') and classifier.quantum_classifier:
    classifier.quantum_classifier.set_gate_config({
        'rotation_scale': 1.069798,
        'interference_strength': 1.000000,
        'feature_importance_boost': 0.553630,
        'uncertainty_threshold': 0.500000,
    })
    print("✅ Quantum gate configuration applied")
