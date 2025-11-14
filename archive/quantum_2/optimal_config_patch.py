"""
Optimal Quantum Configuration (Auto-generated)

This configuration was calculated by quantum optimization.
Import this module and call apply_optimal_config() to apply it.
"""

QUANTUM_CONFIG = {
    'rotation_scale': 1.069798,
    'interference_strength': 1.000000,
    'feature_importance_boost': 0.553630,
    'uncertainty_threshold': 0.500000,
}

SYSTEM_CONFIG = {
    'phi_variance_target': 0.055363,
    'temperature_min': 0.672422,
    'metahead_confidence_threshold': 0.500000,
}

def apply_optimal_config(classifier=None):
    """
    Apply optimal quantum configuration to a classifier.
    
    Args:
        classifier: GeometricClassifier instance (optional)
    """
    if classifier is None:
        # Try to get from MetaHead if available
        try:
            from layers.layer3.meta.metahead import MetaHead
            # This would need to be called from context where MetaHead exists
            print("⚠️  Please pass classifier instance directly")
            return
        except ImportError:
            print("⚠️  Cannot auto-detect classifier")
            return
    
    if hasattr(classifier, 'quantum_classifier') and classifier.quantum_classifier:
        classifier.quantum_classifier.set_gate_config(QUANTUM_CONFIG)
        print("✅ Optimal quantum configuration applied")
        print(f"   Rotation scale: {QUANTUM_CONFIG['rotation_scale']:.6f}")
        print(f"   Interference strength: {QUANTUM_CONFIG['interference_strength']:.6f}")
    else:
        print("⚠️  Classifier does not have quantum_classifier. Enable quantum mode first.")

def get_system_config_recommendations():
    """Get recommendations for system parameter updates."""
    return {
        'phi_variance_target': SYSTEM_CONFIG['phi_variance_target'],
        'temperature_min': SYSTEM_CONFIG['temperature_min'],
        'metahead_confidence_threshold': SYSTEM_CONFIG['metahead_confidence_threshold'],
    }
