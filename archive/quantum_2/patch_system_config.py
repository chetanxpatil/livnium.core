"""
Patch System with Optimal Quantum Configuration

This script applies the optimal quantum configuration to the system files.
Run this after quantum optimization to automatically apply the best configuration.
"""

import json
import sys
from pathlib import Path


def load_config():
    """Load optimal configuration."""
    config_path = Path("quantum/optimal_config.json")
    if not config_path.exists():
        print("❌ Configuration file not found. Run quantum/optimize_config.py first.")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        return json.load(f)


def patch_geometric_classifier(config):
    """Patch GeometricClassifier to apply optimal quantum gate configuration."""
    print("🔧 Patching GeometricClassifier...")
    
    classifier_path = Path("layers/layer3/meta/geometric_classifier.py")
    if not classifier_path.exists():
        print(f"⚠️  File not found: {classifier_path}")
        return False
    
    quantum_config = config['quantum_config']
    
    # Read current file
    with open(classifier_path, 'r') as f:
        content = f.read()
    
    # Check if quantum classifier is initialized
    if 'self.quantum_classifier = QuantumClassifier(' not in content:
        print("⚠️  Quantum classifier initialization not found. Quantum mode may not be enabled.")
        return False
    
    # Find the initialization and add configuration
    init_pattern = 'self.quantum_classifier = QuantumClassifier('
    if init_pattern in content:
        # Find the end of QuantumClassifier initialization
        start_idx = content.find(init_pattern)
        end_idx = content.find('\n', start_idx + len(init_pattern))
        if end_idx == -1:
            end_idx = len(content)
        
        # Check if configuration is already applied
        if 'set_gate_config' in content:
            print("✅ Quantum gate configuration already applied.")
            return True
        
        # Insert configuration after initialization
        insert_pos = content.find('\n', end_idx)
        if insert_pos == -1:
            insert_pos = len(content)
        
        config_code = f"""
                # Apply optimal quantum gate configuration (from quantum optimization)
                self.quantum_classifier.set_gate_config({{
                    'rotation_scale': {quantum_config['rotation_scale']:.6f},
                    'interference_strength': {quantum_config['interference_strength']:.6f},
                }})
"""
        
        # Insert after the quantum classifier initialization block
        # Find the closing of the if self.use_quantum block
        if_block_start = content.rfind('if self.use_quantum:', 0, insert_pos)
        if if_block_start != -1:
            # Find the end of the if block (look for next unindented line or end of __init__)
            lines = content[:insert_pos].split('\n')
            if_start_line = content[:insert_pos].count('\n', 0, if_block_start)
            indent_level = len(lines[if_start_line]) - len(lines[if_start_line].lstrip())
            
            # Find where to insert (after the print statement in the if block)
            for i in range(if_start_line + 1, len(lines)):
                if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= indent_level:
                    # End of if block
                    insert_line = i
                    break
            else:
                insert_line = len(lines)
            
            # Insert the configuration code
            lines.insert(insert_line, config_code.rstrip())
            content = '\n'.join(lines) + content[insert_pos:]
        else:
            # Fallback: insert after quantum classifier creation
            content = content[:insert_pos] + config_code + content[insert_pos:]
        
        # Write back
        with open(classifier_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Patched {classifier_path}")
        print("   Applied optimal quantum gate configuration:")
        print(f"     rotation_scale: {quantum_config['rotation_scale']:.6f}")
        print(f"     interference_strength: {quantum_config['interference_strength']:.6f}")
        return True
    
    return False


def generate_config_patch_file(config):
    """Generate a Python file that can be imported to apply configuration."""
    print("📝 Generating configuration patch file...")
    
    quantum_config = config['quantum_config']
    system_config = config['system_config']
    
    patch_code = f'''"""
Optimal Quantum Configuration (Auto-generated)

This configuration was calculated by quantum optimization.
Import this module and call apply_optimal_config() to apply it.
"""

QUANTUM_CONFIG = {{
    'rotation_scale': {quantum_config['rotation_scale']:.6f},
    'interference_strength': {quantum_config['interference_strength']:.6f},
    'feature_importance_boost': {quantum_config.get('feature_importance_boost', 1.0):.6f},
    'uncertainty_threshold': {quantum_config.get('uncertainty_threshold', 0.5):.6f},
}}

SYSTEM_CONFIG = {{
    'phi_variance_target': {system_config['phi_variance_target']:.6f},
    'temperature_min': {system_config['temperature_min']:.6f},
    'metahead_confidence_threshold': {system_config['metahead_confidence_threshold']:.6f},
}}

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
        print(f"   Rotation scale: {{QUANTUM_CONFIG['rotation_scale']:.6f}}")
        print(f"   Interference strength: {{QUANTUM_CONFIG['interference_strength']:.6f}}")
    else:
        print("⚠️  Classifier does not have quantum_classifier. Enable quantum mode first.")

def get_system_config_recommendations():
    """Get recommendations for system parameter updates."""
    return {{
        'phi_variance_target': SYSTEM_CONFIG['phi_variance_target'],
        'temperature_min': SYSTEM_CONFIG['temperature_min'],
        'metahead_confidence_threshold': SYSTEM_CONFIG['metahead_confidence_threshold'],
    }}
'''
    
    patch_path = Path("quantum/optimal_config_patch.py")
    with open(patch_path, 'w') as f:
        f.write(patch_code)
    
    print(f"✅ Generated {patch_path}")
    return patch_path


def main():
    """Main function."""
    print("=" * 70)
    print("APPLY OPTIMAL QUANTUM CONFIGURATION TO SYSTEM")
    print("=" * 70)
    print()
    
    # Load configuration
    config = load_config()
    print(f"✅ Loaded optimal configuration (score: {config['best_score']:.4f})")
    print()
    
    # Patch GeometricClassifier
    patched = patch_geometric_classifier(config)
    
    if not patched:
        print("\n⚠️  Could not automatically patch GeometricClassifier.")
        print("   You can manually apply the configuration using:")
        print("   - quantum/apply_quantum_config.py")
        print("   - quantum/optimal_config_patch.py")
    
    # Generate patch file
    patch_path = generate_config_patch_file(config)
    
    print("\n" + "=" * 70)
    print("✅ CONFIGURATION PATCH COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review quantum/CONFIGURATION_RESULTS.md for details")
    print("2. Test the system with the new configuration")
    print("3. Monitor accuracy improvements")
    print(f"\nTo apply configuration programmatically:")
    print(f"   from quantum.optimal_config_patch import apply_optimal_config")
    print(f"   apply_optimal_config(classifier_instance)")


if __name__ == "__main__":
    main()

