"""
Apply Optimal Quantum Configuration to System

Reads optimal configuration from quantum optimization and applies it to debug the system.
"""

import json
import sys
import numpy as np
from pathlib import Path


def load_optimal_config(config_path: str = "quantum/optimal_config.json") -> dict:
    """Load optimal configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("   Run quantum/optimize_config.py first to generate optimal configuration.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing configuration file: {e}")
        sys.exit(1)


def apply_quantum_config(config: dict):
    """Apply quantum gate configuration to GeometricClassifier."""
    print("🔧 Applying Quantum Gate Configuration...")
    
    quantum_config = config.get('quantum_config', {})
    
    print("\n   Quantum Gate Parameters:")
    for key, value in quantum_config.items():
        print(f"     {key}: {value:.6f}")
    
    # Generate code to apply configuration
    code = """
# Apply optimal quantum configuration to GeometricClassifier
# Add this to your training script or main.py

from layers.layer3.meta.geometric_classifier import GeometricClassifier

# Update classifier configuration
if hasattr(classifier, 'quantum_classifier') and classifier.quantum_classifier:
    classifier.quantum_classifier.set_gate_config({
"""
    
    for key, value in quantum_config.items():
        code += f"        '{key}': {value:.6f},\n"
    
    code += """    })
    print("✅ Quantum gate configuration applied")
"""
    
    print("\n   Generated code:")
    print(code)
    
    return code


def apply_system_config(config: dict):
    """Apply system configuration parameters."""
    print("\n🔧 Applying System Configuration...")
    
    system_config = config.get('system_config', {})
    
    print("\n   System Parameters:")
    for key, value in system_config.items():
        print(f"     {key}: {value:.6f}")
    
    # Generate code to apply system configuration
    code = """
# Apply optimal system configuration
# Update config.py or relevant configuration files

"""
    
    for key, value in system_config.items():
        if key == 'phi_variance_target':
            code += f"# Target phi variance: {value:.6f}\n"
            code += "# Update phi variance computation to target this value\n"
        elif key == 'temperature_min':
            code += f"# Minimum temperature: {value:.6f}\n"
            code += "# Update temperature clamping in main.py\n"
        elif key == 'metahead_confidence_threshold':
            code += f"# MetaHead confidence threshold: {value:.6f}\n"
            code += "# Update MetaHead usage threshold\n"
    
    print("\n   Configuration recommendations:")
    print(code)
    
    return code


def generate_debug_report(config: dict):
    """Generate a debug report with configuration insights."""
    print("\n📊 Generating Debug Report...")
    
    metrics = config.get('metrics', {})
    quantum_config = config.get('quantum_config', {})
    system_config = config.get('system_config', {})
    
    report = f"""
# Quantum Configuration Debug Report
Generated: {config.get('timestamp', 'Unknown')}

## Optimal Configuration Found

### Performance Metrics
- Accuracy: {metrics.get('accuracy', 0):.4f}
- Uncertainty: {metrics.get('uncertainty', 0):.4f}
- Importance Variance: {metrics.get('importance_variance', 0):.4f}
- Overall Score: {metrics.get('score', 0):.4f}

### Quantum Gate Configuration
"""
    
    for key, value in quantum_config.items():
        report += f"- {key}: {value:.6f}\n"
    
    report += "\n### System Configuration\n"
    for key, value in system_config.items():
        report += f"- {key}: {value:.6f}\n"
    
    report += """
## Debugging Insights

### 1. Rotation Scale Analysis
"""
    
    rotation_scale = quantum_config.get('rotation_scale', np.pi / 2)
    if rotation_scale < np.pi / 4:
        report += "- ⚠️  Low rotation scale detected - features may not rotate enough\n"
        report += "  → Consider increasing rotation_scale for better feature discrimination\n"
    elif rotation_scale > np.pi:
        report += "- ⚠️  High rotation scale detected - may cause over-rotation\n"
        report += "  → Consider decreasing rotation_scale for more stable predictions\n"
    else:
        report += "- ✅ Rotation scale is in optimal range\n"
    
    report += "\n### 2. Interference Strength Analysis\n"
    
    interference = quantum_config.get('interference_strength', 1.0)
    if interference < 0.5:
        report += "- ⚠️  Low interference strength - quantum effects may be weak\n"
        report += "  → Consider increasing interference_strength for better feature interactions\n"
    elif interference > 1.5:
        report += "- ⚠️  High interference strength - may cause instability\n"
        report += "  → Consider decreasing interference_strength for more stable predictions\n"
    else:
        report += "- ✅ Interference strength is in optimal range\n"
    
    report += "\n### 3. System Parameter Analysis\n"
    
    phi_variance = system_config.get('phi_variance_target', 0.1)
    if phi_variance < 0.05:
        report += "- ⚠️  Low phi variance target - may cause low feature importance\n"
        report += "  → Current system has phi variance issues - this confirms the problem\n"
    else:
        report += f"- ✅ Phi variance target ({phi_variance:.4f}) should improve feature discrimination\n"
    
    temperature = system_config.get('temperature_min', 0.8)
    if temperature < 0.7:
        report += "- ⚠️  Low temperature minimum - may cause phi collapse\n"
        report += "  → System needs higher temperature for exploration\n"
    else:
        report += f"- ✅ Temperature minimum ({temperature:.4f}) should prevent collapse\n"
    
    report += """
## Recommended Actions

1. **Apply Quantum Configuration**
   - Update GeometricClassifier to use optimal gate parameters
   - Test with new rotation_scale and interference_strength

2. **Update System Parameters**
   - Adjust phi variance computation to target optimal value
   - Update temperature clamping if needed
   - Adjust MetaHead confidence threshold

3. **Monitor Performance**
   - Track accuracy improvements
   - Monitor uncertainty metrics
   - Check feature importance distribution

4. **Iterate**
   - If accuracy improves, fine-tune parameters
   - If accuracy doesn't improve, check other system issues
"""
    
    return report


def main():
    """Main function to apply optimal configuration."""
    print("=" * 70)
    print("APPLY OPTIMAL QUANTUM CONFIGURATION")
    print("=" * 70)
    print()
    
    # Load configuration
    config_path = "quantum/optimal_config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    config = load_optimal_config(config_path)
    
    print(f"✅ Loaded configuration from: {config_path}")
    print(f"   Best score: {config.get('best_score', 0):.4f}")
    print()
    
    # Apply configurations
    quantum_code = apply_quantum_config(config)
    system_code = apply_system_config(config)
    
    # Generate debug report
    debug_report = generate_debug_report(config)
    
    # Save outputs
    output_dir = Path("quantum")
    output_dir.mkdir(exist_ok=True)
    
    # Save application code
    apply_code_path = output_dir / "apply_quantum_config.py"
    with open(apply_code_path, 'w') as f:
        f.write(quantum_code)
    print(f"\n💾 Application code saved to: {apply_code_path}")
    
    # Save debug report
    debug_report_path = output_dir / "debug_report.md"
    with open(debug_report_path, 'w') as f:
        f.write(debug_report)
    print(f"💾 Debug report saved to: {debug_report_path}")
    
    print("\n" + "=" * 70)
    print("✅ CONFIGURATION APPLICATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review debug_report.md for insights")
    print("2. Apply quantum configuration to GeometricClassifier")
    print("3. Update system parameters as recommended")
    print("4. Run training and monitor improvements")


if __name__ == "__main__":
    main()

