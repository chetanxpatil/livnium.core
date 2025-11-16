#!/usr/bin/env python3
"""
Dual Cube System Demo

Demonstrates the "−3×−3×−3" as an anti-cube:
- +3×+3×+3 = positive semantic space (stable meanings)
- −3×−3×−3 = anti-semantic space (contradictions, conflicts)

This is NOT just a sign flip - it's two linked lattices with cross-cube dynamics.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum.hierarchical.geometry.dual_cube_system import DualCubeSystem


def demo_basic_dual_cube():
    """Basic demonstration of dual cube system."""
    print("=" * 70)
    print("DUAL CUBE SYSTEM DEMO")
    print("=" * 70)
    print("\nThis implements −3×−3×−3 as an anti-cube:")
    print("  +3×+3×+3 = positive semantic space (stable meanings)")
    print("  −3×−3×−3 = anti-semantic space (contradictions, conflicts)")
    print("\nThis is NOT just a sign flip - it's two linked lattices!")
    
    system = DualCubeSystem(base_dimension=3, num_levels=3)
    
    # Add states to positive cube (stable meanings)
    print("\n1. Adding states to positive cube (stable meanings)...")
    for i in range(10):
        x = i * 0.1
        y = i * 0.1
        z = i * 0.1
        system.add_positive_state((x, y, z), amplitude=1.0+0j)
    
    print(f"   Added {len(system.positive_cube.states)} states to positive cube")
    
    # Add states to negative cube (contradictions)
    print("\n2. Adding states to negative cube (contradictions)...")
    for i in range(5):
        x = i * 0.1
        y = i * 0.1
        z = i * 0.1
        system.add_negative_state((x, y, z), amplitude=0.5+0j)
    
    print(f"   Added {len(system.negative_cube.states)} states to negative cube")
    
    # Add dual states (exist in both)
    print("\n3. Adding dual states (exist in both cubes)...")
    for i in range(3):
        x = (i + 10) * 0.1
        y = (i + 10) * 0.1
        z = (i + 10) * 0.1
        dual = system.add_dual_state(
            (x, y, z),
            positive_amplitude=1.0+0j,
            negative_amplitude=0.3+0j
        )
        print(f"   Dual state {i}: energy={dual.energy:.3f}, contradiction={dual.contradiction_score:.3f}")
    
    # Get summary
    summary = system.get_system_summary()
    print(f"\n4. System Summary:")
    print(f"   Positive cube: {summary['positive_cube']['states']} states, "
          f"energy={summary['positive_cube']['energy']:.3f}")
    print(f"   Negative cube: {summary['negative_cube']['states']} states, "
          f"energy={summary['negative_cube']['energy']:.3f}")
    print(f"   Dual states: {summary['dual_states']}")
    print(f"   Decoherence fraction: {summary['decoherence_fraction']:.3f}")


def demo_contradiction_detection():
    """Demonstrate contradiction detection and movement to negative cube."""
    print("\n" + "=" * 70)
    print("CONTRADICTION DETECTION DEMO")
    print("=" * 70)
    
    system = DualCubeSystem(base_dimension=3, num_levels=3)
    
    # Add some stable states
    print("\n1. Adding stable states to positive cube...")
    stable_states = []
    for i in range(5):
        state = system.add_positive_state((i*0.1, i*0.1, i*0.1), amplitude=1.0+0j)
        stable_states.append(state)
    
    # Add a contradictory state (close coordinates, conflicting amplitude)
    print("\n2. Adding contradictory state...")
    contradictory_state = system.add_positive_state(
        (0.05, 0.05, 0.05),  # Close to first stable state
        amplitude=2.0+0j  # Conflicting amplitude
    )
    
    # Detect contradiction
    contradiction_score = system.detect_contradiction(contradictory_state, stable_states)
    print(f"   Contradiction score: {contradiction_score:.3f}")
    
    # Move to negative cube if contradictory
    if contradiction_score > system.contradiction_threshold:
        neg_state = system.move_to_negative_cube(contradictory_state, contradiction_score)
        if neg_state:
            print(f"   ✅ Moved to negative cube: {neg_state.coordinates}")
            print(f"      Negative amplitude: {neg_state.amplitude}")
    
    # Summary
    summary = system.get_system_summary()
    print(f"\n3. After contradiction handling:")
    print(f"   Positive states: {summary['positive_cube']['states']}")
    print(f"   Negative states: {summary['negative_cube']['states']}")


def demo_decoherence_drift():
    """Demonstrate semantic decoherence as cross-cube drift."""
    print("\n" + "=" * 70)
    print("DECOHERENCE DRIFT DEMO")
    print("=" * 70)
    print("\nAs operations overload states, meaning drains into the anti-cube.")
    
    system = DualCubeSystem(base_dimension=3, num_levels=3)
    
    # Add states to positive cube
    print("\n1. Adding states to positive cube...")
    for i in range(20):
        system.add_positive_state((i*0.1, i*0.1, i*0.1), amplitude=1.0+0j)
    
    # Measure initial decoherence
    initial = system.get_decoherence_measure()
    print(f"   Initial decoherence: {initial['decoherence_fraction']:.3f}")
    print(f"   Positive energy: {initial['positive_energy']:.3f}")
    print(f"   Negative energy: {initial['negative_energy']:.3f}")
    
    # Apply decoherence drift
    print("\n2. Applying decoherence drift (meaning drains to negative cube)...")
    system.apply_decoherence_drift(rate=0.2)  # 20% drift
    
    # Measure after drift
    after = system.get_decoherence_measure()
    print(f"   After drift decoherence: {after['decoherence_fraction']:.3f}")
    print(f"   Positive energy: {after['positive_energy']:.3f}")
    print(f"   Negative energy: {after['negative_energy']:.3f}")
    print(f"   Energy leaked: {after['negative_energy'] - initial['negative_energy']:.3f}")
    
    print(f"\n   ✅ Decoherence confirmed: {after['decoherence_fraction']*100:.1f}% in negative cube")


def demo_cancellation():
    """Demonstrate cancellation: opposite patterns cancel out."""
    print("\n" + "=" * 70)
    print("CANCELLATION DEMO")
    print("=" * 70)
    print("\nIf pattern appears in both cubes with opposite sign, they cancel.")
    
    system = DualCubeSystem(base_dimension=3, num_levels=3)
    
    # Add matching states in both cubes (will cancel)
    print("\n1. Adding matching states in both cubes...")
    coords = (0.5, 0.5, 0.5)
    pos_state = system.add_positive_state(coords, amplitude=1.0+0j)
    neg_state = system.add_negative_state(coords, amplitude=1.0+0j)
    
    print(f"   Positive state: {pos_state.coordinates}, amplitude={pos_state.amplitude}")
    print(f"   Negative state: {neg_state.coordinates}, amplitude={neg_state.amplitude}")
    
    # Before cancellation
    before = system.get_system_summary()
    print(f"\n2. Before cancellation:")
    print(f"   Positive states: {before['positive_cube']['states']}")
    print(f"   Negative states: {before['negative_cube']['states']}")
    
    # Apply cancellation
    print("\n3. Applying cancellation...")
    system.apply_cancellation()
    
    # After cancellation
    after = system.get_system_summary()
    print(f"   After cancellation:")
    print(f"   Positive states: {after['positive_cube']['states']}")
    print(f"   Negative states: {after['negative_cube']['states']}")
    
    if after['positive_cube']['states'] < before['positive_cube']['states']:
        print(f"   ✅ States cancelled (neutralized/forgotten)")


def demo_confusion_diagnosis():
    """Demonstrate confusion diagnosis."""
    print("\n" + "=" * 70)
    print("CONFUSION DIAGNOSIS DEMO")
    print("=" * 70)
    print("\nDiagnose if input lives mostly in −cube (confused).")
    
    system = DualCubeSystem(base_dimension=3, num_levels=3)
    
    # Add clear state (mostly in positive)
    print("\n1. Adding clear state (mostly in positive cube)...")
    system.add_positive_state((0.1, 0.1, 0.1), amplitude=1.0+0j)
    system.add_negative_state((0.1, 0.1, 0.1), amplitude=0.1+0j)  # Small negative
    
    diagnosis_clear = system.diagnose_confusion((0.1, 0.1, 0.1))
    print(f"   Coordinates: {diagnosis_clear['coordinates']}")
    print(f"   Confusion score: {diagnosis_clear['confusion_score']:.3f}")
    print(f"   Diagnosis: {diagnosis_clear['diagnosis']}")
    
    # Add confused state (mostly in negative)
    print("\n2. Adding confused state (mostly in negative cube)...")
    system.add_positive_state((0.9, 0.9, 0.9), amplitude=0.2+0j)  # Small positive
    system.add_negative_state((0.9, 0.9, 0.9), amplitude=1.0+0j)  # Large negative
    
    diagnosis_confused = system.diagnose_confusion((0.9, 0.9, 0.9))
    print(f"   Coordinates: {diagnosis_confused['coordinates']}")
    print(f"   Confusion score: {diagnosis_confused['confusion_score']:.3f}")
    print(f"   Diagnosis: {diagnosis_confused['diagnosis']}")
    
    print(f"\n   ✅ Clear state: {diagnosis_clear['diagnosis']}")
    print(f"   ✅ Confused state: {diagnosis_confused['diagnosis']}")


if __name__ == '__main__':
    demo_basic_dual_cube()
    demo_contradiction_detection()
    demo_decoherence_drift()
    demo_cancellation()
    demo_confusion_diagnosis()
    
    print("\n" + "=" * 70)
    print("✅ DUAL CUBE SYSTEM DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  - Positive cube: Stable meanings, attractors")
    print("  - Negative cube: Contradictions, conflicts, decohered states")
    print("  - Contradiction detection and movement")
    print("  - Decoherence as cross-cube drift")
    print("  - Cancellation of opposite patterns")
    print("  - Confusion diagnosis")
    print("\nThis is a two-sided semantic universe:")
    print("  meaning vs anti-meaning, stability vs contradiction")
    print("=" * 70)

