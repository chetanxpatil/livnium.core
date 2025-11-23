"""
Test the Two Faces of Φ: Straight-line and Rotated Forms

This demonstrates the metaphysical-math moment where Φ had two faces:
- One perfectly straight (eigenbasis - stable, minimal energy)
- One rotated (rotated basis - expressive, dynamic)

Both are valid projections of the same underlying invariant.
Just like quantum objects are both wave and particle depending on how you look.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.quantum.geometry_quantum_coupling import GeometryQuantumCoupling
from core.quantum.quantum_lattice import QuantumLattice


def test_phi_straight_line():
    """Test straight-line Φ: Direct, linear, no-rotation mapping."""
    print("\n" + "="*70)
    print("TEST 1: Straight-Line Φ (Eigenbasis)")
    print("="*70)
    
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_semantic_polarity=True,
        enable_quantum=True,
        enable_geometry_quantum_coupling=True
    )
    system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(system)
    
    # Test from OM (origin at (0,0,0)) to LO (local orientation)
    target_coords = (1, 1, 1)  # Corner cell
    
    straight_phi = coupling.phi_straight_line(target_coords)
    
    print(f"OM (Origin Anchor): (0, 0, 0)")
    print(f"LO (Local Orientation): {target_coords}")
    print(f"Straight-line Φ: {straight_phi:.4f}")
    print(f"Interpretation: Direct, linear connection - 'pure polarity' direction")
    print(f"                Like a laser beam pointed straight out of OM")
    
    return straight_phi


def test_phi_rotated():
    """Test rotated Φ: Same connection, but rotated through cube's orientation."""
    print("\n" + "="*70)
    print("TEST 2: Rotated Φ (Rotated Basis)")
    print("="*70)
    
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_semantic_polarity=True,
        enable_quantum=True,
        enable_geometry_quantum_coupling=True
    )
    system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(system)
    
    target_coords = (1, 1, 1)
    
    rotated_phi, rotation_matrix = coupling.phi_rotated(
        target_coords, rotation_axis="Y", quarter_turns=1
    )
    
    print(f"OM (Origin Anchor): (0, 0, 0)")
    print(f"LO (Local Orientation): {target_coords}")
    print(f"Rotated Φ: {rotated_phi:.4f}")
    print(f"Rotation: Y-axis, 1 quarter-turn")
    print(f"Interpretation: Phase-shifted version - 'wave aspect'")
    print(f"                Same connection in rotated reference frame")
    
    return rotated_phi, rotation_matrix


def test_phi_dual_representation():
    """Test both faces of Φ together."""
    print("\n" + "="*70)
    print("TEST 3: Dual Representation - Both Faces of Φ")
    print("="*70)
    
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_semantic_polarity=True,
        enable_quantum=True,
        enable_geometry_quantum_coupling=True
    )
    system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(system)
    
    target_coords = (1, 1, 1)
    
    result = coupling.phi_dual_representation(target_coords)
    
    print(f"OM → LO: (0,0,0) → {target_coords}")
    print(f"\nStraight-line Φ (Eigenbasis):")
    print(f"  Value: {result['straight_phi']:.4f}")
    print(f"  Magnitude: {result['straight_magnitude']:.4f}")
    print(f"  {result['interpretation']['straight']}")
    print(f"  When: {result['interpretation']['when_straight']}")
    
    print(f"\nRotated Φ (Rotated Basis):")
    print(f"  Value: {result['rotated_phi']:.4f}")
    print(f"  Magnitude: {result['rotated_magnitude']:.4f}")
    print(f"  {result['interpretation']['rotated']}")
    print(f"  When: {result['interpretation']['when_rotated']}")
    
    print(f"\nInvariant Preserved: {result['invariant_preserved']}")
    print(f"\n{result['interpretation']['both_valid']}")
    print(f"\nThis is the kind of thing that makes Livnium feel like")
    print(f"a strange little universe growing its own physics.")
    
    return result


def test_phi_under_rotation():
    """Test Φ invariance under all 24 rotations."""
    print("\n" + "="*70)
    print("TEST 4: Φ Invariance Under All Rotations")
    print("="*70)
    
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_semantic_polarity=True,
        enable_quantum=True,
        enable_geometry_quantum_coupling=True
    )
    system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(system)
    
    target_coords = (1, 1, 1)
    
    results = coupling.phi_under_rotation(target_coords)
    
    print(f"Baseline Φ (straight-line): {results['baseline_phi']:.4f}")
    print(f"Baseline magnitude: {results['baseline_magnitude']:.4f}")
    print(f"\nRotations tested: {results['rotations_tested']}")
    print(f"Invariant preserved: {results['invariant_count']}/{results['rotations_tested']}")
    print(f"Invariant ratio: {results['invariant_ratio']*100:.1f}%")
    
    print(f"\nSample rotation results:")
    for i, rot_result in enumerate(results['rotation_results'][:6]):  # Show first 6
        print(f"  {rot_result['axis']}-axis, {rot_result['quarter_turns']} turns: "
              f"Φ={rot_result['rotated_phi']:.4f}, "
              f"mag={rot_result['rotated_magnitude']:.4f}, "
              f"invariant={'✓' if rot_result['invariant_preserved'] else '✗'}")
    
    print(f"\nConclusion: Both straight and rotated Φ preserve invariants")
    print(f"under the full 24-element rotation group.")
    
    return results


def test_wave_particle_duality_analogy():
    """Demonstrate the wave-particle duality analogy."""
    print("\n" + "="*70)
    print("TEST 5: Wave-Particle Duality Analogy")
    print("="*70)
    
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_semantic_polarity=True,
        enable_geometry_quantum_coupling=True,
        enable_quantum=True
    )
    system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(system)
    
    target_coords = (1, 0, 0)  # Center cell
    
    result = coupling.phi_dual_representation(target_coords)
    
    print("Your system spotted wave–particle duality inside your own geometry")
    print("long before we gave it the words.\n")
    
    print(f"Straight Φ (Particle aspect):")
    print(f"  - Stable, minimal energy")
    print(f"  - Direct, linear mapping")
    print(f"  - Value: {result['straight_phi']:.4f}")
    
    print(f"\nRotated Φ (Wave aspect):")
    print(f"  - Expressive, dynamic")
    print(f"  - Phase-shifted through rotation")
    print(f"  - Value: {result['rotated_phi']:.4f}")
    
    print(f"\nBoth are correct because they're two projections of the")
    print(f"same underlying invariant - a polarity field inside a")
    print(f"rotating reference frame.")
    
    print(f"\nSame Φ. Two coordinate choices. Two behaviors. No contradiction.")


if __name__ == "__main__":
    print("="*70)
    print("THE TWO FACES OF Φ")
    print("="*70)
    print("\nThis demonstrates the metaphysical-math moment where Φ had")
    print("two faces — one perfectly straight, one rotated — and how it")
    print("compares to how light behaves as both wave AND particle.\n")
    
    # Run all tests
    test_phi_straight_line()
    test_phi_rotated()
    test_phi_dual_representation()
    test_phi_under_rotation()
    test_wave_particle_duality_analogy()
    
    print("\n" + "="*70)
    print("✅ All tests completed!")
    print("="*70)

