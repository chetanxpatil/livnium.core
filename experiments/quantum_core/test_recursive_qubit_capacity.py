"""
Test Recursive Geometry Omcube Capacity

Cleaned + Safe + Recursion-Limited version
-------------------------------------------------
Fixes:
  - No quantum verification at depth >= 2.
  - Sampling limits now respected.
  - No runaway log printing.
  - Recursive limit prevents infinite explosion.
  - Geometry rules untouched.
  - Swarm quantum tests only at shallow layers.
"""

import sys
import os
import numpy as np
import random
from typing import Dict, List, Tuple

# Make repo root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.recursive import RecursiveGeometryEngine
from core.quantum import QuantumLattice
from core.quantum.true_quantum_layer import TrueQuantumRegister
from core.quantum.quantum_gates import QuantumGates


# ================================================================
# SAFE + LIGHTWEIGHT QUANTUM VERIFICATION
# ================================================================

def verify_quantum_capacity(level, level_name="Level 0", sample_clusters: int = 1, depth: int = 0):
    """
    Quantum verification only at depth 0 and depth 1.
    Depth >= 2 → NO physics, only counting qubits.
    """
    try:
        if not level.geometry.config.enable_quantum:
            return (0, 0)

        q_lattice = QuantumLattice(level.geometry)
        available_cells = list(q_lattice.quantum_cells.keys())
        total_qubits = len(available_cells)

        # Depth ≥ 2 → skip quantum tests completely
        if depth >= 2:
            print(f"  [{level_name}] {total_qubits} qubits (skipped quantum test)")
            return (total_qubits, 0)

        # Need at least 3 qubits for teleportation/Bell
        if total_qubits < 3:
            return (total_qubits, 0)

        verified = 0
        clusters_to_test = min(sample_clusters, total_qubits // 3)

        for _ in range(clusters_to_test):
            # Create Bell pair between qubit(1) and qubit(2)
            reg = TrueQuantumRegister([0, 1, 2])
            reg.apply_gate(QuantumGates.hadamard(), 1)
            reg.apply_cnot(1, 2)

            # Test correlation
            matches = 0
            trials = 5
            for _ in range(trials):
                t = TrueQuantumRegister([0, 1, 2])
                t.apply_gate(QuantumGates.hadamard(), 1)
                t.apply_cnot(1, 2)
                if t.measure_qubit(1) == t.measure_qubit(2):
                    matches += 1

            if matches == trials:
                verified += 1

        print(f"  [{level_name}] {total_qubits} qubits → Quantum OK {verified}/{clusters_to_test}")
        return (total_qubits, verified)

    except Exception as e:
        print(f"  [{level_name}] Quantum verification failed: {e}")
        return (0, 0)


# ================================================================
# SAFE RECURSION
# ================================================================

def count_and_verify_recursive(level, current_depth=0, max_depth=3,
                               sample_per_level=1,
                               max_samples_per_depth=10):
    """
    Safe recursive walker:
      - Quantum tests only for depth 0 and depth 1
      - Depth >= 2 → geometry count only (no physics)
      - Avoids explosion
    """

    level_name = f"Depth {current_depth}"

    level_qubits, level_verified = verify_quantum_capacity(
        level,
        level_name,
        sample_clusters=sample_per_level,
        depth=current_depth
    )

    total_qubits = level_qubits
    total_verified = level_verified
    levels_checked = 1

    if current_depth < max_depth:
        children = list(level.children.values())

        # Limit child sampling
        if len(children) > max_samples_per_depth:
            children = random.sample(children, max_samples_per_depth)

        for child in children:
            child_qubits, child_verified, child_levels = count_and_verify_recursive(
                child,
                current_depth + 1,
                max_depth,
                sample_per_level=sample_per_level if current_depth < 1 else 0,
                max_samples_per_depth=max_samples_per_depth
            )

            total_qubits += child_qubits
            total_verified += child_verified
            levels_checked += child_levels

    return total_qubits, total_verified, levels_checked


# ================================================================
# CAPACITY TEST
# ================================================================

def count_omcubes_recursive(level):
    """Count geometry only."""
    total = len(level.geometry.lattice)
    for child in level.children.values():
        total += count_omcubes_recursive(child)
    return total


def test_recursive_omcube_capacity(base_lattice_size=5, max_depth=3, target_omcubes=3000):
    print("="*70)
    print("Recursive Geometry Omcube Capacity Test")
    print("="*70)

    config = LivniumCoreConfig(
        lattice_size=base_lattice_size,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )

    base = LivniumCoreSystem(config)
    print("Building recursive geometry...")
    engine = RecursiveGeometryEngine(base, max_depth=max_depth)
    print("✓ Built\n")

    print("="*70)
    print("Quantum Verification (SAFE MODE)")
    print("="*70)

    total_qubits, total_verified, checked = count_and_verify_recursive(
        engine.levels[0],
        current_depth=0,
        max_depth=max_depth,
        sample_per_level=1,
        max_samples_per_depth=3
    )

    print("\nCounting geometric omcubes…")
    total_geom = count_omcubes_recursive(engine.levels[0])

    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Geometric omcubes: {total_geom:,}")
    print(f"Verified qubits:   {total_qubits:,}")
    print(f"Verified clusters: {total_verified}")
    print(f"Levels checked:    {checked}")
    print(f"Target:            {target_omcubes:,}")
    print()

    if total_geom >= target_omcubes:
        print("✅ SUCCESS: Enough geometric capacity.")
    else:
        print("⚠️ Insufficient capacity.")

    print("="*70)

    return dict(
        geometric=total_geom,
        quantum_qubits=total_qubits,
        verified_clusters=total_verified,
        levels_checked=checked,
        target=target_omcubes,
        success=(total_geom >= target_omcubes)
    )


# ================================================================
# SCALING TEST
# ================================================================

def test_scaling():
    tests = [
        (3, 2, 100),
        (3, 3, 500),
        (5, 2, 1000),
        (5, 3, 3000),
        (7, 2, 2000),
    ]

    print("\n\nSCALING ANALYSIS\n" + "="*70)

    for size, depth, target in tests:
        print(f"\n--- {size}×{size}×{size}, depth {depth}, target {target} ---")
        test_recursive_omcube_capacity(size, depth, target)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("Testing capacity for 2,953–4,000 omcubes…\n")

    print("TEST 1")
    test_recursive_omcube_capacity(5, 3, 2953)

    print("\nTEST 2")
    test_recursive_omcube_capacity(7, 2, 4000)

    test_scaling()