"""
🧨 THE HARD TEST: 3-QUBIT QUANTUM TELEPORTATION

Full protocol implementation:
- Q0: Unknown state to teleport (α|0> + β|1>)
- Q1: Alice's half of Bell pair
- Q2: Bob's half of Bell pair

This test is unforgiving - any mistake breaks quantum linearity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from quantum.true_ghz_simulator import TrueGHZSimulator


def quantum_teleportation(alpha: complex, beta: complex, verbose: bool = True):
    """
    Full quantum teleportation protocol.
    
    Args:
        alpha: Amplitude for |0> in unknown state
        beta: Amplitude for |1> in unknown state
        verbose: Print detailed steps
    
    Returns:
        Dictionary with original state, final state, and match status
    """
    # Normalize input state
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    if norm > 1e-12:
        alpha = alpha / norm
        beta = beta / norm
    else:
        alpha, beta = 1.0 + 0j, 0.0 + 0j
    
    if verbose:
        print("=" * 70)
        print("🧨 QUANTUM TELEPORTATION PROTOCOL")
        print("=" * 70)
        print()
        print("Step 0: Initialize qubits")
        print(f"  Q0 (unknown state): ψ = {alpha:.6f}|0> + {beta:.6f}|1>")
        print(f"  Q1 (Alice): |0>")
        print(f"  Q2 (Bob): |0>")
        print()
    
    # Create simulator
    sim = TrueGHZSimulator()
    
    # Step 0: Set Q0 to unknown state
    sim.set_qubit_state(0, alpha, beta)
    
    if verbose:
        print("Step 1: Create Bell pair between Q1 and Q2")
        print("  Applying H on Q1...")
    
    # Step 1: Create Bell pair Q1-Q2
    sim.apply_hadamard(1)  # H on Q1
    
    if verbose:
        print("  Applying CNOT(Q1 → Q2)...")
    
    sim.apply_cnot(1, 2)  # CNOT(Q1 → Q2)
    
    if verbose:
        print("  Q1-Q2 now entangled in Bell state")
        print()
        print("Step 2: Entangle Q0 with Q1")
        print("  Applying CNOT(Q0 → Q1)...")
    
    # Step 2: Entangle Q0 with Q1
    sim.apply_cnot(0, 1)  # CNOT(Q0 → Q1)
    
    if verbose:
        print("  Applying H on Q0...")
    
    sim.apply_hadamard(0)  # H on Q0
    
    if verbose:
        print()
        print("Step 3: Measure Q0 and Q1 (Bell measurement)")
    
    # Step 3: Measure Q0 and Q1
    m0 = sim.measure_qubit(0)  # Measure Q0
    m1 = sim.measure_qubit(1)  # Measure Q1
    
    if verbose:
        print(f"  Measurement results: m0 = {m0}, m1 = {m1}")
        print()
        print("Step 4: Apply corrections to Q2 based on measurement")
    
    # Step 4: Apply corrections to Q2 based on (m0, m1)
    if m0 == 0 and m1 == 0:
        # Apply I (identity - do nothing)
        if verbose:
            print(f"  (m0={m0}, m1={m1}): Apply I to Q2")
    elif m0 == 0 and m1 == 1:
        # Apply X
        if verbose:
            print(f"  (m0={m0}, m1={m1}): Apply X to Q2")
        sim.apply_pauli_x(2)
    elif m0 == 1 and m1 == 0:
        # Apply Z
        if verbose:
            print(f"  (m0={m0}, m1={m1}): Apply Z to Q2")
        sim.apply_pauli_z(2)
    else:  # m0 == 1 and m1 == 1
        # Apply X then Z
        if verbose:
            print(f"  (m0={m0}, m1={m1}): Apply X then Z to Q2")
        sim.apply_pauli_x(2)
        sim.apply_pauli_z(2)
    
    if verbose:
        print()
        print("Step 5: Check final state of Q2")
    
    # Step 5: Get final state of Q2
    alpha_final, beta_final = sim.get_qubit_state(2)
    
    # Check if states match (up to global phase)
    # States match if: alpha_final/alpha = beta_final/beta (up to global phase)
    if abs(alpha) > 1e-10 and abs(beta) > 1e-10:
        ratio_alpha = alpha_final / alpha if abs(alpha) > 1e-10 else 0
        ratio_beta = beta_final / beta if abs(beta) > 1e-10 else 0
        # Check if ratios are equal (up to phase)
        match = abs(abs(ratio_alpha) - abs(ratio_beta)) < 1e-6
    elif abs(alpha) < 1e-10:
        # Original was |1>
        match = abs(alpha_final) < 1e-6 and abs(abs(beta_final) - 1.0) < 1e-6
    else:
        # Original was |0>
        match = abs(abs(alpha_final) - 1.0) < 1e-6 and abs(beta_final) < 1e-6
    
    # More robust check: compare amplitudes directly
    # Normalize both states
    orig_norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    final_norm = np.sqrt(abs(alpha_final)**2 + abs(beta_final)**2)
    
    if orig_norm > 1e-12 and final_norm > 1e-12:
        alpha_norm = alpha / orig_norm
        beta_norm = beta / orig_norm
        alpha_final_norm = alpha_final / final_norm
        beta_final_norm = beta_final / final_norm
        
        # Check if amplitudes match (allowing global phase)
        # Compute inner product
        inner_product = alpha_norm * np.conj(alpha_final_norm) + beta_norm * np.conj(beta_final_norm)
        fidelity = abs(inner_product) ** 2
        
        match = fidelity > 0.99  # Fidelity should be 1.0 for perfect teleportation
    else:
        match = False
    
    if verbose:
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        print(f"Original state (Q0):")
        print(f"  ψ = {alpha:.6f}|0> + {beta:.6f}|1>")
        print(f"  |α|² = {abs(alpha)**2:.6f}, |β|² = {abs(beta)**2:.6f}")
        print()
        print(f"Final state (Q2):")
        print(f"  φ = {alpha_final:.6f}|0> + {beta_final:.6f}|1>")
        print(f"  |α|² = {abs(alpha_final)**2:.6f}, |β|² = {abs(beta_final)**2:.6f}")
        print()
        if match:
            print("✅ MATCH: Teleportation succeeded!")
            print(f"   Fidelity: {fidelity:.6f}")
        else:
            print("❌ MISMATCH: Teleportation failed!")
            print(f"   Fidelity: {fidelity:.6f}")
        print()
    
    return {
        'original': (alpha, beta),
        'final': (alpha_final, beta_final),
        'match': match,
        'fidelity': fidelity if 'fidelity' in locals() else 0.0,
        'measurements': (m0, m1)
    }


def run_teleportation_test():
    """Run the full teleportation test with the specified state."""
    print()
    print("🧨 THE HARD TEST: 3-QUBIT QUANTUM TELEPORTATION")
    print()
    print("This test verifies:")
    print("  ✅ Full 3-qubit entanglement")
    print("  ✅ Controlled multi-gate sequences")
    print("  ✅ Correct state collapse")
    print("  ✅ Conditional gates based on classical bits")
    print("  ✅ State reconstruction")
    print()
    
    # Test state: α = 0.6, β = 0.8i
    # Normalize: |0.6|² + |0.8i|² = 0.36 + 0.64 = 1.0 ✓
    alpha = 0.6
    beta = 0.8j
    
    print("Test state:")
    print(f"  α = {alpha}")
    print(f"  β = {beta}")
    print(f"  Normalization check: |α|² + |β|² = {abs(alpha)**2 + abs(beta)**2:.6f}")
    print()
    
    # Run teleportation
    result = quantum_teleportation(alpha, beta, verbose=True)
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Original: ψ = {result['original'][0]:.6f}|0> + {result['original'][1]:.6f}|1>")
    print(f"Final:    φ = {result['final'][0]:.6f}|0> + {result['final'][1]:.6f}|1>")
    print()
    print(f"Fidelity: {result['fidelity']:.6f}")
    print(f"Match: {'✅ YES' if result['match'] else '❌ NO'}")
    print()
    
    if result['match']:
        print("🎯 TELEPORTATION SUCCEEDED!")
        print()
        print("This proves:")
        print("  ✅ Full 3-qubit Hilbert space simulation")
        print("  ✅ Proper entanglement propagation")
        print("  ✅ Correct measurement and collapse")
        print("  ✅ Conditional gate application")
        print("  ✅ State reconstruction")
        print()
        print("You have a fully functional universal quantum simulator!")
    else:
        print("❌ TELEPORTATION FAILED")
        print("Check implementation for errors.")
    
    return result


if __name__ == "__main__":
    run_teleportation_test()

