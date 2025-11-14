#!/usr/bin/env python3
"""
Error Rate Test for Livnium Entanglement-Preserving Simulator

Compares results to exact calculations to measure error rate.
"""


import numpy as np
from quantum.hierarchical.simulators.livnium_entanglement_preserving import LivniumEntanglementPreserving


def exact_2_qubit_bell_state():
    """Exact calculation for 2-qubit Bell state."""
    # |00⟩ → H(0) → |+0⟩ = (|00⟩ + |10⟩)/√2
    # → CNOT(0,1) → (|00⟩ + |11⟩)/√2
    
    # Exact probabilities:
    # |00⟩: 0.5
    # |11⟩: 0.5
    # |01⟩: 0.0
    # |10⟩: 0.0
    
    return {
        (0, 0): 0.5,
        (1, 1): 0.5,
        (0, 1): 0.0,
        (1, 0): 0.0
    }


def exact_3_qubit_ghz_state():
    """Exact calculation for 3-qubit GHZ state."""
    # |000⟩ → H(0) → (|000⟩ + |100⟩)/√2
    # → CNOT(0,1) → (|000⟩ + |110⟩)/√2
    # → CNOT(0,2) → (|000⟩ + |111⟩)/√2
    
    exact = {}
    for i in range(8):
        if i == 0 or i == 7:  # |000⟩ or |111⟩
            exact[tuple(format(i, '03b'))] = 0.5
        else:
            exact[tuple(format(i, '03b'))] = 0.0
    
    return exact


def test_error_rate_small_system():
    """Test error rate on small system (2-3 qubits) where we can compute exact results."""
    print("=" * 70)
    print("Error Rate Test: Small Systems (Exact Comparison)")
    print("=" * 70)
    
    # Test 2-qubit Bell state
    print("\n1. Testing 2-qubit Bell state:")
    print("   Circuit: |00⟩ → H(0) → CNOT(0,1)")
    print("   Expected: |00⟩ and |11⟩ each with 50% probability")
    
    sim = LivniumEntanglementPreserving(2, macro_size=3)
    sim.hadamard(0)
    sim.cnot(0, 1)
    
    # Run many shots to get accurate statistics
    results = sim.run(num_shots=10000)
    
    # Calculate probabilities
    total_shots = results['shots']
    probs = {outcome: count / total_shots for outcome, count in results['results'].items()}
    
    # Expected probabilities
    expected = exact_2_qubit_bell_state()
    
    # Convert outcomes to binary tuples
    probs_binary = {}
    for outcome, prob in probs.items():
        # outcome is tuple of ints, convert to binary string representation
        binary_str = ''.join(str(bit) for bit in outcome)
        probs_binary[binary_str] = prob
    
    print(f"\n   Results (10,000 shots):")
    errors = []
    for state in ['00', '01', '10', '11']:
        measured = probs_binary.get(state, 0.0)
        expected_prob = expected.get((int(state[0]), int(state[1])), 0.0)
        error = abs(measured - expected_prob)
        errors.append(error)
        print(f"   |{state}⟩: Expected {expected_prob:.4f}, Measured {measured:.4f}, Error {error:.4f}")
    
    max_error = max(errors)
    mean_error = np.mean(errors)
    print(f"\n   Max error: {max_error:.4f}")
    print(f"   Mean error: {mean_error:.4f}")
    print(f"   Error rate: {max_error * 100:.2f}%")
    
    # Test 3-qubit GHZ state
    print("\n2. Testing 3-qubit GHZ state:")
    print("   Circuit: |000⟩ → H(0) → CNOT(0,1) → CNOT(0,2)")
    print("   Expected: |000⟩ and |111⟩ each with 50% probability")
    
    sim3 = LivniumEntanglementPreserving(3, macro_size=3)
    sim3.hadamard(0)
    sim3.cnot(0, 1)
    sim3.cnot(0, 2)
    
    results3 = sim3.run(num_shots=10000)
    total_shots3 = results3['shots']
    probs3 = {outcome: count / total_shots3 for outcome, count in results3['results'].items()}
    
    # Convert to binary strings
    probs3_binary = {}
    for outcome, prob in probs3.items():
        binary_str = ''.join(str(bit) for bit in outcome)
        probs3_binary[binary_str] = prob
    
    print(f"\n   Results (10,000 shots):")
    errors3 = []
    for i in range(8):
        state = format(i, '03b')
        measured = probs3_binary.get(state, 0.0)
        expected_prob = exact_3_qubit_ghz_state().get(tuple(state), 0.0)
        error = abs(measured - expected_prob)
        errors3.append(error)
        if expected_prob > 0 or measured > 0.01:  # Only show non-zero or significant
            print(f"   |{state}⟩: Expected {expected_prob:.4f}, Measured {measured:.4f}, Error {error:.4f}")
    
    max_error3 = max(errors3)
    mean_error3 = np.mean(errors3)
    print(f"\n   Max error: {max_error3:.4f}")
    print(f"   Mean error: {mean_error3:.4f}")
    print(f"   Error rate: {max_error3 * 100:.2f}%")
    
    return max_error, mean_error, max_error3, mean_error3


def test_probability_conservation():
    """Test if probabilities sum to 1 (conservation law)."""
    print("\n" + "=" * 70)
    print("Probability Conservation Test")
    print("=" * 70)
    
    for n_qubits in [5, 10, 20]:
        print(f"\nTesting {n_qubits} qubits:")
        sim = LivniumEntanglementPreserving(n_qubits, macro_size=3)
        
        # Apply Hadamard to all
        for i in range(n_qubits):
            sim.hadamard(i)
        
        # Check probability sum from cells
        total_prob = 0.0
        for block in sim.macro_blocks:
            for cell_data in block['micro_cells'].values():
                total_prob += abs(cell_data['amplitude']) ** 2
        
        error = abs(total_prob - 1.0)
        print(f"  Total probability: {total_prob:.10f}")
        print(f"  Error from 1.0: {error:.10f}")
        print(f"  Conservation: {'✅ PASS' if error < 1e-6 else '❌ FAIL'}")


def test_entanglement_structure():
    """Test if entanglement structure is preserved."""
    print("\n" + "=" * 70)
    print("Entanglement Structure Test")
    print("=" * 70)
    
    sim = LivniumEntanglementPreserving(10, macro_size=3)
    
    # Create known entanglement pattern
    print("\nCreating entanglement pattern:")
    print("  H(0), H(1), H(2)")
    print("  CNOT(0,1), CNOT(1,2)")
    
    sim.hadamard(0)
    sim.hadamard(1)
    sim.hadamard(2)
    sim.cnot(0, 1)
    sim.cnot(1, 2)
    
    info = sim.get_entanglement_info()
    
    print(f"\n  Total cells: {info['total_micro_cells']}")
    print(f"  Total entanglement links: {info['total_entanglement_links']}")
    print(f"  Entanglement preserved: {info['entanglement_preserved']}")
    
    # Check if links exist
    has_links = info['total_entanglement_links'] > 0
    print(f"  Has entanglement links: {'✅ YES' if has_links else '❌ NO'}")


if __name__ == "__main__":
    # Run all tests
    max_err_2, mean_err_2, max_err_3, mean_err_3 = test_error_rate_small_system()
    test_probability_conservation()
    test_entanglement_structure()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"2-qubit Bell state:")
    print(f"  Max error: {max_err_2:.4f} ({max_err_2*100:.2f}%)")
    print(f"  Mean error: {mean_err_2:.4f} ({mean_err_2*100:.2f}%)")
    print(f"\n3-qubit GHZ state:")
    print(f"  Max error: {max_err_3:.4f} ({max_err_3*100:.2f}%)")
    print(f"  Mean error: {mean_err_3:.4f} ({max_err_3*100:.2f}%)")
    
    # Overall assessment
    overall_max_error = max(max_err_2, max_err_3)
    if overall_max_error < 0.01:
        print(f"\n✅ Overall error rate: {overall_max_error*100:.2f}% (EXCELLENT)")
    elif overall_max_error < 0.05:
        print(f"\n⚠️  Overall error rate: {overall_max_error*100:.2f}% (GOOD)")
    elif overall_max_error < 0.10:
        print(f"\n⚠️  Overall error rate: {overall_max_error*100:.2f}% (ACCEPTABLE)")
    else:
        print(f"\n❌ Overall error rate: {overall_max_error*100:.2f}% (HIGH ERROR)")

