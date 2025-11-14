"""
GHZ State Comparison Test

Compares:
1. Geometric simulator (pairwise entanglement) - can produce illegal states
2. True GHZ simulator (8D state vector) - only produces |000> or |111>

This demonstrates the trade-off:
- Geometric: Efficient for 105+ qubits, approximate entanglement
- True: Correct physics for 3 qubits, exponential memory for more qubits
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from quantum.geometric_quantum_simulator import GeometricQuantumSimulator
from quantum.true_ghz_simulator import TrueGHZSimulator


def test_geometric_simulator():
    """Test geometric simulator (pairwise entanglement)."""
    print("=" * 70)
    print("GEOMETRIC SIMULATOR TEST (Pairwise Entanglement)")
    print("=" * 70)
    print()
    print("This simulator uses pairwise entanglement (A↔B, B↔C)")
    print("and can produce illegal GHZ states like |110>.")
    print()
    
    # Measure multiple times
    outcomes = {'000': 0, '001': 0, '010': 0, '011': 0,
                '100': 0, '101': 0, '110': 0, '111': 0}
    
    print("Running 1000 measurements...")
    for _ in range(1000):
        # Create fresh simulator for each measurement
        sim = GeometricQuantumSimulator(grid_size=3)
        
        # Create three qubits
        qubit_a = sim.add_qubit((0, 0, 0), value=0.0, qubit_id="A")
        qubit_b = sim.add_qubit((1, 0, 0), value=0.0, qubit_id="B")
        qubit_c = sim.add_qubit((1, 1, 0), value=0.0, qubit_id="C")
        
        # GHZ circuit
        sim.apply_hadamard_at_position((0, 0, 0), qubit_idx=0)
        sim.apply_cnot_between_positions((0, 0, 0), (1, 0, 0), 0, 0)
        sim.apply_cnot_between_positions((1, 0, 0), (1, 1, 0), 0, 0)
        
        # Measure
        measurements = sim.measure_all()
        result_a = measurements[(0, 0, 0)][0]
        result_b = measurements[(1, 0, 0)][0]
        result_c = measurements[(1, 1, 0)][0]
        outcome = f"{result_a}{result_b}{result_c}"
        outcomes[outcome] += 1
    
    print()
    print("Results:")
    illegal_states = []
    for state, count in sorted(outcomes.items()):
        if count > 0:
            print(f"  |{state}>: {count} times ({count/10:.1f}%)")
            if state not in ['000', '111']:
                illegal_states.append((state, count))
    
    print()
    if illegal_states:
        print("⚠️  ILLEGAL STATES DETECTED:")
        for state, count in illegal_states:
            print(f"  |{state}>: {count} times (should be 0 for true GHZ state)")
        print()
        print("This confirms: Geometric simulator uses pairwise entanglement,")
        print("not a global 3-qubit wavefunction.")
    else:
        print("✅ No illegal states (unexpected for geometric simulator)")
    
    return outcomes


def test_true_ghz_simulator():
    """Test true GHZ simulator (8D state vector)."""
    print()
    print("=" * 70)
    print("TRUE GHZ SIMULATOR TEST (8D State Vector)")
    print("=" * 70)
    print()
    
    sim = TrueGHZSimulator()
    
    # Create GHZ state
    sim.create_ghz_state()
    
    print("State: (|000> + |111>)/√2")
    print()
    
    probs = sim.get_probabilities()
    print("Pre-measurement probabilities:")
    for state, prob in sorted(probs.items()):
        if prob > 1e-6:
            print(f"  P(|{state}>) = {prob:.6f}")
    print()
    
    # Measure multiple times
    outcomes = {'000': 0, '001': 0, '010': 0, '011': 0,
                '100': 0, '101': 0, '110': 0, '111': 0}
    
    print("Running 1000 measurements...")
    for _ in range(1000):
        sim.create_ghz_state()
        a, b, c = sim.measure()
        outcome = f"{a}{b}{c}"
        outcomes[outcome] += 1
    
    print()
    print("Results:")
    illegal_states = []
    for state, count in sorted(outcomes.items()):
        if count > 0:
            print(f"  |{state}>: {count} times ({count/10:.1f}%)")
            if state not in ['000', '111']:
                illegal_states.append((state, count))
    
    print()
    if illegal_states:
        print("❌ ERROR: Illegal states detected in true simulator!")
        for state, count in illegal_states:
            print(f"  |{state}>: {count} times")
    else:
        print("✅ CORRECT: Only |000> and |111> outcomes (true GHZ state)")
        print()
        print("This confirms: True simulator maintains global 3-qubit wavefunction,")
        print("enforcing quantum mechanical constraints.")
    
    return outcomes


def compare_simulators():
    """Compare both simulators side by side."""
    print()
    print("=" * 70)
    print("SIMULATOR COMPARISON")
    print("=" * 70)
    print()
    
    print("Geometric Simulator:")
    print("  ✅ Efficient: ~5 KB for 105 qubits")
    print("  ✅ Scalable: Linear memory growth")
    print("  ⚠️  Approximate: Uses pairwise entanglement")
    print("  ⚠️  Can produce illegal GHZ states (e.g., |110>)")
    print()
    
    print("True GHZ Simulator:")
    print("  ✅ Correct: Maintains global wavefunction")
    print("  ✅ Verified: Only produces |000> or |111>")
    print("  ❌ Limited: Only works for 3 qubits")
    print("  ❌ Exponential: Would need 2^n memory for n qubits")
    print()
    
    print("Trade-off:")
    print("  - For Livnium's geometric/AI goals: Geometric simulator is fine")
    print("  - For strict physics verification: True simulator is required")
    print("  - For 105+ qubits: Geometric simulator is the only option")
    print()


if __name__ == "__main__":
    print()
    print("🧪 GHZ STATE COMPARISON TEST")
    print()
    print("This test demonstrates the difference between:")
    print("  1. Geometric simulator (pairwise entanglement)")
    print("  2. True GHZ simulator (8D state vector)")
    print()
    
    # Test geometric simulator
    geo_outcomes = test_geometric_simulator()
    
    # Test true simulator
    true_outcomes = test_true_ghz_simulator()
    
    # Compare
    compare_simulators()
    
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

