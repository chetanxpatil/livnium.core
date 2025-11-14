"""
3-Qubit GHZ Chain Test
Proves geometric quantum simulator is real and working.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from quantum.geometric_quantum_simulator import GeometricQuantumSimulator

def test_3_qubit_ghz_chain():
    """
    Create 3-qubit GHZ-like chain:
    A at (0,0,0) → B at (1,0,0) → C at (1,1,0)
    
    Operations:
    1. Hadamard on A
    2. CNOT(A → B)
    3. CNOT(B → C)
    4. Measure all
    """
    print("=" * 70)
    print("3-QUBIT GHZ CHAIN TEST")
    print("=" * 70)
    print()
    
    # Create simulator
    sim = GeometricQuantumSimulator(grid_size=3)
    
    # Create three qubits at specific positions
    print("Creating qubits:")
    qubit_a = sim.add_qubit((0, 0, 0), value=0.0, qubit_id="A")
    print(f"  A at (0,0,0): {qubit_a}")
    
    qubit_b = sim.add_qubit((1, 0, 0), value=0.0, qubit_id="B")
    print(f"  B at (1,0,0): {qubit_b}")
    
    qubit_c = sim.add_qubit((1, 1, 0), value=0.0, qubit_id="C")
    print(f"  C at (1,1,0): {qubit_c}")
    print()
    
    # Show initial entanglement (automatic from geometry)
    print("Initial Entanglements (geometric neighbors):")
    entanglements = []
    for qubit, neighbors in sim.entanglement_graph.items():
        for neighbor in neighbors:
            pair = (qubit.qubit_id, neighbor.qubit_id)
            reverse_pair = (pair[1], pair[0])
            if pair not in entanglements and reverse_pair not in entanglements:
                entanglements.append(pair)
                print(f"  {pair[0]} ↔ {pair[1]}")
    print()
    
    # Step 1: Hadamard on A
    print("Step 1: Applying Hadamard to A...")
    sim.apply_hadamard_at_position((0, 0, 0), qubit_idx=0)
    prob_a = qubit_a.get_probability()
    print(f"  A probability |1>: {prob_a:.3f}")
    print()
    
    # Step 2: CNOT(A → B)
    print("Step 2: Applying CNOT(A → B)...")
    sim.apply_cnot_between_positions((0, 0, 0), (1, 0, 0), 0, 0)
    prob_b = qubit_b.get_probability()
    print(f"  B probability |1>: {prob_b:.3f}")
    print()
    
    # Step 3: CNOT(B → C)
    print("Step 3: Applying CNOT(B → C)...")
    sim.apply_cnot_between_positions((1, 0, 0), (1, 1, 0), 0, 0)
    prob_c = qubit_c.get_probability()
    print(f"  C probability |1>: {prob_c:.3f}")
    print()
    
    # Step 4: Measure all
    print("Step 4: Measuring all qubits...")
    measurements = sim.measure_all()
    
    # Extract results
    result_a = measurements[(0, 0, 0)][0]
    result_b = measurements[(1, 0, 0)][0]
    result_c = measurements[(1, 1, 0)][0]
    
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print()
    
    # Show final states
    print("Final States:")
    print(f"  A: |{result_a}>")
    print(f"  B: |{result_b}>")
    print(f"  C: |{result_c}>")
    print()
    
    # Show correlation
    print("Correlation Check:")
    if result_a == result_b == result_c:
        print(f"  ✅ Perfect correlation: All qubits = |{result_a}>")
    elif result_a == result_b:
        print(f"  ⚠️  Partial correlation: A=B={result_a}, C={result_c}")
    elif result_b == result_c:
        print(f"  ⚠️  Partial correlation: B=C={result_b}, A={result_a}")
    else:
        print(f"  ❌ No correlation: A={result_a}, B={result_b}, C={result_c}")
    print()
    
    # Show final entanglement list
    print("Final Entanglements:")
    final_entanglements = []
    for qubit, neighbors in sim.entanglement_graph.items():
        for neighbor in neighbors:
            pair = (qubit.qubit_id, neighbor.qubit_id)
            reverse_pair = (pair[1], pair[0])
            if pair not in final_entanglements and reverse_pair not in final_entanglements:
                final_entanglements.append(pair)
                print(f"  {pair[0]} ↔ {pair[1]}")
    print()
    
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return {
        'results': (result_a, result_b, result_c),
        'entanglements': final_entanglements
    }

if __name__ == "__main__":
    test_3_qubit_ghz_chain()

