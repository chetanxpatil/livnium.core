#!/usr/bin/env python3
"""
Example: 500-Qubit Quantum Circuit

Demonstrates how to use the MPS hierarchical geometry simulator
to run quantum circuits on 500+ qubits.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantum_computer.simulators.mps_hierarchical_simulator import MPSHierarchicalGeometrySimulator


def example_500_qubit_circuit():
    """Example: Simple 500-qubit circuit."""
    print("=" * 70)
    print("500-Qubit Quantum Circuit Example")
    print("=" * 70)
    
    # Create 500-qubit simulator
    print("\n1. Creating 500-qubit simulator...")
    sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)
    
    # Check capacity
    info = sim.get_capacity_info()
    print(f"\n   Capacity Info:")
    print(f"   - Qubits: {info['num_qubits']}")
    print(f"   - Memory: {info['memory_mb']:.2f} MB")
    print(f"   - Scaling: {info['scaling']}")
    
    # Build circuit
    print("\n2. Building quantum circuit...")
    print("   - Applying Hadamard to qubits 0-9...")
    for i in range(10):
        sim.hadamard(i)
    
    print("   - Creating Bell pairs (qubits 0-1, 2-3, 4-5, 6-7, 8-9)...")
    for i in range(0, 10, 2):
        sim.cnot(i, i+1)
    
    print("   - Applying Pauli-X to qubit 100...")
    sim.pauli_x(100)
    
    print("   - Creating entanglement (qubits 100-101)...")
    sim.cnot(100, 101)
    
    print("   ✅ Circuit built!")
    
    # Run simulation
    print("\n3. Running simulation (100 shots)...")
    results = sim.run(num_shots=100)
    
    print(f"\n   Results:")
    print(f"   - Shots: {results['shots']}")
    print(f"   - Unique outcomes: {len(results['results'])}")
    print(f"\n   Top 5 outcomes:")
    for outcome, count in list(results['results'].items())[:5]:
        print(f"   {outcome[:20]}...: {count} ({count/results['shots']*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ 500-qubit circuit simulation complete!")
    print("=" * 70)


def example_large_scale():
    """Example: Large-scale 500-qubit circuit."""
    print("\n" + "=" * 70)
    print("Large-Scale 500-Qubit Circuit")
    print("=" * 70)
    
    sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)
    
    print("\nInitializing all 500 qubits in superposition...")
    for i in range(500):
        if i % 100 == 0:
            print(f"  Progress: {i}/500 qubits...")
        sim.hadamard(i)
    
    print("\nCreating 1D chain entanglement...")
    for i in range(499):
        if i % 100 == 0:
            print(f"  Progress: {i}/499 connections...")
        sim.cnot(i, i+1)
    
    print("\n✅ Large-scale circuit complete!")
    print(f"   Applied {500 + 499} gates to 500 qubits")
    
    info = sim.get_capacity_info()
    print(f"   Memory: {info['memory_mb']:.2f} MB")


if __name__ == "__main__":
    # Run examples
    example_500_qubit_circuit()
    
    # THE REAL TEST: Maximum entanglement
    print("\n" + "=" * 70)
    print("RUNNING THE REAL TEST: Maximum Entanglement")
    print("=" * 70)
    example_large_scale()

