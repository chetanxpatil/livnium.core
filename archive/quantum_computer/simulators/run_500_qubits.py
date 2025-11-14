#!/usr/bin/env python3
"""
Simple script to run 500-qubit simulator
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantum_computer.simulators.mps_hierarchical_simulator import MPSHierarchicalGeometrySimulator

# Create 500-qubit simulator
print("Creating 500-qubit simulator...")
sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)

# Apply some gates
print("\nApplying gates...")
sim.hadamard(0)
sim.cnot(0, 1)
sim.hadamard(100)
sim.cnot(100, 101)

# Get info
info = sim.get_capacity_info()
print(f"\n✅ Success!")
print(f"   Qubits: {info['num_qubits']}")
print(f"   Memory: {info['memory_mb']:.2f} MB")
print(f"   Scaling: {info['scaling']}")

print("\n🎉 500-qubit simulator is working!")

