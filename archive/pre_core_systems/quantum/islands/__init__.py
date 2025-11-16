"""
Quantum-Inspired Islands Engine

This is a quantum-inspired classical system. It is NOT a physical quantum computer.
"""

from quantum.islands.core.kernel import LivniumQubit, EntangledPair, normalize
from quantum.islands.core.geometric_quantum_simulator import (
    GeometricQuantumSimulator, 
    create_105_qubit_geometric_system,
    create_large_geometric_system
)
from quantum.islands.core.quantum_islands import QuantumIsland, QuantumIslandArchitecture

__all__ = [
    'LivniumQubit',
    'EntangledPair',
    'normalize',
    'GeometricQuantumSimulator',
    'create_105_qubit_geometric_system',
    'create_large_geometric_system',  # For 5000+ qubits!
    'QuantumIsland',
    'QuantumIslandArchitecture',
]

