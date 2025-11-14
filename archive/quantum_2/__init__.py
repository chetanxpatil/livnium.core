"""
Quantum Computing Module for Livnium

This module provides quantum-mechanical representations for features and classification,
enabling uncertainty modeling, feature entanglement, and probabilistic operations.

Modules:
- gates: Quantum gate operations (Pauli, Hadamard, rotations, etc.)
- features: Quantum feature representation and entanglement
"""

from quantum.gates import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    HADAMARD,
    apply_gate,
    hadamard_gate,
    pauli_x_gate,
    pauli_z_gate,
    phase_shift_gate,
    rotate_y_gate,
    rotate_z_gate,
    get_probabilities,
    normalize_state,
    cnot_gate,
    create_superposition_from_value,
    measure_qubit
)

from quantum.features import (
    QuantumFeature,
    QuantumFeatureSet,
    convert_features_to_quantum
)

from quantum.classifier import (
    QuantumClassifier
)

# Upgraded kernel with true entanglement
from quantum.kernel import (
    LivniumQubit,
    EntangledPair,
    normalize as normalize_state_kernel,
    X as PAULI_X_KERNEL,
    Y as PAULI_Y_KERNEL,
    Z as PAULI_Z_KERNEL,
    H as HADAMARD_KERNEL,
    CNOT as CNOT_KERNEL,
)

# Upgraded features with true entanglement
from quantum.features_v2 import (
    QuantumFeatureV2,
    QuantumFeatureSetV2,
    convert_features_to_quantum_v2,
)

__all__ = [
    # Gates
    'PAULI_X',
    'PAULI_Y',
    'PAULI_Z',
    'HADAMARD',
    'apply_gate',
    'hadamard_gate',
    'pauli_x_gate',
    'pauli_z_gate',
    'phase_shift_gate',
    'rotate_y_gate',
    'rotate_z_gate',
    'get_probabilities',
    'normalize_state',
    'cnot_gate',
    'create_superposition_from_value',
    'measure_qubit',
    # Features
    'QuantumFeature',
    'QuantumFeatureSet',
    'convert_features_to_quantum',
    # Classifier
    'QuantumClassifier',
    # Upgraded Kernel (True Entanglement)
    'LivniumQubit',
    'EntangledPair',
    'normalize_state_kernel',
    'PAULI_X_KERNEL',
    'PAULI_Y_KERNEL',
    'PAULI_Z_KERNEL',
    'HADAMARD_KERNEL',
    'CNOT_KERNEL',
    # Upgraded Features (True Entanglement)
    'QuantumFeatureV2',
    'QuantumFeatureSetV2',
    'convert_features_to_quantum_v2',
]

