"""
Quantum Gates: Unitary Operations for Livnium-T

Standard quantum gates: Hadamard, Pauli, Phase, Rotation, CNOT, etc.
Same as Livnium Core, adapted for Livnium-T.
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum

# Try to import numba for acceleration (optional dependency)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class GateType(Enum):
    """Types of quantum gates."""
    HADAMARD = "H"
    PAULI_X = "X"
    PAULI_Y = "Y"
    PAULI_Z = "Z"
    PHASE = "P"
    ROTATION_X = "Rx"
    ROTATION_Y = "Ry"
    ROTATION_Z = "Rz"
    CNOT = "CNOT"
    CZ = "CZ"
    SWAP = "SWAP"
    CUSTOM = "CUSTOM"


class QuantumGates:
    """
    Library of quantum gates (unitary matrices).
    
    All gates are unitary: U†U = I
    """
    
    @staticmethod
    def hadamard() -> np.ndarray:
        """Hadamard gate: H = (1/√2)[[1, 1], [1, -1]]"""
        return (1.0 / np.sqrt(2)) * np.array([
            [1, 1],
            [1, -1]
        ], dtype=complex)
    
    @staticmethod
    def pauli_x() -> np.ndarray:
        """Pauli X gate: X = [[0, 1], [1, 0]]"""
        return np.array([
            [0, 1],
            [1, 0]
        ], dtype=complex)
    
    @staticmethod
    def pauli_y() -> np.ndarray:
        """Pauli Y gate: Y = [[0, -i], [i, 0]]"""
        return np.array([
            [0, -1j],
            [1j, 0]
        ], dtype=complex)
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        """Pauli Z gate: Z = [[1, 0], [0, -1]]"""
        return np.array([
            [1, 0],
            [0, -1]
        ], dtype=complex)
    
    @staticmethod
    def phase(phi: float) -> np.ndarray:
        """
        Phase gate: P(φ) = [[1, 0], [0, e^(iφ)]]
        
        Args:
            phi: Phase angle in radians
        """
        return np.array([
            [1, 0],
            [0, np.exp(1j * phi)]
        ], dtype=complex)
    
    @staticmethod
    def rotation_x(theta: float) -> np.ndarray:
        """
        Rotation about X-axis: Rx(θ) = e^(-iθX/2)
        
        Args:
            theta: Rotation angle in radians
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([
            [c, -1j * s],
            [-1j * s, c]
        ], dtype=complex)
    
    @staticmethod
    def rotation_y(theta: float) -> np.ndarray:
        """
        Rotation about Y-axis: Ry(θ) = e^(-iθY/2)
        
        Args:
            theta: Rotation angle in radians
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([
            [c, -s],
            [s, c]
        ], dtype=complex)
    
    @staticmethod
    def rotation_z(theta: float) -> np.ndarray:
        """
        Rotation about Z-axis: Rz(θ) = e^(-iθZ/2)
        
        Args:
            theta: Rotation angle in radians
        """
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)
    
    @staticmethod
    def cnot(control: int = 0, target: int = 1) -> np.ndarray:
        """
        CNOT gate (controlled-NOT).
        
        Args:
            control: Control qubit index (0 or 1)
            target: Target qubit index (0 or 1)
            
        Returns:
            4x4 CNOT matrix
        """
        # Standard CNOT: |control, target⟩ → |control, target⊕control⟩
        cnot_matrix = np.eye(4, dtype=complex)
        if control == 0 and target == 1:
            # Standard CNOT: flips target if control is |1⟩
            cnot_matrix[2, 2] = 0  # |10⟩ → |11⟩
            cnot_matrix[3, 3] = 0  # |11⟩ → |10⟩
            cnot_matrix[2, 3] = 1
            cnot_matrix[3, 2] = 1
        elif control == 1 and target == 0:
            # Reverse CNOT: flips target if control is |1⟩
            cnot_matrix[1, 1] = 0  # |01⟩ → |11⟩
            cnot_matrix[3, 3] = 0  # |11⟩ → |01⟩
            cnot_matrix[1, 3] = 1
            cnot_matrix[3, 1] = 1
        
        return cnot_matrix
    
    @staticmethod
    def cz() -> np.ndarray:
        """Controlled-Z gate: CZ = diag(1, 1, 1, -1)"""
        return np.diag([1, 1, 1, -1]).astype(complex)
    
    @staticmethod
    def swap() -> np.ndarray:
        """SWAP gate: swaps two qubits"""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    @staticmethod
    def get_gate(gate_type: GateType, **params) -> np.ndarray:
        """
        Get gate matrix by type.
        
        Args:
            gate_type: Type of gate
            **params: Gate parameters (e.g., phi, theta)
            
        Returns:
            Unitary matrix
        """
        if gate_type == GateType.HADAMARD:
            return QuantumGates.hadamard()
        elif gate_type == GateType.PAULI_X:
            return QuantumGates.pauli_x()
        elif gate_type == GateType.PAULI_Y:
            return QuantumGates.pauli_y()
        elif gate_type == GateType.PAULI_Z:
            return QuantumGates.pauli_z()
        elif gate_type == GateType.PHASE:
            phi = params.get('phi', 0.0)
            return QuantumGates.phase(phi)
        elif gate_type == GateType.ROTATION_X:
            theta = params.get('theta', 0.0)
            return QuantumGates.rotation_x(theta)
        elif gate_type == GateType.ROTATION_Y:
            theta = params.get('theta', 0.0)
            return QuantumGates.rotation_y(theta)
        elif gate_type == GateType.ROTATION_Z:
            theta = params.get('theta', 0.0)
            return QuantumGates.rotation_z(theta)
        elif gate_type == GateType.CNOT:
            control = params.get('control', 0)
            target = params.get('target', 1)
            return QuantumGates.cnot(control, target)
        elif gate_type == GateType.CZ:
            return QuantumGates.cz()
        elif gate_type == GateType.SWAP:
            return QuantumGates.swap()
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

