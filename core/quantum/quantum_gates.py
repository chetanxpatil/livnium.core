"""
Quantum Gates: Unitary Operations for Livnium Core

Standard quantum gates: Hadamard, Pauli, Phase, Rotation, CNOT, etc.
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
    # Create dummy decorators if numba not available
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
    
    # Original implementation (commented for reference)
    # @staticmethod
    # def phase(phi: float) -> np.ndarray:
    #     """
    #     Phase gate: P(φ) = [[1, 0], [0, e^(iφ)]]
    #     
    #     Args:
    #         phi: Phase angle in radians
    #     """
    #     return np.array([
    #         [1, 0],
    #         [0, np.exp(1j * phi)]
    #     ], dtype=complex)
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _phase_numba(phi: float):
        """Phase gate (numba-accelerated)."""
        exp_phi = np.exp(1j * phi)
        return ((1.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, exp_phi))
    
    @staticmethod
    def phase(phi: float) -> np.ndarray:
        """
        Phase gate: P(φ) = [[1, 0], [0, e^(iφ)]]
        
        Args:
            phi: Phase angle in radians
        """
        if NUMBA_AVAILABLE:
            result = QuantumGates._phase_numba(phi)
            return np.array([[result[0][0], result[0][1]], 
                           [result[1][0], result[1][1]]], dtype=complex)
        else:
            # Fallback to original implementation
            return np.array([
                [1, 0],
                [0, np.exp(1j * phi)]
            ], dtype=complex)
    
    # Original implementation (commented for reference)
    # @staticmethod
    # def rotation_x(theta: float) -> np.ndarray:
    #     """
    #     Rotation about X-axis: Rx(θ) = e^(-iθX/2)
    #     
    #     Args:
    #         theta: Rotation angle in radians
    #     """
    #     c = np.cos(theta / 2)
    #     s = np.sin(theta / 2)
    #     return np.array([
    #         [c, -1j * s],
    #         [-1j * s, c]
    #     ], dtype=complex)
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _rotation_x_numba(theta: float):
        """Rotation about X-axis (numba-accelerated)."""
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        # Return as tuple of tuples (numba-friendly)
        return ((c, -1j * s), (-1j * s, c))
    
    @staticmethod
    def rotation_x(theta: float) -> np.ndarray:
        """
        Rotation about X-axis: Rx(θ) = e^(-iθX/2)
        
        Args:
            theta: Rotation angle in radians
        """
        if NUMBA_AVAILABLE:
            result = QuantumGates._rotation_x_numba(theta)
            return np.array([[result[0][0], result[0][1]], 
                           [result[1][0], result[1][1]]], dtype=complex)
        else:
            # Fallback to original implementation
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            return np.array([
                [c, -1j * s],
                [-1j * s, c]
            ], dtype=complex)
    
    # Original implementation (commented for reference)
    # @staticmethod
    # def rotation_y(theta: float) -> np.ndarray:
    #     """
    #     Rotation about Y-axis: Ry(θ) = e^(-iθY/2)
    #     
    #     Args:
    #         theta: Rotation angle in radians
    #     """
    #     c = np.cos(theta / 2)
    #     s = np.sin(theta / 2)
    #     return np.array([
    #         [c, -s],
    #         [s, c]
    #     ], dtype=complex)
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _rotation_y_numba(theta: float):
        """Rotation about Y-axis (numba-accelerated)."""
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        return ((c, -s), (s, c))
    
    @staticmethod
    def rotation_y(theta: float) -> np.ndarray:
        """
        Rotation about Y-axis: Ry(θ) = e^(-iθY/2)
        
        Args:
            theta: Rotation angle in radians
        """
        if NUMBA_AVAILABLE:
            result = QuantumGates._rotation_y_numba(theta)
            return np.array([[result[0][0], result[0][1]], 
                           [result[1][0], result[1][1]]], dtype=complex)
        else:
            # Fallback to original implementation
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            return np.array([
                [c, -s],
                [s, c]
            ], dtype=complex)
    
    # Original implementation (commented for reference)
    # @staticmethod
    # def rotation_z(theta: float) -> np.ndarray:
    #     """
    #     Rotation about Z-axis: Rz(θ) = e^(-iθZ/2)
    #     
    #     Args:
    #         theta: Rotation angle in radians
    #     """
    #     return np.array([
    #         [np.exp(-1j * theta / 2), 0],
    #         [0, np.exp(1j * theta / 2)]
    #     ], dtype=complex)
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _rotation_z_numba(theta: float):
        """Rotation about Z-axis (numba-accelerated)."""
        angle = theta / 2.0
        exp_neg = np.exp(-1j * angle)
        exp_pos = np.exp(1j * angle)
        return ((exp_neg, 0.0 + 0.0j), (0.0 + 0.0j, exp_pos))
    
    @staticmethod
    def rotation_z(theta: float) -> np.ndarray:
        """
        Rotation about Z-axis: Rz(θ) = e^(-iθZ/2)
        
        Args:
            theta: Rotation angle in radians
        """
        if NUMBA_AVAILABLE:
            result = QuantumGates._rotation_z_numba(theta)
            return np.array([[result[0][0], result[0][1]], 
                           [result[1][0], result[1][1]]], dtype=complex)
        else:
            # Fallback to original implementation
            return np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]
            ], dtype=complex)
    
    @staticmethod
    def cnot(control: int = 0, target: int = 1) -> np.ndarray:
        """
        CNOT gate (2-qubit): flips target if control is |1⟩
        
        Args:
            control: Control qubit index (0 or 1)
            target: Target qubit index (0 or 1)
            
        Returns:
            4×4 unitary matrix
        """
        # For 2-qubit system: |control, target⟩
        U = np.eye(4, dtype=complex)
        
        # Flip target when control is |1⟩
        if control == 0 and target == 1:
            # Standard CNOT: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
            U[2, 2] = 0  # |10⟩ → |11⟩
            U[2, 3] = 1
            U[3, 2] = 1  # |11⟩ → |10⟩
            U[3, 3] = 0
        
        return U
    
    @staticmethod
    def cz() -> np.ndarray:
        """
        Controlled-Z gate (2-qubit): applies Z to target if control is |1⟩
        
        Returns:
            4×4 unitary matrix
        """
        U = np.eye(4, dtype=complex)
        U[3, 3] = -1  # |11⟩ → -|11⟩
        return U
    
    @staticmethod
    def swap() -> np.ndarray:
        """
        SWAP gate (2-qubit): swaps two qubits
        
        Returns:
            4×4 unitary matrix
        """
        U = np.eye(4, dtype=complex)
        # |01⟩ ↔ |10⟩
        U[1, 1] = 0
        U[1, 2] = 1
        U[2, 1] = 1
        U[2, 2] = 0
        return U
    
    @staticmethod
    def identity(num_levels: int = 2) -> np.ndarray:
        """Identity gate: I"""
        return np.eye(num_levels, dtype=complex)
    
    @staticmethod
    def is_unitary(U: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if matrix is unitary: U†U = I
        
        Args:
            U: Matrix to check
            tolerance: Numerical tolerance
            
        Returns:
            True if unitary
        """
        U_dagger = np.conj(U.T)
        product = U_dagger @ U
        identity = np.eye(U.shape[0])
        return np.allclose(product, identity, atol=tolerance)
    
    @staticmethod
    def get_gate(gate_type: GateType, **params) -> np.ndarray:
        """
        Get gate by type.
        
        Args:
            gate_type: Type of gate
            **params: Gate parameters (theta, phi, etc.)
            
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
            return QuantumGates.phase(params.get('phi', 0.0))
        elif gate_type == GateType.ROTATION_X:
            return QuantumGates.rotation_x(params.get('theta', 0.0))
        elif gate_type == GateType.ROTATION_Y:
            return QuantumGates.rotation_y(params.get('theta', 0.0))
        elif gate_type == GateType.ROTATION_Z:
            return QuantumGates.rotation_z(params.get('theta', 0.0))
        elif gate_type == GateType.CNOT:
            return QuantumGates.cnot(
                control=params.get('control', 0),
                target=params.get('target', 1)
            )
        elif gate_type == GateType.CZ:
            return QuantumGates.cz()
        elif gate_type == GateType.SWAP:
            return QuantumGates.swap()
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

