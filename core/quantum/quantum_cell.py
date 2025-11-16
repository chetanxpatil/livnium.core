"""
Quantum Cell: Superposition State for Livnium Core

Adds quantum state (complex amplitudes) to each lattice cell.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Try to import numba for acceleration (optional dependency)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@dataclass
class QuantumCell:
    """
    Quantum state for a single lattice cell.
    
    State: |ψ⟩ = α|0⟩ + β|1⟩ (2-level)
    Or: |ψ⟩ = Σᵢ αᵢ|i⟩ (N-level)
    
    Normalization: Σ|αᵢ|² = 1
    """
    coordinates: Tuple[int, int, int]
    amplitudes: np.ndarray  # Complex amplitudes [α₀, α₁, ..., αₙ₋₁]
    num_levels: int = 2  # 2 for qubit, N for qudit
    
    def __post_init__(self):
        """Initialize and normalize quantum state."""
        if self.amplitudes is None:
            # Default: |0⟩ state
            self.amplitudes = np.zeros(self.num_levels, dtype=complex)
            self.amplitudes[0] = 1.0 + 0j
        else:
            self.amplitudes = np.array(self.amplitudes, dtype=complex)
        
        # Normalize
        self.normalize()
    
    # Original implementation (commented for reference)
    # def normalize(self):
    #     """Normalize state: Σ|αᵢ|² = 1"""
    #     norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
    #     if norm > 1e-10:
    #         self.amplitudes /= norm
    #     else:
    #         # Zero state: set to |0⟩
    #         self.amplitudes = np.zeros(self.num_levels, dtype=complex)
    #         self.amplitudes[0] = 1.0 + 0j
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _normalize_numba(amplitudes: np.ndarray, num_levels: int):
        """Normalize state vector (numba-accelerated)."""
        # Calculate norm
        norm_sq = 0.0
        for i in range(len(amplitudes)):
            re = amplitudes[i].real
            im = amplitudes[i].imag
            norm_sq += re * re + im * im
        
        norm = np.sqrt(norm_sq)
        
        if norm > 1e-10:
            # Normalize
            for i in range(len(amplitudes)):
                amplitudes[i] = amplitudes[i] / norm
        else:
            # Zero state: set to |0⟩
            for i in range(len(amplitudes)):
                amplitudes[i] = 0.0 + 0.0j
            amplitudes[0] = 1.0 + 0.0j
    
    def normalize(self):
        """Normalize state: Σ|αᵢ|² = 1"""
        if NUMBA_AVAILABLE:
            self._normalize_numba(self.amplitudes, self.num_levels)
        else:
            # Fallback to original implementation
            norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
            if norm > 1e-10:
                self.amplitudes /= norm
            else:
                # Zero state: set to |0⟩
                self.amplitudes = np.zeros(self.num_levels, dtype=complex)
                self.amplitudes[0] = 1.0 + 0j
    
    # Original implementation (commented for reference)
    # def get_probabilities(self) -> np.ndarray:
    #     """Get measurement probabilities: P(i) = |αᵢ|²"""
    #     return np.abs(self.amplitudes) ** 2
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_probabilities_numba(amplitudes: np.ndarray) -> np.ndarray:
        """Get probabilities (numba-accelerated)."""
        probs = np.zeros(len(amplitudes), dtype=np.float64)
        for i in range(len(amplitudes)):
            re = amplitudes[i].real
            im = amplitudes[i].imag
            probs[i] = re * re + im * im
        return probs
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities: P(i) = |αᵢ|²"""
        if NUMBA_AVAILABLE:
            return self._get_probabilities_numba(self.amplitudes)
        else:
            # Fallback to original implementation
            return np.abs(self.amplitudes) ** 2
    
    def measure(self) -> int:
        """
        Measure the cell (Born rule + collapse).
        
        Returns:
            Measured state index (0, 1, ..., n-1)
        """
        probs = self.get_probabilities()
        measured = np.random.choice(self.num_levels, p=probs)
        
        # Collapse to measured state
        self.amplitudes = np.zeros(self.num_levels, dtype=complex)
        self.amplitudes[measured] = 1.0 + 0j
        
        return measured
    
    # Original implementation (commented for reference)
    # def apply_unitary(self, U: np.ndarray):
    #     """
    #     Apply unitary gate: |ψ'⟩ = U|ψ⟩
    #     
    #     Args:
    #         U: Unitary matrix (n×n for n-level system)
    #     """
    #     if U.shape != (self.num_levels, self.num_levels):
    #         raise ValueError(f"Unitary shape {U.shape} doesn't match num_levels {self.num_levels}")
    #     
    #     self.amplitudes = U @ self.amplitudes
    #     self.normalize()
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _apply_unitary_numba(amplitudes: np.ndarray, U: np.ndarray):
        """Apply unitary gate (numba-accelerated)."""
        n = len(amplitudes)
        result = np.zeros(n, dtype=np.complex128)
        
        for i in range(n):
            for j in range(n):
                result[i] += U[i, j] * amplitudes[j]
        
        # Copy result back
        for i in range(n):
            amplitudes[i] = result[i]
    
    def apply_unitary(self, U: np.ndarray):
        """
        Apply unitary gate: |ψ'⟩ = U|ψ⟩
        
        Args:
            U: Unitary matrix (n×n for n-level system)
        """
        if U.shape != (self.num_levels, self.num_levels):
            raise ValueError(f"Unitary shape {U.shape} doesn't match num_levels {self.num_levels}")
        
        if NUMBA_AVAILABLE:
            self._apply_unitary_numba(self.amplitudes, U)
            self.normalize()
        else:
            # Fallback to original implementation
            self.amplitudes = U @ self.amplitudes
            self.normalize()
    
    def get_state_vector(self) -> np.ndarray:
        """Get state vector (copy)."""
        return self.amplitudes.copy()
    
    def set_state_vector(self, state: np.ndarray):
        """Set state vector and normalize."""
        self.amplitudes = np.array(state, dtype=complex)
        self.normalize()
    
    def is_entangled_with(self, other: 'QuantumCell') -> bool:
        """
        Check if this cell is entangled with another.
        
        For now, returns False (entanglement handled at lattice level).
        """
        return False
    
    # Original implementation (commented for reference)
    # def get_fidelity(self, target_state: np.ndarray) -> float:
    #     """
    #     Calculate fidelity with target state: |⟨ψ|φ⟩|²
    #     
    #     Args:
    #         target_state: Target state vector
    #         
    #     Returns:
    #         Fidelity value [0, 1]
    #     """
    #     target = np.array(target_state, dtype=complex)
    #     target = target / np.linalg.norm(target)  # Normalize
    #     
    #     overlap = np.abs(np.vdot(self.amplitudes, target)) ** 2
    #     return float(overlap)
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_fidelity_numba(amplitudes: np.ndarray, target: np.ndarray) -> float:
        """Calculate fidelity (numba-accelerated)."""
        # Normalize target
        target_norm_sq = 0.0
        for i in range(len(target)):
            re = target[i].real
            im = target[i].imag
            target_norm_sq += re * re + im * im
        target_norm = np.sqrt(target_norm_sq)
        
        if target_norm < 1e-10:
            return 0.0
        
        # Calculate overlap: ⟨ψ|φ⟩
        overlap_re = 0.0
        overlap_im = 0.0
        for i in range(len(amplitudes)):
            # Conjugate of amplitudes (bra)
            amp_re = amplitudes[i].real
            amp_im = -amplitudes[i].imag  # Conjugate
            target_re = target[i].real / target_norm
            target_im = target[i].imag / target_norm
            
            # (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
            overlap_re += amp_re * target_re - amp_im * target_im
            overlap_im += amp_re * target_im + amp_im * target_re
        
        # |overlap|²
        fidelity = overlap_re * overlap_re + overlap_im * overlap_im
        return fidelity
    
    def get_fidelity(self, target_state: np.ndarray) -> float:
        """
        Calculate fidelity with target state: |⟨ψ|φ⟩|²
        
        Args:
            target_state: Target state vector
            
        Returns:
            Fidelity value [0, 1]
        """
        target = np.array(target_state, dtype=complex)
        
        if NUMBA_AVAILABLE:
            return float(self._get_fidelity_numba(self.amplitudes, target))
        else:
            # Fallback to original implementation
            target = target / np.linalg.norm(target)  # Normalize
            overlap = np.abs(np.vdot(self.amplitudes, target)) ** 2
            return float(overlap)
    
    def __repr__(self) -> str:
        """String representation."""
        probs = self.get_probabilities()
        return f"QuantumCell({self.coordinates}, |ψ⟩, probs={probs})"

