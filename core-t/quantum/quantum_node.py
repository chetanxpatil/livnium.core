"""
Quantum Node: Superposition State for Livnium-T

Adds quantum state (complex amplitudes) to each node in the 5-node topology.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

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


@dataclass
class QuantumNode:
    """
    Quantum state for a single node in Livnium-T.
    
    State: |ψ⟩ = α|0⟩ + β|1⟩ (2-level qubit)
    Or: |ψ⟩ = Σᵢ αᵢ|i⟩ (N-level qudit)
    
    Normalization: Σ|αᵢ|² = 1
    """
    node_id: int  # Node ID (0-4): 0 = Core, 1-4 = Vertices
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
        
        # Validate node_id
        if self.node_id < 0 or self.node_id > 4:
            raise ValueError(f"Node ID must be in [0, 4], got {self.node_id}")
        
        # Normalize
        self.normalize()
    
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
            # Fallback implementation
            norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
            if norm > 1e-10:
                self.amplitudes /= norm
            else:
                # Zero state: set to |0⟩
                self.amplitudes = np.zeros(self.num_levels, dtype=complex)
                self.amplitudes[0] = 1.0 + 0j
    
    def apply_unitary(self, unitary: np.ndarray):
        """
        Apply unitary gate: |ψ'⟩ = U|ψ⟩
        
        Args:
            unitary: Unitary matrix (must match num_levels)
        """
        if unitary.shape != (self.num_levels, self.num_levels):
            raise ValueError(
                f"Unitary shape {unitary.shape} doesn't match num_levels {self.num_levels}"
            )
        
        # Apply: |ψ'⟩ = U|ψ⟩
        self.amplitudes = unitary @ self.amplitudes
        self.normalize()
    
    def get_probability(self, level: int) -> float:
        """
        Get probability of measuring |level⟩: P(level) = |α_level|²
        
        Args:
            level: Level to measure (0, 1, ..., num_levels-1)
            
        Returns:
            Probability
        """
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Level must be in [0, {self.num_levels-1}], got {level}")
        
        return float(np.abs(self.amplitudes[level]) ** 2)
    
    def get_probabilities(self) -> np.ndarray:
        """
        Get all probabilities: P(i) = |αᵢ|²
        
        Returns:
            Array of probabilities
        """
        return np.abs(self.amplitudes) ** 2
    
    def measure(self) -> int:
        """
        Measure the quantum state (Born rule).
        
        Returns:
            Measured level (0, 1, ..., num_levels-1)
        """
        probabilities = self.get_probabilities()
        measured = np.random.choice(self.num_levels, p=probabilities)
        
        # Collapse to measured state
        self.amplitudes = np.zeros(self.num_levels, dtype=complex)
        self.amplitudes[measured] = 1.0 + 0j
        
        return int(measured)
    
    def __repr__(self) -> str:
        """String representation."""
        probs = self.get_probabilities()
        return (
            f"QuantumNode(node_id={self.node_id}, "
            f"state=|ψ⟩, "
            f"P(0)={probs[0]:.3f}, "
            f"P(1)={probs[1]:.3f})"
        )

