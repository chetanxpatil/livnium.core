"""
Measurement Engine: Born Rule + Collapse for Livnium-T

Real quantum measurement with Born rule probabilities and state collapse.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class MeasurementBasis(Enum):
    """Measurement basis."""
    COMPUTATIONAL = "Z"  # |0⟩, |1⟩
    X_BASIS = "X"  # |+⟩, |-⟩
    Y_BASIS = "Y"  # |+i⟩, |-i⟩


@dataclass
class MeasurementResult:
    """Result of quantum measurement."""
    node_id: int
    measured_level: int  # 0 or 1 for qubit
    probability: float  # Probability of this outcome
    basis: MeasurementBasis
    collapsed_state: np.ndarray  # State after collapse


class MeasurementEngine:
    """
    Quantum measurement engine using Born rule.
    
    P(level) = |⟨level|ψ⟩|²
    After measurement: |ψ⟩ → |level⟩ (collapse)
    """
    
    def __init__(self):
        """Initialize measurement engine."""
        self.measurement_history: List[MeasurementResult] = []
    
    def measure(self, amplitudes: np.ndarray, 
                node_id: int,
                basis: MeasurementBasis = MeasurementBasis.COMPUTATIONAL) -> MeasurementResult:
        """
        Measure quantum state using Born rule.
        
        Args:
            amplitudes: Complex amplitudes [α₀, α₁]
            node_id: Node ID being measured
            basis: Measurement basis
            
        Returns:
            MeasurementResult
        """
        # Get probabilities in the measurement basis
        if basis == MeasurementBasis.COMPUTATIONAL:
            # Standard basis: |0⟩, |1⟩
            probabilities = np.abs(amplitudes) ** 2
            basis_states = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
        elif basis == MeasurementBasis.X_BASIS:
            # X basis: |+⟩ = (|0⟩ + |1⟩)/√2, |-⟩ = (|0⟩ - |1⟩)/√2
            plus = (amplitudes[0] + amplitudes[1]) / np.sqrt(2)
            minus = (amplitudes[0] - amplitudes[1]) / np.sqrt(2)
            probabilities = np.array([np.abs(plus)**2, np.abs(minus)**2])
            probabilities = probabilities / np.sum(probabilities)  # Normalize
            basis_states = [
                np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
                np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
            ]
        elif basis == MeasurementBasis.Y_BASIS:
            # Y basis: |+i⟩ = (|0⟩ + i|1⟩)/√2, |-i⟩ = (|0⟩ - i|1⟩)/√2
            plus_i = (amplitudes[0] + 1j * amplitudes[1]) / np.sqrt(2)
            minus_i = (amplitudes[0] - 1j * amplitudes[1]) / np.sqrt(2)
            probabilities = np.array([np.abs(plus_i)**2, np.abs(minus_i)**2])
            probabilities = probabilities / np.sum(probabilities)  # Normalize
            basis_states = [
                np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex),
                np.array([1/np.sqrt(2), -1j/np.sqrt(2)], dtype=complex)
            ]
        else:
            raise ValueError(f"Unknown basis: {basis}")
        
        # Sample from probability distribution
        measured_level = np.random.choice(len(probabilities), p=probabilities)
        
        # Collapse to measured state
        collapsed_state = basis_states[measured_level].copy()
        
        # Create result
        result = MeasurementResult(
            node_id=node_id,
            measured_level=measured_level,
            probability=float(probabilities[measured_level]),
            basis=basis,
            collapsed_state=collapsed_state
        )
        
        # Record measurement
        self.measurement_history.append(result)
        
        return result
    
    def get_measurement_history(self) -> List[MeasurementResult]:
        """Get all measurement results."""
        return self.measurement_history.copy()
    
    def clear_history(self):
        """Clear measurement history."""
        self.measurement_history.clear()

