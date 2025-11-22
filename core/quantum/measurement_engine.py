"""
Measurement Engine: Born Rule and Collapse

Handles quantum measurement with collapse to classical states.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

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

from .quantum_cell import QuantumCell
from .entanglement_manager import EntangledPair


@dataclass
class MeasurementResult:
    """Result of a quantum measurement."""
    cell: Tuple[int, int, int]
    measured_state: int  # Measured basis state (0, 1, ..., n-1)
    probability: float  # Probability of this outcome
    collapsed: bool  # Whether state was collapsed
    
    def __repr__(self) -> str:
        return f"MeasurementResult({self.cell}, state={self.measured_state}, P={self.probability:.3f})"


class MeasurementEngine:
    """
    Quantum measurement engine with Born rule and collapse.
    
    Born Rule: P(i) = |⟨i|ψ⟩|² = |αᵢ|²
    Collapse: |ψ⟩ → |i⟩ (measured state)
    """
    
    def __init__(self):
        """Initialize measurement engine."""
        self.measurement_history: List[MeasurementResult] = []
    
    def measure_cell(self, cell: QuantumCell, collapse: bool = True) -> MeasurementResult:
        """
        Measure a single quantum cell.
        
        Args:
            cell: Quantum cell to measure
            collapse: Whether to collapse state after measurement
            
        Returns:
            Measurement result
        """
        # Get probabilities (Born rule)
        probs = cell.get_probabilities()
        
        # Sample from distribution
        measured_state = np.random.choice(len(probs), p=probs)
        probability = probs[measured_state]
        
        # Collapse if requested
        if collapse:
            cell.amplitudes = np.zeros(len(cell.amplitudes), dtype=complex)
            cell.amplitudes[measured_state] = 1.0 + 0j
        
        result = MeasurementResult(
            cell=cell.coordinates,
            measured_state=int(measured_state),
            probability=float(probability),
            collapsed=collapse
        )
        
        self.measurement_history.append(result)
        return result
    
    def measure_entangled_pair(self, pair: EntangledPair,
                               basis: str = "computational") -> Tuple[int, int]:
        """
        DEPRECATED: Measures "fake" entanglement metadata.
        
        ⚠️ WARNING: This measures the metadata label, NOT the actual qubits! ⚠️
        
        This method measures the EntangledPair's state_vector which is just
        metadata and does NOT affect the actual QuantumCell amplitudes. This
        is why teleportation fidelity was 0.0 - the measurement result didn't
        update the Source qubit's reality.
        
        For true quantum protocols, use `TrueQuantumRegister.measure_qubit()`
        from `true_quantum_layer.py`.
        
        This method is kept for backward compatibility but should not be used
        for quantum protocols.
        
        Measure an entangled pair (2-qubit measurement).
        
        Args:
            pair: Entangled pair (metadata only)
            basis: Measurement basis ("computational", "bell", etc.)
            
        Returns:
            Tuple of (cell1_state, cell2_state) - from metadata, not real qubits
        """
        if basis == "computational":
            # Measure in computational basis
            probs = np.abs(pair.state_vector) ** 2
            
            # Sample
            outcome = np.random.choice(4, p=probs)
            
            # Map to 2-qubit states: 0=|00⟩, 1=|01⟩, 2=|10⟩, 3=|11⟩
            cell1_state = outcome // 2
            cell2_state = outcome % 2
            
            # Collapse
            pair.state_vector = np.zeros(4, dtype=complex)
            pair.state_vector[outcome] = 1.0 + 0j
            
            return (cell1_state, cell2_state)
        else:
            raise ValueError(f"Unknown basis: {basis}")
    
    def measure_all_cells(self, cells: Dict[Tuple[int, int, int], QuantumCell],
                         collapse: bool = True) -> Dict[Tuple[int, int, int], MeasurementResult]:
        """
        Measure all cells in the lattice.
        
        Args:
            cells: Dictionary of quantum cells
            collapse: Whether to collapse states
            
        Returns:
            Dictionary of measurement results
        """
        results = {}
        for coords, cell in cells.items():
            result = self.measure_cell(cell, collapse=collapse)
            results[coords] = result
        return results
    
    # Original implementation (commented for reference)
    # def get_expectation_value(self, cell: QuantumCell, 
    #                         operator: np.ndarray) -> float:
    #     """
    #     Calculate expectation value: ⟨ψ|O|ψ⟩
    #     
    #     Args:
    #         cell: Quantum cell
    #         operator: Hermitian operator (matrix)
    #         
    #     Returns:
    #         Expectation value
    #     """
    #     state = cell.get_state_vector()
    #     result = np.vdot(state, operator @ state)
    #     return float(np.real(result))  # Expectation values are real
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_expectation_value_numba(state: np.ndarray, operator: np.ndarray) -> float:
        """Calculate expectation value (numba-accelerated)."""
        n = len(state)
        # Compute O|ψ⟩
        O_psi = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                O_psi[i] += operator[i, j] * state[j]
        
        # Compute ⟨ψ|O|ψ⟩
        result_re = 0.0
        result_im = 0.0
        for i in range(n):
            # Conjugate of state (bra)
            state_re = state[i].real
            state_im = -state[i].imag  # Conjugate
            O_psi_re = O_psi[i].real
            O_psi_im = O_psi[i].imag
            
            # (a-ib)*(c+id) = (ac+bd) + i(ad-bc)
            result_re += state_re * O_psi_re + state_im * O_psi_im
            result_im += state_re * O_psi_im - state_im * O_psi_re
        
        # Expectation values are real
        return result_re
    
    def get_expectation_value(self, cell: QuantumCell, 
                            operator: np.ndarray) -> float:
        """
        Calculate expectation value: ⟨ψ|O|ψ⟩
        
        Args:
            cell: Quantum cell
            operator: Hermitian operator (matrix)
            
        Returns:
            Expectation value
        """
        state = cell.get_state_vector()
        
        if NUMBA_AVAILABLE:
            return float(self._get_expectation_value_numba(state, operator))
        else:
            # Fallback to original implementation
            result = np.vdot(state, operator @ state)
            return float(np.real(result))  # Expectation values are real
    
    def get_variance(self, cell: QuantumCell, operator: np.ndarray) -> float:
        """
        Calculate variance: ⟨O²⟩ - ⟨O⟩²
        
        Args:
            cell: Quantum cell
            operator: Hermitian operator
            
        Returns:
            Variance
        """
        O_squared = operator @ operator
        expectation_O = self.get_expectation_value(cell, operator)
        expectation_O2 = self.get_expectation_value(cell, O_squared)
        variance = expectation_O2 - expectation_O ** 2
        return float(variance)
    
    def get_measurement_statistics(self) -> Dict:
        """Get statistics about measurements."""
        if not self.measurement_history:
            return {'total_measurements': 0}
        
        # Count outcomes
        outcome_counts = {}
        for result in self.measurement_history:
            key = (result.cell, result.measured_state)
            outcome_counts[key] = outcome_counts.get(key, 0) + 1
        
        return {
            'total_measurements': len(self.measurement_history),
            'unique_outcomes': len(outcome_counts),
            'outcome_counts': outcome_counts
        }

