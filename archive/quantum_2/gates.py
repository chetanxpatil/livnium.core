"""
Quantum Gates Module for Livnium

Provides reusable quantum gate operations optimized for feature representation
and classification. All gates are unitary matrices preserving quantum state normalization.
"""

import numpy as np
from typing import Tuple

# Try to import Numba for JIT compilation
try:
    import sys
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 14:
        NUMBA_AVAILABLE = False
        NUMBA_REASON = f"Python {python_version.major}.{python_version.minor} not supported"
    else:
        from numba import jit
        NUMBA_AVAILABLE = True
        NUMBA_REASON = None
except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_REASON = "Numba not installed"
except Exception as e:
    NUMBA_AVAILABLE = False
    NUMBA_REASON = f"Numba import failed: {e}"

# Create no-op decorator if Numba not available
if not NUMBA_AVAILABLE:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# Standard Quantum Gates (Unitary Matrices)
# ============================================================================

PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
PHASE_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
PHASE_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)


@jit(nopython=True, cache=True)
def apply_gate(state_vector: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """
    Apply a unitary gate to a quantum state vector.
    
    Args:
        state_vector: [α, β] complex amplitudes
        gate: 2x2 unitary matrix
        
    Returns:
        New state vector after gate application
    """
    return gate @ state_vector


@jit(nopython=True, cache=True)
def hadamard_gate(state_vector: np.ndarray) -> np.ndarray:
    """Apply Hadamard gate: creates superposition."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    return h @ state_vector


@jit(nopython=True, cache=True)
def pauli_x_gate(state_vector: np.ndarray) -> np.ndarray:
    """Apply Pauli-X gate (quantum NOT): |0> <-> |1>"""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return x @ state_vector


@jit(nopython=True, cache=True)
def pauli_z_gate(state_vector: np.ndarray) -> np.ndarray:
    """Apply Pauli-Z gate (phase flip)."""
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return z @ state_vector


@jit(nopython=True, cache=True)
def phase_shift_gate(state_vector: np.ndarray, angle: float) -> np.ndarray:
    """Apply phase shift gate: |1> -> e^(iθ)|1>"""
    gate = np.array([[1.0, 0.0], [0.0, np.exp(1j * angle)]], dtype=np.complex128)
    return gate @ state_vector


@jit(nopython=True, cache=True)
def rotate_y_gate(state_vector: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around Y-axis on Bloch sphere."""
    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
    return gate @ state_vector


@jit(nopython=True, cache=True)
def rotate_z_gate(state_vector: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around Z-axis on Bloch sphere."""
    gate = np.array([[np.exp(-1j * angle / 2.0), 0.0], 
                      [0.0, np.exp(1j * angle / 2.0)]], dtype=np.complex128)
    return gate @ state_vector


@jit(nopython=True, cache=True)
def get_probabilities(state_vector: np.ndarray) -> Tuple[float, float]:
    """
    Get measurement probabilities: (P(|0>), P(|1>))
    
    Args:
        state_vector: [α, β] complex amplitudes
        
    Returns:
        (P(|0>), P(|1>)) where P(|0>) = |α|², P(|1>) = |β|²
    """
    p0 = np.abs(state_vector[0]) ** 2
    p1 = np.abs(state_vector[1]) ** 2
    # Normalize (should already be normalized, but ensure)
    total = p0 + p1
    if total > 1e-10:
        p0 /= total
        p1 /= total
    return p0, p1


@jit(nopython=True, cache=True)
def normalize_state(state_vector: np.ndarray) -> np.ndarray:
    """Normalize state vector to ensure |α|² + |β|² = 1"""
    norm = np.sqrt(np.abs(state_vector[0])**2 + np.abs(state_vector[1])**2)
    if norm > 1e-10:
        return state_vector / norm
    else:
        # Return |0> state if norm is zero
        return np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)


def cnot_gate(control_state: np.ndarray, target_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    CNOT gate: flips target if control is |1>
    
    Note: This is a simplified version. True CNOT requires a 2-qubit state space.
    This version measures control probabilistically and applies X to target.
    
    Args:
        control_state: [α_c, β_c] control qubit state
        target_state: [α_t, β_t] target qubit state
        
    Returns:
        (control_state, new_target_state)
    """
    # Get probability that control is |1>
    _, p1 = get_probabilities(control_state)
    
    # Apply X gate to target with probability p1
    if np.random.rand() < p1:
        target_state = pauli_x_gate(target_state)
    
    return control_state, target_state


def create_superposition_from_value(value: float, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Convert a deterministic value to a quantum superposition state.
    
    Maps value to probability amplitude: higher value → higher probability of |1>
    
    Args:
        value: Deterministic value to convert
        min_val: Minimum possible value
        max_val: Maximum possible value
        
    Returns:
        [α, β] state vector where |β|² represents normalized value
    """
    # Normalize value to [0, 1]
    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Create superposition: |β|² = normalized value
    # |α|² = 1 - normalized value
    alpha = np.sqrt(1.0 - normalized)
    beta = np.sqrt(normalized)
    
    return np.array([alpha + 0j, beta + 0j], dtype=np.complex128)


def measure_qubit(state_vector: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Measure a qubit, collapsing it to |0> or |1>.
    
    Args:
        state_vector: [α, β] quantum state
        
    Returns:
        (result, collapsed_state) where result is 0 or 1
    """
    p0, p1 = get_probabilities(state_vector)
    
    # Sample measurement outcome
    result = 1 if np.random.rand() < p1 else 0
    
    # Collapse state
    if result == 0:
        collapsed = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
    else:
        collapsed = np.array([0.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    
    return result, collapsed

