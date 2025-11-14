"""
Livnium Quantum Kernel (Upgraded Version)

True multi-qubit entanglement support with proper 4D state vectors.
"""

import numpy as np
from cmath import phase
from typing import Optional, Tuple, List


# ==========================================================
#      LIVNIUM QUANTUM KERNEL (UPGRADED VERSION)
# ==========================================================

def normalize(state: np.ndarray) -> np.ndarray:
    """
    Normalize any quantum state vector.
    
    Optimized: Uses efficient norm calculation and avoids division by zero.
    """
    norm = np.linalg.norm(state)
    if norm > 1e-12:
        return state / norm
    else:
        # Return |0> state if norm is zero
        return np.array([1.0 + 0j, 0.0 + 0j], dtype=complex) if len(state) == 2 else state


# ------------------------------
#  Single-Qubit Gate Matrices
# ------------------------------
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


# ------------------------------
#  Two-Qubit Gate Matrices
# ------------------------------
CNOT = np.array([
    [1, 0, 0, 0],  # |00> -> |00>
    [0, 1, 0, 0],  # |01> -> |01>
    [0, 0, 0, 1],  # |10> -> |11>
    [0, 0, 1, 0],  # |11> -> |10>
], dtype=complex)


# ==========================================================
#                SINGLE QUBIT OBJECT
# ==========================================================

class LivniumQubit:
    """
    Single qubit with true entanglement support.
    
    Can exist independently or as part of an EntangledPair.
    """
    
    def __init__(self, position: Tuple[int, int, int], f: int,
                 initial_state: Optional[np.ndarray] = None):
        """
        Initialize a Livnium qubit.
        
        Args:
            position: 3D position (x, y, z)
            f: face exposure (0-3)
            initial_state: Optional [α, β] complex amplitudes. Defaults to |0>
        """
        self.position = position
        self.f = f
        self.SW = 9 * f

        if initial_state is None:
            self.state = np.array([1 + 0j, 0 + 0j])
        else:
            self.state = normalize(np.array(initial_state, dtype=complex))

        self.entangled = False
        self.entangled_state = None   # Reference to EntangledPair if entangled
        self.partner = None
        self.gate_history = []
        self.measurement_history = []

    # ---------------------------------------
    # Apply gate to single-qubit state
    # ---------------------------------------
    def apply_gate(self, gate, name="G"):
        """Apply a single-qubit gate."""
        if self.entangled:
            raise ValueError("Cannot apply single-qubit gate to entangled qubit. Use pair.apply_gate()")

        self.state = normalize(gate @ self.state)
        self.gate_history.append(name)

    # Shorthands
    def hadamard(self): 
        """Apply Hadamard gate."""
        self.apply_gate(H, "H")
    
    def pauli_x(self): 
        """Apply Pauli-X gate."""
        self.apply_gate(X, "X")
    
    def pauli_y(self): 
        """Apply Pauli-Y gate."""
        self.apply_gate(Y, "Y")
    
    def pauli_z(self): 
        """Apply Pauli-Z gate."""
        self.apply_gate(Z, "Z")

    def phase_shift(self, angle):
        """Apply phase shift gate."""
        gate = np.array([[1, 0], [0, np.exp(1j * angle)]])
        self.apply_gate(gate, f"P({angle:.2f})")

    # ---------------------------------------
    # Bloch sphere coordinates
    # ---------------------------------------
    def get_bloch(self):
        """Get (θ, φ) coordinates on Bloch sphere."""
        α, β = self.state
        α_mag = min(max(abs(α), 0), 1)
        θ = 2 * np.arccos(α_mag)
        φ = phase(β) - phase(α)
        return float(θ), float(φ)

    # ---------------------------------------
    # Probabilities
    # ---------------------------------------
    def get_probabilities(self):
        """Get measurement probabilities: (P(|0>), P(|1>))"""
        α, β = self.state
        return float(abs(α) ** 2), float(abs(β) ** 2)

    # ---------------------------------------
    # Measurement
    # ---------------------------------------
    def measure(self, basis="z"):
        """Measure the qubit, collapsing to |0> or |1>."""
        if self.entangled:
            raise ValueError("Use EntangledPair.measure()")

        p0, p1 = self.get_probabilities()
        result = 1 if np.random.rand() < p1 else 0

        self.state = np.array([1, 0], dtype=complex) if result == 0 else np.array([0, 1], dtype=complex)
        self.measurement_history.append(result)
        return result

    # ---------------------------------------
    def state_string(self):
        """Get quantum state as string: α|0> + β|1>"""
        α, β = self.state
        s = lambda c: f"{c.real:.3f}{'+' if c.imag>=0 else ''}{c.imag:.3f}i"
        return f"{s(α)}|0> + {s(β)}|1>"

    def info(self):
        """Get full qubit information."""
        p0, p1 = self.get_probabilities()
        θ, φ = self.get_bloch()
        return {
            "position": self.position,
            "SW": self.SW,
            "state": self.state_string(),
            "probabilities": (p0, p1),
            "bloch": (θ, φ),
            "entangled": self.entangled,
            "gate_history": self.gate_history
        }


# ==========================================================
#                TRUE ENTANGLED PAIR
# ==========================================================

class EntangledPair:
    """
    True 2-qubit entangled system with 4D state vector.
    
    Supports Bell states and proper entanglement operations.
    """
    
    def __init__(self, q1: LivniumQubit, q2: LivniumQubit, state4: np.ndarray):
        """
        Create an entangled pair from two qubits.
        
        Args:
            q1: First qubit
            q2: Second qubit
            state4: 4D state vector [α₀₀, α₀₁, α₁₀, α₁₁] for |00>, |01>, |10>, |11>
        """
        self.q1 = q1
        self.q2 = q2
        self.state4 = normalize(state4.astype(complex))

        q1.entangled = True
        q2.entangled = True
        q1.partner = q2
        q2.partner = q1
        q1.entangled_state = self
        q2.entangled_state = self

        self.history = []

    @staticmethod
    def bell(phi_plus=True, sign_plus=True):
        """
        Create a Bell state entangled pair.
        
        Args:
            phi_plus: If True, create |Φ+> or |Φ->, else |Ψ+> or |Ψ->
            sign_plus: If True, use + sign, else use - sign
            
        Returns:
            EntangledPair in Bell state
        """
        q1 = LivniumQubit((0, 0, 0), 1)
        q2 = LivniumQubit((0, 0, 0), 1)

        if phi_plus:
            s = 1 if sign_plus else -1
            state = np.array([1, 0, 0, s], dtype=complex) / np.sqrt(2)   # |00> ± |11>
        else:
            s = 1 if sign_plus else -1
            state = np.array([0, 1, 1*s, 0], dtype=complex) / np.sqrt(2) # |01> ± |10>

        return EntangledPair(q1, q2, state)

    @staticmethod
    def create_from_qubits(q1: LivniumQubit, q2: LivniumQubit):
        """
        Create entangled pair from two independent qubits using CNOT.
        
        Args:
            q1: Control qubit
            q2: Target qubit
            
        Returns:
            EntangledPair with q1 and q2 entangled
        """
        # Create tensor product state
        α1, β1 = q1.state
        α2, β2 = q2.state
        
        # Tensor product: |q1> ⊗ |q2>
        state4 = np.array([
            α1 * α2,  # |00>
            α1 * β2,  # |01>
            β1 * α2,  # |10>
            β1 * β2   # |11>
        ], dtype=complex)
        
        # Apply CNOT to entangle
        state4 = CNOT @ state4
        
        return EntangledPair(q1, q2, state4)

    # ---------------------------------------
    # 2-qubit gate application
    # ---------------------------------------
    def apply_gate(self, gate4, name="2G"):
        """
        Apply a 2-qubit gate to the entangled pair.
        
        Args:
            gate4: 4×4 unitary matrix
            name: Gate name for history
        """
        self.state4 = normalize(gate4 @ self.state4)
        self.history.append(name)

    def apply_cnot(self):
        """Apply CNOT gate (idempotent if already applied)."""
        self.apply_gate(CNOT, "CNOT")

    # ---------------------------------------
    # Measurement of both
    # ---------------------------------------
    def measure(self):
        """
        Measure both qubits simultaneously.
        
        Optimized: Efficient probability calculation and sampling.
        
        Returns:
            (result1, result2) where each is 0 or 1
        """
        # Optimized: Use in-place operations where possible
        probs = np.abs(self.state4) ** 2
        # Normalize probabilities (avoid division by zero)
        prob_sum = np.sum(probs)
        if prob_sum > 1e-12:
            probs = probs / prob_sum
        else:
            # Fallback: uniform distribution
            probs = np.ones(4) / 4.0
        
        result = np.random.choice(4, p=probs)

        # Collapse state
        collapsed = np.zeros(4, dtype=complex)
        collapsed[result] = 1
        self.state4 = collapsed

        # Decode result: |00>, |01>, |10>, |11>
        if result == 0: 
            q1_result, q2_result = 0, 0
        elif result == 1: 
            q1_result, q2_result = 0, 1
        elif result == 2: 
            q1_result, q2_result = 1, 0
        else:  # result == 3
            q1_result, q2_result = 1, 1
        
        # Update individual qubit states
        self.q1.state = np.array([1, 0], dtype=complex) if q1_result == 0 else np.array([0, 1], dtype=complex)
        self.q2.state = np.array([1, 0], dtype=complex) if q2_result == 0 else np.array([0, 1], dtype=complex)
        
        # Record measurements
        self.q1.measurement_history.append(q1_result)
        self.q2.measurement_history.append(q2_result)
        
        return (q1_result, q2_result)

    def get_probabilities(self):
        """Get probabilities for all 4 states: (P(|00>), P(|01>), P(|10>), P(|11>))"""
        probs = np.abs(self.state4) ** 2
        probs = probs / np.sum(probs)  # Normalize
        return tuple(float(p) for p in probs)

    def state_string(self):
        """Get entangled state as string."""
        p00, p01, p10, p11 = self.get_probabilities()
        α00, α01, α10, α11 = self.state4
        s = lambda c: f"{c.real:.3f}{'+' if c.imag>=0 else ''}{c.imag:.3f}i"
        return f"{s(α00)}|00> + {s(α01)}|01> + {s(α10)}|10> + {s(α11)}|11>"

    def info(self):
        """Get full entangled pair information."""
        p00, p01, p10, p11 = self.get_probabilities()
        return {
            "q1": self.q1.info(),
            "q2": self.q2.info(),
            "entangled_state": self.state_string(),
            "probabilities": {
                "|00>": p00,
                "|01>": p01,
                "|10>": p10,
                "|11>": p11
            },
            "history": self.history
        }


# ==========================================================
#                    DEMO
# ==========================================================

if __name__ == "__main__":
    print("\n=== TRUE ENTANGLEMENT DEMO ===\n")

    # Create Bell state
    bell = EntangledPair.bell(phi_plus=True, sign_plus=True)
    print("Initial Bell state: (|00> + |11>)/√2")
    print(f"State: {bell.state_string()}")
    
    p00, p01, p10, p11 = bell.get_probabilities()
    print(f"Probabilities: P(|00>)={p00:.3f}, P(|01>)={p01:.3f}, P(|10>)={p10:.3f}, P(|11>)={p11:.3f}")

    print("\nMeasuring both qubits:")
    r1, r2 = bell.measure()
    print(f"  Results: q1={r1}, q2={r2}")
    print(f"  Correlated: {r1 == r2} (should be True for Bell state)")
    
    print("\n=== Creating Entanglement from Independent Qubits ===\n")
    
    # Create two independent qubits
    q1 = LivniumQubit((1, 1, 1), f=1)
    q1.hadamard()  # Put in superposition
    
    q2 = LivniumQubit((2, 2, 2), f=2)
    
    print(f"q1 before: {q1.state_string()}")
    print(f"q2 before: {q2.state_string()}")
    
    # Entangle them
    pair = EntangledPair.create_from_qubits(q1, q2)
    print(f"\nAfter entanglement:")
    print(f"Entangled state: {pair.state_string()}")
    
    p00, p01, p10, p11 = pair.get_probabilities()
    print(f"Probabilities: P(|00>)={p00:.3f}, P(|01>)={p01:.3f}, P(|10>)={p10:.3f}, P(|11>)={p11:.3f}")
    
    print("\nMeasuring entangled pair:")
    r1, r2 = pair.measure()
    print(f"  Results: q1={r1}, q2={r2}")

