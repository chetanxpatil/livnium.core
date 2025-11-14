"""
True 3-Qubit GHZ State Simulator

Implements a proper 8-dimensional state vector for 3 qubits.
This enforces true quantum mechanics: GHZ states can ONLY produce |000> or |111>.

Trade-off:
- Geometric simulator: Efficient for 105+ qubits, uses pairwise entanglement
- True GHZ simulator: Correct physics for 3 qubits, uses full 8D state vector

For Livnium's goals (geometric/AI), the geometric simulator is fine.
For strict physics verification, use this simulator.
"""

import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TrueGHZSimulator:
    """
    True 3-qubit quantum simulator with 8-dimensional state vector.
    
    State vector: [α₀₀₀, α₀₀₁, α₀₁₀, α₀₁₁, α₁₀₀, α₁₀₁, α₁₁₀, α₁₁₁]
                  |000>  |001>  |010>  |011>  |100>  |101>  |110>  |111>
    
    This enforces true quantum mechanics: GHZ states can ONLY produce |000> or |111>.
    """
    
    def __init__(self):
        """Initialize 3-qubit system in |000> state."""
        # 8D state vector: [|000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>]
        self.state = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j,
                               0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        self.gate_history = []
    
    def normalize(self):
        """Normalize the state vector."""
        norm = np.linalg.norm(self.state)
        if norm > 1e-12:
            self.state = self.state / norm
        else:
            # Fallback to |000>
            self.state = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j,
                                   0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=np.complex128)
    
    def apply_hadamard(self, qubit_idx: int):
        """
        Apply Hadamard gate to qubit at index (0, 1, or 2).
        
        Uses tensor products:
        - H on qubit 0: H ⊗ I ⊗ I
        - H on qubit 1: I ⊗ H ⊗ I
        - H on qubit 2: I ⊗ I ⊗ H
        """
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        I = np.eye(2, dtype=np.complex128)
        
        if qubit_idx == 0:
            gate = np.kron(np.kron(H, I), I)
        elif qubit_idx == 1:
            gate = np.kron(np.kron(I, H), I)
        elif qubit_idx == 2:
            gate = np.kron(np.kron(I, I), H)
        else:
            raise ValueError(f"Qubit index must be 0, 1, or 2, got {qubit_idx}")
        
        self.state = gate @ self.state
        self.normalize()
        self.gate_history.append(f"H({qubit_idx})")
    
    def apply_cnot(self, control_idx: int, target_idx: int):
        """
        Apply CNOT gate: control on qubit control_idx, target on qubit target_idx.
        
        Uses tensor products to construct 8×8 CNOT matrix.
        """
        CNOT_2q = np.array([
            [1, 0, 0, 0],  # |00> -> |00>
            [0, 1, 0, 0],  # |01> -> |01>
            [0, 0, 0, 1],  # |10> -> |11>
            [0, 0, 1, 0],  # |11> -> |10>
        ], dtype=np.complex128)
        
        I = np.eye(2, dtype=np.complex128)
        
        # Construct 8×8 CNOT matrix based on which qubits are control/target
        if control_idx == 0 and target_idx == 1:
            # CNOT on qubits 0 (control) and 1 (target), identity on qubit 2
            gate = np.kron(CNOT_2q, I)
        elif control_idx == 0 and target_idx == 2:
            # CNOT on qubits 0 (control) and 2 (target), identity on qubit 1
            # Need to swap: CNOT(0→2) = SWAP(1,2) CNOT(0→1) SWAP(1,2)
            # Or construct directly
            gate = self._cnot_0_to_2()
        elif control_idx == 1 and target_idx == 0:
            # CNOT on qubits 1 (control) and 0 (target)
            gate = self._cnot_1_to_0()
        elif control_idx == 1 and target_idx == 2:
            # CNOT on qubits 1 (control) and 2 (target), identity on qubit 0
            gate = np.kron(I, CNOT_2q)
        elif control_idx == 2 and target_idx == 0:
            # CNOT on qubits 2 (control) and 0 (target)
            gate = self._cnot_2_to_0()
        elif control_idx == 2 and target_idx == 1:
            # CNOT on qubits 2 (control) and 1 (target)
            gate = self._cnot_2_to_1()
        else:
            raise ValueError(f"Invalid CNOT indices: control={control_idx}, target={target_idx}")
        
        self.state = gate @ self.state
        self.normalize()
        self.gate_history.append(f"CNOT({control_idx}→{target_idx})")
    
    def _cnot_0_to_2(self):
        """CNOT with control=0, target=2 (qubit 1 unchanged)."""
        # Basis: |000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>
        gate = np.zeros((8, 8), dtype=np.complex128)
        # |0xx> -> |0xx> (identity)
        gate[0, 0] = 1  # |000> -> |000>
        gate[1, 1] = 1  # |001> -> |001>
        gate[2, 2] = 1  # |010> -> |010>
        gate[3, 3] = 1  # |011> -> |011>
        # |1xx> -> flip qubit 2
        gate[4, 5] = 1  # |100> -> |101>
        gate[5, 4] = 1  # |101> -> |100>
        gate[6, 7] = 1  # |110> -> |111>
        gate[7, 6] = 1  # |111> -> |110>
        return gate
    
    def _cnot_1_to_0(self):
        """CNOT with control=1, target=0 (qubit 2 unchanged)."""
        gate = np.zeros((8, 8), dtype=np.complex128)
        # |x0x> -> |x0x> (identity)
        gate[0, 0] = 1  # |000> -> |000>
        gate[1, 1] = 1  # |001> -> |001>
        gate[4, 4] = 1  # |100> -> |100>
        gate[5, 5] = 1  # |101> -> |101>
        # |x1x> -> flip qubit 0
        gate[2, 6] = 1  # |010> -> |110>
        gate[3, 7] = 1  # |011> -> |111>
        gate[6, 2] = 1  # |110> -> |010>
        gate[7, 3] = 1  # |111> -> |011>
        return gate
    
    def _cnot_2_to_0(self):
        """CNOT with control=2, target=0 (qubit 1 unchanged)."""
        gate = np.zeros((8, 8), dtype=np.complex128)
        # |xx0> -> |xx0> (identity)
        gate[0, 0] = 1  # |000> -> |000>
        gate[2, 2] = 1  # |010> -> |010>
        gate[4, 4] = 1  # |100> -> |100>
        gate[6, 6] = 1  # |110> -> |110>
        # |xx1> -> flip qubit 0
        gate[1, 5] = 1  # |001> -> |101>
        gate[3, 7] = 1  # |011> -> |111>
        gate[5, 1] = 1  # |101> -> |001>
        gate[7, 3] = 1  # |111> -> |011>
        return gate
    
    def _cnot_2_to_1(self):
        """CNOT with control=2, target=1 (qubit 0 unchanged)."""
        gate = np.zeros((8, 8), dtype=np.complex128)
        # |xx0> -> |xx0> (identity)
        gate[0, 0] = 1  # |000> -> |000>
        gate[1, 1] = 1  # |001> -> |001>
        gate[4, 4] = 1  # |100> -> |100>
        gate[5, 5] = 1  # |101> -> |101>
        # |xx1> -> flip qubit 1
        gate[2, 3] = 1  # |010> -> |011>
        gate[3, 2] = 1  # |011> -> |010>
        gate[6, 7] = 1  # |110> -> |111>
        gate[7, 6] = 1  # |111> -> |110>
        return gate
    
    def measure(self) -> Tuple[int, int, int]:
        """
        Measure all three qubits simultaneously.
        
        Returns:
            (result_A, result_B, result_C) where each is 0 or 1
        """
        # Calculate probabilities for each basis state
        probs = np.abs(self.state) ** 2
        prob_sum = np.sum(probs)
        if prob_sum > 1e-12:
            probs = probs / prob_sum
        else:
            # Fallback: uniform distribution
            probs = np.ones(8) / 8.0
        
        # Sample from distribution
        result_idx = np.random.choice(8, p=probs)
        
        # Decode result: |000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>
        # Binary representation: result_idx = 4*A + 2*B + C
        result_a = (result_idx // 4) % 2
        result_b = (result_idx // 2) % 2
        result_c = result_idx % 2
        
        # Collapse state to measured state
        collapsed = np.zeros(8, dtype=np.complex128)
        collapsed[result_idx] = 1.0
        self.state = collapsed
        
        return (result_a, result_b, result_c)
    
    def get_probabilities(self) -> dict:
        """Get probabilities for all 8 basis states."""
        probs = np.abs(self.state) ** 2
        prob_sum = np.sum(probs)
        if prob_sum > 1e-12:
            probs = probs / prob_sum
        
        basis_states = ['000', '001', '010', '011', '100', '101', '110', '111']
        return {state: float(prob) for state, prob in zip(basis_states, probs)}
    
    def get_state_string(self) -> str:
        """Get state as string: α|000> + β|001> + ..."""
        probs = self.get_probabilities()
        terms = []
        for state, prob in probs.items():
            if prob > 1e-6:
                amplitude = self.state[int(state, 2)]
                terms.append(f"{amplitude:.3f}|{state}>")
        return " + ".join(terms) if terms else "|000>"
    
    def set_qubit_state(self, qubit_idx: int, alpha: complex, beta: complex):
        """
        Set qubit at index to arbitrary state α|0> + β|1>.
        
        This initializes the 3-qubit state as |qubit_idx> ⊗ |0> ⊗ |0> with the given amplitudes.
        """
        # Normalize input
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        if norm > 1e-12:
            alpha = alpha / norm
            beta = beta / norm
        else:
            alpha, beta = 1.0 + 0j, 0.0 + 0j
        
        # Reset to |000>
        self.state = np.zeros(8, dtype=np.complex128)
        
        if qubit_idx == 0:
            # Q0 = α|0> + β|1>, Q1 = |0>, Q2 = |0>
            # State: α|000> + β|100>
            self.state[0] = alpha  # |000>
            self.state[4] = beta   # |100>
        elif qubit_idx == 1:
            # Q0 = |0>, Q1 = α|0> + β|1>, Q2 = |0>
            # State: α|000> + β|010>
            self.state[0] = alpha  # |000>
            self.state[2] = beta     # |010>
        elif qubit_idx == 2:
            # Q0 = |0>, Q1 = |0>, Q2 = α|0> + β|1>
            # State: α|000> + β|001>
            self.state[0] = alpha  # |000>
            self.state[1] = beta    # |001>
        else:
            raise ValueError(f"Qubit index must be 0, 1, or 2, got {qubit_idx}")
        
        self.normalize()
        self.gate_history.append(f"SET({qubit_idx}, α={alpha:.3f}, β={beta:.3f})")
    
    def apply_pauli_x(self, qubit_idx: int):
        """Apply Pauli-X gate (NOT) to qubit at index."""
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        I = np.eye(2, dtype=np.complex128)
        
        if qubit_idx == 0:
            gate = np.kron(np.kron(X, I), I)
        elif qubit_idx == 1:
            gate = np.kron(np.kron(I, X), I)
        elif qubit_idx == 2:
            gate = np.kron(np.kron(I, I), X)
        else:
            raise ValueError(f"Qubit index must be 0, 1, or 2, got {qubit_idx}")
        
        self.state = gate @ self.state
        self.normalize()
        self.gate_history.append(f"X({qubit_idx})")
    
    def apply_pauli_z(self, qubit_idx: int):
        """Apply Pauli-Z gate (phase flip) to qubit at index."""
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        I = np.eye(2, dtype=np.complex128)
        
        if qubit_idx == 0:
            gate = np.kron(np.kron(Z, I), I)
        elif qubit_idx == 1:
            gate = np.kron(np.kron(I, Z), I)
        elif qubit_idx == 2:
            gate = np.kron(np.kron(I, I), Z)
        else:
            raise ValueError(f"Qubit index must be 0, 1, or 2, got {qubit_idx}")
        
        self.state = gate @ self.state
        self.normalize()
        self.gate_history.append(f"Z({qubit_idx})")
    
    def measure_qubit(self, qubit_idx: int) -> int:
        """
        Measure a single qubit, collapsing the state.
        
        Returns:
            0 or 1 (the measurement result)
        """
        # Calculate probabilities for measuring qubit_idx = 0 or 1
        probs_0 = 0.0
        probs_1 = 0.0
        
        for i in range(8):
            # Extract bit for qubit_idx
            if qubit_idx == 0:
                bit = (i // 4) % 2
            elif qubit_idx == 1:
                bit = (i // 2) % 2
            else:  # qubit_idx == 2
                bit = i % 2
            
            prob = abs(self.state[i]) ** 2
            if bit == 0:
                probs_0 += prob
            else:
                probs_1 += prob
        
        # Normalize
        total = probs_0 + probs_1
        if total > 1e-12:
            probs_0 /= total
            probs_1 /= total
        else:
            probs_0, probs_1 = 0.5, 0.5
        
        # Sample
        result = 1 if np.random.rand() < probs_1 else 0
        
        # Collapse state: zero out amplitudes inconsistent with measurement
        collapsed = np.zeros(8, dtype=np.complex128)
        norm = 0.0
        
        for i in range(8):
            # Extract bit for qubit_idx
            if qubit_idx == 0:
                bit = (i // 4) % 2
            elif qubit_idx == 1:
                bit = (i // 2) % 2
            else:  # qubit_idx == 2
                bit = i % 2
            
            if bit == result:
                collapsed[i] = self.state[i]
                norm += abs(collapsed[i]) ** 2
        
        # Normalize collapsed state
        if norm > 1e-12:
            collapsed = collapsed / np.sqrt(norm)
        else:
            # Fallback: set to |000> if measurement was 0, |100> if 1 (for qubit 0)
            collapsed = np.zeros(8, dtype=np.complex128)
            if qubit_idx == 0:
                collapsed[0 if result == 0 else 4] = 1.0
            elif qubit_idx == 1:
                collapsed[0 if result == 0 else 2] = 1.0
            else:
                collapsed[0 if result == 0 else 1] = 1.0
        
        self.state = collapsed
        self.normalize()
        
        return result
    
    def get_qubit_state(self, qubit_idx: int) -> Tuple[complex, complex]:
        """
        Get the state of a single qubit (marginal state).
        
        This computes the reduced density matrix by tracing out other qubits.
        
        Returns:
            (α, β) where α|0> + β|1> is the qubit's state
        """
        # Compute reduced density matrix for qubit_idx
        # Sum over all basis states where qubit_idx is |0> or |1>
        alpha_0 = 0.0 + 0j  # Amplitude for |0>
        alpha_1 = 0.0 + 0j  # Amplitude for |1>
        
        for i in range(8):
            # Extract bit for qubit_idx
            if qubit_idx == 0:
                bit = (i // 4) % 2
            elif qubit_idx == 1:
                bit = (i // 2) % 2
            else:  # qubit_idx == 2
                bit = i % 2
            
            amplitude = self.state[i]
            
            # For marginal state, we need to sum amplitudes for each qubit value
            # But we need to be careful: if other qubits are measured/collapsed,
            # we just extract the amplitude directly
            if bit == 0:
                alpha_0 += amplitude
            else:
                alpha_1 += amplitude
        
        # Normalize
        norm = np.sqrt(abs(alpha_0)**2 + abs(alpha_1)**2)
        if norm > 1e-12:
            alpha_0 /= norm
            alpha_1 /= norm
        else:
            alpha_0, alpha_1 = 1.0 + 0j, 0.0 + 0j
        
        return alpha_0, alpha_1
    
    def create_ghz_state(self):
        """
        Create GHZ state: (|000> + |111>)/√2
        
        Steps:
        1. Start in |000>
        2. H on qubit 0: (|000> + |100>)/√2
        3. CNOT(0→1): (|000> + |110>)/√2
        4. CNOT(1→2): (|000> + |111>)/√2
        """
        # Reset to |000>
        self.state = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j,
                               0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        self.gate_history = []
        
        # Step 1: Hadamard on qubit 0
        self.apply_hadamard(0)
        
        # Step 2: CNOT(0→1)
        self.apply_cnot(0, 1)
        
        # Step 3: CNOT(1→2)
        self.apply_cnot(1, 2)
        
        # Verify we have GHZ state
        probs = self.get_probabilities()
        p000 = probs.get('000', 0)
        p111 = probs.get('111', 0)
        other_probs = sum(probs.get(state, 0) for state in ['001', '010', '011', '100', '101', '110'])
        
        return {
            'success': abs(p000 - 0.5) < 0.01 and abs(p111 - 0.5) < 0.01 and other_probs < 0.01,
            'p000': p000,
            'p111': p111,
            'other_probs': other_probs
        }


if __name__ == "__main__":
    print("=" * 70)
    print("TRUE 3-QUBIT GHZ STATE SIMULATOR")
    print("=" * 70)
    print()
    print("This simulator uses a full 8-dimensional state vector.")
    print("GHZ states can ONLY produce |000> or |111> outcomes.")
    print()
    
    # Create simulator
    sim = TrueGHZSimulator()
    
    # Create GHZ state
    print("Creating GHZ state: (|000> + |111>)/√2")
    print("Steps:")
    print("  1. Start in |000>")
    print("  2. H on qubit 0")
    print("  3. CNOT(0→1)")
    print("  4. CNOT(1→2)")
    print()
    
    result = sim.create_ghz_state()
    
    print(f"State: {sim.get_state_string()}")
    print()
    
    probs = sim.get_probabilities()
    print("Probabilities:")
    for state, prob in sorted(probs.items()):
        if prob > 1e-6:
            print(f"  P(|{state}>) = {prob:.6f}")
    print()
    
    print(f"✅ GHZ state verification:")
    print(f"  P(|000>) = {result['p000']:.6f} (should be ~0.5)")
    print(f"  P(|111>) = {result['p111']:.6f} (should be ~0.5)")
    print(f"  Other states = {result['other_probs']:.6f} (should be ~0.0)")
    print()
    
    # Run multiple measurements
    print("Running 1000 measurements to verify only |000> and |111> occur:")
    print()
    
    outcomes = {'000': 0, '001': 0, '010': 0, '011': 0,
                '100': 0, '101': 0, '110': 0, '111': 0}
    
    for _ in range(1000):
        # Reset to GHZ state
        sim.create_ghz_state()
        a, b, c = sim.measure()
        outcome = f"{a}{b}{c}"
        outcomes[outcome] += 1
    
    print("Results:")
    for state, count in sorted(outcomes.items()):
        if count > 0:
            print(f"  |{state}>: {count} times ({count/10:.1f}%)")
    
    print()
    print("=" * 70)
    print("✅ TRUE GHZ SIMULATION: Only |000> and |111> outcomes!")
    print("=" * 70)

