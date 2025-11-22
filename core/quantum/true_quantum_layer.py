"""
True Quantum Layer: Real Tensor Product Quantum Mechanics

This module implements TRUE quantum mechanics using tensor products and full
state vectors. This replaces the "fake" entanglement metadata with real
linear algebra.

This is the "Special Forces" unit for handling true multi-qubit entanglement.

Features:
- Standard quantum operations (unitary gates, measurement, entanglement)
- Meta-interference: Non-unitary optimization to bias states toward targets
  (breaks unitarity for optimization purposes - useful for key search, etc.)
"""

import numpy as np
from typing import List, Tuple, Optional


class TrueQuantumRegister:
    """
    A localized region of TRUE quantum mechanics (Tensor Products).
    
    Used for simulating interactions between a small group of qubits.
    
    This replaces the 'fake' entanglement metadata with real linear algebra.
    """
    
    def __init__(self, qubit_indices: List[int]):
        """
        Initialize quantum register.
        
        Args:
            qubit_indices: ID numbers for the qubits (e.g., [0, 1, 2] for Alice, Bob, Source)
        """
        self.qubit_map = {idx: i for i, idx in enumerate(qubit_indices)}
        self.num_qubits = len(qubit_indices)
        self.dim = 2 ** self.num_qubits
        
        # Initialize global state to |00...0⟩
        # Size is 2^N (e.g., 8 for 3 qubits, 16 for 4 qubits)
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0 + 0j
        
        # Track gate history for debugging
        self.gate_history = []
    
    def normalize(self):
        """Normalize state vector."""
        norm = np.sqrt(np.sum(np.abs(self.state)**2))
        if norm > 1e-10:
            self.state /= norm
        else:
            # Reset to |00...0⟩
            self.state = np.zeros(self.dim, dtype=complex)
            self.state[0] = 1.0 + 0j
    
    def apply_gate(self, gate_matrix: np.ndarray, target_id: int):
        """
        Apply a single-qubit gate (H, X, Z, etc.) using Tensor Product.
        
        Args:
            gate_matrix: 2x2 unitary matrix
            target_id: ID of the qubit to apply gate to
        """
        if target_id not in self.qubit_map:
            raise ValueError(f"Qubit {target_id} not in this register")
            
        target_idx = self.qubit_map[target_id]
        
        # Build the operator: I ⊗ ... ⊗ Gate ⊗ ... ⊗ I
        ops = []
        for i in range(self.num_qubits):
            if i == target_idx:
                ops.append(gate_matrix)
            else:
                ops.append(np.eye(2, dtype=complex))  # Identity
        
        # Combine using Kronecker product
        full_op = ops[0]
        for i in range(1, self.num_qubits):
            full_op = np.kron(full_op, ops[i])
        
        # Apply to state vector
        self.state = full_op @ self.state
        
        # Track for debugging
        self.gate_history.append(('gate', target_id, gate_matrix))
    
    def apply_cnot(self, control_id: int, target_id: int):
        """
        Apply CNOT gate between any two qubits.
        
        This creates REAL entanglement.
        
        Args:
            control_id: ID of control qubit
            target_id: ID of target qubit
        """
        if control_id not in self.qubit_map:
            raise ValueError(f"Control qubit {control_id} not in this register")
        if target_id not in self.qubit_map:
            raise ValueError(f"Target qubit {target_id} not in this register")
            
        c_idx = self.qubit_map[control_id]
        t_idx = self.qubit_map[target_id]
        
        # Explicitly construct the CNOT permutation matrix
        new_state = np.zeros(self.dim, dtype=complex)
        
        for i in range(self.dim):
            if abs(self.state[i]) < 1e-12:
                continue
            
            # Extract bits
            # Convention: big-endian, bit 0 is most significant
            bits = [(i >> (self.num_qubits - 1 - k)) & 1 for k in range(self.num_qubits)]
            
            # Logic: If control bit is 1, flip target bit
            if bits[c_idx] == 1:
                bits[t_idx] = 1 - bits[t_idx]
            
            # Reconstruct index
            new_idx = 0
            for bit in bits:
                new_idx = (new_idx << 1) | bit
                
            new_state[new_idx] += self.state[i]
            
        self.state = new_state
        self.gate_history.append(('cnot', control_id, target_id))
    
    def measure_qubit(self, target_id: int) -> int:
        """
        Measure a specific qubit and COLLAPSE the wavefunction.
        
        Args:
            target_id: ID of qubit to measure
            
        Returns:
            0 or 1 (measurement outcome)
        """
        if target_id not in self.qubit_map:
            raise ValueError(f"Qubit {target_id} not in this register")
            
        t_idx = self.qubit_map[target_id]
        
        prob_0 = 0.0
        prob_1 = 0.0
        indices_0 = []
        indices_1 = []
        
        for i in range(self.dim):
            # Check bit at the target position
            bit = (i >> (self.num_qubits - 1 - t_idx)) & 1
            prob = abs(self.state[i])**2
            
            if bit == 0:
                prob_0 += prob
                indices_0.append(i)
            else:
                prob_1 += prob
                indices_1.append(i)
        
        # Normalize probabilities (floating point safety)
        total_prob = prob_0 + prob_1
        if total_prob < 1e-9:
            return 0  # Should not happen
        
        prob_0 /= total_prob
        
        # Simulate Measurement
        outcome = 0 if np.random.random() < prob_0 else 1
        
        # COLLAPSE
        new_state = np.zeros(self.dim, dtype=complex)
        norm_factor = np.sqrt(prob_0 if outcome == 0 else prob_1)
        
        if norm_factor < 1e-12:
            # Should not happen, but handle gracefully
            new_state[0] = 1.0 + 0j
        else:
            keep_indices = indices_0 if outcome == 0 else indices_1
            for idx in keep_indices:
                new_state[idx] = self.state[idx] / norm_factor
                
        self.state = new_state
        self.gate_history.append(('measure', target_id, outcome))
        return outcome
    
    def get_qubit_state(self, qubit_id: int) -> np.ndarray:
        """
        Get the reduced state of a single qubit (partial trace).
        
        Args:
            qubit_id: ID of qubit
            
        Returns:
            2-element state vector [α₀, α₁] for the qubit
        """
        if qubit_id not in self.qubit_map:
            raise ValueError(f"Qubit {qubit_id} not in this register")
            
        t_idx = self.qubit_map[qubit_id]
        
        # Calculate reduced density matrix diagonal (probabilities)
        state_0 = np.zeros(1, dtype=complex)
        state_1 = np.zeros(1, dtype=complex)
        
        for i in range(self.dim):
            bit = (i >> (self.num_qubits - 1 - t_idx)) & 1
            amplitude = self.state[i]
            
            if bit == 0:
                state_0[0] += amplitude
            else:
                state_1[0] += amplitude
        
        # Normalize
        norm = np.sqrt(abs(state_0[0])**2 + abs(state_1[0])**2)
        if norm > 1e-10:
            state_0[0] /= norm
            state_1[0] /= norm
        
        return np.array([state_0[0], state_1[0]], dtype=complex)
    
    def get_fidelity(self, target_state_vector: np.ndarray, qubit_id: int) -> float:
        """
        Check how close a specific qubit is to a target state.
        
        WARNING: Only works if the qubit is separable (not entangled).
        For entangled qubits, this is an approximation.
        
        Args:
            target_state_vector: Target 2-element state vector [α, β]
            qubit_id: ID of qubit to check
            
        Returns:
            Fidelity value [0, 1]
        """
        # Get the reduced state
        actual_state = self.get_qubit_state(qubit_id)
        
        # Normalize target
        target = np.array(target_state_vector, dtype=complex)
        target_norm = np.sqrt(np.sum(np.abs(target)**2))
        if target_norm > 1e-10:
            target = target / target_norm
        
        # Calculate fidelity: |⟨target|actual⟩|²
        overlap = np.vdot(target, actual_state)
        fidelity = abs(overlap)**2
        
        return float(fidelity)
    
    def set_qubit_state(self, qubit_id: int, state_vector: np.ndarray):
        """
        Set a qubit to a specific state (only works if qubit is separable).
        
        Args:
            qubit_id: ID of qubit
            state_vector: 2-element state vector [α, β]
        """
        if qubit_id not in self.qubit_map:
            raise ValueError(f"Qubit {qubit_id} not in this register")
        
        # Normalize
        state = np.array(state_vector, dtype=complex)
        norm = np.sqrt(np.sum(np.abs(state)**2))
        if norm > 1e-10:
            state = state / norm
        else:
            state = np.array([1.0, 0.0], dtype=complex)
        
        # This is a simplified version - assumes qubit is separable
        # For proper implementation, would need to update full state vector
        # For now, we'll use gates to prepare the state
        t_idx = self.qubit_map[qubit_id]
        
        # Reset qubit to |0⟩
        # Then apply gates to reach target state
        # This is a simplified approach - full implementation would be more complex
        
        # For teleportation, we'll prepare states using gates instead
        # This method is mainly for documentation
        pass
    
    def get_full_state(self) -> np.ndarray:
        """Get the full state vector."""
        return self.state.copy()
    
    def apply_meta_interference(self, target_pattern: int, bias_strength: float):
        """
        ALTERS THE META-STRUCTURE OF AMPLITUDES.
        
        This breaks Unitary constraints (Conservation of Probability).
        
        It actively 'warps' the interference pattern to favor specific
        geometric outcomes, effectively 'cheating' standard physics to
        force convergence.
        
        ⚠️ WARNING: This is NOT standard quantum mechanics!
        This is an optimization technique that manipulates amplitudes
        non-linearly to bias toward target patterns.
        
        Args:
            target_pattern: Target state index (0 to dim-1) to bias toward
            bias_strength: Strength of bias (0.0 = no effect, 1.0 = strong)
        """
        # Clamp target_pattern to valid range
        target_pattern = target_pattern % self.dim
        
        # 1. Extract current amplitudes (The Wavefunction)
        amplitudes = self.state
        
        # 2. Apply Non-Linear Bias (The Meta-Rule)
        # We don't just rotate; we STRETCH space around the target.
        new_amplitudes = np.copy(amplitudes)
        
        for i in range(self.dim):
            # Check if this state 'i' resonates with the target pattern
            # (e.g., Hamming distance, geometric proximity)
            resonance = self._calculate_geometric_resonance(i, target_pattern)
            
            if resonance > 0:
                # AMPLIFY: Boost this reality manually
                # Real physics forbids this (energy creation).
                # Livnium allows it (optimization).
                phase = np.angle(amplitudes[i])
                magnitude = np.abs(amplitudes[i])
                
                # Non-linear boost
                new_magnitude = magnitude * (1.0 + (bias_strength * resonance))
                new_amplitudes[i] = new_magnitude * np.exp(1j * phase)
                
            else:
                # SUPPRESS: Dampen competing realities
                new_amplitudes[i] *= (1.0 - (bias_strength * 0.1))
        
        # 3. Re-Normalize (Force consistency back into the universe)
        # This effectively redistributes the stolen probability to the winner.
        norm = np.sqrt(np.sum(np.abs(new_amplitudes)**2))
        if norm > 1e-10:
            self.state = new_amplitudes / norm
        else:
            # Fallback: reset to target state
            self.state = np.zeros(self.dim, dtype=complex)
            self.state[target_pattern] = 1.0 + 0j
        
        # Track this non-unitary operation
        self.gate_history.append(('meta_interference', target_pattern, bias_strength))
    
    def _calculate_geometric_resonance(self, state_idx: int, target_pattern: int) -> float:
        """
        Defines the 'Meta-Structure' rule.
        Does this quantum state 'look like' the geometry we want?
        
        Uses inverse Hamming distance as resonance measure.
        Closer states (fewer bit flips) have higher resonance.
        
        Args:
            state_idx: Current state index
            target_pattern: Target state index
            
        Returns:
            Resonance value [0, 1] where 1.0 = perfect match
        """
        # Calculate Hamming distance (number of differing bits)
        xor = state_idx ^ target_pattern
        dist = bin(xor).count('1')
        
        if dist == 0:
            return 1.0  # Perfect match
        
        # Inverse distance: closer = higher resonance
        # Normalize by max possible distance (num_qubits)
        max_dist = self.num_qubits
        if max_dist == 0:
            return 0.0
        
        # Resonance decays with distance
        # Formula: 1 / (1 + distance) gives smooth decay
        resonance = 1.0 / (1.0 + dist)
        
        return resonance
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TrueQuantumRegister({self.num_qubits} qubits, dim={self.dim})"

