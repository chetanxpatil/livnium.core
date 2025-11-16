"""
Grover's Search Algorithm Implementation

Implements Grover's algorithm for searching a marked item in an unstructured database.
This is a real quantum computation test for the hierarchical geometry quantum computer.
"""

import numpy as np
from typing import List, Tuple, Optional


class GroversSearch:
    """
    Grover's Search Algorithm implementation.
    
    Searches for a marked item in an unstructured database of N items using
    quantum amplitude amplification.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize Grover's search algorithm.
        
        Args:
            num_qubits: Number of qubits (database size = 2^num_qubits)
        """
        self.num_qubits = num_qubits
        self.database_size = 2 ** num_qubits
        self.state_vector = np.zeros(self.database_size, dtype=complex)
        
    def initialize_uniform_superposition(self):
        """Initialize all qubits in uniform superposition using Hadamard gates."""
        # Uniform superposition: |ψ⟩ = (1/√N) Σ|x⟩
        amplitude = 1.0 / np.sqrt(self.database_size)
        self.state_vector.fill(amplitude)
        
    def create_oracle(self, winner_state: int):
        """
        Create quantum oracle that marks the winner state.
        
        The oracle flips the phase of the winner state:
        |x⟩ → -|x⟩ if x is the winner, |x⟩ otherwise
        
        Args:
            winner_state: The state index to mark (0 to database_size-1)
        """
        # Phase flip oracle: U_ω|x⟩ = -|x⟩ if x=ω, |x⟩ otherwise
        self.state_vector[winner_state] *= -1
        
    def create_diffuser(self):
        """
        Create Grover diffuser (inversion about the mean).
        
        The diffuser reflects the state vector about the mean amplitude.
        This amplifies the amplitude of the marked state.
        """
        # Compute mean amplitude
        mean_amplitude = np.mean(self.state_vector)
        
        # Inversion about the mean: |x⟩ → 2⟨a⟩|x⟩ - |x⟩
        # where ⟨a⟩ is the mean amplitude
        self.state_vector = 2 * mean_amplitude - self.state_vector
        
    def grover_iteration(self, winner_state: int):
        """
        Perform one Grover iteration (Oracle + Diffuser).
        
        Args:
            winner_state: The state index to search for
        """
        # Step 1: Apply oracle (phase flip)
        self.create_oracle(winner_state)
        
        # Step 2: Apply diffuser (inversion about mean)
        self.create_diffuser()
        
    def run_grovers_algorithm(self, winner_state: int, num_iterations: Optional[int] = None) -> float:
        """
        Run complete Grover's algorithm.
        
        Args:
            winner_state: The state index to search for (0 to database_size-1)
            num_iterations: Number of Grover iterations. If None, uses optimal number.
            
        Returns:
            Final probability of measuring the winner state (as percentage)
        """
        # Step 1: Initialize uniform superposition
        self.initialize_uniform_superposition()
        
        # Step 2: Determine optimal number of iterations
        if num_iterations is None:
            # Optimal: π/4 * sqrt(N) where N = database_size
            optimal_iterations = int(np.pi / 4 * np.sqrt(self.database_size))
            num_iterations = optimal_iterations
        
        # Step 3: Apply Grover iterations
        for _ in range(num_iterations):
            self.grover_iteration(winner_state)
        
        # Step 4: Compute probability of winner state
        winner_probability = abs(self.state_vector[winner_state]) ** 2
        
        return winner_probability * 100  # Return as percentage
        
    def get_state_probabilities(self) -> np.ndarray:
        """
        Get probability distribution over all states.
        
        Returns:
            Array of probabilities for each state
        """
        return np.abs(self.state_vector) ** 2
        
    def get_winner_probability(self, winner_state: int) -> float:
        """
        Get probability of measuring the winner state.
        
        Args:
            winner_state: The state index to check
            
        Returns:
            Probability as percentage
        """
        return abs(self.state_vector[winner_state]) ** 2 * 100


def binary_string_to_int(binary_string: str) -> int:
    """
    Convert binary string to integer.
    
    Args:
        binary_string: Binary string (e.g., "1101001011")
        
    Returns:
        Integer value
    """
    return int(binary_string, 2)


def solve_grovers_10_qubit(winner_binary: str = "1101001011") -> float:
    """
    Solve the 10-qubit Grover's search problem.
    
    Args:
        winner_binary: Binary string of the winner state (default: "1101001011")
        
    Returns:
        Final probability (%) of measuring the winner state
    """
    num_qubits = len(winner_binary)
    
    if num_qubits != 10:
        raise ValueError(f"Expected 10 qubits, got {num_qubits}")
    
    # Convert binary string to integer
    winner_state = binary_string_to_int(winner_binary)
    
    print("=" * 70)
    print("Grover's Search Algorithm - 10 Qubit Test")
    print("=" * 70)
    print(f"\nDatabase size: 2^{num_qubits} = {2**num_qubits} states")
    print(f"Winner state: {winner_binary} (decimal: {winner_state})")
    
    # Calculate optimal iterations
    database_size = 2 ** num_qubits
    optimal_iterations = int(np.pi / 4 * np.sqrt(database_size))
    print(f"Optimal Grover iterations: {optimal_iterations}")
    print(f"\nRunning Grover's algorithm...")
    
    # Create Grover's search instance
    grover = GroversSearch(num_qubits)
    
    # Run algorithm
    probability = grover.run_grovers_algorithm(winner_state)
    
    print(f"\n✅ Algorithm complete!")
    print(f"Final probability of measuring winner state: {probability:.6f}%")
    
    # Show some statistics
    probabilities = grover.get_state_probabilities()
    max_prob_idx = np.argmax(probabilities)
    max_prob = probabilities[max_prob_idx] * 100
    
    print(f"\nHighest probability state: {max_prob_idx} ({max_prob:.6f}%)")
    print(f"Winner state probability: {probability:.6f}%")
    
    # Verify the winner state matches
    if max_prob_idx == winner_state:
        print(f"✅ Verified: Highest probability state matches winner!")
    else:
        print(f"⚠️  Warning: Highest probability state ({max_prob_idx}) != winner ({winner_state})")
    
    return probability


if __name__ == "__main__":
    # Run the test
    result = solve_grovers_10_qubit("1101001011")
    print(f"\n{'='*70}")
    print(f"FINAL ANSWER: {result:.6f}%")
    print(f"{'='*70}")

