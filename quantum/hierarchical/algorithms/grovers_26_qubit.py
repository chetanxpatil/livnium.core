"""
26-Qubit Grover's Search - REAL Simulation

This is a REAL quantum simulation that handles the full 2^26 state vector.
No shortcuts, no formulas - actual state vector manipulation.
"""

import numpy as np
from typing import Optional
import time


def binary_string_to_int(binary_string: str) -> int:
    """Convert binary string to integer."""
    return int(binary_string, 2)


def int_to_binary_string(value: int, num_bits: int) -> str:
    """Convert integer to binary string with padding."""
    return format(value, f'0{num_bits}b')


class RealGroversSearch:
    """
    REAL Grover's Search implementation with full state vector simulation.
    
    This actually simulates 2^n states - no shortcuts!
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize Grover's search with REAL state vector.
        
        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.database_size = 2 ** num_qubits
        
        # REAL state vector - this is the actual simulation
        print(f"Allocating state vector: {self.database_size:,} states")
        print(f"Memory: ~{self.database_size * 16 / (1024**3):.2f} GB")
        print("Allocating memory...", end=" ", flush=True)
        
        self.state_vector = np.zeros(self.database_size, dtype=np.complex128)
        print("✅ Done!")
        
    def initialize_uniform_superposition(self):
        """Initialize REAL uniform superposition."""
        print("Initializing uniform superposition...")
        amplitude = 1.0 / np.sqrt(self.database_size)
        self.state_vector.fill(amplitude)
        print(f"  ✅ All {self.database_size:,} states initialized")
        
    def create_oracle(self, winner_state: int):
        """
        REAL oracle - phase flip the winner state.
        
        Args:
            winner_state: State index to mark
        """
        # Phase flip: multiply winner state amplitude by -1
        self.state_vector[winner_state] *= -1
        
    def create_diffuser(self):
        """
        REAL diffuser - inversion about the mean.
        
        This actually computes the mean and reflects all amplitudes.
        """
        # Compute REAL mean amplitude
        mean_amplitude = np.mean(self.state_vector)
        
        # Inversion about the mean: |x⟩ → 2⟨a⟩|x⟩ - |x⟩
        # This is applied to ALL states in the vector
        self.state_vector = 2 * mean_amplitude - self.state_vector
        
    def grover_iteration(self, winner_state: int):
        """
        Perform one REAL Grover iteration.
        
        Args:
            winner_state: State index to search for
        """
        # Step 1: Apply oracle (phase flip winner)
        self.create_oracle(winner_state)
        
        # Step 2: Apply diffuser (inversion about mean)
        self.create_diffuser()
        
    def run_grovers_algorithm(self, winner_state: int, num_iterations: Optional[int] = None) -> float:
        """
        Run REAL Grover's algorithm with full state vector.
        
        Args:
            winner_state: State index to search for
            num_iterations: Number of iterations (None = optimal)
            
        Returns:
            Final probability (%) of measuring winner state
        """
        print("\n" + "=" * 70)
        print("Running REAL Grover's Algorithm")
        print("=" * 70)
        
        # Step 1: Initialize uniform superposition
        print("\nStep 1: Initializing uniform superposition...")
        start_time = time.time()
        self.initialize_uniform_superposition()
        init_time = time.time() - start_time
        print(f"  Time: {init_time:.3f} seconds")
        
        # Verify initialization
        initial_prob = abs(self.state_vector[winner_state]) ** 2 * 100
        print(f"  Initial probability of winner: {initial_prob:.10f}%")
        
        # Step 2: Determine optimal iterations
        if num_iterations is None:
            optimal_iterations = int(np.pi / 4 * np.sqrt(self.database_size))
            num_iterations = optimal_iterations
        
        print(f"\nStep 2: Running {num_iterations} Grover iterations...")
        print(f"  Optimal iterations: {num_iterations}")
        
        # Step 3: Apply REAL Grover iterations
        iteration_times = []
        for i in range(num_iterations):
            iter_start = time.time()
            self.grover_iteration(winner_state)
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            
            if (i + 1) % max(1, num_iterations // 10) == 0:
                current_prob = abs(self.state_vector[winner_state]) ** 2 * 100
                print(f"  Iteration {i+1}/{num_iterations}: probability = {current_prob:.6f}%")
        
        avg_iter_time = np.mean(iteration_times)
        total_iter_time = sum(iteration_times)
        print(f"\n  ✅ All iterations complete")
        print(f"  Average time per iteration: {avg_iter_time:.4f} seconds")
        print(f"  Total iteration time: {total_iter_time:.3f} seconds")
        
        # Step 4: Compute final probability
        winner_probability = abs(self.state_vector[winner_state]) ** 2 * 100
        
        # Verify normalization
        total_prob = np.sum(np.abs(self.state_vector) ** 2)
        print(f"\nStep 3: Verification")
        print(f"  Total probability (should be ~1.0): {total_prob:.10f}")
        print(f"  Winner state probability: {winner_probability:.10f}%")
        
        return winner_probability
        
    def get_state_probabilities(self) -> np.ndarray:
        """Get probability distribution over all states."""
        return np.abs(self.state_vector) ** 2


def solve_grovers_26_qubit(winner_binary: str = "10101010101010101010101010") -> float:
    """
    Solve 26-qubit Grover's search with REAL simulation.
    
    Args:
        winner_binary: Binary string of winner state
        
    Returns:
        Final probability (%) of measuring winner state
    """
    num_qubits = len(winner_binary)
    
    if num_qubits != 26:
        raise ValueError(f"Expected 26 qubits, got {num_qubits}")
    
    # Convert binary string to integer
    winner_state = binary_string_to_int(winner_binary)
    
    print("=" * 70)
    print("26-Qubit Grover's Search - REAL Simulation")
    print("=" * 70)
    print(f"\nDatabase size: 2^{num_qubits} = {2**num_qubits:,} states")
    print(f"Winner state: {winner_binary} (decimal: {winner_state:,})")
    
    # Create REAL Grover's search instance
    grover = RealGroversSearch(num_qubits)
    
    # Run REAL algorithm
    start_total = time.time()
    probability = grover.run_grovers_algorithm(winner_state)
    total_time = time.time() - start_total
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Winner state: {winner_binary}")
    print(f"Final probability: {probability:.10f}%")
    print(f"Total computation time: {total_time:.3f} seconds")
    print("=" * 70)
    
    return probability


if __name__ == "__main__":
    # Run the REAL 26-qubit Grover's search
    result = solve_grovers_26_qubit("10101010101010101010101010")
    
    print(f"\n📊 FINAL ANSWER: {result:.10f}%")

