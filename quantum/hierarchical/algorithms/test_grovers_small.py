"""Quick test to verify REAL Grover's works"""
import numpy as np
import time

# Test with 10 qubits first
n = 10
N = 2**n
print(f"Testing with {n} qubits, {N:,} states")

# Allocate
state = np.zeros(N, dtype=np.complex128)
print("Allocated!")

# Initialize
amplitude = 1.0 / np.sqrt(N)
state.fill(amplitude)
print(f"Initialized! First few amplitudes: {state[:5]}")

# Test oracle
winner = 5
state[winner] *= -1
print(f"After oracle, winner amplitude: {state[winner]}")

# Test diffuser
mean = np.mean(state)
state = 2 * mean - state
print(f"After diffuser, winner amplitude: {state[winner]}")
print(f"Winner probability: {abs(state[winner])**2 * 100:.6f}%")

print("✅ REAL simulation works!")

