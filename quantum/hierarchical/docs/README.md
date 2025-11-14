# Quantum Algorithms

Implementation of quantum algorithms for the hierarchical geometry quantum computer.

## Grover's Search Algorithm

### Overview

Grover's algorithm is a quantum search algorithm that can find a marked item in an unstructured database of N items using only O(√N) queries, compared to O(N) for classical search.

### Implementation

**File**: `grovers_search.py`

**Key Features:**
- Full quantum state vector simulation
- Oracle that marks the winner state (phase flip)
- Diffuser (inversion about the mean)
- Optimal iteration calculation
- Probability measurement

### Usage

```python
from quantum_computer.algorithms.grovers_search import solve_grovers_10_qubit

# Solve the 10-qubit Grover's search problem
probability = solve_grovers_10_qubit("1101001011")
print(f"Probability: {probability}%")
```

### Test Results: 10-Qubit Grover's Search

**Problem:**
- Database size: 2^10 = 1024 states
- Winner state: `1101001011` (decimal: 843)
- Optimal iterations: 25 (π/4 * √1024)

**Result:**
- **Final probability: 99.946124%**
- ✅ Verified: Highest probability state matches winner
- ✅ Algorithm successfully amplifies winner state from ~0.031% to ~99.95%

### Algorithm Steps

1. **Initialize uniform superposition**: Apply Hadamard gates to all qubits
   - Creates state: |ψ⟩ = (1/√N) Σ|x⟩

2. **Grover iteration** (repeated optimal number of times):
   - **Oracle**: Phase flip the winner state
   - **Diffuser**: Inversion about the mean amplitude

3. **Measurement**: Measure the final state

### Mathematical Details

- **Optimal iterations**: ⌊π/4 * √N⌋ where N = 2^n (n = number of qubits)
- **Oracle**: U_ω|x⟩ = -|x⟩ if x=ω, |x⟩ otherwise
- **Diffuser**: Reflects state vector about mean amplitude
- **Final probability**: After optimal iterations, probability ≈ 1

### Verification

The implementation correctly:
- ✅ Creates uniform superposition
- ✅ Marks the winner state with phase flip
- ✅ Applies inversion about mean
- ✅ Uses optimal number of iterations
- ✅ Achieves ~99.95% probability for winner state

This demonstrates that the hierarchical geometry quantum computer can handle real quantum algorithms!

