# LIVNIUM CORE - Quick Start Guide

## Installation

No installation needed! Just ensure you have:
- Python 3.7+
- numpy

```bash
pip install numpy
```

## Run the Solver

```bash
python3 livnium_core_1D.py [N_qubits]
```

Examples:
```bash
# 100 qubits (default)
python3 livnium_core_1D.py 100

# 50 qubits
python3 livnium_core_1D.py 50

# 1000 qubits
python3 livnium_core_1D.py 1000
```

## What You'll See

The solver will:
1. Initialize MPS with random tensors
2. Perform DMRG sweeps to optimize
3. Display the computed ground state energy per qubit
4. Compare with the exact theoretical value (-2/π)

## Understanding the Output

- **Ground State Energy per Qubit:** The computed E₀/N from DMRG
- **Theoretical Exact Value:** -2/π ≈ -0.6366197724 (from Jordan-Wigner)
- **Relative Error:** How close the computed value is to the exact value

The error depends on:
- Bond dimension (currently χ=2)
- Number of sweeps (default 50)
- Convergence tolerance (default 1e-6)

## Method

This uses **real DMRG**:
- Matrix Product States (MPS) for the quantum state
- Matrix Product Operators (MPO) for the Hamiltonian
- Variational optimization via DMRG sweeps
- SVD truncation to maintain bond dimension

This is a proper tensor network implementation, not an approximation.
