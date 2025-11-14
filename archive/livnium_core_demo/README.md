# LIVNIUM CORE - 1D Quantum Ground State Solver

A real tensor network implementation using **DMRG (Density Matrix Renormalization Group)** with **Matrix Product States (MPS)** to solve the 1D Transverse-Field Ising Model.

## Quick Start

```bash
python3 livnium_core_1D.py 100
```

## What Problem Does This Solve?

The **1D Transverse-Field Ising Model (TFIM)** is a fundamental quantum many-body problem:

**Hamiltonian:** H = -J Σ σᵢᶻ σᵢ₊₁ᶻ - gJ Σ σᵢˣ

At the quantum critical point (g=1, J=1), the exact ground state energy per site is:

**E₀/N = -2/π ≈ -0.6366197724**

This is a **theoretically known exact result** from Jordan-Wigner transformation.

## How It Works

This solver uses **real tensor network methods**:

- **Matrix Product State (MPS)** representation with bond dimension χ=2
- **Matrix Product Operator (MPO)** for the TFIM Hamiltonian
- **DMRG sweeps** with SVD truncation to optimize the ground state
- **Proper tensor network contractions** to compute energy expectation values

This is a **real DMRG implementation**, not an approximation or ad-hoc formula.

## Requirements

- Python 3.7+
- numpy

## Method Details

The solver:
1. Creates an MPO representation of the TFIM Hamiltonian
2. Initializes a random MPS with bond dimension χ=2
3. Performs DMRG sweeps (left-to-right and right-to-left)
4. Uses SVD truncation to maintain bond dimension
5. Computes energy from MPS/MPO tensor contractions
6. Converges to the ground state energy

## Accuracy

The accuracy depends on:
- Bond dimension (χ=2 is used here)
- Number of DMRG sweeps
- Convergence tolerance

For the critical point, the exact solution is E₀/N = -2/π. The DMRG method should converge close to this value.

## References

- Density Matrix Renormalization Group (DMRG) algorithm
- Matrix Product States (MPS) and Matrix Product Operators (MPO)
- 1D TFIM exact solution via Jordan-Wigner transformation
