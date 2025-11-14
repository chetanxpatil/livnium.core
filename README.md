# LIVNIUM CORE - 1D Quantum Ground State Solver

A real tensor network implementation using DMRG (Density Matrix Renormalization Group) with Matrix Product States (MPS) to solve the 1D Transverse-Field Ising Model.

## Quick Start

```bash
cd livnium_core_demo
python3 livnium_core_1D.py 100
```

## What It Does

Solves the 1D TFIM ground state:
- **Hamiltonian:** H = -J Σ σᵢᶻ σᵢ₊₁ᶻ - gJ Σ σᵢˣ
- **Critical point:** g=1, J=1
- **Exact solution:** E₀/N = -2/π ≈ -0.6366197724

## Method

Uses **real DMRG/MPS tensor network methods**:
- Matrix Product State (MPS) representation
- Matrix Product Operator (MPO) for Hamiltonian
- DMRG sweeps with SVD truncation
- Proper tensor network contractions

## Requirements

- Python 3.7+
- numpy

## Project Structure

```
livnium_core_demo/
  ├── livnium_core_1D.py    # Main DMRG solver
  ├── README.md             # Documentation
  └── QUICKSTART.md         # Quick start guide
```

