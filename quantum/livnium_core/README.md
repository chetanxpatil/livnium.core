# Livnium Core: 1D TFIM Tensor-Network Solver

## Overview

Livnium Core is a **real physics solver** using DMRG (Density Matrix Renormalization Group) / MPS (Matrix Product States) tensor-network algorithms to solve the 1D Transverse Field Ising Model (TFIM).

## What This Is

This is a **legitimate tensor network implementation** used in quantum many-body physics. Unlike the quantum-inspired systems in `islands/` and `hierarchical/`, this module performs actual physics simulation using tensor network methods.

## Key Features

- DMRG/MPS tensor network implementation
- 1D TFIM ground state optimization
- Bond dimension control
- Energy minimization

## Usage

```python
from quantum.livnium_core.livnium_core_1D import LivniumCore1D

solver = LivniumCore1D(n_qubits=20, J=1.0, g=1.0)
energy = solver.optimize_ground_state(n_iterations=50)
```

## Documentation

See `docs/QUICKSTART.md` for detailed usage instructions.
