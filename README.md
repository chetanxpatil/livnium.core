# LIVNIUM - Quantum-Inspired Classical Systems

A collection of quantum-inspired classical computational systems and real tensor network physics solvers.

## Overview

LIVNIUM contains **three fundamentally different computational systems**:

### 1. **Islands System** (`quantum/islands/`)
Quantum-inspired information processing engine
- 105–500+ qubit-analogous units
- Information-theoretic quantum-inspired system
- Geometric cube structure (3×3×3)
- Use cases: Feature representation, semantic reasoning, classification

### 2. **Hierarchical System** (`quantum/hierarchical/`)
Hierarchical geometry machine with 3-level architecture
- 5000+ qubit-analogue capacity
- Geometry-in-geometry architecture (Level 0 → Level 1 → Level 2)
- Linear memory scaling (~400 bytes per qubit analogue)
- Use cases: Large-scale geometric reasoning, hierarchical state management

### 3. **Livnium Core** (`quantum/livnium_core/`)
Real tensor network physics solver
- DMRG (Density Matrix Renormalization Group) / MPS (Matrix Product States)
- 1D Transverse Field Ising Model (TFIM) ground state solver
- Legitimate quantum many-body physics method
- Use cases: Physics research, ground state finding, tensor network studies

## Important Clarification

⚠️ **These are quantum-inspired classical systems (islands & hierarchical), NOT physical quantum computers.**

- **`islands/`** and **`hierarchical/`**: Quantum-inspired classical systems
  - Use quantum language and concepts
  - Operate on classical hardware
  - Do NOT perform actual quantum computation

- **`livnium_core/`**: Real physics solver
  - Uses legitimate tensor network methods (MPS/DMRG)
  - Solves actual physics problems
  - Part of quantum many-body physics research

## Quick Start

### Islands System
```python
from quantum.islands.core.kernel import LivniumQubit

# Create qubit-analogue
qubit = LivniumQubit()
```

### Hierarchical System
```python
from quantum.hierarchical.core.quantum_processor import QuantumProcessor

# Create processor
processor = QuantumProcessor(base_dimension=3)
qubit_id = processor.create_qubit((0.1, 0.2, 0.3))
```

### Livnium Core (Real Physics)
```python
from quantum.livnium_core.livnium_core_1D import LivniumCore1D

# Solve 1D TFIM ground state
solver = LivniumCore1D(n_qubits=20, J=1.0, g=1.0)
energy = solver.optimize_ground_state(n_iterations=50)
```

## Project Structure

```
quantum/
├── islands/          # Quantum-inspired information processing
├── hierarchical/     # Geometry-in-geometry architecture
├── livnium_core/     # Real MPS/DMRG tensor networks
└── shared/           # Shared utilities

docs/                 # Comprehensive documentation
archive/              # Archived experimental/broken components
```

## Documentation

- **Architecture Overview**: `docs/project/ARCHITECTURE_OVERVIEW.md`
- **System Explanation**: `docs/project/QUANTUM_SYSTEM_EXPLANATION.md`
- **Roadmap**: `docs/project/ROADMAP.md`
- **Module READMEs**: Each system has its own README in its directory

## Requirements

- Python 3.7+
- numpy

## Key Principles

- **Clear separation**: Each system maintains its own namespace
- **No quantum confusion**: All documentation clearly states "NOT a quantum computer" (for islands/hierarchical)
- **Professional structure**: Follows best practices for research software
- **Architectural purity**: Real physics (livnium_core) separate from quantum-inspired systems

## License

[Add your license here]

---

*LIVNIUM: Quantum-inspired classical systems and real tensor network physics*
