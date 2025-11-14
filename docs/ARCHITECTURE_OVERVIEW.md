# LIVNIUM Architecture Overview

## Three Quantum-Inspired Classical Engines

This project contains three fundamentally different computational systems:

### 1. Quantum-Inspired Islands Engine (`quantum/islands/`)

**Type**: Information-theoretic quantum-inspired system

**Capacity**: 105–500+ qubit-analogous units

**Purpose**: Classical computation using quantum-inspired concepts (qubit analogues, entanglement-like correlations)

**Key Characteristics**:
- Independent qubit analogues (linear O(n) scaling)
- Pairwise entanglement-like correlations
- Geometric cube structure (3×3×3)
- Feature integration and classification

**Use Cases**: Feature representation, semantic reasoning, classification tasks

### 2. Hierarchical Geometry Machine (`quantum/hierarchical/`)

**Type**: Multi-level geometric architecture

**Capacity**: 5000+ qubit-analogue capacity

**Purpose**: Classical computation using hierarchical geometric structures

**Key Characteristics**:
- 3-level "geometry-in-geometry" architecture
- Linear memory scaling (~400 bytes per qubit analogue)
- Geometric state representation
- Meta-geometric operations

**Use Cases**: Large-scale geometric reasoning, hierarchical state management

### 3. Livnium Core Physics Solver (`quantum/livnium_core/`)

**Type**: Real tensor network physics solver

**Purpose**: Actual physics simulation using DMRG/MPS methods

**Key Characteristics**:
- DMRG (Density Matrix Renormalization Group) implementation
- MPS (Matrix Product States) tensor networks
- 1D TFIM ground state optimization
- Legitimate quantum many-body physics method

**Use Cases**: Physics research, ground state finding, tensor network studies

## Important Distinctions

### Quantum-Inspired vs. Real Quantum

- **islands/** and **hierarchical/**: Quantum-inspired classical systems
  - Use quantum language and concepts
  - Operate on classical hardware
  - Do NOT perform actual quantum computation

- **livnium_core/**: Real physics solver
  - Uses legitimate tensor network methods
  - Solves actual physics problems
  - Part of quantum many-body physics research

## System Comparison

| System | Type | Capacity | Memory Scaling | Purpose |
|--------|------|----------|----------------|---------|
| islands | Quantum-inspired | 105–500+ | Linear O(n) | Information processing |
| hierarchical | Quantum-inspired | 5000+ | Linear O(n) | Geometric reasoning |
| livnium_core | Physics solver | Variable | O(χ² × n) | Physics simulation |

## Future Extensions

The architecture is designed to allow easy addition of new quantum-inspired systems:

- `livnium_vector/` - Vector-based quantum-inspired system
- `livnium_field/` - Field-based quantum-inspired system
- `livnium_wave/` - Wave-based quantum-inspired system

Each new system follows the same structure pattern: core/, simulators/, tests/, docs/

