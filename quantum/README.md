# Quantum-Inspired Classical Systems

This module contains **three fundamentally different quantum-inspired classical engines**:

1. **islands/** - Quantum-Inspired Islands Engine (105–500+ qubit-analogous units)
2. **hierarchical/** - Hierarchical Geometry Machine (5000+ qubit-analogue capacity)
3. **livnium_core/** - Real 1D Physics Solver (DMRG/MPS tensor-network algorithm)

## Important Clarification

⚠️ **These are quantum-inspired classical systems, NOT physical quantum computers.**

They borrow the **language of qubits**, but are not governed by quantum mechanics. They are classical computational systems that use quantum-inspired concepts for information processing, geometric reasoning, and physics simulation.

## Structure

- `islands/` - Information-theoretic quantum-inspired system
- `hierarchical/` - Multi-level geometric architecture
- `livnium_core/` - Tensor network physics solver
- `shared/` - Shared classical utilities

## Usage

```python
from quantum.islands.core.kernel import LivniumQubit
from quantum.hierarchical.geometry.level0.base_geometry import BaseGeometry
from quantum.livnium_core.livnium_core_1D import LivniumCore1D
```

