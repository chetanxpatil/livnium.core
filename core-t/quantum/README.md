# Quantum Layer for Livnium-T

The bridge between geometric Livnium-T and tensor-product quantum mechanics.

## What is the Quantum Layer?

**The quantum field for the tetrahedral universe.**

This layer provides:
- **Real physics**: Born rule, unitaries, tensor products, collapse
- **Livnium-T physics**: Exposure-based coupling, symbolic weight modulation, simplex entanglement
- **The bridge**: Between your 5-node geometric universe and the tensor-product quantum universe

## Contents

- **`quantum_node.py`**: Individual quantum node with complex amplitudes (adapted for node IDs 0-4)
- **`quantum_gates.py`**: Standard quantum gates (H, X, Y, Z, CNOT, etc.)
- **`quantum_system.py`**: Full integration with Livnium-T geometry + optional entanglement
- **`entanglement_manager.py`**: Manages entanglement between nodes (5-node topology)
- **`measurement_engine.py`**: Real Born rule + collapse (stable, physical core)
- **`geometry_quantum_coupling.py`**: Connects geometry to quantum (exposure f, SW, node class)

## Key Differences from Livnium Core Quantum Layer

| Feature | Livnium Core | Livnium-T |
|---------|--------------|-----------|
| **Topology** | 3×3×3 lattice (27 cells) | 5-node simplex (1 core + 4 vertices) |
| **Coordinates** | (x, y, z) tuples | Node IDs (0-4) |
| **Geometry Coupling** | Face exposure (f ∈ {0,1,2,3}) | Exposure (f ∈ {0,3}) |
| **SW Range** | 0-27 | 0-27 (same formula) |
| **Entanglement** | Between lattice cells | Between nodes |

## Usage

```python
from classical import LivniumTSystem
from quantum import QuantumSystem, GateType, MeasurementBasis

# Create systems
t_system = LivniumTSystem()
q_system = QuantumSystem(t_system)

# Apply quantum gates
q_system.apply_gate(1, GateType.HADAMARD)  # Apply H to node 1

# Entangle nodes
q_system.entangle_nodes(1, 2, 'phi_plus')  # Create Bell pair

# Measure
result = q_system.measure_node(1, MeasurementBasis.COMPUTATIONAL)
print(f"Measured: {result.measured_level}, P={result.probability}")
```

## Geometry-Quantum Coupling Rules

1. **Core (f=0, SW=0)**: Ground state |0⟩
2. **Vertex (f=3, SW=27)**: Superposition |+⟩ = (|0⟩ + |1⟩)/√2
3. **Exposure → Entanglement**: f=3 → maximum entanglement strength
4. **SW → Amplitude**: Higher SW → stronger amplitudes

## Status

✅ **All components implemented and tested**
- Quantum nodes working
- Gates working
- Entanglement working
- Measurement working
- Geometry coupling working

## Future Directions

1. **Simplex Entanglement Patterns**: Use tetrahedral structure to guide entanglement topology
2. **Rotation-Quantum Coupling**: Link A₄ rotations to quantum phase
3. **Base-5 Quantum Encoding**: Use base-5 for quantum state representation
4. **Multi-Node Entanglement**: Entangle all 4 vertices simultaneously

