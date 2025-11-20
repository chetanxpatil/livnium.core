# Livnium Core System — Fully Generalized N×N×N

Complete implementation of the **Livnium Core System specification** with feature switches.

**Fully Generalized:** Works for any odd N ≥ 3 (not just 3×3×3)

## Folder Structure

```
core/
├── __init__.py              # Main package exports
├── config.py                # Configuration with feature switches
├── README.md                # This file
├── ARCHITECTURE.md          # Complete 8-layer architecture
├── CORE_STRUCTURE.md        # Layer-by-layer structure guide
├── QUANTUM_LAYER.md         # Quantum layer documentation
├── LAYER_0.md               # Recursive geometry engine
├── MOKSHA.md                # Fixed-point convergence
│
├── classical/               # Classical geometric system
│   ├── __init__.py
│   └── livnium_core_system.py
│
├── quantum/                 # Quantum layer (optional)
│   ├── __init__.py
│   ├── quantum_cell.py
│   ├── quantum_gates.py
│   ├── quantum_lattice.py
│   ├── entanglement_manager.py
│   ├── measurement_engine.py
│   └── geometry_quantum_coupling.py
│
└── tests/                   # Test suite
    ├── __init__.py
    ├── test_livnium_core.py
    ├── test_generalized_n.py
    └── test_quantum.py
```

## Overview

This is the **full implementation** of the Livnium Core System as specified in the canonical specification:

- **A1**: Canonical Spatial Alphabet (N×N×N lattice, Σ(N) with N³ symbols)
- **A2**: Observer Anchor (Global Observer at (0,0,0))
- **A3**: Symbolic Weight Law (SW = 9·f)
- **A4**: Dynamic Law (90° rotations only)
- **A5**: Semantic Polarity (cos(θ))
- **A6**: Activation Rule (Local Observer)
- **A7**: Cross-Lattice Coupling (Wreath-product)

**Plus Quantum Layer:**
- Superposition (complex amplitudes)
- Quantum gates (H, X, Y, Z, rotations, CNOT, etc.)
- Entanglement (Bell states, geometric)
- Measurement (Born rule + collapse)
- Geometry-Quantum coupling (Livnium-specific)

## Quick Start

### Classical System Only

```python
from core import LivniumCoreSystem, LivniumCoreConfig

# Create system with all features
config = LivniumCoreConfig()
system = LivniumCoreSystem(config)

# Get cell information
cell = system.get_cell((0, 0, 0))
print(f"Face exposure: {cell.face_exposure}")
print(f"Symbolic Weight: {cell.symbolic_weight}")
print(f"Class: {cell.cell_class}")
```

### With Quantum Layer

```python
from core import (
    LivniumCoreSystem,
    LivniumCoreConfig,
    QuantumLattice,
    GateType,
)

# Enable quantum features
config = LivniumCoreConfig(
    enable_quantum=True,
    enable_superposition=True,
    enable_quantum_gates=True,
    enable_entanglement=True,
    enable_measurement=True,
    enable_geometry_quantum_coupling=True
)

# Create systems
core = LivniumCoreSystem(config)
qlattice = QuantumLattice(core)

# Apply quantum gates
qlattice.apply_gate((0, 0, 0), GateType.HADAMARD)

# Entangle cells
qlattice.entangle_cells((0, 0, 0), (1, 0, 0))

# Measure
result = qlattice.measure_cell((0, 0, 0))
```

## Features

### Classical Features

1. **N×N×N Lattice** - Works for any odd N ≥ 3
2. **Symbol Alphabet** - Σ(N) with exactly N³ symbols
3. **Symbolic Weight** - SW = 9·f (face exposure)
4. **Class Structure** - Core, Centers, Edges, Corners
5. **90° Rotations** - 24-element rotation group
6. **Observer System** - Global and Local observers
7. **Semantic Polarity** - cos(θ) between motion and observer
8. **Invariants** - ΣSW and class counts conservation

### Quantum Features (Optional)

1. **Superposition** - Complex amplitudes per cell
2. **Quantum Gates** - Full unitary gate library
3. **Entanglement** - Multi-cell correlations
4. **Measurement** - Born rule + collapse
5. **Geometry Coupling** - Face exposure → entanglement, etc.

## Invariants (General N)

### Total Symbolic Weight
```
ΣSW(N) = 54(N-2)² + 216(N-2) + 216
```

**Verified Values:**
- N=3: ΣSW = 486
- N=5: ΣSW = 1350
- N=7: ΣSW = 2646

### Class Counts (Closed Form)
- Core: (N-2)³
- Centers: 6(N-2)²
- Edges: 12(N-2)
- Corners: 8

**All rotations preserve these invariants for any N.**

## Running Tests

```bash
# Classical system tests
python3 core/tests/test_livnium_core.py

# Generalized N×N×N tests
python3 core/tests/test_generalized_n.py

# Quantum layer tests
python3 core/tests/test_quantum.py
```

## Documentation

- **README.md** - This file (overview)
- **ARCHITECTURE.md** - Complete 8-layer architecture documentation
- **CORE_STRUCTURE.md** - Layer-by-layer structure guide
- **QUANTUM_LAYER.md** - Quantum layer details
- **LAYER_0.md** - Recursive geometry engine (Layer 0)
- **MOKSHA.md** - Fixed-point convergence engine

## Comparison with Other Systems

| Feature | Livnium Core | Islands | Omcube | DualCube |
|---------|--------------|---------|--------|----------|
| SW = 9·f | ✅ | ✅ | ❌ | ❌ |
| Face Exposure | ✅ | ✅ | ❌ | ❌ |
| 90° Rotations | ✅ | ❌ | ❌ | ❌ |
| Observer | ✅ | ❌ | ❌ | ❌ |
| Class Structure | ✅ | ❌ | ❌ | ❌ |
| Invariants | ✅ | ❌ | ❌ | ❌ |
| Quantum Layer | ✅ | ⚠️ | ❌ | ❌ |
| Geometry-Quantum | ✅ | ❌ | ❌ | ❌ |

**This is the ONLY complete implementation of the Livnium Core System specification with full quantum capabilities.**
