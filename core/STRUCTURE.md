# Core Folder Structure

## Overview

The `core` folder is organized into three main sections:

1. **Classical** - Geometric lattice system
2. **Quantum** - Quantum layer (optional)
3. **Tests** - Test suite

## Directory Structure

```
core/
├── __init__.py                 # Main package exports
├── config.py                   # Configuration with feature switches
├── README.md                   # Main documentation
├── QUANTUM_LAYER.md            # Quantum layer documentation
├── STRUCTURE.md                # This file
│
├── classical/                  # Classical geometric system
│   ├── __init__.py
│   └── livnium_core_system.py  # Main Livnium Core System
│
├── quantum/                     # Quantum layer (optional)
│   ├── __init__.py
│   ├── quantum_cell.py         # Quantum state per cell
│   ├── quantum_gates.py        # Unitary gate library
│   ├── quantum_lattice.py      # Quantum lattice integration
│   ├── entanglement_manager.py # Multi-cell entanglement
│   ├── measurement_engine.py   # Born rule + collapse
│   └── geometry_quantum_coupling.py # Geometry ↔ Quantum mapping
│
└── tests/                       # Test suite
    ├── __init__.py
    ├── test_livnium_core.py    # Basic system tests
    ├── test_generalized_n.py   # N×N×N generalization tests
    └── test_quantum.py         # Quantum layer tests
```

## Module Organization

### Classical (`classical/`)

**Purpose**: Geometric lattice with symbolic weight, rotations, and observer system.

**Key Components**:
- `LivniumCoreSystem` - Main system class
- `LatticeCell` - Single cell in lattice
- `Observer` - Global/Local observer
- `RotationAxis`, `CellClass` - Enums
- `RotationGroup` - 90° rotation operations

**Features**:
- N×N×N lattice (any odd N ≥ 3)
- Symbolic Weight (SW = 9·f)
- Face exposure classification
- 90° rotation group
- Observer system
- Semantic polarity
- Invariants conservation

### Quantum (`quantum/`)

**Purpose**: Quantum layer that integrates with classical geometry.

**Key Components**:
- `QuantumCell` - Quantum state (complex amplitudes)
- `QuantumGates` - Unitary gate library
- `QuantumLattice` - Integration layer
- `EntanglementManager` - Multi-cell entanglement
- `MeasurementEngine` - Born rule + collapse
- `GeometryQuantumCoupling` - Livnium-specific coupling

**Features**:
- Superposition (complex amplitudes)
- Quantum gates (H, X, Y, Z, rotations, CNOT, etc.)
- Entanglement (Bell states, geometric)
- Measurement (Born rule, collapse)
- Geometry-Quantum coupling (face exposure → entanglement, etc.)

### Tests (`tests/`)

**Purpose**: Comprehensive test suite.

**Test Files**:
- `test_livnium_core.py` - Basic functionality tests
- `test_generalized_n.py` - N×N×N generalization (N=3,5,7,9...)
- `test_quantum.py` - Quantum layer tests

## Import Examples

### Classical Only

```python
from core import LivniumCoreSystem, LivniumCoreConfig

config = LivniumCoreConfig()
system = LivniumCoreSystem(config)
```

### With Quantum

```python
from core import (
    LivniumCoreSystem,
    LivniumCoreConfig,
    QuantumLattice,
    QuantumGates,
    GateType,
)

config = LivniumCoreConfig(
    enable_quantum=True,
    enable_superposition=True,
    enable_quantum_gates=True,
)

core = LivniumCoreSystem(config)
qlattice = QuantumLattice(core)
```

### Direct Imports

```python
# Classical components
from core.classical import LivniumCoreSystem, RotationAxis, CellClass

# Quantum components
from core.quantum import QuantumCell, QuantumGates, GateType
```

## File Responsibilities

| File | Purpose |
|------|---------|
| `config.py` | Feature switches and configuration |
| `classical/livnium_core_system.py` | Main geometric system |
| `quantum/quantum_cell.py` | Quantum state representation |
| `quantum/quantum_gates.py` | Unitary gate library |
| `quantum/quantum_lattice.py` | Quantum-geometry integration |
| `quantum/entanglement_manager.py` | Entanglement management |
| `quantum/measurement_engine.py` | Measurement and collapse |
| `quantum/geometry_quantum_coupling.py` | Geometry ↔ Quantum rules |

## Benefits of This Structure

1. **Clear Separation**: Classical and quantum are separate modules
2. **Modular**: Can use classical without quantum
3. **Organized**: Related files grouped together
4. **Testable**: Tests in dedicated folder
5. **Scalable**: Easy to add new features

## Adding New Features

### Add Classical Feature
1. Add to `classical/livnium_core_system.py` or create new file in `classical/`
2. Update `classical/__init__.py` to export
3. Add tests to `tests/`

### Add Quantum Feature
1. Add to appropriate file in `quantum/` or create new file
2. Update `quantum/__init__.py` to export
3. Add tests to `tests/test_quantum.py`

### Add Configuration
1. Add feature switch to `config.py`
2. Update validation in `config.py.__post_init__()`
3. Document in `README.md`

