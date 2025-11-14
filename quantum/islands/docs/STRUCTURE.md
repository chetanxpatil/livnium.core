# Quantum Module Structure

## 📁 Folder Organization

All quantum-related code is now organized in the `quantum/` folder at the project root.

```
quantum/
├── __init__.py          # Module exports and public API
├── gates.py            # Quantum gate operations
├── features.py         # Quantum feature representation
├── README.md          # Module overview and usage
└── STRUCTURE.md       # This file
```

## 📦 Module Contents

### `gates.py`
Core quantum gate operations:
- Standard gates: Pauli X/Y/Z, Hadamard, Phase, Rotations
- State operations: Normalization, probability computation
- Entanglement: CNOT gate
- Value conversion: Deterministic → quantum superposition
- Numba-optimized (with fallback)

### `features.py`
Quantum feature representation:
- `QuantumFeature`: Single feature as qubit
- `QuantumFeatureSet`: Collection with entanglement support
- `convert_features_to_quantum()`: Convert deterministic features
- Feature importance as quantum amplitudes
- Measurement and state collapse

### `__init__.py`
Public API exports:
- All gate functions
- Feature classes and utilities
- Clean import interface

## 🔄 Migration from Old Locations

**Old locations** (removed):
- `layers/layer3/meta/quantum_gates.py` → `quantum/gates.py`
- `layers/layer4/evaluation/quantum_features.py` → `quantum/features.py`

**New import style**:
```python
# Old (no longer works)
from layers.layer3.meta.quantum_gates import ...
from layers.layer4.evaluation.quantum_features import ...

# New (current)
from quantum.gates import ...
from quantum.features import ...
# Or use the main module
from quantum import convert_features_to_quantum
```

## ✅ Benefits of Separate Folder

1. **Organization**: All quantum code in one place
2. **Clarity**: Clear separation from layer-specific code
3. **Reusability**: Can be used across all layers
4. **Maintainability**: Easier to find and update quantum code
5. **Scalability**: Easy to add new quantum modules (classifier, circuits, etc.)

## 🚀 Usage

```python
from quantum import convert_features_to_quantum

# Convert features
quantum_features = convert_features_to_quantum(
    feature_dict,
    entanglement_pairs=[('phi_adjusted', 'negation_flag')]
)

# Get importance
importance = quantum_features.get_feature_importance()

# Measure
results = quantum_features.measure_all()
```

## 📝 Future Modules

Planned additions to `quantum/`:
- `classifier.py`: Quantum-enhanced GeometricClassifier
- `circuits.py`: Quantum circuit construction
- `optimization.py`: Quantum optimization algorithms

