# Quantum Module

This module provides quantum-mechanical representations for the Livnium system, enabling:

- **Uncertainty Modeling**: Features exist in superposition states
- **Feature Entanglement**: Correlated features can be entangled
- **Probabilistic Classification**: Quantum measurement for decisions
- **Quantum Feature Importance**: Feature importance as quantum amplitudes

## Structure

```
quantum/
├── __init__.py          # Module exports
├── gates.py            # Quantum gate operations
├── features.py         # Quantum feature representation
└── README.md          # This file
```

## Modules

### `gates.py`
Core quantum gate operations:
- Standard gates: Pauli X/Y/Z, Hadamard, Phase, Rotations
- State operations: Normalization, probability computation
- Entanglement: CNOT gate
- Value conversion: Deterministic → quantum superposition

### `features.py`
Quantum feature representation:
- `QuantumFeature`: Single feature as qubit
- `QuantumFeatureSet`: Collection with entanglement support
- `convert_features_to_quantum()`: Convert deterministic features

## Usage

```python
from quantum import convert_features_to_quantum

# Convert deterministic features to quantum
feature_dict = {
    'phi_adjusted': 0.5,
    'negation_flag': 0.8
}

quantum_features = convert_features_to_quantum(
    feature_dict,
    entanglement_pairs=[('phi_adjusted', 'negation_flag')]
)

# Get feature importance
importance = quantum_features.get_feature_importance()

# Measure features
results = quantum_features.measure_all()
```

## Integration

See `docs/QUANTUM_INTEGRATION_PLAN.md` for full integration details.

## Testing

Run the integration test:
```bash
python3 test_quantum_integration.py
```

