# Quantum Module Quick Start Guide

## 🚀 How to Run

### Option 1: Run Integration Test (Basic Features)

```bash
python3 test_quantum_integration.py
```

**What it does:**
- Creates quantum features from deterministic values
- Shows feature entanglement
- Demonstrates measurement and importance
- Tests uncertainty quantification

**Expected output:**
```
============================================================
QUANTUM FEATURE INTEGRATION TEST
============================================================
1. Creating quantum features...
2. Quantum feature states...
3. Feature importance...
4. Measuring features...
5. Converting back to deterministic...
6. Entanglement effects...
✅ Quantum feature integration test complete!
```

---

### Option 2: Run Classifier Test (Full Classification)

```bash
python3 test_quantum_classifier.py
```

**What it does:**
- Trains a quantum classifier
- Tests 3-class NLI classification
- Shows prediction uncertainty
- Demonstrates gate configuration

**Expected output:**
```
============================================================
QUANTUM CLASSIFIER TEST (Phase 2)
============================================================
1. Creating training data...
2. Initializing quantum classifier...
3. Training classifier...
4. Testing predictions...
5. Feature uncertainty...
6. Testing gate configuration...
✅ Quantum classifier test complete!
```

---

### Option 3: Run Original Quantum Model

```bash
python3 "Livnium Qubit Model.py"
```

**What it does:**
- Demonstrates full quantum mechanics
- Runs Test of Faith simulation
- Monte Carlo analysis (1000 trials)
- Shows quantum interference

**Expected output:**
```
============================================================
QUANTUM SUPERPOSITION & GATES
============================================================
...
============================================================
TEST OF FAITH: GEOMETRIC SIGNAL ISOLATION
============================================================
...
============================================================
MONTE CARLO SIMULATION: STATISTICAL ANALYSIS
============================================================
...
```

---

## 📝 Basic Usage Examples

### Example 1: Convert Features to Quantum

```python
from quantum import convert_features_to_quantum

# Your deterministic features
features = {
    'phi_adjusted': 0.5,
    'negation_flag': 0.8,
    'embedding_proximity': 0.6
}

# Convert to quantum
quantum_features = convert_features_to_quantum(
    features,
    feature_ranges={'phi_adjusted': (-1.0, 1.0)},
    entanglement_pairs=[('phi_adjusted', 'negation_flag')],
    random_seed=42  # For reproducibility
)

# Get feature importance
importance = quantum_features.get_feature_importance()
print(importance)
# {'phi_adjusted': 0.75, 'negation_flag': 0.80, ...}

# Get uncertainty
uncertainty = quantum_features.get_uncertainty()
print(uncertainty)
# {'phi_adjusted': 0.61, 'negation_flag': 0.72, ...}
```

---

### Example 2: Quantum Classification

```python
from quantum import QuantumClassifier, convert_features_to_quantum
import numpy as np

# Training data
training_features = [
    {'phi_adjusted': 0.7, 'negation_flag': 0.2},  # Entailment
    {'phi_adjusted': 0.1, 'negation_flag': 0.5},  # Neutral
    {'phi_adjusted': -0.6, 'negation_flag': 0.9},  # Contradiction
]
labels = np.array([0, 1, 2])  # Entailment, Neutral, Contradiction

# Convert to quantum
quantum_training = [
    convert_features_to_quantum(feat, random_seed=42)
    for feat in training_features
]

# Train classifier
classifier = QuantumClassifier(n_classes=3, random_seed=42)
classifier.fit(quantum_training, labels)

# Predict
test_features = {'phi_adjusted': 0.8, 'negation_flag': 0.1}
test_quantum = convert_features_to_quantum(test_features, random_seed=42)

probabilities, uncertainty = classifier.predict_proba(
    test_quantum, 
    return_uncertainty=True
)
predicted = classifier.predict(test_quantum)

print(f"Probabilities: {probabilities}")
print(f"Predicted: {predicted}")
print(f"Uncertainty: {uncertainty}")
```

---

### Example 3: Custom Gate Configuration

```python
from quantum import QuantumClassifier

classifier = QuantumClassifier(random_seed=42)

# Customize gate parameters
classifier.set_gate_config({
    'rotation_scale': np.pi / 4,  # Smaller rotations
    'interference_strength': 0.5   # Less interference
})

# Use classifier with custom config
probabilities, uncertainty = classifier.predict_proba(quantum_features)
```

---

## 🔧 Integration with Livnium

### Step 1: Import Quantum Module

```python
from quantum import QuantumClassifier, convert_features_to_quantum
```

### Step 2: Convert Your Features

```python
# Get features from your existing pipeline
feature_dict = extract_features_from_reasoning_path(...)

# Convert to quantum
quantum_features = convert_features_to_quantum(
    feature_dict,
    feature_ranges={'phi_adjusted': (-1.0, 1.0)},
    entanglement_pairs=[
        ('phi_adjusted', 'sw_distribution'),
        ('phi_adjusted', 'concentration')
    ]
)
```

### Step 3: Classify

```python
# Use quantum classifier
classifier = QuantumClassifier(n_classes=3, random_seed=42)
probabilities, uncertainty = classifier.predict_proba(
    quantum_features,
    return_uncertainty=True
)

# Or convert back to deterministic for existing classifier
deterministic_vector = quantum_features.to_deterministic_vector(feature_names)
```

---

## 🐛 Troubleshooting

### Issue: Import Error
```bash
ModuleNotFoundError: No module named 'quantum'
```

**Solution:** Make sure you're in the project root directory:
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 test_quantum_integration.py
```

### Issue: Numba Not Available
```
NUMBA_REASON: Numba not installed
```

**Solution:** This is OK! The module falls back to pure NumPy. Numba is optional but improves performance.

### Issue: Random Results
**Solution:** Use `random_seed` parameter for reproducibility:
```python
quantum_features = convert_features_to_quantum(features, random_seed=42)
```

---

## 📊 Expected Performance

- **Feature Conversion**: ~1ms per feature set
- **Classification**: ~2-5ms per prediction
- **Monte Carlo (1000 trials)**: ~5-10 seconds

---

## 🎯 Next Steps

1. **Run tests** to verify installation
2. **Try examples** above
3. **Integrate** into your feature pipeline
4. **Benchmark** quantum vs deterministic
5. **Tune** gate parameters for your data

---

## 📚 More Information

- `quantum/README.md` - Module overview
- `quantum/STRUCTURE.md` - File organization
- `quantum/IMPROVEMENTS.md` - Recent improvements
- `quantum/PROBLEMS_SOLVED.md` - Problems addressed
- `docs/QUANTUM_INTEGRATION_PLAN.md` - Full integration plan

