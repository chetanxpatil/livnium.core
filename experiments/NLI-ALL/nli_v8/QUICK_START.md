# Quick Start: Livnium v8

## What's New in v8

**Clean architecture with our best understanding:**

1. **Semantic Warp**: Dynamic programming alignment (no hardcoded rules)
2. **Fracture Detection**: Collision-based negation detection
3. **Geometry-First**: Geometry is the teacher, not the student

---

## Training

```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v8/training/train_v8.py \
  --train 1000 \
  --test 100
```

**Output includes:**
- Training accuracy
- Fracture detection statistics
- Confusion matrices

---

## Test Individual Examples

```python
from experiments.nli_v8 import ChainEncoder, LivniumV8Classifier

encoder = ChainEncoder()
pair = encoder.encode_pair("A dog is barking", "A dog is not barking")
classifier = LivniumV8Classifier(pair)
result = classifier.classify()

print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.3f}")

# Check fracture detection
opposition = result.layer_states['layer_opposition']
if opposition['fracture_detected']:
    print(f"🔥 Fracture detected (strength: {opposition['fracture_strength']:.4f})")
```

---

## Key Features

### Semantic Warp
- Finds optimal alignment automatically
- No hardcoded words
- Handles paraphrases, word order differences

### Fracture Detection
- Detects negation via collision analysis
- Negation = maximal alignment tension
- Pure physics, no word lists

### Geometry-First
- Geometry zones determine classification
- Classifier learns to read geometry
- Geometry is stable and invariant

---

## Architecture

```
Input → Semantic Warp → Fracture Detection → Divergence → Geometry Zones → Decision
```

**Clean. Minimal. Physics-based.**

---

## Files

- `core/encoder.py` - Chain encoder
- `core/semantic_warp.py` - DP alignment
- `core/fracture_dynamics.py` - Collision-based fracture detection
- `core/layers.py` - Clean layer stack
- `core/classifier.py` - Main classifier
- `training/train_v8.py` - Training script

---

## Philosophy

**This is computational geometry, not machine learning.**

- Meaning from field equations
- Physics-based alignment
- Geometry discovers negation automatically
- No hardcoded rules

