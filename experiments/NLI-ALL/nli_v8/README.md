# Livnium NLI v8: Clean Architecture

## Philosophy

**Geometry is stable and invariant. It refuses lies.**

- Geometry produces meaning, labeling describes it
- Train classifier to read geometry, not force it
- Negation = alignment tension (collision-based detection)
- Alignment = optimal path (semantic warp, no hardcoded rules)

---

## Key Features

### 1. Semantic Warp Alignment
- **Dynamic Programming** alignment (like DTW)
- Finds optimal alignment automatically
- No hardcoded words, no rules, pure geometry
- Handles paraphrases, word order, etc.

### 2. Collision-Based Fracture Detection
- Negation = maximal alignment tension between premise & hypothesis
- Detects negation by colliding vectors
- No word lists, pure physics

### 3. Geometry-First Classification
- Geometry zones: E (div < -0.1), C (div > +0.1), N (near zero)
- Classifier learns to read geometry zones
- Geometry is the teacher, not the student

### 4. Clean Architecture
- Layer 0: Resonance
- Layer 1.5: Opposition + Fracture Detection (with semantic warp)
- Layer 1: Curvature
- Layer 2: Basins
- Layer 3: Valley
- Layer 4: Decision

---

## Quick Start

### Training

```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v8/training/train_v8.py \
  --train 1000 \
  --test 100
```

### Test Individual Examples

```python
from experiments.nli_v8 import ChainEncoder, LivniumV8Classifier

encoder = ChainEncoder()
pair = encoder.encode_pair("A dog is barking", "A dog is not barking")
classifier = LivniumV8Classifier(pair)
result = classifier.classify()

print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Fracture detected: {result.layer_states['layer_opposition']['fracture_detected']}")
```

---

## Architecture

```
Input: Premise + Hypothesis
  ↓
Semantic Warp (DP alignment)
  ↓
Fracture Detection (collision analysis)
  ↓
Angle-Based Divergence
  ↓
Geometry Zones (E/C/N)
  ↓
Basins + Valley
  ↓
Decision
  ↓
Output: Label + Confidence
```

---

## Files

- `core/encoder.py` - Chain encoder
- `core/semantic_warp.py` - DP alignment (no hardcoded rules)
- `core/fracture_dynamics.py` - Collision-based negation detection
- `core/geometry_teacher.py` - Geometry-first classification
- `core/layers.py` - Clean layer stack
- `core/classifier.py` - Main classifier
- `training/train_v8.py` - Training script

---

## Differences from v5

1. **Semantic Warp Integrated**: Alignment is automatic (DP), not position-based
2. **Fracture Detection**: Collision-based, not internal analysis
3. **Cleaner Layers**: Simplified, focused on geometry-first
4. **Better Organization**: Clear separation of concerns

---

## Philosophy

**This is not machine learning. This is computational geometry.**

- Meaning from field equations, not statistical datasets
- Physics-based, not probability-based
- Geometry discovers negation automatically
- Alignment is physical, not lexical

---

## References

- `docs/SEMANTIC_WARP.md` - Semantic warp documentation
- `docs/FRACTURE_DYNAMICS.md` - Fracture detection documentation
- `docs/GEOMETRY_FIRST_PHILOSOPHY.md` - Geometry-first philosophy

