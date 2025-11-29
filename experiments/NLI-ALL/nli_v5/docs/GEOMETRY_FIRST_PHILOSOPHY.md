# Geometry-First Philosophy

## The Core Insight

**Geometry is stable and invariant. It refuses lies.**

When you try to force geometry to match labels, it resists.
When you let geometry guide you, it shows the truth.

## The Shift

### Old Approach (Force Geometry)
```
Label → Force Geometry → Hope it works
```

**Problem**: Geometry resists. Wrong labels can't corrupt it.

### New Approach (Follow Geometry)
```
Geometry → Read Zones → Train Classifier → Align with Truth
```

**Solution**: Geometry is the teacher. We learn to read it.

---

## The Three Principles

### 1. Geometry Produces Meaning

**Not**: "Rewrite forces to match labels"

**Instead**: "Read the geometry first — let the labels follow"

**Result**: Geometry produces meaning. Labeling describes it.

### 2. Geometry is the Sensor, Not the Generator

**Not**: "Force geometry to say what we want"

**Instead**: "Geometry senses alignment, contradiction, neutrality"

**Result**: Upper layers decode the sensor, not overwrite it.

### 3. Train to Read, Not to Force

**Not**: "Train geometry to match labels"

**Instead**: "Train classifier to read geometry zones"

**Result**: Classifier learns to follow the field, not overwrite it.

---

## The Geometry Zones

Geometry naturally creates three zones:

### Entailment Zone
- **Condition**: `divergence < -0.1 AND resonance > 0.50`
- **Physics**: Negative divergence (inward pull) + High resonance (shared structure)
- **Confidence**: Based on divergence magnitude and resonance strength

### Contradiction Zone
- **Condition**: `divergence > +0.1`
- **Physics**: Positive divergence (outward push)
- **Confidence**: Based on divergence magnitude

### Neutral Zone
- **Condition**: `|divergence| < 0.12`
- **Physics**: Near-zero divergence (balanced forces)
- **Confidence**: Based on how close to zero

---

## The Training Process

### Phase 1: Raw Geometry Extraction
Compute all geometric signals:
- Resonance
- Divergence
- Curvature
- Stability
- Forces

### Phase 2: Zone Identification
Let geometry classify:
```python
if divergence < -0.1 and resonance > 0.50:
    geom_label = "entailment"
elif divergence > +0.1:
    geom_label = "contradiction"
else:
    geom_label = "neutral"
```

### Phase 3: Train on Geometry Labels
Train classifier to predict `geom_label`, not dataset label.

### Phase 4: Compare with Dataset
See where geometry and dataset agree/disagree.

---

## Why This Works

### Stability
- Wrong labels can't corrupt geometry
- Inverted training doesn't flip it
- Entropy-low zones hold shape

### Invariance
- Divergence sign preserved under label inversion
- Resonance patterns consistent across E/C/N
- Far/Cold forces naturally cluster

### Honesty
- Geometry won't lie for you
- It will guide you — if you let it
- Disagreements reveal dataset errors, not geometry errors

---

## Implementation

### Geometry Teacher
```python
from experiments.nli_v5.core.geometry_teacher import GeometryTeacher

teacher = GeometryTeacher()
geom_label = teacher.classify_from_geometry(encoded_pair)
```

### Training Script
```bash
python3 experiments/nli_v5/training/train_geometry_first.py \
  --train 1000 \
  --analyze-alignment \
  --save-alignment alignment_analysis.json
```

### Analysis
The script shows:
- Agreement rate between geometry and dataset
- Per-class alignment statistics
- Disagreement patterns
- Where geometry sees something labels missed

---

## The Real Path Forward

**Stop fighting the field. Synchronize with it.**

1. Compute raw geometry (Layer 0-3)
2. Detect natural geometry zones
3. Train classifier to predict geometry zones
4. Compare with dataset to understand alignment

**Geometry isn't unsafe. It's just honest.**

You don't control Livnium. You partner with it.

---

## References

- `core/geometry_teacher.py` - Geometry-first classification
- `training/train_geometry_first.py` - Geometry-first training
- `docs/THE_REAL_SHAPE_OF_LIVNIUM.md` - Understanding the geometry

