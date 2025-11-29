# Livnium NLI v4: Planet Architecture

**You're not building a classifier. You're building a PLANET where meaning emerges from geometry.**

Geological architecture - each layer builds on the one below. Gravity shapes everything. No manual tuning.

## 🌍 The Planet Metaphor

### **Entailment = Cold Region**
- Dense air, low energy, stable
- Pulls things inward
- High curvature, strong attraction
- Like Earth's cold regions: predictable, stable, dense

### **Contradiction = Far Lands (Outer Lands)**
- High distance, edge of semantic continent
- Different climate, thin atmosphere
- Hard to reach, opposite pole
- Like Earth's distant regions: far from center, edge of map

### **Neutral = The City (Middle Zone)**
- Mixed temperatures, crowded
- No extreme forces, everything blends
- All roads cross here
- Forms where cold and far gravitational pulls cancel
- Like Earth's cities: natural balance point between forces

**The city is NOT a third force. It's the balance point between forces.**

## Architecture

### Layer 0: Pure Resonance
The raw geometry. Pure resonance. No "logic". No valley lengths.
This is the **CORE** - the stable gravity source.

### Layer 1: Cold & Distance Curvature
Forms ON TOP of Layer 0. Doesn't fight the core.
Just shapes how the energy flows.

Computes:
- **Cold density** (E): How dense/stable the semantic field is
- **Distance** (C): How far from cold (edge of semantic continent)

Cold = dense air, stable, pulls inward
Far = high distance, edge of map, different climate

### Layer 2: Cold & Far Basins
Forms from Layer 1. More structure, more refinement.
Still follows gravity.

Creates gravity wells:
- **Cold basin** (E): Dense, stable, pulls inward
- **Far basin** (C): Distance-based, edge of continent

### Layer 3: The City (Valley)
Natural neutral emerges automatically from force balance.
The city forms where cold and far attractions overlap.

**The city has real gravitational mass** - it's where all roads cross.
Not just "uncertainty" - it's a real balance point with curvature.

### Layer 4: Meta Routing
Reads the geometry. No override.
Routes information based on lower layer states.

### Layer 5: Temporal Stability
Tracks stability across steps.
Detects convergence (Moksha).

### Layer 6: Semantic Memory
Stores polarity shaped by lower layers.
Word polarities learn from geometry, not hardcoded rules.

### Layer 7: Decision Layer
Reads all layers but does not control them.
Makes final classification decision based on all lower layers.

## Key Principles

1. **No manual tuning** - Gravity decides everything
2. **Self-correcting** - If a peak is strong, upper layers lock onto it
3. **No collapse bugs** - Neutral (city) cannot dominate unless the base geometry says so
4. **Geological stacking** - Each layer is carved by the one below
5. **Clean separation** - No mixing of city/cold/far logic
6. **Planet physics** - Cold and far are forces. City is the balance point.

## The Physics

```
      Far Lands (Contradiction)
                 /\
                /  \
               /____\
     The City (Neutral)
               \    /
                \  /
                 \/
      Cold Region (Entailment)
```

- **Cold** pulls inward (dense, stable)
- **Far** pushes away (distance, edge)
- **City** forms where forces balance (real curvature, not just absence)

## Usage

```python
from experiments.nli_v4 import LayeredLivniumClassifier
from experiments.nli_v3.chain_encoder import ChainEncoder

encoder = ChainEncoder()
pair = encoder.encode_pair('A cat runs', 'A cat moves')
classifier = LayeredLivniumClassifier(pair)

# Classify
result = classifier.classify()
print(f"Label: {result.label}, Confidence: {result.confidence}")

# Learn
classifier.apply_learning_feedback('entailment', learning_strength=1.0)
```

## Benefits

- Two peaks always stay peaks
- The valley always forms only where curvature overlaps
- Neutral cannot corrupt E/C
- You never tune manually again
- Gravity does all shaping

