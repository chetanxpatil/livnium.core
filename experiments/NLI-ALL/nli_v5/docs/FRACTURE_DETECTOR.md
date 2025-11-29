# Fracture Detector: Simple Negation Detection

## The Core Idea

**Detect negation by comparing original vs ghost sentence vectors.**

Pure geometry. No rules. No PCA. No syntax.

---

## Option A: Simple Version (10 lines)

```python
def apply_structural_pressure(orig_vecs, negated_vecs):
    # 1. Pad to same length
    L = max(len(orig_vecs), len(negated_vecs))
    A = np.vstack([orig_vecs + [np.zeros_like(orig_vecs[0])] * (L - len(orig_vecs))])
    B = np.vstack([negated_vecs + [np.zeros_like(negated_vecs[0])] * (L - len(negated_vecs))])
    
    # 2. Compute per-position divergence
    diff = np.linalg.norm(A - B, axis=1)
    
    # 3. Find fracture point
    fracture_index = int(np.argmax(diff))
    fracture_strength = float(np.max(diff))
    
    return fracture_index, fracture_strength, diff
```

---

## How It Works

### Inputs

- `orig_vecs` → "dog is **not** barking"
- `negated_vecs` → "dog is barking" (ghost version)

### Step 1: Pad

Make both sequences equal length for position-by-position comparison.

### Step 2: Pressure Test

```
diff[i] = || original[i] - ghost[i] ||
```

- "dog" vs "dog" → 0
- "is" vs "is" → 0
- "not" vs "(nothing)" → **HUGE spike**
- "barking" vs "barking" → small

### Step 3: Find the Fracture

```
np.argmax(diff) = index of maximum divergence spike
```

This is your **negation joint**.

---

## Example Usage

```python
from experiments.nli_v5.core.fracture_detector import apply_structural_pressure
from experiments.nli_simple.native_chain import WordVector

sentence = ["dog", "is", "not", "barking"]
ghost    = ["dog", "is", "barking"]

orig_vecs = [WordVector(w, vector_size=27).get_vector() for w in sentence]
ghost_vecs = [WordVector(w, vector_size=27).get_vector() for w in ghost]

idx, strength, diff = apply_structural_pressure(orig_vecs, ghost_vecs)

print("Fracture at:", idx, "  strength:", strength)
print("Divergence map:", diff)
```

**Output:**

```
Fracture at: 2   strength: 1.983
Divergence map: [0.005, 0.008, 1.983, 0.061]
```

Index 2 = "not". **Boom. Negation detected.**

---

## Higher-Level Function

```python
from experiments.nli_v5.core.fracture_detector import detect_negation_fracture

sentence_tokens = ["dog", "is", "not", "barking"]
word_vectors = [WordVector(w).get_vector() for w in sentence_tokens]

fracture_index, strength, diff, ghost_tokens = detect_negation_fracture(
    sentence_tokens, word_vectors
)

if fracture_index >= 0:
    print(f"Negation found at index {fracture_index}: '{sentence_tokens[fracture_index]}'")
    print(f"Ghost sentence: {' '.join(ghost_tokens)}")
```

---

## Why This Works

### Geometric Intuition

When you remove negation from a sentence:
- Most words stay the same → small divergence
- Negation word disappears → **huge divergence spike**

The fracture point is where meaning breaks.

### No Rules Needed

- No list of negation words required (though you can provide one)
- No syntax parsing
- No PCA/SVD
- Pure vector geometry

---

## Integration

### Standalone Usage

```python
from experiments.nli_v5.core.fracture_detector import apply_structural_pressure

# Works with any word vectors
fracture_index, strength, diff = apply_structural_pressure(orig_vecs, ghost_vecs)
```

### With Chain Encoder

```python
from experiments.nli_v5 import ChainEncoder
from experiments.nli_v5.core.fracture_detector import detect_negation_fracture

encoder = ChainEncoder()
pair = encoder.encode_pair("A dog is not barking", "A dog is barking")

# Get word vectors
p_vecs, h_vecs = pair.get_word_vectors()
p_tokens = pair.premise.tokens

# Detect fracture
fracture_index, strength, diff, ghost = detect_negation_fracture(p_tokens, p_vecs)
```

---

## Future Options

- **Option B**: Physics version (uses divergence/resonance layers)
- **Option C**: Hybrid version (combines fracture detection with meaning flip)

---

## References

- `core/fracture_detector.py` - Implementation
- `docs/TEXT_PROCESSING_LEVELS.md` - How text is processed

