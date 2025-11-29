# Semantic Warp: Dynamic Programming Alignment

## The Core Principle

**Let geometry choose the alignment automatically.**

No hardcoded words. No rules. No heuristics. Pure physics + optimization.

---

## Why Warp Works Without Hardcoding

### The Problem

Current alignment forces position-by-position matching:

```
Premise:    p1, p2, p3, p4
Hypothesis: h1, h2, h3

Forced alignment:
p1 ↔ h1
p2 ↔ h2
p3 ↔ h3
p4 ↔ (padding)
```

This creates **false fractures** when words don't align perfectly.

### The Solution

Build a **distance matrix** and let geometry find the optimal path:

```
Distance Matrix:
      h1   h2   h3
p1   0.1  0.9  1.0
p2   0.2  0.8  0.95
p3   0.9  0.1  0.2
p4   1.0  0.3  0.2
```

Geometry automatically finds the "valley" (minimum-energy path):
- p1 aligns with h1 (low distance)
- p2 skips (no good match)
- p3 aligns with h2 (low distance)
- p4 aligns with h3 (low distance)

---

## How It Works

### 1. Build Distance Matrix

```
D[i][j] = distance(premise[i], hypothesis[j])
```

Uses cosine distance (semantic similarity) or Euclidean distance.

### 2. Find Minimum-Energy Path

Uses **Dynamic Programming** (like DTW - Dynamic Time Warping):

- Start at (0, 0)
- Find path to (m, n) with minimum total energy
- Can move: down, right, or diagonally
- Choose the path that minimizes total distance

### 3. Warp Alignment

The optimal path becomes the alignment:

```
Warp path: [(0,0), (1,1), (2,2), (2,3)]
```

This means:
- premise[0] ↔ hypothesis[0]
- premise[1] ↔ hypothesis[1]
- premise[2] ↔ hypothesis[2] (then also hypothesis[3])

### 4. Fracture Detection on Warped Alignment

Run fracture detection on the **warped alignment**, not raw tokens.

This finds negation where alignment **breaks**, not where positions don't match.

---

## Why This Is Safe

You are **not** telling Livnium:
- "sleeping ≈ sleeps"
- "moves ≈ walks"
- "never = negation"

The model **discovers**:
- Similar vectors stay together (low distance)
- Different vectors stay apart (high distance)
- Negation fractures alignment (high distance spike)
- Paraphrases warp smoothly (low total energy)

**Automatically. No rules.**

---

## Usage

### Standalone

```python
from experiments.nli_v5.core.semantic_warp import SemanticWarp

warp = SemanticWarp(use_cosine_distance=True)
alignment = warp.align(premise_vectors, hypothesis_vectors)

print(f"Warp path: {alignment.warp_path}")
print(f"Total energy: {alignment.total_energy}")
```

### With Fracture Detection

```python
from experiments.nli_v5.core.fracture_dynamics import FractureDynamics

fracture_detector = FractureDynamics(fracture_threshold=0.5)
# use_warp=True: Use semantic warp before fracture detection
fracture = fracture_detector.detect_alignment_fracture(
    premise_vecs, hypothesis_vecs, use_warp=True
)
```

### Integrated in Classifier

Semantic warp is **automatically enabled** in the classifier.

Fracture detection uses warp alignment by default.

---

## Examples

### Example 1: Negation

```
Premise:    "dog is barking"
Hypothesis: "dog is not barking"

Warp path: [(0,0), (1,1), (2,2), (2,3)]
- 'dog' ↔ 'dog' (0.0000)
- 'is' ↔ 'is' (0.0000)
- 'barking' ↔ 'not' (0.6962) ← HIGH (fracture!)
- 'barking' ↔ 'barking' (0.0000)
```

Fracture detected at "not" - alignment breaks there.

### Example 2: Paraphrase

```
Premise:    "man walks"
Hypothesis: "person moves"

Warp path: [(0,0), (1,1)]
- 'man' ↔ 'person' (0.7223)
- 'walks' ↔ 'moves' (0.8214)
```

No fracture - alignment is smooth (paraphrase, not negation).

### Example 3: Word Order

```
Premise:    "bird flies"
Hypothesis: "bird can fly"

Warp path: [(0,0), (0,1), (1,2)]
- 'bird' ↔ 'bird' (0.0000)
- 'bird' ↔ 'can' (0.7799)
- 'flies' ↔ 'fly' (0.3641)
```

Geometry handles word order differences automatically.

---

## The Algorithm

### Dynamic Programming Table

```
dp[i][j] = minimum energy to reach (i, j)

Base case:
dp[0][0] = distance_matrix[0][0]

Recurrence:
dp[i][j] = min(
    dp[i-1][j] + distance_matrix[i][j],  # Move from above
    dp[i][j-1] + distance_matrix[i][j],  # Move from left
    dp[i-1][j-1] + distance_matrix[i][j] # Move diagonally
)
```

### Traceback

After filling DP table, trace back from (m-1, n-1) to (0, 0) to find optimal path.

---

## Why This Improves Accuracy

### Before Warp

- False fractures from position mismatches
- "sleeps" vs "sleeping" creates fracture (wrong!)
- Word order differences break alignment

### After Warp

- True fractures only (negation breaks alignment)
- Paraphrases align smoothly
- Word order handled automatically
- Geometry chooses optimal path

**Accuracy rises because alignment becomes physical, not lexical.**

---

## References

- `core/semantic_warp.py` - Implementation
- `core/fracture_dynamics.py` - Fracture detection with warp
- `test_semantic_warp.py` - Test script

---

## The Bigger Picture

This is exactly like:
- **Protein folding** - finding optimal 3D structure
- **Speech alignment** - aligning phonemes across speakers
- **Dynamic Time Warping** - aligning time series

All use the same principle: **find minimum-energy path through distance space**.

Livnium applies this to **semantic alignment**.

No rules. Pure geometry.

