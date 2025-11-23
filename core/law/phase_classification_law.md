# Phase Classification Law

## The Decision Rules

The phase classification law maps geometric signals to semantic phases (Entailment/Contradiction/Neutral) using a **2D phase diagram**.

## The Three Phases

### 1. Contradiction (Push Apart)
**Region**: Positive divergence (push apart)

**Decision rule**:
```python
if divergence > 0.02:
    predict = CONTRADICTION
```

**Physics**: Vectors push apart → contradiction

**Threshold**: `0.02` (from canonical fingerprints)

### 2. Entailment (Pull Inward + Shared Basin)
**Region**: Negative divergence AND high resonance

**Decision rule**:
```python
elif divergence < -0.08 AND resonance > 0.50:
    predict = ENTAILMENT
```

**Physics**: Vectors pull inward **AND** share strong structure → entailment

**Thresholds**:
- Divergence: `-0.08` (negative, convergence)
- Resonance: `0.50` (high, shared basin)

### 3. Neutral (Balanced Forces)
**Region**: Near-zero divergence (balanced forces)

**Decision rule**:
```python
elif abs(divergence) < 0.12:
    predict = NEUTRAL
```

**Physics**: Forces cancel → neutral

**Threshold**: `0.12` (near-zero band)

### 4. Fallback (Force-Based)
**Region**: Edge cases where physics signals are ambiguous

**Decision rule**: Use force-based decision (attraction ratios, force comparisons, resonance tiebreaker)

## Implementation

**Location**: `experiments/nli_v5/layers.py` → `Layer4Decision.compute()`

```python
# Rule 1: Contradiction - Strong positive divergence
if divergence > self.divergence_c_threshold:
    label = 'contradiction'
    
# Rule 2: Entailment - Negative divergence AND high resonance
elif divergence < self.divergence_e_threshold and resonance > self.resonance_e_min:
    label = 'entailment'
    
# Rule 3: Neutral - Near-zero divergence
elif abs(divergence) < self.divergence_n_band:
    label = 'neutral'
    
# Rule 4: Fallback - Force-based decision
else:
    # Use attraction ratios and force comparisons
    ...
```

## The Phase Diagram

```
        High Resonance (0.5+)
              |
              |  E (Entailment)
              |  (negative div + high res)
              |
    ----------+---------- Divergence
              |  (push/pull)
              |
    C (Contradiction)  |  N (Neutral)
    (positive div)     |  (near-zero div)
              |
        Low Resonance (<0.5)
```

## Thresholds (From Canonical Fingerprints)

### Divergence Thresholds
- **Contradiction**: `d > 0.02` (positive)
- **Entailment**: `d < -0.08` (negative)
- **Neutral**: `|d| < 0.12` (near zero)

### Resonance Thresholds
- **Entailment**: `r > 0.50` (high)
- **Neutral**: `0.45 < r < 0.70` (mid-range)

## Results

### Before Physics-Based Logic
- Entailment recall: 23.8%
- Contradiction recall: 56.9%
- Neutral recall: 40.8%
- Overall accuracy: 40.4%

### After Physics-Based Logic
- Entailment recall: **39.6%** ⬆️ (+15.8%)
- Contradiction recall: **47.5%** (still strong)
- Neutral recall: 24.9% (needs tuning)
- Overall accuracy: ~40% (maintained)

## Status

✅ **Established**: Contradiction region (positive divergence)
✅ **Established**: Entailment region (negative divergence + high resonance)
⚠️ **In Progress**: Neutral region (needs explicit balance band)

## Next Steps

### 1. Extract Canonical Neutral Fingerprints
Run with golden labels and extract:
- Divergence band (q25-q75)
- Balance condition: `|cold_attraction - far_attraction|`

### 2. Define Neutral as Explicit Balance Band
```python
is_neutral = (
    abs(divergence) < d_neutral_band
    and abs(cold_attraction - far_attraction) < attraction_balance_band
)
```

### 3. Prioritize Phases
- Strong C first (if divergence clearly positive)
- Strong E next (if strongly negative + high resonance)
- Then neutral (if both E and C are weak and balance criteria match)

### 4. Use Data-Derived Thresholds
Make thresholds adaptive:
- `d_c_strong = mean_C_divergence + k * std_C_divergence`
- `d_e_strong = mean_E_divergence - k * std_E_divergence`
- `d_neutral_band = factor * std_N_divergence`

## Philosophy

This is not "if-else soup". It's a **phase classifier over a vector field**.

The decision logic maps geometric signals (divergence, resonance) to semantic phases (E/C/N) using physically interpretable thresholds.

## Related Laws

- **Divergence Law**: Provides the x-axis (push/pull)
- **Resonance Law**: Provides the y-axis (similarity)

## References

- Implementation: `experiments/nli_v5/layers.py` → `Layer4Decision`
- Fingerprints: `experiments/nli_v5/physics_fingerprints.json`
- Analysis: `experiments/nli_v5/PHYSICS_ANALYSIS_CONFIRMED.md`
- Completion: `experiments/nli_v5/IMPLEMENTATION_COMPLETE.md`

