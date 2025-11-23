# Law of Resonance

## The Second Axis

Resonance measures how strongly a sentence pair shares geometric structure. It's the **second axis** of the phase diagram, alongside divergence.

## Definition

**Resonance** = Raw geometric similarity signal from chain structure

**Range**: Typically 0.0 to 1.0 (can be negative for strong opposition)

## Physical Meaning

### High Resonance
- **Meaning**: Sentences share strong geometric structure
- **Physics**: Vectors align in similar directions
- **Interpretation**: "Genuinely sharing structure" / "Living in shared basin"

### Low Resonance
- **Meaning**: Sentences have weak geometric similarity
- **Physics**: Vectors point in different directions
- **Interpretation**: "Different structures" / "Separate basins"

## Role in Phase Classification

### Entailment Requires Both Signals

Entailment is **not** just "not contradiction". It requires:

1. **Negative divergence** (pull inward)
2. **AND high resonance** (shared structure)

**Decision rule**: `divergence < -0.08 AND resonance > 0.50`

This makes entailment its own region, not just the complement of contradiction.

### Why Resonance Matters

From canonical fingerprints (golden labels):
- **Entailment**: resonance = 0.6186 ± 0.1369 (highest)
- **Contradiction**: resonance = 0.5808 ± 0.1201 (mid-range)
- **Neutral**: resonance = 0.5853 ± 0.1262 (mid-range)

**Separation**:
- E-C separation: 0.0378 (adequate)
- E-N separation: 0.0333 (adequate)

Resonance provides the **second dimension** needed to separate entailment from contradiction and neutral.

## Implementation

**Location**: `experiments/nli_v5/layers.py` → `Layer0Resonance.compute()`

```python
resonance = encoded_pair.get_resonance()
```

Used in Layer 4 decision logic:

```python
# Entailment: Negative divergence AND high resonance
elif divergence < -0.08 and resonance > 0.50:
    predict = ENTAILMENT
```

## Impact

### Before Resonance Promotion
- Entailment recall: 23.8%
- Entailment had no strong geometric signature

### After Resonance Promotion
- Entailment recall: **39.6%** (+15.8% improvement!)
- Entailment now uses both divergence AND resonance

## The 2D Phase Diagram

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

## Thresholds

From canonical fingerprints:
- **Entailment**: `resonance > 0.50` (mean - 1 std ≈ 0.48, rounded to 0.50)
- **Neutral**: `0.45 < resonance < 0.70` (mid-range)

## Status

✅ **Established**: Resonance as second axis
✅ **Implemented**: Used in Layer 4 decision logic
✅ **Verified**: Entailment recall improved significantly
✅ **Working**: 2D phase diagram operational

## Related Laws

- **Divergence Law**: First axis (push/pull)
- **Phase Classification Law**: Uses both axes to classify phases

## Notes

- Resonance separation is small (0.0378) but adequate
- May need boosting if separation weakens with more data
- Could use `cold_density` (which incorporates resonance) as alternative y-axis

## References

- Implementation: `experiments/nli_v5/layers.py` → `Layer4Decision`
- Fingerprints: `experiments/nli_v5/physics_fingerprints.json`
- Analysis: `experiments/nli_v5/PHYSICS_ANALYSIS_CONFIRMED.md`

