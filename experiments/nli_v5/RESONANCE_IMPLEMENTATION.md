# Resonance Implementation: Physics-Based Decision Logic

## What Was Implemented

Promoted **resonance as a first-class citizen** alongside divergence in Layer 4 decision logic, using canonical fingerprints from golden labels.

## The Physics-Based Decision Rules

### Rule 1: Contradiction (Strong Positive Divergence)
```python
if divergence > 0.05:
    predict = CONTRADICTION
```
- **Primary signal**: Positive divergence (push apart)
- **Threshold**: 0.05 (from fingerprints)
- **Confidence**: Based on divergence strength

### Rule 2: Entailment (Negative Divergence AND High Resonance)
```python
elif divergence < -0.05 AND resonance > 0.48:
    predict = ENTAILMENT
```
- **Requires BOTH signals**:
  - Negative divergence (convergence, pull inward)
  - High resonance (similarity, resonance > 0.48)
- **Confidence**: Combined strength of both signals

### Rule 3: Neutral (Near-Zero Divergence)
```python
elif abs(divergence) < 0.15:
    predict = NEUTRAL
```
- **Primary signal**: Near-zero divergence (balanced forces)
- **Optional check**: Resonance in mid-range (0.46-0.71)
- **Confidence**: Based on force balance

### Rule 4: Fallback (Force-Based)
- For edge cases where physics signals are ambiguous
- Uses attraction ratios and force comparisons
- Resonance as tiebreaker

## Thresholds (From Canonical Fingerprints)

- **Divergence thresholds**:
  - Contradiction: d > 0.05 (positive)
  - Entailment: d < -0.05 (negative)
  - Neutral: |d| < 0.15 (near zero)

- **Resonance thresholds**:
  - Entailment: r > 0.48 (high, mean - 1 std)
  - Neutral: 0.46 < r < 0.71 (mid-range)

## Initial Results

### Before (Force-Based Only)
- Entailment recall: 23.8%
- Contradiction recall: 56.9%
- Neutral recall: 40.8%
- Overall accuracy: 40.4%

### After (Physics-Based with Resonance)
- Entailment recall: **38.0%** ⬆️ (+14.2%)
- Contradiction recall: 50.0% (slight drop)
- Neutral recall: 24.8% (needs tuning)
- Overall accuracy: 37.7% (needs tuning)

## Key Improvements

✅ **Entailment recall nearly doubled** (23.8% → 38.0%)
- Resonance axis is working!
- E now uses both divergence AND resonance

⚠️ **Neutral recall dropped** (40.8% → 24.8%)
- Neutral band may be too strict
- May need to adjust thresholds

⚠️ **Contradiction recall slightly down** (56.9% → 50.0%)
- Still strong, but may need threshold tuning

## Next Steps

1. **Tune thresholds** based on validation performance
2. **Adjust neutral band** to improve neutral recall
3. **Fine-tune resonance threshold** for entailment
4. **Run full training** (10k examples) to see final performance

## The Physics

The decision logic now uses a **2D phase diagram**:
- **x-axis**: Divergence (push/pull)
- **y-axis**: Resonance (similarity)

**Three regions**:
1. **Contradiction**: Positive divergence (push apart)
2. **Entailment**: Negative divergence **AND** high resonance (pull inward + similarity)
3. **Neutral**: Near-zero divergence (balanced forces)

This matches the canonical fingerprints from golden labels.

## Files Modified

- `layers.py`: Layer4Decision class
  - Added physics-based thresholds
  - Implemented 2D phase diagram logic
  - Kept fallback to force-based decision

