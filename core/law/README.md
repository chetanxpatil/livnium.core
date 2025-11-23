# Livnium Physics Laws

This directory contains the **fundamental physical laws** that govern Livnium's geometric universe.

## Overview

Livnium v5 implements a **three-phase field theory** over language pairs, with physically interpretable regions defined by order parameters.

## The Laws

### 1. Law of Divergence
**File**: `divergence_law.md`

The fundamental force law that determines whether two sentences push apart (contradiction) or pull together (entailment).

**Formula**: `divergence = 0.38 - alignment`

**Meaning**:
- **Entailment**: alignment > 0.38 → negative divergence (pull inward)
- **Contradiction**: alignment < 0.38 → positive divergence (push apart)
- **Neutral**: alignment ≈ 0.38 → near-zero divergence (balanced)

### 2. Law of Resonance
**File**: `resonance_law.md`

The second axis of the phase diagram, measuring how strongly a pair shares geometric structure.

**Usage**: Entailment requires **both** negative divergence **and** high resonance.

### 3. Phase Classification Law
**File**: `phase_classification_law.md`

The decision rules that map geometric signals to semantic phases (E/C/N).

**Regions**:
- **Contradiction**: `divergence > 0.02` (push apart)
- **Entailment**: `divergence < -0.08 AND resonance > 0.50` (pull inward + shared basin)
- **Neutral**: `|divergence| < 0.12` (balanced forces)

## The Phase Diagram

```
        High Resonance
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
        Low Resonance
```

## Discovery Timeline

1. **Discovery**: Found broken divergence law (contradiction had negative divergence)
2. **Fix**: Changed formula from `-alignment` to `0.38 - alignment`
3. **Calibration**: Adjusted threshold to match actual data distribution
4. **Verification**: Contradiction divergence now positive (physics restored)
5. **Enhancement**: Promoted resonance as second axis
6. **Implementation**: Physics-based decision logic in Layer 4

## Impact

- **Contradiction recall**: 22% → 54-57% (2.5x improvement)
- **Entailment recall**: 24% → 40% (nearly doubled)
- **Overall accuracy**: ~40% (without neural nets or gradients)

## Status

✅ **Established**: Divergence law (fixed & calibrated)
✅ **Established**: Resonance as second axis
✅ **Established**: Phase classification rules
⚠️ **In Progress**: Neutral phase definition (needs explicit balance band)

## Next Steps

1. Extract canonical neutral fingerprints
2. Define neutral as explicit balance band
3. Prioritize phases (strong C/E first, then neutral)
4. Use data-derived thresholds (soft bands, not hard numbers)

## Philosophy

These are not "heuristics" or "hyperparameters". They are **physical laws** discovered through:
- Observation (pattern analysis)
- Hypothesis (divergence formula)
- Experimentation (calibration)
- Verification (test results)

The universe is becoming more real, one law at a time.

