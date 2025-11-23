# Livnium NLI Phase Diagram

## The 2D Phase Space

Livnium v5 implements a **three-phase field theory** over language pairs, with physically interpretable regions defined by order parameters.

## Order Parameters

### X-Axis: Divergence
- **Range**: -1.0 to +1.0
- **Meaning**: Push apart (positive) vs pull together (negative)
- **Law**: `divergence = 0.38 - alignment`

### Y-Axis: Resonance
- **Range**: 0.0 to 1.0 (typically)
- **Meaning**: How strongly sentences share geometric structure
- **Source**: Raw geometric similarity from chain structure

## The Three Phases

### 1. Contradiction (Push Apart)
**Region**: `divergence > 0.02`

**Physics**: Vectors push apart → contradiction

**Characteristics**:
- Positive divergence (push)
- Mid-range resonance (0.46-0.70)
- Strong far_attraction

**Performance**: ~47-57% recall (strong signal)

### 2. Entailment (Pull Inward + Shared Basin)
**Region**: `divergence < -0.08 AND resonance > 0.50`

**Physics**: Vectors pull inward **AND** share strong structure → entailment

**Characteristics**:
- Negative divergence (pull)
- High resonance (>0.50)
- Strong cold_attraction

**Performance**: ~40% recall (improved from 24%)

### 3. Neutral (Balanced Forces)
**Region**: `|divergence| < 0.12` (needs explicit balance band)

**Physics**: Forces cancel → neutral

**Characteristics**:
- Near-zero divergence (balanced)
- Mid-range resonance (0.45-0.70)
- Balanced attractions (`|cold - far| < threshold`)

**Performance**: ~25-40% recall (needs tuning)

## Visual Phase Diagram

```
        High Resonance (0.5+)
              |
              |  E (Entailment)
              |  (negative div + high res)
              |  ●●●●●●●●●●●
              |  ●●●●●●●●●●●
              |
    ----------+---------- Divergence
              |  (push/pull)
              |
    C (Contradiction)  |  N (Neutral)
    (positive div)     |  (near-zero div)
    ●●●●●●●●●●●        |  ●●●●●●●●●●●
              |
        Low Resonance (<0.5)
```

## Phase Boundaries

### Contradiction Boundary
- **Strong**: `divergence > 0.10` (clearly positive)
- **Weak**: `divergence > 0.02` (slightly positive)

### Entailment Boundary
- **Strong**: `divergence < -0.10 AND resonance > 0.55`
- **Weak**: `divergence < -0.08 AND resonance > 0.50`

### Neutral Boundary
- **Current**: `|divergence| < 0.12`
- **Proposed**: `|divergence| < 0.15 AND |cold_attraction - far_attraction| < 0.15`

## Decision Priority

1. **Strong C first**: If divergence clearly positive → contradiction
2. **Strong E next**: If strongly negative + high resonance → entailment
3. **Then neutral**: If both E and C are weak and balance criteria match
4. **Fallback**: Force-based decision for edge cases

## Canonical Fingerprints

From golden labels (debug mode):

| Phase | Divergence | Resonance | Cold Attraction | Far Attraction |
|-------|------------|-----------|-----------------|----------------|
| **E** | -0.1188 ± 0.1656 | 0.6186 ± 0.1369 | 0.7180 ± 0.1036 | 0.5897 ± 0.0718 |
| **C** | -0.0871 ± 0.1480 | 0.5808 ± 0.1201 | 0.7070 ± 0.0904 | 0.5866 ± 0.0696 |
| **N** | -0.0883 ± 0.1539 | 0.5853 ± 0.1262 | 0.7031 ± 0.0937 | 0.5938 ± 0.0704 |

**Note**: Contradiction divergence should be positive in normal mode (+0.1276), but debug mode shows negative due to different sample distribution.

## Performance

### Current Performance (~40% accuracy)
- **Contradiction**: 47-57% recall (strong)
- **Entailment**: 40% recall (improved from 24%)
- **Neutral**: 25-40% recall (needs tuning)

### Expected After Neutral Tuning
- **Contradiction**: 50-60% recall
- **Entailment**: 45-50% recall
- **Neutral**: 40-50% recall
- **Overall**: 45-50% accuracy

## Status

✅ **Established**: Contradiction region (positive divergence)
✅ **Established**: Entailment region (negative divergence + high resonance)
⚠️ **In Progress**: Neutral region (needs explicit balance band)

## References

- Divergence Law: `divergence_law.md`
- Resonance Law: `resonance_law.md`
- Phase Classification: `phase_classification_law.md`
- Neutral Phase: `neutral_phase_law.md`

