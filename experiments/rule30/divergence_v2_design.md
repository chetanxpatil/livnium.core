# Divergence V2: Structure-Aware Design

## Problem with V1

Current divergence is constant for all sequences because:
- Self-comparison (premise == hypothesis) → angle = 0
- Divergence = (0 - θ_eq) * scale = constant
- No dependency on actual sequence structure

## Goal: Structure-Aware Divergence

Design divergence that:
- ✅ Varies with sequence structure
- ✅ Captures pattern transitions
- ✅ Reflects geometric differences
- ✅ Enables real invariant hunting
- ✅ Can discover Rule 30-specific properties

## Candidate Divergence Laws

### Candidate 1: Transition-Based Divergence

Measure divergence from **transitions** between consecutive bits:

```
D(s) = (1/n) Σ_i |s[i] - s[i+1]| * geometric_factor(i)
```

Where `geometric_factor` depends on:
- Local pattern context
- Angle changes
- Vector alignment shifts

### Candidate 2: Neighborhood Divergence

Compare **neighboring windows** instead of self-comparison:

```
D(s) = (1/n) Σ_i angle(mean(window_i), mean(window_{i+1}))
```

This captures:
- Local structure changes
- Pattern transitions
- Geometric variations

### Candidate 3: Pattern-Weighted Angular Divergence

Weight divergence by **pattern frequencies**:

```
D(s) = Σ_p freq_p(s) * angle_pattern(p)
```

Where `angle_pattern(p)` is the geometric angle associated with pattern `p`.

## Implementation Plan

1. **Design new divergence function** that depends on sequence structure
2. **Test on diverse sequences** to ensure variation
3. **Re-run Rule 30** to see if invariants appear
4. **Build invariant hunter** for structure-aware divergence
5. **Extract symbolic form** of new divergence law

## Next Steps

- Implement Candidate 1, 2, or 3 (or combination)
- Test that divergence varies across sequences
- Hunt for Rule 30-specific invariants
- Extract exact formula

