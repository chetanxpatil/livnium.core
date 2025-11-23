# Pattern Comparison Results: Debug Mode vs Normal Mode

## Summary

Comparison of geometric patterns between debug mode (golden labels) and normal mode reveals critical issues with divergence computation.

## Key Findings

### 1. **CRITICAL: Contradiction Divergence is Wrong**

**Problem**: Contradiction examples produce **NEGATIVE divergence** in both modes, but they should produce **POSITIVE divergence**.

- Debug mode: -0.3115 (should be positive) ❌
- Normal mode: -0.2441 (should be positive) ❌

**Root Cause**: The divergence formula `-alignment` produces negative values even for contradiction cases with low positive alignment (e.g., alignment=0.3 → divergence=-0.3).

**Fix Applied**: Changed formula to `0.5 - alignment`:
- High alignment (0.8) → divergence = -0.3 (negative, convergence, E) ✓
- Low alignment (0.3) → divergence = 0.2 (positive, divergence, C) ✓
- Negative alignment (-0.5) → divergence = 1.0 (positive, divergence, C) ✓

### 2. **Entailment Divergence is Correct (but weak)**

- Debug mode: -0.3548 (negative, correct) ✓
- Normal mode: -0.2802 (negative, correct but weaker) ⚠️

The normal mode produces less negative divergence, indicating geometry needs strengthening.

### 3. **Forces Not Recorded Correctly in Debug Mode**

**Problem**: Debug mode forces don't match expected values:
- Expected: E(cold=0.7, far=0.2), C(cold=0.2, far=0.7), N(cold=0.33, far=0.33)
- Actual debug: E(cold=0.475, far=0.331), C(cold=0.473, far=0.331), N(cold=0.468, far=0.333)

**Root Cause**: Debug forces were set as local variables but not included in the return dict, so old forces from Layer3 were returned instead.

**Fix Applied**: Explicitly include debug forces in Layer4 return dict.

### 4. **Geometric Signals Are Consistent**

Resonance, divergence, and convergence values are similar between modes (as expected), confirming:
- Geometry computation is consistent
- The issue is in how divergence is computed, not in signal propagation

## Comparison Table

| Signal | Entailment (Debug) | Entailment (Normal) | Contradiction (Debug) | Contradiction (Normal) |
|--------|-------------------|---------------------|----------------------|------------------------|
| Resonance | 0.5953 ± 0.1484 | 0.5492 ± 0.1582 | 0.5532 ± 0.1266 | 0.5097 ± 0.1377 |
| **Divergence** | **-0.3548** ✓ | **-0.2802** ✓ | **-0.3115** ❌ | **-0.2441** ❌ |
| Convergence | 0.3548 ± 0.2160 | 0.2802 ± 0.2338 | 0.3115 ± 0.1868 | 0.2441 ± 0.2033 |
| Cold Force | 0.475 (should be 0.7) | 0.441 | 0.473 (should be 0.2) | 0.439 |
| Far Force | 0.331 (should be 0.2) | 0.344 | 0.331 (should be 0.7) | 0.343 |

## Fixes Applied

1. **Divergence Formula**: Changed from `-alignment` to `0.5 - alignment`
   - Ensures low alignment produces positive divergence (contradiction)
   - High alignment produces negative divergence (entailment)

2. **Debug Forces**: Explicitly include forces in Layer4 return dict
   - Ensures pattern learner records correct debug forces

3. **Cross-word Weight**: Reduced from 30% to 15%
   - Reduces noise from comparing different words

## Expected Improvements

After fixes:
- Contradiction divergence should be **positive** (not negative)
- Debug mode forces should match expected values (0.7, 0.2, 0.1)
- Normal mode accuracy should improve from ~36% toward higher values

## Next Steps

1. Re-run comparison to verify fixes
2. Check if contradiction divergence is now positive
3. Verify debug forces match expected values
4. Monitor accuracy improvement in normal mode

## How to Re-run Comparison

```bash
# Run comparison (will train in both modes)
python3 experiments/nli_v5/compare_patterns.py --samples 1000

# Or use existing pattern files
python3 experiments/nli_v5/compare_patterns.py \
    --debug-file experiments/nli_v5/patterns_debug.json \
    --normal-file experiments/nli_v5/patterns_normal.json
```

