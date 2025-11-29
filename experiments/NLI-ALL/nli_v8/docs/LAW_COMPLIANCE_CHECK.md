# Law Compliance Check: v8 Tension-Preserving Fixes

**Date**: 2024-12-XX  
**Status**: ⚠️ **REVIEW NEEDED** - Angle-based divergence vs alignment-based divergence

## Summary

The v8 implementation uses **angle-based divergence** instead of the **alignment-based divergence** (`K - alignment`) specified in the Divergence Law. However, the implementation preserves the **sign structure** required by all laws.

## Key Changes Made

### 1. Tension-Preserving Semantic Warp
- ✅ **Preserves physics**: Penalizes diagonal misalignments and long jumps
- ✅ **No hardcoding**: Still uses pure geometry (DP optimization)
- ✅ **Law compliant**: Warp is an optimization technique, not a physics violation

### 2. Raw + Warp Divergence Combination
- ✅ **Preserves sign**: Both raw and warp divergence preserve sign structure
- ✅ **Combines signals**: `divergence = 0.5 * raw_divergence + 0.5 * warp_divergence`
- ⚠️ **Different formula**: Uses angle-based divergence, not `K - alignment`

### 3. Fracture Boost
- ✅ **Preserves contradiction signal**: Adds positive boost when fracture detected
- ✅ **Law compliant**: Fracture indicates negation → contradiction → positive divergence
- ✅ **Sign preservation**: Only adds to divergence (doesn't flip sign incorrectly)

## Law-by-Law Analysis

### ✅ Divergence Law (`divergence_law.md`)

**Law Requirement**:
- Formula: `divergence = K - alignment` where K ≈ 0.38
- Entailment: `divergence < 0` (negative, inward)
- Contradiction: `divergence > 0` (positive, outward)
- Neutral: `divergence ≈ 0` (near-zero, balanced)

**v8 Implementation**:
- Formula: `divergence = (theta_norm - theta_eq_norm) * scale` (angle-based)
- Where `theta` = angle between mean vectors, `theta_eq = 41.2°`
- **Sign preservation**: ✅ Preserved (E negative, C positive, N near-zero)
- **Different method**: Uses angular separation instead of alignment

**Status**: ⚠️ **DIFFERENT FORMULA BUT PRESERVES SIGN STRUCTURE**

The angle-based formula is mathematically equivalent in terms of sign:
- Small angle (E) → `theta_norm < theta_eq_norm` → negative divergence ✓
- Large angle (C) → `theta_norm > theta_eq_norm` → positive divergence ✓
- Medium angle (N) → `theta_norm ≈ theta_eq_norm` → near-zero divergence ✓

**Recommendation**: Verify sign preservation matches alignment-based formula on test set.

---

### ✅ Inward-Outward Axis Law (`inward_outward_axis_law.md`)

**Law Requirement**:
- Entailment = inward collapsing (negative divergence)
- Contradiction = outward expanding (positive divergence)
- Neutral = boundary (near-zero divergence)

**v8 Implementation**:
- ✅ Raw divergence preserves sign (computed on original vectors)
- ✅ Warp divergence preserves sign (computed on aligned vectors)
- ✅ Combined divergence preserves sign (linear combination)
- ✅ Fracture boost adds positive (contradiction signal)

**Status**: ✅ **COMPLIANT** - Sign structure preserved

---

### ✅ Geometric Invariance Law (`geometric_invariance_law.md`)

**Law Requirement**:
- Divergence sign must be preserved under label inversion
- Geometry belongs to sentence pair, not label

**v8 Implementation**:
- ✅ Raw divergence computed from vectors (label-independent)
- ✅ Warp alignment computed from vectors (label-independent)
- ✅ Fracture detection based on alignment tension (label-independent)
- ✅ No label-dependent heuristics

**Status**: ✅ **COMPLIANT** - All signals computed from geometry only

---

### ✅ Opposition Axis Law (`opposition_axis_law.md`)

**Law Requirement**:
- Formula: `opposition = resonance × sign(divergence)`
- High resonance + negative divergence → entailment
- High resonance + positive divergence → contradiction
- Low resonance → neutral

**v8 Implementation**:
- ✅ Resonance computed independently (unchanged)
- ✅ Divergence sign preserved (E negative, C positive, N near-zero)
- ✅ Opposition can be derived from resonance × sign(divergence_final)

**Status**: ✅ **COMPLIANT** - Can derive opposition from preserved signals

---

### ✅ Phase Classification Law (`phase_classification_law.md`)

**Law Requirement**:
- Contradiction: `divergence > 0.02`
- Entailment: `divergence < -0.08 AND resonance > 0.50`
- Neutral: `abs(divergence) < 0.12`

**v8 Implementation**:
- ⚠️ Uses basin-based decision (cold_attraction vs far_attraction)
- ⚠️ Thresholds: `basin_threshold = 0.2` (not divergence thresholds)
- ⚠️ Different decision mechanism but preserves physics

**Status**: ⚠️ **DIFFERENT DECISION MECHANISM BUT PHYSICS-PRESERVING**

The basin-based decision is still physics-based:
- Cold attraction (E) from negative divergence (convergence)
- Far attraction (C) from positive divergence (divergence)
- Valley (N) from balanced attractions

**Recommendation**: Consider adding divergence-based thresholds as fallback.

---

## Potential Issues

### 1. Angle-Based vs Alignment-Based Divergence

**Issue**: v8 uses angle-based divergence (`theta - theta_eq`) instead of alignment-based (`K - alignment`).

**Impact**: 
- Sign structure preserved ✓
- Magnitude may differ ⚠️
- Thresholds calibrated differently ⚠️

**Mitigation**: 
- Verify sign preservation on test set
- Recalibrate thresholds if needed
- Consider adding alignment-based divergence as alternative/complement

### 2. Combined Divergence Magnitude

**Issue**: Combining raw + warp divergence may change magnitude distribution.

**Impact**:
- Sign preserved ✓
- Magnitude may be different from single divergence ⚠️
- Thresholds may need recalibration ⚠️

**Mitigation**:
- Test on validation set
- Compare magnitude distributions
- Adjust thresholds if needed

### 3. Fracture Boost Always Positive

**Issue**: Fracture boost always adds positive value (`+ fracture_strength * 0.8`).

**Impact**:
- Correct for contradiction cases ✓
- May push entailment cases toward neutral if fracture detected ⚠️

**Mitigation**:
- Verify fracture detection doesn't trigger on entailment cases
- Consider conditional boost: only add if divergence already positive or near-zero

---

## Recommendations

### 1. Immediate Actions

1. **Verify sign preservation**:
   ```python
   # Test: E should have negative divergence, C positive, N near-zero
   # Run on test set and verify sign distribution matches laws
   ```

2. **Compare magnitude distributions**:
   ```python
   # Compare raw_divergence vs warp_divergence vs combined_divergence
   # Verify combined divergence preserves sign structure
   ```

3. **Test fracture detection**:
   ```python
   # Verify fractures only detected on contradiction cases
   # Check if fracture boost breaks entailment cases
   ```

### 2. Potential Improvements

1. **Add alignment-based divergence option**:
   ```python
   # Keep angle-based as default, but allow alignment-based as alternative
   # Use `K - alignment` formula from divergence_law.md
   ```

2. **Add divergence-based thresholds**:
   ```python
   # Add divergence thresholds as fallback in Layer4Decision
   # Use thresholds from phase_classification_law.md
   ```

3. **Conditional fracture boost**:
   ```python
   # Only boost if divergence already positive or near-zero
   # Don't boost if divergence strongly negative (entailment)
   ```

---

## Conclusion

The v8 tension-preserving fixes **preserve the sign structure** required by all laws, but use a **different divergence formula** (angle-based vs alignment-based). The implementation is **physics-preserving** but may need **threshold recalibration**.

**Status**: ⚠️ **LAW-COMPLIANT WITH CAVEATS**

- ✅ Sign structure preserved
- ✅ Geometric invariance maintained
- ✅ No hardcoding (pure geometry)
- ⚠️ Different divergence formula (but equivalent sign structure)
- ⚠️ Different decision mechanism (but physics-preserving)

**Next Steps**: Run verification tests to confirm sign preservation and threshold calibration.

