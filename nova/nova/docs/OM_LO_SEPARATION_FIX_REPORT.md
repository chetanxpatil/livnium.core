# OM/LO Separation Fix: Why It Should Work (Or Not)

## Executive Summary

**Problem**: SNLI accuracy stuck at ~32% (random baseline) because alignment ≈ 0.976 for all label types (entailment, contradiction, neutral), making divergence constant (~-0.596) and impossible to distinguish E/C/N.

**Root Cause**: OM (premise) and LO (hypothesis) were computed from the same geometry system (just reset between calls), causing them to collapse into nearly identical states.

**Fix**: Compute OM and LO from completely independent, fresh geometry instances for each premise/hypothesis pair.

**Expected Outcome**: Alignment should now vary (0.7-0.99 range), divergence should separate E/C/N, and accuracy should jump to 45-55% (physics mode).

---

## 1. Why This Should Work

### 1.1 The Physics Is Correct

The divergence law `divergence = 0.38 - alignment` is mathematically sound and aligns with Livnium's core laws. The problem was not the law itself, but the inputs to the law.

### 1.2 Independent Geometries Create Genuine Separation

**Before (Broken)**:
```
premise → geometry → signature_A → reset → hypothesis → same_geometry → signature_B
                                                              ↑
                                                      Same collapse basin
                                                      → signature_B ≈ signature_A
                                                      → alignment ≈ 1.0
```

**After (Fixed)**:
```
premise → fresh_geometry_1 → signature_A (OM)
hypothesis → fresh_geometry_2 → signature_B (LO)
                                                      ↑
                                              Different collapse basins
                                              → signature_B ≠ signature_A
                                              → alignment varies (0.7-0.99)
```

### 1.3 Evidence From Training Logs

The training logs showed:
- **All labels**: alignment ≈ 0.976, divergence ≈ -0.596
- **No separation**: E/C/N distributions completely overlap

This is the smoking gun: if premise and hypothesis were genuinely different, we would see:
- **Entailment**: High alignment (0.95-0.99) → negative divergence
- **Contradiction**: Low alignment (0.5-0.7) → positive divergence  
- **Neutral**: Medium alignment (0.8-0.9) → near-zero divergence

The fact that all three had identical alignment means the geometries were collapsing to the same state.

### 1.4 The Fix Addresses The Core Issue

By creating completely independent geometries:
1. **Premise** collapses into its own unique basin (OM direction)
2. **Hypothesis** collapses into its own unique basin (LO direction)
3. **Alignment** = dot(OM, LO) now reflects genuine semantic relationship
4. **Divergence** = 0.38 - alignment now varies meaningfully

---

## 2. Why It Might Not Work

### 2.1 Signature Similarity Despite Independent Geometries

**Risk**: Even with independent geometries, premise and hypothesis might still produce similar signatures if:
- The sentences are semantically similar (e.g., "A man walks" vs "A person walks")
- The collapse process converges to similar basins regardless of input
- The SW distribution is dominated by sentence length rather than meaning

**Mitigation**: 
- The LO tilt (epsilon=0.05) adds angular variation
- Fracture metric captures geometric inconsistency
- If this fails, we may need to use different collapse strategies for premise vs hypothesis

### 2.2 The Divergence Law Constant (0.38) May Be Wrong

**Risk**: The constant `0.38` in `divergence = 0.38 - alignment` was calibrated from the broken data (constant alignment). If the true relationship is different, the law won't work.

**Mitigation**:
- After retraining, check if divergence distributions show the expected ordering: `E < N < C`
- If not, we may need to recalibrate the constant or use a different functional form

### 2.3 Collapse Steps May Not Be Sufficient

**Risk**: With only 12 collapse steps, the geometries might not fully converge to their basins, leading to noisy signatures.

**Mitigation**:
- Increase collapse_steps to 20-30 if needed
- Monitor signature stability across multiple runs

### 2.4 The Lattice Size (5×5×5) May Be Too Small

**Risk**: 125 dimensions might not provide enough resolution to distinguish subtle semantic differences between premise and hypothesis.

**Mitigation**:
- If separation is still poor, try 7×7×7 (343 dimensions)
- Monitor signature variance to see if we're hitting resolution limits

### 2.5 Token Hashing May Create Collisions

**Risk**: The MD5-based token hashing might map different tokens to the same geometric coordinates, causing information loss.

**Mitigation**:
- Check token collision rates
- Consider using a more sophisticated hashing scheme if collisions are high

---

## 3. Success Criteria

### 3.1 Divergence Statistics Must Show Separation

**After retraining, we should see**:

```
ENTAILMENT:
  Divergence: mean ~ -0.6, std ~ 0.1
  Alignment: mean ~ 0.98, std ~ 0.02

NEUTRAL:
  Divergence: mean ~ 0.0, std ~ 0.1
  Alignment: mean ~ 0.85, std ~ 0.05

CONTRADICTION:
  Divergence: mean ~ +0.6, std ~ 0.1
  Alignment: mean ~ 0.5, std ~ 0.1
```

**Critical Check**: The ordering must be `div_mean(E) < div_mean(N) < div_mean(C)`

### 3.2 Accuracy Must Exceed Random Baseline

- **Physics Mode**: Should jump from ~32% to 45-55%
- **Cluster+Grammar Mode**: Should improve from ~33% to 35-40%

If physics mode improves but cluster+grammar doesn't, the issue is in the unsupervised pipeline, not the physics.

### 3.3 Alignment Must Vary Meaningfully

- **Before**: alignment std ≈ 0.003 (essentially constant)
- **After**: alignment std should be > 0.05 (meaningful variation)

---

## 4. Diagnostic Steps

### Step 1: Retrain and Check Divergence Stats

```bash
python3 nova/training/train_snli_phase1.py \
  --snli-train nova/data/snli/snli_1.0_train.jsonl \
  --max-samples 10000 \
  --lattice-size 5 \
  --collapse-steps 12 \
  --output-dir nova/model/snli_phase1_om_lo_fixed
```

**Look for**: Divergence statistics showing separation (see 3.1)

### Step 2: Test Physics Mode

```bash
python3 nova/chat/test_snli_phase1.py \
  --model-dir nova/model/snli_phase1_om_lo_fixed \
  --max-samples 1000 \
  --physics
```

**Look for**: Accuracy > 40% (ideally 45-55%)

### Step 3: Analyze Alignment Distribution

Add debug output to see alignment distribution:
- Plot histogram of alignment values per label
- Check if distributions overlap or are separated

### Step 4: If Still Failing, Check Signature Similarity

Compute cosine similarity between premise and hypothesis signatures directly:
- If similarity is still > 0.95 for all pairs, the geometries are still too similar
- May need different collapse strategies or impulse scales

---

## 5. Fallback Strategies (If Fix Doesn't Work)

### 5.1 Different Collapse Strategies

Use different collapse parameters for premise vs hypothesis:
- Premise: `collapse_steps=15`, `impulse_scale=0.1`
- Hypothesis: `collapse_steps=10`, `impulse_scale=0.15`

This forces them into different basins.

### 5.2 Principal Component Analysis (PCA)

Instead of using raw SW distributions, extract principal directions:
- Premise: `OM = first_PC(premise_SW)`
- Hypothesis: `LO = first_PC(hypothesis_SW)`

This focuses on the most significant geometric patterns.

### 5.3 Multi-Scale Signatures

Combine signatures at different collapse depths:
- Premise: `[sig_at_step_5, sig_at_step_10, sig_at_step_15]`
- Hypothesis: `[sig_at_step_5, sig_at_step_10, sig_at_step_15]`

This captures temporal evolution differences.

### 5.4 Recalibrate Divergence Law

If alignment varies but divergence doesn't separate E/C/N, the law constant may be wrong:
- Try: `divergence = 0.5 - alignment`
- Or: `divergence = 0.3 - alignment`
- Or: Use a learned function instead of a constant

---

## 6. Conclusion

**Why It Should Work**:
1. The physics (divergence law) is correct
2. Independent geometries ensure genuine separation
3. Training logs show the problem is identical signatures, not the law
4. The fix directly addresses the root cause

**Why It Might Not Work**:
1. Signatures might still be too similar despite independent geometries
2. The divergence law constant (0.38) may need recalibration
3. Collapse process might converge to similar basins regardless of input
4. Lattice resolution might be insufficient

**Next Steps**:
1. Retrain with the fix
2. Check divergence statistics for separation
3. Test physics mode accuracy
4. If still failing, try fallback strategies

**Expected Timeline**:
- Retraining: ~30-60 minutes
- Testing: ~5-10 minutes
- Analysis: ~15 minutes

**Success Probability**: 70-80% (the fix addresses the core issue, but there may be secondary problems)

---

## 7. Technical Details

### 7.1 Implementation Changes

**File**: `nova/core/text_to_geometry.py`

**New Method**: `_get_signature_from_fresh_geometry()`
- Creates completely fresh `LivniumCoreSystem` instance
- Applies symmetry breaking only to hypothesis (LO)
- Returns signature from independent geometry

**Modified Method**: `get_signature_with_divergence()`
- Calls `_get_signature_from_fresh_geometry()` for both premise and hypothesis
- Ensures no state leakage between them
- Increased LO tilt epsilon from 0.02 to 0.05 for more variation

### 7.2 Memory Considerations

Creating fresh geometries for each pair increases memory usage:
- **Before**: 1 geometry instance (reused)
- **After**: 2 geometry instances per pair (premise + hypothesis)

For 10,000 training samples: ~20,000 geometry instances created (but garbage collected immediately).

**Impact**: Minimal - geometries are small (125 cells each) and Python GC handles cleanup.

### 7.3 Performance Impact

**Before**: 
- Premise: inject → collapse → extract → reset
- Hypothesis: inject → collapse → extract → reset

**After**:
- Premise: create_geometry → inject → collapse → extract → destroy
- Hypothesis: create_geometry → inject → collapse → extract → destroy

**Overhead**: ~2x geometry creation time per pair (negligible compared to collapse time)

**Total Impact**: < 5% slowdown (geometry creation is fast, collapse is slow)

---

## 8. References

- **Divergence Law**: `divergence = 0.38 - alignment` (from Livnium core laws)
- **Training Logs**: Show constant alignment ≈ 0.976 for all labels
- **User Analysis**: Identified OM/LO identical problem
- **Fix Location**: `nova/core/text_to_geometry.py::get_signature_with_divergence()`

