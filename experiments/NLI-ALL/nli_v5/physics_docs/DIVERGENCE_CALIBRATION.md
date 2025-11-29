# Divergence Calibration: The 0.5 Threshold Issue

## Discovery

After fixing the divergence formula to `0.5 - alignment`, we discovered:

### The Problem
- **Entailment**: Mean alignment = 0.4047 (below 0.5!) → Gets positive divergence ❌
- **Contradiction**: Mean alignment = 0.3724 (below 0.5) → Gets positive divergence ✓
- **Neutral**: Mean alignment = 0.3686 (below 0.5) → Gets positive divergence

**Result**: 70% of entailment cases have alignment < 0.5, so they're being treated as contradictions!

## Root Cause

The **0.5 threshold** assumes word vectors have high cosine similarity for entailment. But actual alignments are:
- Entailment: Mean 0.40 (range: -0.34 to 0.92)
- Contradiction: Mean 0.37 (range: -0.17 to 0.91)
- Neutral: Mean 0.37 (range: -0.20 to 0.91)

**The threshold is too high** for the actual distribution of alignments.

## Solutions

### Option 1: Lower the Threshold
Change from `0.5` to `0.3` or `0.35`:
```python
divergence = 0.3 - alignment  # or 0.35 - alignment
```

This would give:
- Entailment (align=0.40) → divergence = -0.10 (negative) ✓
- Contradiction (align=0.37) → divergence = -0.07 (still negative, but closer to zero)

**Problem**: Contradiction might still be negative.

### Option 2: Use Relative Threshold
Use the mean alignment as threshold:
```python
# Compute mean alignment per batch
mean_alignment = np.mean([alignment for all examples])
divergence = mean_alignment - alignment
```

### Option 3: Use Percentile-Based Threshold
Use median or 33rd percentile:
```python
# For entailment: use 33rd percentile as threshold
# For contradiction: use 67th percentile as threshold
```

### Option 4: Boost Alignment Values
Multiply alignment by a factor before computing divergence:
```python
boosted_alignment = alignment * 1.2  # Boost by 20%
divergence = 0.5 - boosted_alignment
```

### Option 5: Use Signed Distance from Mean
```python
mean_align = 0.38  # Approximate mean
divergence = mean_align - alignment
```

This gives:
- Entailment (align=0.40) → divergence = -0.02 (slightly negative) ✓
- Contradiction (align=0.37) → divergence = +0.01 (slightly positive) ✓
- Neutral (align=0.37) → divergence = +0.01 (near zero) ✓

## Recommended Fix

**Use Option 5** (signed distance from mean) with adaptive mean:

```python
# In Layer0Resonance
divergence = self.mean_alignment - alignment

# Update mean_alignment during training
self.mean_alignment = 0.9 * self.mean_alignment + 0.1 * alignment
```

Or simpler: **Use 0.38 as threshold** (close to actual mean):

```python
divergence = 0.38 - alignment
```

This gives:
- Entailment (align=0.40) → divergence = -0.02 (negative) ✓
- Contradiction (align=0.37) → divergence = +0.01 (positive) ✓
- Neutral (align=0.37) → divergence = +0.01 (near zero) ✓

## Current Status

- ✅ Contradiction divergence is now **positive** (0.1276)
- ❌ Entailment divergence is **positive** (0.0953) - should be negative
- ⚠️  The 0.5 threshold doesn't match actual alignment distribution

## Next Steps

1. **Lower threshold** from 0.5 to 0.38 (or use adaptive mean)
2. **Re-run training** and verify divergence signs
3. **Monitor accuracy** - should improve further
4. **Fine-tune** threshold based on validation set performance

## The Physics

The threshold represents the **equilibrium point** where convergence and divergence balance. Currently set at 0.5 (perfectly neutral), but actual word vector geometry has a different equilibrium around 0.38.

Adjusting the threshold is like **recalibrating the zero point** of the divergence field.

