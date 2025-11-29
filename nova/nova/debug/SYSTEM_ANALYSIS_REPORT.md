# Nova System Analysis Report: Token Hash Collision Impact

## Executive Summary

**ChatGPT's Claim**: ✅ **VERIFIED**
- Collision rate: **92.06%** (ChatGPT claimed 92.4%)
- Average words per coordinate: **12.60** (ChatGPT calculated ~13)
- Total unique tokens: 1575
- Unique hash coordinates: 125 (5×5×5)

**However**: Collisions at the **token level** don't necessarily mean signatures are identical at the **sentence level**. We need to verify if OM ≈ LO.

---

## 1. Token Hash Collision Confirmed

### Measurement Results

```
Tokens analyzed: 1575
Unique hash coordinates: 125
Collision rate: 92.06%
Average words per coordinate: 12.60
```

### Collision Examples

- Coordinate (4, 0, 3): 'shoes', 'pakistani', 'writting', 'artist', 'umbrellas'
- Coordinate (4, 2, 1): 'sled', 'she', 'forest', 'capital', 'coffeshop'
- Coordinate (0, 3, 3): 'handover', 'beginners', 'next', 'favorite', 'little'

**Impact**: Many semantically different words map to the same geometric coordinate.

---

## 2. How Our System Actually Works

### Architecture

1. **Token Hashing** (`geometric_token_learner.py::token_hash()`)
   - Token → MD5 hash → (x, y, z) coordinate in [0, 4]³
   - Only 125 possible coordinates
   - **Collision rate: 92.06%** ✓

2. **Impulse Injection** (`text_to_geometry.py::inject_sentence()`)
   - Each token adds an impulse to its hashed coordinate
   - Multiple tokens → multiple impulses at different coordinates
   - Impulses accumulate per coordinate

3. **Collapse Process** (`text_to_geometry.py::inject_sentence()`)
   - Geometry rotates and redistributes SW across ALL 125 cells
   - Final signature = SW value at each of the 125 cells
   - **Signature is 125-dimensional, not just hash coordinates**

4. **OM/LO Separation** (`text_to_geometry.py::get_signature_with_divergence()`)
   - OM = normalized SW distribution from premise
   - LO = normalized SW distribution from hypothesis
   - Alignment = dot(OM, LO)

### Key Insight

**Even with 92% token collisions, sentence signatures can still differ** because:
- Different sentences have different token sequences
- Different impulse values (from hash)
- Collapse redistributes SW differently
- Final signature is 125-dimensional continuous vector

---

## 3. The Critical Question: Does Collision Cause OM ≈ LO?

### Hypothesis

**If collisions are the root cause**:
- OM and LO signatures should be nearly identical
- Cosine similarity ≈ 0.99998
- Alignment ≈ 0.976 (constant)
- Divergence ≈ -0.596 (constant)

**If collisions are NOT the root cause**:
- OM and LO signatures should vary
- Cosine similarity ≈ 0.6-0.95 (varies)
- Alignment varies
- Divergence varies

### What We Need to Measure

Run the diagnostic script:

```bash
python3 nova/debug/snli_physics_report.py
```

**Key Sections**:
- **Section 1**: OM vs LO cosine similarity distribution
- **Section 3**: Alignment/divergence distributions per label
- **Section 5**: Physics signal strength (ranges)

---

## 4. Why Increasing Lattice Size Would Help

### Current State (5×5×5 = 125 cells)

- Collision rate: 92.06%
- 12.60 words per coordinate on average
- Many semantically different words share coordinates

### If We Increase to 15×15×15 = 3375 cells

- Collision rate: ~48.7% (1645 / 3375)
- ~0.49 words per coordinate on average
- Much better token separation
- Signatures become 3375-dimensional

### If We Increase to 21×21×21 = 9261 cells

- Collision rate: ~17.8% (1645 / 9261)
- ~0.18 words per coordinate on average
- Most tokens get unique coordinates
- Signatures become 9261-dimensional

### Trade-offs

**Pros**:
- Better token separation
- More distinct signatures
- Potentially better OM/LO separation

**Cons**:
- Larger memory footprint (125 → 3375 or 9261 dimensions)
- Slower computation (more cells to process)
- More training data needed

---

## 5. Alternative Explanations for OM ≈ LO

### If Collisions Are NOT the Root Cause

**Possible causes**:
1. **Collapse too deterministic**: All sentences collapse to similar basins
2. **SW distribution too uniform**: All signatures look similar
3. **Normalization washes out differences**: L2 normalization makes signatures too similar
4. **Insufficient collapse steps**: Geometry doesn't fully converge
5. **Impulse scale too small**: Differences are too subtle

### How to Distinguish

**Run diagnostic Section 2** (Collapse Stability):
- If same text run twice gives cosine ≈ 1.0 → collapse is deterministic
- If cosine varies → collapse is noisy

**Run diagnostic Section 8** (Collapse Evolution):
- If sig(step 0) ≈ sig(step 12) → collapse converges instantly
- If they differ → collapse is meaningful

---

## 6. Recommended Action Plan

### Step 1: Run Full Diagnostic

```bash
python3 nova/debug/snli_physics_report.py
```

**Check**:
- Section 1: OM/LO cosine similarity mean and distribution
- Section 3: Alignment/divergence distributions
- Section 5: Signal strength ranges

### Step 2: Interpret Results

**If cosine similarity ≈ 0.99998**:
- ✅ ChatGPT is right: collisions are the problem
- ✅ Increase lattice size to 15×15×15 or 21×21×21
- ✅ This should fix OM ≈ LO

**If cosine similarity varies (0.6-0.95)**:
- ❌ ChatGPT is wrong: collisions are NOT the problem
- ❌ Problem is elsewhere (collapse, normalization, etc.)
- ❌ Increasing lattice size won't help

### Step 3: Apply Fix Based on Results

**If collisions are the problem**:
```python
# Change in train_snli_phase1.py and test_snli_phase1.py
lattice_size = 15  # or 21
```

**If collisions are NOT the problem**:
- Investigate collapse process
- Check normalization effects
- Adjust impulse scale
- Increase collapse steps

---

## 7. Conclusion

**ChatGPT's Analysis**:
- ✅ **Math is correct**: Collision rate ≈ 92%
- ✅ **Observation is correct**: High collision rate
- ❓ **Conclusion is unverified**: Need to check if this causes OM ≈ LO

**Our System's Reality**:
- Signatures are 125-dimensional SW distributions (not just hash coordinates)
- Even with collisions, different sentences can create different signatures
- The collapse process redistributes SW across all cells

**The Verdict**:
- **We need diagnostic data to know for sure**
- Run `python3 nova/debug/snli_physics_report.py`
- Check Section 1 & 3 results
- If OM/LO cosine ≈ 0.99998 → increase lattice size
- If OM/LO cosine varies → problem is elsewhere

**Next Step**: Run the diagnostic and check the actual OM/LO similarity distribution.

