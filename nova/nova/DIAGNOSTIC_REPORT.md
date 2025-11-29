# Nova Dialogue Engine - Diagnostic Report

## Executive Summary

**Current Status:** ❌ **CRITICAL ISSUES DETECTED**

The chat responses are incoherent ("Nope I !", "Re just a good to the m pretty sure...") due to fundamental configuration problems in the training pipeline.

---

## 🔴 CRITICAL ISSUE #1: Severe Cluster Under-Segmentation

### Problem
- **Training:** 31,429 signatures → **Only 50 clusters**
- **Average:** 1,465 tokens per cluster
- **41 out of 50 clusters** have >1,000 tokens
- **Largest cluster:** 2,104 tokens

### Why This Breaks Everything

With only 50 clusters for 31,429 signatures:
1. **Massive vocabulary overlap:** Each cluster contains ~1,500 unrelated words
2. **No semantic separation:** "hey", "why", "what" all map to the same huge cluster
3. **Grammar intersection fails:** When grammar can't find valid transitions, it falls back to the entire 1,500-word cluster vocabulary → **word salad**
4. **Random word selection:** With 1,500 candidates, the system picks random words

### Evidence from Chat
```
You: hey
Bot: Nope I !  ← Random words from massive cluster

You: why  
Bot: That?  ← Different random words from same cluster

You: what?
Bot: Re just a good to the m pretty sure...  ← Complete word salad
```

### Solution
**Increase clusters to 1,000-3,000:**
```bash
--num-clusters 2000  # Recommended: 2,000 clusters
```

**Rule of thumb:** ~10-30 signatures per cluster for good separation
- 31,429 signatures ÷ 2,000 clusters = ~15 signatures/cluster ✅
- 31,429 signatures ÷ 50 clusters = ~628 signatures/cluster ❌

---

## 🔴 CRITICAL ISSUE #2: Lattice Size Too Small

### Problem
- **Current:** Lattice size 3 → **27 dimensions**
- **Result:** Very limited signature space

### Why This Breaks Everything

1. **Signature collision:** Many different sentences collapse to similar 27-dimensional vectors
2. **Poor clustering:** K-Means can't separate meanings in such low-dimensional space
3. **Limited expressiveness:** 27 dimensions can't capture semantic diversity

### Solution
**Use lattice size 5 or 7:**
```bash
--lattice-size 5  # 125 dimensions (recommended)
--lattice-size 7  # 343 dimensions (better, but slower)
```

**Why 5 is better:**
- 5×5×5 = 125 dimensions (4.6× more expressive)
- Still fast to train
- Better signature separation

---

## 🟡 ISSUE #3: Collapse Steps Mismatch

### Problem
- **Training:** 20 collapse steps
- **Chat:** 12 collapse steps
- **Result:** Different signature distributions

### Why This Matters

Different collapse depths produce different signature patterns:
- 20 steps → More collapsed, smoother signatures
- 12 steps → Less collapsed, more detailed signatures

When chat uses different collapse steps than training, signatures don't match the learned clusters.

### Solution
**Match collapse steps:**
```bash
# Training
--collapse-steps 12

# Chat  
--collapse-steps 12
```

**Or use 15-20 for both** (more stable, but slower)

---

## 🟡 ISSUE #4: Impulse Scale Mismatch

### Problem
- **Training:** `--impulse-scale 0.4` (4× stronger)
- **Chat:** `--impulse-scale 0.1` (default)
- **Result:** Different signature magnitudes

### Why This Matters

Higher impulse scale = stronger geometric perturbations:
- Training with 0.4 → Signatures have larger values
- Chat with 0.1 → Signatures have smaller values
- K-Means clusters learned on 0.4-scale signatures won't match 0.1-scale signatures

### Solution
**Match impulse scales:**
```bash
# Training
--impulse-scale 0.1  # Recommended (stable)

# Chat
--impulse-scale 0.1  # Match training
```

**Note:** 0.4 is too high and can cause unstable signatures. Stick with 0.1.

---

## 🟡 ISSUE #5: Grammar Intersection Failure

### Problem
In `cluster_decoder.py` line 117-123:
```python
# INTERSECTION: Grammar AND Cluster
valid_candidates = [w for w in possible_next_words if w in cluster_vocab]

if not valid_candidates:
    # Fallback: Grammar failed, pick from cluster
    valid_candidates = words  # ← FALLS BACK TO ENTIRE 1,500-WORD CLUSTER!
```

### Why This Breaks Everything

When grammar can't find valid transitions:
1. Falls back to **entire cluster vocabulary** (1,500 words)
2. Picks random words from this massive set
3. Result: **Word salad**

### Evidence
```
You: what?
Bot: Re just a good to the m pretty sure that you can do is a
```
This is the fallback mechanism picking random words from a 1,500-word cluster.

### Solution
**Improve fallback logic:**
1. **Limit fallback vocabulary** to top-N most frequent words in cluster
2. **Add minimum grammar match threshold** before falling back
3. **Use sentence-level patterns** instead of word-level when grammar fails

---

## 🟡 ISSUE #6: No Signature Normalization

### Problem
Signatures are not normalized before clustering:
- Different sentence lengths → Different signature magnitudes
- K-Means clustering on unnormalized data → Clusters based on magnitude, not pattern

### Why This Matters

K-Means is sensitive to scale:
- Long sentences → Large signature values → One cluster
- Short sentences → Small signature values → Another cluster
- **Not based on semantic similarity!**

### Solution
**Add normalization before clustering:**
```python
# In geometric_token_learner.py, learn_clusters():
from sklearn.preprocessing import normalize

# Normalize signatures to unit vectors
signatures_normalized = normalize(signatures, norm='l2', axis=1)
cluster_ids = self.kmeans.fit_predict(signatures_normalized)
```

This makes clustering focus on **patterns**, not magnitudes.

---

## 📊 Current Training Configuration Analysis

### What You Trained With:
```bash
--lattice-size 3          ❌ Too small (27 dims)
--collapse-steps 20       ⚠️  Mismatch with chat (12)
--num-clusters 50         ❌ CRITICAL: Way too few
--impulse-scale 0.4       ⚠️  Too high, mismatch with chat (0.1)
```

### Model Statistics:
- **Signatures:** 31,429
- **Clusters:** 50
- **Avg tokens/cluster:** 1,465
- **Clusters >1000 tokens:** 41 (82% of clusters!)
- **Largest cluster:** 2,104 tokens

### Result:
**82% of clusters are massive word bags** → Word salad generation

---

## ✅ RECOMMENDED FIXES

### 1. Retrain with Correct Parameters

```bash
python3 nova/training/train_text_to_geometry.py \
  --csv nova/data/empathetic_train.csv \
  --dataset nova \
  --max-dialogues 10000 \
  --lattice-size 5 \              # ✅ Increased from 3
  --collapse-steps 12 \           # ✅ Match chat default
  --num-clusters 2000 \           # ✅ CRITICAL: Increased from 50
  --impulse-scale 0.1 \           # ✅ Match chat, stable value
  --output-dir nova/model
```

### 2. Update Chat to Match

```bash
python3 nova/chat/chat_demo.py \
  --lattice-size 5 \              # ✅ Match training
  --collapse-steps 12 \           # ✅ Match training
  --impulse-scale 0.1 \           # ✅ Match training
  --temperature 0.7 \
  --repetition-penalty 0.1
```

### 3. Add Signature Normalization

**File:** `nova/core/geometric_token_learner.py`

Add normalization in `learn_clusters()`:
```python
from sklearn.preprocessing import normalize

def learn_clusters(self, signatures: np.ndarray, tokens_list: List[List[str]]):
    # ... existing code ...
    
    # NORMALIZE SIGNATURES (NEW)
    print("  Normalizing signatures...")
    signatures_normalized = normalize(signatures, norm='l2', axis=1)
    
    # Fit KMeans on normalized signatures
    cluster_ids = self.kmeans.fit_predict(signatures_normalized)
```

### 4. Improve Fallback Logic

**File:** `nova/core/cluster_decoder.py`

Replace line 119-123 with:
```python
if not valid_candidates:
    # Fallback: Use top-N most frequent words from cluster (not all!)
    # Limit to top 50 words to prevent word salad
    top_words = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:50]
    valid_candidates = [word for word, count in top_words]
    candidate_probs = np.array([count for word, count in top_words], dtype=float)
    if candidate_probs.sum() > 0:
        candidate_probs /= candidate_probs.sum()
    else:
        candidate_probs = np.ones(len(valid_candidates)) / len(valid_candidates)
```

---

## 📈 Expected Improvements

### After Fixes:

**Before (Current):**
- 50 clusters → 1,465 tokens/cluster → Word salad
- 27 dimensions → Poor separation
- Mismatched parameters → Wrong signatures

**After (Fixed):**
- 2,000 clusters → ~15 tokens/cluster → **Semantic coherence**
- 125 dimensions → Better separation
- Matched parameters → Correct signatures

**Expected Chat Quality:**
```
You: hey
Bot: Hi there! How are you? [context: ON]

You: why
Bot: That's a good question. Let me think... [context: ON]

You: what?
Bot: I'm not sure what you mean. Can you clarify? [context: ON]
```

---

## 🔍 Step-by-Step Pipeline Analysis

### Step 1: Text → Geometry ✅
- **Status:** Working correctly
- **Issue:** None
- Tokenization and hashing work fine

### Step 2: Geometry → Signature ✅
- **Status:** Working correctly
- **Issue:** Parameter mismatches cause wrong signatures
- **Fix:** Match collapse steps and impulse scale

### Step 3: Signature → Cluster ID ❌
- **Status:** BROKEN
- **Issue:** Only 50 clusters for 31k signatures
- **Fix:** Increase to 2,000 clusters

### Step 4: Cluster → Vocabulary ❌
- **Status:** BROKEN
- **Issue:** Clusters have 1,500+ words (word salad)
- **Fix:** More clusters = smaller vocabularies

### Step 5: Vocabulary → Grammar ❌
- **Status:** PARTIALLY BROKEN
- **Issue:** Grammar intersection fails → Falls back to entire cluster
- **Fix:** Improve fallback logic

### Step 6: Grammar → Text ❌
- **Status:** BROKEN
- **Issue:** Picking random words from massive clusters
- **Fix:** All above fixes combined

---

## 🎯 Priority Fix Order

1. **CRITICAL:** Increase `--num-clusters` to 2,000
2. **CRITICAL:** Increase `--lattice-size` to 5
3. **HIGH:** Match collapse steps (12 for both)
4. **HIGH:** Match impulse scale (0.1 for both)
5. **MEDIUM:** Add signature normalization
6. **MEDIUM:** Improve fallback logic

---

## 📝 Summary

**Root Cause:** The model was trained with **50 clusters for 31,429 signatures**, creating massive word bags that produce word salad.

**Primary Fix:** Retrain with **2,000 clusters** and **lattice size 5**.

**Secondary Fixes:** Match parameters between training and chat, add normalization, improve fallback logic.

**Expected Result:** Coherent, contextually appropriate responses instead of word salad.

