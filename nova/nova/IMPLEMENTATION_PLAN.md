# Nova Implementation Plan - High-Resolution Fix

## ✅ Phase 1: Code Patches (COMPLETED)

### 1. Signature Normalization ✅
**File:** `nova/core/geometric_token_learner.py`

**Changes:**
- Added `from sklearn.preprocessing import normalize`
- Normalize signatures before clustering (L2 normalization)
- Normalize signatures before prediction (matching training)

**Why:** Clusters based on geometric angle (meaning), not vector magnitude (sentence length).

### 2. Intelligent Fallback Logic ✅
**File:** `nova/core/cluster_decoder.py`

**Changes:**
- Replaced fallback to entire cluster vocabulary
- Now falls back to top 50 most frequent words in cluster
- Prevents word salad when grammar intersection fails

**Why:** Limits fallback vocabulary to high-probability words instead of random selection from 1,500+ words.

---

## 📋 Phase 2: Retraining (READY TO EXECUTE)

### High-Resolution Training Command

```bash
python3 nova/training/train_text_to_geometry.py \
  --csv nova/data/empathetic_train.csv \
  --dataset nova \
  --max-dialogues 10000 \
  --lattice-size 5 \
  --collapse-steps 12 \
  --num-clusters 2000 \
  --impulse-scale 0.1 \
  --output-dir nova/model_v2_highres
```

### Key Parameter Changes

| Parameter | Old Value | New Value | Impact |
|-----------|-----------|-----------|--------|
| `--lattice-size` | 3 (27 dims) | **5 (125 dims)** | 4.6× more expressive space |
| `--num-clusters` | 50 | **2000** | 40× more granularity |
| `--collapse-steps` | 20 | **12** | Matches chat default |
| `--impulse-scale` | 0.4 | **0.1** | Stable, matches chat |

### Expected Results

**Before:**
- 50 clusters → ~1,465 tokens/cluster
- 41/50 clusters have >1,000 tokens
- Word salad generation

**After:**
- 2,000 clusters → ~15 tokens/cluster
- Small, coherent vocabularies per cluster
- Semantic coherence

---

## 💬 Phase 3: Chat Testing (AFTER TRAINING)

### Chat Command (Matching Parameters)

```bash
python3 nova/chat/chat_demo.py \
  --lattice-size 5 \
  --collapse-steps 12 \
  --impulse-scale 0.1 \
  --temperature 0.6 \
  --repetition-penalty 0.1
```

**Note:** The chat script will auto-detect `lattice_size` from the saved model, but it's good to specify explicitly.

---

## 🔍 Verification Checklist

After training completes, verify:

1. **Cluster Statistics:**
   ```bash
   python3 -c "
   import json
   from pathlib import Path
   data = json.load(open('nova/model_v2_highres/geometric_clusters.json'))
   clusters = data['cluster_tokens']
   sizes = [len(tokens) for tokens in clusters.values()]
   print(f'Clusters: {len(clusters)}')
   print(f'Avg tokens/cluster: {sum(sizes)/len(sizes):.1f}')
   print(f'Max cluster size: {max(sizes)}')
   print(f'Clusters >100 tokens: {sum(1 for s in sizes if s > 100)}')
   "
   ```

2. **Expected Output:**
   - Average tokens/cluster: ~15-20
   - Max cluster size: <100
   - Clusters >100 tokens: <10% of total

3. **Chat Quality:**
   - Responses should be coherent
   - No word salad
   - Context-aware replies

---

## 📊 Performance Expectations

### Training Time
- **Old:** ~33 seconds (lattice 3, 50 clusters)
- **New:** ~2-5 minutes (lattice 5, 2000 clusters)
  - Larger lattice = more computation per signature
  - More clusters = longer KMeans fitting

### Memory Usage
- **Old:** ~50 cluster centroids × 27 dims = small
- **New:** ~2000 cluster centroids × 125 dims = moderate
  - Still manageable on most systems

### Chat Response Quality
- **Before:** Word salad, incoherent
- **After:** Coherent, contextually appropriate responses

---

## 🎯 Success Criteria

1. ✅ Code patches applied (normalization + fallback)
2. ⏳ Training completes with 2000 clusters
3. ⏳ Average cluster size <20 tokens
4. ⏳ Chat produces coherent responses
5. ⏳ No word salad in test conversations

---

## 🚀 Next Steps

1. **Run the training command** (Phase 2)
2. **Monitor cluster statistics** during training
3. **Test chat** with matching parameters (Phase 3)
4. **Compare** before/after response quality

---

## 📝 Notes

- The normalization patch ensures signatures are compared by direction, not magnitude
- The fallback patch prevents word salad when grammar fails
- The 2000 clusters provide semantic granularity
- The lattice size 5 provides sufficient expressive space

**Ready to execute Phase 2!**

