# Unsupervised Mode: Geometry Discovers Meaning

**Force-driven meaning emergence. No labels required.**

---

## What It Does

In unsupervised mode, Livnium becomes a **pure semantic physics engine**:

- **No label comparisons** - no accuracy, no confusion matrices
- **No reward shaping** - no reinforcement tied to ground truth
- **Only physics** - resonance, curvature, repulsion, entropy, basins, memory

**Text in → geometry self-organizes → basins drift → meaning emerges.**

---

## How It Works

### Step 1: Predict Basin, Not Label

Layer 7 now returns `basin_index` (0, 1, or 2) instead of just labels:

- **Basin 0** = Cold (entailment-like patterns)
- **Basin 1** = Far (contradiction-like patterns)  
- **Basin 2** = City (neutral-like patterns)

This is **geometry discovering clusters**, not label prediction.

### Step 2: Track Clusters

Every sentence pair is assigned to a basin based on force competition:

```python
basin_index = argmax([cold_force, far_force, city_force])
```

The `ClusterTracker` records which basin each sentence falls into.

### Step 3: Word Polarities Self-Organize

Words update their polarity based on which basin sentences fall into:

- Basin 0 → `polarity[0]++` (cold/entailment-like)
- Basin 1 → `polarity[1]++` (far/contradiction-like)
- Basin 2 → `polarity[2]++` (city/neutral-like)

No labels needed - just physics.

### Step 4: Clusters Emerge

Over time, you'll see:

- **Basin 0** = "causal / implies / positive relation"
- **Basin 1** = "opposition / negation / contradiction"
- **Basin 2** = "descriptions / ambiguous / observational"

These match human categories - but they're **not forced**. They appear naturally.

---

## Usage

### Basic Unsupervised Training

```bash
python3 train_v4.py --unsupervised --train 10000 --data-dir experiments/nli/data
```

### Export Clusters

```bash
python3 train_v4.py --unsupervised --train 10000 \
    --data-dir experiments/nli/data \
    --cluster-output experiments/nli_v4/clusters
```

This creates:
- `cluster_0_cold.json` - Cold basin entries
- `cluster_1_far.json` - Far basin entries
- `cluster_2_city.json` - City basin entries
- `cluster_summary.json` - Statistics

---

## What You'll See

### During Training

```
Step 500: Basin 0=167 | Basin 1=198 | Basin 2=135 | Moksha=0.023 | Entropy=0.0145 | Imbalance=0.126 | Temp=0.252
```

- **Basin counts** - how many sentences fell into each basin
- **Moksha** - convergence rate
- **Entropy** - current thermal noise
- **Imbalance** - class distribution imbalance
- **Temp** - system temperature

### After Training

```
GEOMETRY-DISCOVERED CLUSTERS
======================================================================

BASIN_0_COLD:
  Count: 3347
  Avg Confidence: 0.7234
  Description: Cold basin (entailment-like patterns)

BASIN_1_FAR:
  Count: 3123
  Avg Confidence: 0.6891
  Description: Far basin (contradiction-like patterns)

BASIN_2_CITY:
  Count: 3530
  Avg Confidence: 0.6543
  Description: City basin (neutral-like patterns)

Total entries: 10000
```

---

## Cluster Files

Each cluster file contains:

```json
{
  "basin_index": 0,
  "basin_name": "cluster_0_cold",
  "count": 3347,
  "statistics": {
    "avg_confidence": 0.7234,
    "total_entries": 3347
  },
  "entries": [
    {
      "premise": "A cat runs",
      "hypothesis": "A cat moves",
      "confidence": 0.85,
      "resonance": 0.72,
      "cold_attraction": 0.89,
      "far_attraction": 0.12,
      "basin_forces": {
        "basin_0_cold": 0.75,
        "basin_1_far": 0.10,
        "basin_2_city": 0.15
      }
    },
    ...
  ]
}
```

---

## The Physics

Meaning emerges because:

1. **Entropy** injects variation (prevents freeze)
2. **Repulsion** carves boundaries (separates clusters)
3. **Cold basins** deepen (attract similar patterns)
4. **City/valley** mediates overlap (handles ambiguity)
5. **Curvature** smooths the field (creates structure)
6. **Stability** shapes long-term structure (Moksha)

The system evolves its own categories.

This is no different from how galaxies, continents, or weather patterns emerge.

**Language becomes geometry.**

---

## Interpretation

After training, you can:

1. **Examine clusters** - see what patterns emerged
2. **Interpret basins** - "This cluster acts like contradiction"
3. **Reuse clusters** - use them for downstream tasks
4. **Watch evolution** - see how clusters change over time

But remember: **these labels are your interpretation, not the system's.**

You're watching a newborn universe classify itself.

---

## Comparison: Supervised vs Unsupervised

### Supervised Mode
- Uses gold labels
- Tracks accuracy
- Applies label-based reinforcement
- Optimizes for human categories

### Unsupervised Mode
- No labels needed
- Tracks cluster distribution
- Applies physics-based organization
- Discovers natural categories

**Both modes use the same physics engine.**
**Unsupervised just removes the human labels.**

---

## Next Steps

1. **Run unsupervised training** - see clusters emerge
2. **Examine cluster files** - understand what patterns formed
3. **Visualize clusters** - plot basin assignments over time
4. **Compare to labels** - see if geometry matches human categories
5. **Extend to new data** - test on different text types

The universe is ready to discover meaning on its own. 🌍

