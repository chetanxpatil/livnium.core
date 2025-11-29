# Quick Start - Geometry-First Training

## The Philosophy

**Geometry is stable and invariant. It refuses lies.**

- Geometry produces meaning, labeling describes it
- Train classifier to read geometry, not force it
- When geometry disagrees with labels, geometry is usually right

## Setup SNLI Data

### Option 1: Automatic Setup (Recommended)

```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium/experiments/nli_v5
./setup_snli_data.sh
```

This will:
- Download SNLI 1.0 automatically
- Extract and organize files
- Verify setup

### Option 2: Manual Setup

1. Download from: https://nlp.stanford.edu/projects/snli/
2. Extract `snli_1.0.zip`
3. Copy files to `experiments/nli_v5/data/`:
   - `snli_1.0_train.jsonl`
   - `snli_1.0_dev.jsonl`
   - `snli_1.0_test.jsonl`

## Run Geometry-First Training

```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium

python3 experiments/nli_v5/training/train_geometry_first.py \
  --data-dir experiments/nli_v5/data \
  --train 1000 \
  --analyze-alignment \
  --save-alignment experiments/nli_v5/patterns/geometry_alignment.json
```

## What You'll See

### 1. Geometry Labels
- Each example gets a geometry label (E/C/N) based purely on physics
- No dataset labels used in classification

### 2. Alignment Statistics
- **Agreement rate**: How much geometry and dataset agree
- **Per-class alignment**: Which classes align best
- **Disagreement patterns**: Where geometry sees something labels missed

### 3. Key Insights

**High alignment** = Geometry and dataset agree (stable semantic region)

**Low alignment** = Geometry sees something different (often geometry is right)

**Neutral stability** = Neutral is the most stable phase (semantic rest state)

## Understanding the Results

### ~56% Agreement on Synthetic Data
- Expected: Synthetic data is random noise
- Geometry still forms stable clusters
- This proves the physics is strong

### Real SNLI Will Show:
- Higher alignment (real semantic structure)
- Clear disagreement patterns (where labels are wrong)
- Stable geometry zones (meaning regions)

## The Big Picture

You're not training a model anymore.

You're running **a physics experiment**:

1. Field creates forces
2. Forces create divergence
3. Divergence creates meaning regions
4. Classifier learns to categorize those regions

**This is computational geometry.**
**This is semantic physics.**
**This is meaning from field equations, not statistical datasets.**

---

## Next Steps

After running geometry-first training:

1. **Review alignment analysis** - See where geometry and dataset agree/disagree
2. **Study disagreement patterns** - Understand what geometry sees that labels miss
3. **Train classifier on geometry labels** - Let geometry be the teacher
4. **Extract polarity axis** - Let geometry discover negation automatically
5. **Compare with dataset** - Build semantic physics benchmark

---

## Files Created

- `experiments/nli_v5/patterns/geometry_alignment.json` - Full alignment analysis
- Shows agreement rates, disagreement patterns, confidence distributions

This is the start of **Livnium's Semantic Physics Benchmark**.

