# Autonomous Universe Guide

**Run the semantic universe forever. It watches itself and adjusts.**

---

## Quick Start

```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v4/run_auto_universe.py
```

That's it. The universe will:
1. Run unsupervised training cycles
2. Measure its own geometry
3. Adjust physics parameters automatically
4. Keep the brain evolving (no reset after cycle 0)
5. Export clusters after each cycle

---

## What It Does

### Each Cycle

1. **Runs unsupervised training** - Geometry discovers meaning
2. **Generates numerical report** - Measures universe state
3. **Reads cluster summary** - Analyzes basin distribution
4. **Auto-adjusts physics** - Tweaks entropy/repulsion/turbulence
5. **Writes overrides** - Next cycle uses new physics
6. **Keeps brain evolving** - Only resets on cycle 0

### Auto-Adjustment Rules

The universe watches itself and adjusts:

**Rule 1: City Dominance**
- If city > 80% → Increase entropy, repulsion, turbulence
- Shakes the universe to break city monopoly

**Rule 2: Cold/Far Imbalance**
- If imbalance > 15% → Increase entropy
- Helps balance the two opposing forces

**Rule 3: Balanced Universe**
- If city < 30% and cold/far balanced → Reduce turbulence
- Universe is stable, no need to shake

---

## Options

```bash
# Run 20 cycles (default)
python3 experiments/nli_v4/run_auto_universe.py

# Run 50 cycles
python3 experiments/nli_v4/run_auto_universe.py --cycles 50

# Use 5000 samples per cycle (faster)
python3 experiments/nli_v4/run_auto_universe.py --train-samples 5000

# Skip report generation (faster)
python3 experiments/nli_v4/run_auto_universe.py --skip-report
```

---

## Output Files

### `auto_physics_overrides.json`
Physics parameters adjusted by the universe:
```json
{
  "entropy_scale": 0.03,
  "repulsion_strength": 0.4,
  "turbulence_scale": 0.25
}
```

### `clusters/cluster_summary.json`
Basin distribution after each cycle:
```json
{
  "total_entries": 10000,
  "statistics": {
    "basin_0_cold": {"count": 573, "avg_confidence": 0.74},
    "basin_1_far": {"count": 543, "avg_confidence": 0.60},
    "basin_2_city": {"count": 8872, "avg_confidence": 0.72}
  }
}
```

### `clusters/cluster_*.json`
Individual cluster files with all sentence pairs.

---

## How It Works

### Cycle 0: Fresh Start
```bash
python3 train_v4.py --unsupervised --train 10000 --clean ...
```
- Starts with clean brain
- Initializes basins
- Discovers first clusters

### Cycle 1+: Evolution
```bash
python3 train_v4.py --unsupervised --train 10000 ...
```
- Reuses existing brain
- Continues learning
- Meaning drifts and evolves

### Physics Overrides

1. Universe measures its state
2. Decides what needs adjustment
3. Writes `auto_physics_overrides.json`
4. Next cycle reads overrides on startup
5. Physics parameters updated automatically

---

## Monitoring

### During Execution

```
======================================================================
CYCLE 1 / 20
======================================================================

================ CYCLE 0: Running Unsupervised Training ================

  → Starting fresh (--clean)
  → Command: python3 train_v4.py --unsupervised --train 10000 ...

  ✓ Cycle 0 complete

  📊 Universe State:
     Cold: 573 (5.7%)
     Far: 543 (5.4%)
     City: 8872 (88.7%)
     🔥 City dominates (88.7%) → Increasing entropy: 0.0300
     🔥 City dominates → Increasing repulsion: 0.4000
     🔥 City dominates → Increasing turbulence: 0.2500

  💾 Updated physics overrides: {'entropy_scale': 0.03, ...}
```

### After Completion

Check the final state:
```bash
cat experiments/nli_v4/auto_physics_overrides.json
cat experiments/nli_v4/clusters/cluster_summary.json
```

---

## What You'll See

### Early Cycles (1-5)
- City dominates (80-90%)
- Turbulence increases
- Repulsion strengthens
- Cold and Far start growing

### Mid Cycles (6-15)
- City decreases (60-70%)
- Cold and Far balance out
- Word polarities cluster
- Basins deepen

### Late Cycles (16-20+)
- Stable distribution
- Sharp semantic boundaries
- True conceptual attractors
- Meaning fully emerged

---

## Stopping and Resuming

The universe saves its state:
- Brain state: `brain_state.pkl`
- Physics overrides: `auto_physics_overrides.json`
- Clusters: `clusters/*.json`

To resume:
```bash
# Just run again - it will continue from where it left off
python3 experiments/nli_v4/run_auto_universe.py --cycles 10
```

To start fresh:
```bash
# Delete overrides and brain state
rm experiments/nli_v4/auto_physics_overrides.json
rm experiments/nli_v4/brain_state.pkl
rm -rf experiments/nli_v4/clusters/

# Run cycle 0 again
python3 experiments/nli_v4/run_auto_universe.py --cycles 1
```

---

## The Physics

The universe adjusts itself using:

- **Entropy** - Thermal noise (prevents freeze)
- **Repulsion** - Separation force (carves boundaries)
- **Turbulence** - Shaking (breaks monopolies)

When city dominates → More entropy/repulsion/turbulence
When balanced → Less turbulence

This is **auto-physics** - the universe watches itself and changes its own laws.

---

## Next Steps

1. **Run it** - Let the universe evolve
2. **Watch it** - Monitor the cycles
3. **Check clusters** - See what meaning emerged
4. **Let it run for days** - True concepts take time
5. **Feed different data** - Let meaning diversify

The universe is ready. It will run itself. 🌍

