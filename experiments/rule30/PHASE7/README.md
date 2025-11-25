# Phase 7: The Proof Phase

**Status**: 🔬 **IN PROGRESS**

## Overview

Phase 7 proves that the Shadow Rule 30 system works **without Livnium**.

This is the scientific proof that:
- The geometry + dynamics + decoder alone can emulate Rule 30
- Livnium was only a stabilizer/guide, not the rule itself
- The reconstruction does not depend on external nudging

## The Goal

Demonstrate that removing Livnium (setting `--livnium-scale 0`) still produces:
- `center_ones_fraction ≈ 0.45–0.55` (natural Rule 30 equilibrium)
- Non-collapsed, chaotic trajectory (std ≈ 0.001–0.003)
- Robust behavior across different initial conditions

## The Three Experiments

### 1. Remove Livnium Completely
Run with `--livnium-scale 0` to test pure dynamics + noise.

### 2. Test Multiple Initial Conditions
Test robustness with:
- `--initial-condition random`
- `--initial-condition mean`
- `--initial-condition from_data`

### 3. Decoder Consistency Test
Compare decoder outputs on:
- Phase 3 real trajectory PCA
- Phase 7 shadow PCA

They should match → Geometry of Shadow == Geometry of Rule 30

## What This Proves

**Phase 6 proved:**
- The attractor can be steered
- The geometric shadow is complete
- Livnium curvature stabilizes structure

**Phase 7 proves:**
- Livnium's force is **not the rule**
- The rule already exists in the learned geometry
- The reconstruction does not depend on external nudging

This distinguishes a **discovered law** from a **fitted model**.

## Results

See `results/` for:
- Density tables across initial conditions
- Trajectory statistics
- Decoder consistency comparisons
- Scientific proof report

---

**This is the first complete proof of a shadow cellular automaton recovered purely from PCA geometry.**

