# Phase 6: Livnium Integration

**Status**: ✅ **IMPLEMENTED**

## Overview

Phase 6 adds a **Livnium geometric influence module** to the Shadow Rule 30 system.

## What is Livnium (Phase 6 Version)?

Livnium, in this experiment, is a **small geometric influence module** that sits on top of the existing PCA dynamics.

It has one job:

```
Modify the PCA state (y_t) with a tiny learned steerable force:

    y_t' = y_t + Ω(y_t)

Where Ω is an 8×8 matrix (or 8D vector).
```

This operator adds **curvature** and **bias** to the learned dynamics.

It does NOT replace the polynomial dynamics or the stochastic driver.

It only nudges the Shadow trajectory toward better regions (e.g. 50% density).

**Important**: This is NOT the full Livnium system. It's a minimal proxy - just a simple function that returns an 8D bias vector. Cursor doesn't need to know about recursive geometry, omcubes, basins, or any of the complex Livnium features. Just this one muscle.

## Purpose

- Prevent collapse into boring eigenvector loops
- Encourage exploration of full attractor
- Push trajectory toward meaningful regions (e.g. 0/1 balance)
- Act as a steering force for the chaotic system

## Implementation

Livnium is integrated into `shadow_rule30_phase6.py`:

```python
y_tp1 = model.predict(...)  # Existing dynamics
y_tp1 += noise  # Stochastic driver
y_tp1 += self.apply_livnium_force(y_t)  # Livnium influence
y_tp1 = normalize(y_tp1)  # Energy conservation
```

The minimal Livnium implementation is in `code/livnium_force.py`:
- `LivniumForce` class with `apply_livnium_force(y_t)` method
- Returns an 8D bias vector
- Can be vector-based (constant) or matrix-based (state-dependent)
- Default scale: 0.01 (1% influence)

## Structure

- `code/` - Implementation files
  - `livnium_force.py` - Minimal Livnium force module
  - `shadow_rule30_phase6.py` - Phase 6 shadow model with Livnium integration
  - `example_usage.py` - Usage examples
- `docs/` - Documentation
  - `LIVNIUM_DEFINITION.md` - Minimal Livnium definition for Cursor
- `results/` - Phase 6 output

## Usage

### Run Phase 6 Simulation

```bash
cd experiments/rule30/PHASE6/code
python shadow_rule30_phase6.py \
    --data-dir ../../PHASE3/results \
    --decoder-dir ../../PHASE4/results \
    --output-dir ../results \
    --num-steps 5000 \
    --livnium-scale 0.01 \
    --livnium-type vector \
    --verbose
```

### Try Examples

```bash
cd experiments/rule30/PHASE6/code
python example_usage.py
```

### Use Livnium in Your Code

```python
from livnium_force import LivniumForce, create_default_livnium

# Create Livnium force
livnium = create_default_livnium(n_components=8, force_scale=0.01)

# Apply to a state
y_t = np.random.randn(8)
bias = livnium.apply_livnium_force(y_t)

# Add to dynamics
y_tp1 = existing_dynamics(y_t) + noise + bias
```

## Dependencies

- Phase 4 complete (Shadow Rule 30 working)
- Phase 3 dynamics model (polynomial degree 3)
- Phase 4 decoder (Random Forest)

---

**Note**: This is the minimal Livnium implementation needed for Phase 6. It uses only the geometric bias generator from the full Livnium core. The real Livnium system (with recursive geometry, omcubes, basins, etc.) is much more complex, but Cursor doesn't need to know about that - just this simple function.

