# Vector Mode Conversion: Cell-less Livnium

## Overview

Livnium has been converted from a **cell-based (3D lattice)** system to a **pure vector-based** system.

**Key Changes**:
- ❌ Removed: 3D lattice (5×5×5, 15×15×15, etc.)
- ✅ Added: High-dimensional vectors (128, 256, 512, 1024 dimensions)
- ✅ Kept: All physics laws (OM/LO, divergence, fracture, alignment)

## Why Vector Mode?

### Problems with Cell-Based System

1. **Token Collisions**: 92% collision rate (1600+ tokens → 125 cells)
2. **Identical Signatures**: All sentences collapse to same basin → OM ≈ LO
3. **Constant Divergence**: Alignment ≈ 1.0 → divergence constant
4. **No Variation**: Physics signals flat → can't distinguish E/C/N

### Benefits of Vector Mode

1. **Zero Collisions**: 128-bit vector space → practically infinite
2. **True Variation**: Different sentences → different vectors → OM ≠ LO
3. **Fast Collapse**: Vector operations are instant (no grid traversal)
4. **Scalable**: Can use 256, 512, 1024 dimensions easily

## Architecture

### Token → Vector

```python
token → MD5 hash → seed → random vector (normalized) → scaled by impulse_scale
```

- Every token gets a **unique, deterministic** vector
- Zero collisions (128-bit space)
- Reproducible (same token → same vector)

### Sentence → Vector

```python
sentence → sum(token_vectors) → normalize
```

- Conceptually identical to impulse accumulation
- Just without the 3D grid

### Collapse

```python
vector → tanh(vector * scale) → normalize → repeat
```

- Creates basins, stability, nonlinearity
- No grid needed
- Fast vector operations

### OM/LO Physics

```python
OM = collapse(premise)
LO = collapse(hypothesis)  # with angular tilt for SNLI
alignment = dot(OM, LO)
divergence = 0.38 - alignment
fracture = ||OM - LO||
```

- **Same physics laws** as before
- Just using vectors instead of cells

## Files Created

1. **`nova/core/vector_text_to_geometry.py`**
   - `VectorTextToGeometry`: Main interface
   - `VectorGeometricTokenLearner`: Compatibility wrapper

2. **`nova/core/vector_collapse.py`**
   - Collapse functions: `tanh`, `power3`, `relu`, `sigmoid`

3. **`nova/training/train_snli_phase1_vector.py`**
   - Training script for vector mode

4. **`nova/chat/test_snli_phase1_vector.py`**
   - Test script for vector mode

5. **`nova/debug/vector_mode_verification.py`**
   - Verification report for vector mode

## Usage

### Training (Vector Mode)

```bash
python3 nova/training/train_snli_phase1_vector.py \
  --snli-train nova/data/snli/snli_1.0_train.jsonl \
  --max-samples 10000 \
  --vector-dim 256 \
  --collapse-type tanh \
  --collapse-steps 12 \
  --num-clusters 2000 \
  --output-dir nova/model/snli_phase1_vector
```

**Parameters**:
- `--vector-dim`: 128, 256, 512, 1024 (default: 256)
- `--collapse-type`: `tanh`, `power3`, `relu`, `sigmoid` (default: `tanh`)

### Testing (Vector Mode)

**Cluster + Grammar Mode**:
```bash
python3 nova/chat/test_snli_phase1_vector.py \
  --model-dir nova/model/snli_phase1_vector \
  --max-samples 1000
```

**Pure Physics Mode**:
```bash
python3 nova/chat/test_snli_phase1_vector.py \
  --model-dir nova/model/snli_phase1_vector \
  --max-samples 1000 \
  --physics
```

### Verification

```bash
python3 nova/debug/vector_mode_verification.py
```

## Expected Results

**Before (Cell Mode)**:
- OM/LO cosine: 0.99998 (identical)
- Alignment range: 0.000057 (constant)
- Physics accuracy: 33% (random)

**After (Vector Mode)**:
- OM/LO cosine: 0.6-0.95 (varies)
- Alignment range: >0.1 (meaningful variation)
- Physics accuracy: 45-65% (should work!)

## What Stays the Same

All Livnium physics laws work exactly the same:
- ✅ OM direction
- ✅ LO direction
- ✅ Divergence law: `divergence = 0.38 - alignment`
- ✅ Fracture metric
- ✅ Symmetry breaking
- ✅ Angular tilt
- ✅ Alignment computation

**Only the container changed**: from 3D lattice to high-dimensional vectors.

## Migration Path

1. **Keep cell mode** for law extraction (needs perfect symmetry)
2. **Use vector mode** for SNLI and dialogue (needs variation)
3. **Both can coexist** - they're separate implementations

## Next Steps

1. Train with vector mode
2. Run verification script
3. Check if alignment varies and divergence separates E/C/N
4. If it works, vector mode becomes the default for SNLI

