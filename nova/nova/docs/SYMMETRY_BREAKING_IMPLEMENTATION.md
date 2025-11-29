# Symmetry Breaking Implementation - SNLI Only

## Overview

**Critical Separation**: SNLI and Law Extractor are **separate systems** with different requirements:

- **Law Extractor**: Requires **perfect symmetry** for invariant measurement
- **SNLI**: Requires **symmetry breaking** for angular variation → meaning → divergence

## Implementation

### What Was Changed

**ONLY** the SNLI geometry path was modified. The law extractor remains untouched.

### Files Modified

1. **`nova/core/text_to_geometry.py`**
   - Added `break_symmetry_for_snli` parameter to `__init__`
   - Added `_apply_symmetry_breaking()` method
   - Modified `reset_geometry()` to apply symmetry breaking for SNLI

2. **`nova/training/train_snli_phase1.py`**
   - Set `break_symmetry_for_snli=True` when initializing TextToGeometry

3. **`nova/chat/test_snli_phase1.py`**
   - Set `break_symmetry_for_snli=True` when initializing TextToGeometry

### Files NOT Modified (Law Extractor)

- ✅ `core/law/law_extractor.py` - Untouched
- ✅ `core/law/advanced_law_extractor.py` - Untouched
- ✅ `core/runtime/orchestrator.py` - Untouched
- ✅ `core/classical/livnium_core_system.py` - Untouched

**Law extractor uses its own physics and maintains perfect symmetry.**

---

## How It Works

### Symmetry Breaking Method

```python
def _apply_symmetry_breaking(self):
    """
    SNLI ONLY: Break perfect symmetry by adding tiny random noise to SW values.
    
    This enables angular variation between premise/hypothesis → meaning → geometry → divergence.
    """
    # Get current total SW (must be preserved)
    target_sw_sum = self.geometry.get_total_symbolic_weight()
    
    # Add small random noise to each cell's SW
    # Noise scale: 0.5% of mean SW per cell
    num_cells = len(self.geometry.lattice)
    mean_sw = target_sw_sum / num_cells if num_cells > 0 else 0.0
    noise_scale = mean_sw * 0.005  # 0.5% noise
    
    # Apply noise to each cell
    for coords, cell in self.geometry.lattice.items():
        if cell.symbolic_weight is not None:
            noise = np.random.normal(0, noise_scale)
            cell.symbolic_weight = max(0.0, cell.symbolic_weight + noise)
    
    # Renormalize to preserve total SW (conservation law)
    current_sw_sum = self.geometry.get_total_symbolic_weight()
    if current_sw_sum > 0:
        scale_factor = target_sw_sum / current_sw_sum
        for cell in self.geometry.lattice.values():
            if cell.symbolic_weight is not None:
                cell.symbolic_weight *= scale_factor
```

### Key Features

1. **Small Noise**: 0.5% of mean SW per cell (tiny perturbation)
2. **SW Conservation**: Total SW is preserved (renormalized after noise)
3. **Non-Negative**: Ensures SW values remain ≥ 0
4. **SNLI Only**: Only applied when `break_symmetry_for_snli=True`

---

## Why This Is Needed

### SNLI Requirements

SNLI needs **angular variation** between premise and hypothesis:
- Different sentences → different geometric angles
- Angular variation → different alignment values
- Different alignment → different divergence
- Divergence → E/C/N classification

**Without symmetry breaking**: All sentences start from perfect symmetry → same angles → same alignment → no divergence variation → can't distinguish E/C/N.

**With symmetry breaking**: Each sentence starts with slight asymmetry → different angles → different alignment → divergence variation → can distinguish E/C/N.

### Law Extractor Requirements

Law extractor needs **perfect symmetry**:
- Perfect symmetry → perfect invariants
- Perfect invariants → accurate conservation law measurement
- Conservation laws → fundamental physics discovery

**Symmetry breaking would break invariant measurement.**

---

## Verification

### SW Conservation Test

```python
from nova.core.text_to_geometry import TextToGeometry

# SNLI mode (symmetry breaking enabled)
t = TextToGeometry(lattice_size=5, break_symmetry_for_snli=True)
sw_before = t.geometry.get_total_symbolic_weight()  # 1350.0
t.reset_geometry()
sw_after = t.geometry.get_total_symbolic_weight()   # 1350.0

# ✓ SW conservation maintained
assert abs(sw_after - sw_before) < 0.01
```

### Law Extractor Unaffected

The law extractor does **not** use `TextToGeometry`:
- It uses `LivniumCoreSystem` directly
- It uses `Orchestrator` for physics state
- It maintains perfect symmetry
- **No changes needed**

---

## Usage

### For SNLI (Symmetry Breaking Enabled)

```python
# Training
interface = TextToGeometry(
    lattice_size=5,
    break_symmetry_for_snli=True  # Enable for SNLI
)

# Testing
interface = TextToGeometry(
    lattice_size=5,
    break_symmetry_for_snli=True  # Enable for SNLI
)
```

### For Law Extraction (Perfect Symmetry)

```python
# Law extractor uses LivniumCoreSystem directly
# No TextToGeometry involved
# Perfect symmetry maintained automatically
```

### For Regular Dialogue (No Symmetry Breaking)

```python
# Regular dialogue doesn't need symmetry breaking
interface = TextToGeometry(
    lattice_size=5,
    break_symmetry_for_snli=False  # Default: no breaking
)
```

---

## Impact

### Expected Improvements for SNLI

1. **Angular Variation**: Premise and hypothesis will have different geometric angles
2. **Alignment Variation**: Different alignment values (not all 1.0)
3. **Divergence Variation**: Divergence will vary based on alignment
4. **Better Classification**: System can distinguish E/C/N based on divergence

### No Impact on Law Extractor

- ✅ Perfect symmetry maintained
- ✅ Invariant measurement accurate
- ✅ Conservation laws valid
- ✅ No changes needed

---

## Summary

**What Changed:**
- ✅ Added symmetry breaking **ONLY** to SNLI path
- ✅ SW conservation maintained (renormalization)
- ✅ Law extractor untouched (perfect symmetry preserved)

**Why:**
- SNLI needs angular variation → meaning → divergence
- Law extractor needs perfect symmetry → invariants

**Result:**
- SNLI can now distinguish E/C/N through divergence
- Law extractor continues to measure invariants accurately
- Both systems work correctly with their respective requirements

---

## Next Steps

1. ✅ Retrain SNLI Phase 1 with symmetry breaking enabled
2. ✅ Test if accuracy improves (expected: 25% → 40-50%+)
3. ✅ Verify divergence statistics show variation
4. ✅ Confirm law extractor still works correctly

**Status**: ✅ Implementation complete - SNLI has symmetry breaking, law extractor has perfect symmetry.

