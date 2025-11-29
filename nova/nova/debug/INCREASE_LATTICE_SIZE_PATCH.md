# Increase Lattice Size Patch

## The Fix

Change `lattice_size` from 5 to 15 (or 21) in all relevant files.

## Files to Update

### 1. Training Script
**File**: `nova/training/train_snli_phase1.py`

**Change**:
```python
# Line ~260 (in argument parser)
parser.add_argument(
    '--lattice-size',
    type=int,
    default=15,  # Changed from 5
    help='Lattice size (must be odd: 3, 5, 7, 9, 11, 13, 15, ...)'
)
```

### 2. Test Script
**File**: `nova/chat/test_snli_phase1.py`

**Change**: No change needed - it auto-detects from saved model

### 3. Diagnostic Script
**File**: `nova/debug/snli_physics_report.py`

**Change**:
```python
# Line ~400 (in main function)
interface = TextToGeometry(
    lattice_size=15,  # Changed from 5
    impulse_scale=0.1,
    num_clusters=2000,
    break_symmetry_for_snli=True
)
```

### 4. Post-Fix Verification Script
**File**: `nova/debug/post_fix_verification.py`

**Already has**: `--lattice-size` argument (default=15)

## Quick Patch Command

```bash
# Update training script default
sed -i '' 's/default=5/default=15/g' nova/training/train_snli_phase1.py

# Update diagnostic script
sed -i '' 's/lattice_size=5/lattice_size=15/g' nova/debug/snli_physics_report.py
```

## Expected Impact

**Before (lattice_size=5)**:
- Cells: 5×5×5 = 125
- Collision rate: ~92%
- OM/LO cosine: ~0.99998
- Physics accuracy: ~33%

**After (lattice_size=15)**:
- Cells: 15×15×15 = 3,375
- Collision rate: ~49%
- OM/LO cosine: Should drop to <0.95
- Physics accuracy: Should jump to >45%

**After (lattice_size=21)**:
- Cells: 21×21×21 = 9,261
- Collision rate: ~18%
- OM/LO cosine: Should drop to <0.90
- Physics accuracy: Should jump to >50%

## Verification

After applying the patch, run:

```bash
python3 nova/debug/post_fix_verification.py --lattice-size 15
```

Check that:
- ✓ Collision rate < 50%
- ✓ OM/LO cosine < 0.95
- ✓ Physics accuracy > 45%

