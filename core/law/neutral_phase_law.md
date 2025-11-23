# Neutral Phase Law (In Progress)

## The Balance Zone

Neutral is the phase where forces balance. Currently defined as "near-zero divergence", but needs explicit balance band definition.

## Current Definition

**Current rule**:
```python
if abs(divergence) < 0.12:
    predict = NEUTRAL
```

**Problem**: This is too simple. Neutral should be defined as a **balance zone** in both divergence AND attractions.

## Proposed Definition

### Explicit Balance Band

```python
is_neutral = (
    abs(divergence) < d_neutral_band
    and abs(cold_attraction - far_attraction) < attraction_balance_band
)
```

### From Canonical Fingerprints

**Neutral statistics** (from golden labels):
- Divergence: -0.0883 ± 0.1539
- Resonance: 0.5853 ± 0.1262
- Cold attraction: 0.7031 ± 0.0937
- Far attraction: 0.5938 ± 0.0704

**Balance condition**:
- `|cold_attraction - far_attraction|` should be small
- Mean difference: 0.7031 - 0.5938 = 0.1093
- Std: ~0.1 (estimated)

## Implementation Plan

### Step 1: Extract Neutral Fingerprints

Run:
```bash
python3 experiments/nli_v5/train_v5.py \
  --clean --train 1000 \
  --debug-golden --learn-patterns
```

Extract from neutral patterns:
- Divergence band (q25-q75): `[-0.18, +0.01]` → band width ≈ 0.19
- Attraction balance: `|cold - far| < 0.15` (mean ± 1 std)

### Step 2: Define Neutral Band

```python
# Neutral: Balanced forces
d_neutral_band = 0.15  # From q75 - q25
attraction_balance_band = 0.15  # From std of |cold - far|

is_neutral = (
    abs(divergence) < d_neutral_band
    and abs(cold_attraction - far_attraction) < attraction_balance_band
    and self.resonance_n_min < resonance < self.resonance_n_max  # Optional
)
```

### Step 3: Prioritize Phases

Order of evaluation:
1. **Strong C first**: `divergence > 0.10` (clearly positive)
2. **Strong E next**: `divergence < -0.10 AND resonance > 0.55` (clearly negative + high res)
3. **Then neutral**: If both E and C are weak and balance criteria match
4. **Fallback**: Force-based decision

This avoids "over-writing" neutral by weak E or C signals.

## Current Status

⚠️ **In Progress**: Neutral recall is low (~25-40%)

**Issues**:
- Neutral band may be too strict
- Balance condition not explicitly checked
- Weak E/C signals may override neutral

## Expected Impact

After implementation:
- Neutral recall should improve to ~40-50%
- Neutral will be a **real physical phase**, not just "whatever is left"
- Phase diagram will be more "physically pretty"

## Related Laws

- **Divergence Law**: Provides divergence signal
- **Phase Classification Law**: Uses neutral band in decision logic

## References

- Current implementation: `experiments/nli_v5/layers.py` → `Layer4Decision`
- Fingerprints: `experiments/nli_v5/physics_fingerprints.json`
- Analysis: `experiments/nli_v5/PHYSICS_ANALYSIS_CONFIRMED.md`

