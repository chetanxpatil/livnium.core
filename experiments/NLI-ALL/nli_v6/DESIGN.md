# Livnium NLI v6: The Corrected 3-Axis Manifold

## The Problem with v5

v5 achieves ~36-37% accuracy, which is the **upper bound of cosine-based geometry**.

The reverse physics discovery revealed:
- ✓ Resonance, Cold Attraction, Curvature are invariant (true laws)
- ✓ Divergence sign is preserved (true law)
- ✗ Divergence magnitude is noisy (artifact)
- ✗ Contradiction is hard because SNLI vectors treat it as "similar topics"

## The Solution: Two Paths Forward

### Path A — Physics-Based (Curvature as Second Axis)

Use curvature to separate contradiction and neutral despite similar resonance:

**Decision Rules**:
- High curvature + negative div → Entailment
- High curvature + positive div → Contradiction
- Low curvature + negative div → Weak entailment
- Low curvature + zero div → Neutral

**Curvature becomes the second axis of contradiction.**

### Path B — Geometry Tools (Simpler, Faster)

Define a new axis:

```python
opposition = resonance * (divergence_sign)
```

Where `divergence_sign` is:
- `+1` if divergence > 0 (push apart)
- `-1` if divergence < 0 (pull together)
- `0` if divergence ≈ 0 (balanced)

**This separates**:
- High resonance + positive sign → Contradiction
- High resonance + negative sign → Entailment
- Low resonance → Neutral

**Expected impact**: Accuracy from ~36% → ~45-50%

This is the simplest way to give geometry a "direction" free from magnitude noise.

## Recommended: Path B (Opposition Axis)

### Why Opposition Works

1. **Uses invariant signals**: Resonance (stable) + Divergence sign (preserved)
2. **Removes noise**: Ignores divergence magnitude (artifact)
3. **Simple**: One multiplication, clear separation
4. **Fast**: No new computations needed

### Implementation

```python
# In Layer 0 or Layer 4
def compute_opposition(resonance, divergence):
    """Compute opposition: resonance weighted by divergence direction."""
    divergence_sign = np.sign(divergence)  # +1, -1, or 0
    opposition = resonance * divergence_sign
    return opposition

# Decision rules
if opposition > threshold_contradiction:  # High res + positive div
    predict = CONTRADICTION
elif opposition < threshold_entailment:  # High res + negative div
    predict = ENTAILMENT
else:  # Low res or zero div
    predict = NEUTRAL
```

### Expected Performance

- **Contradiction**: Better separation from neutral (opposition > 0)
- **Entailment**: Better separation from contradiction (opposition < 0)
- **Neutral**: Clearer definition (opposition ≈ 0)
- **Overall**: 45-50% accuracy (up from 36-37%)

## The v6 Architecture

### Core Principles

1. **Use only invariant signals**: Resonance, Cold Attraction, Curvature, Divergence Sign
2. **Ignore artifacts**: Divergence magnitude, convergence magnitude, force ratios
3. **Simple combinations**: Opposition = Resonance × Divergence Sign
4. **Clear phase boundaries**: Based on invariant geometry, not noisy magnitudes

### Layer Structure

**Layer 0: Resonance** (unchanged)
- Raw geometric similarity signal

**Layer 1: Curvature** (unchanged)
- How meaning bends through chain

**Layer 2: Opposition** (NEW)
- `opposition = resonance * sign(divergence)`
- Separates E/C/N using invariant signals only

**Layer 3: Cold Attraction** (unchanged)
- Semantic gravity (invariant)

**Layer 4: Decision** (simplified)
- Use opposition + cold attraction
- Ignore divergence magnitude noise

## Status

⚠️ **Design Phase** - Ready for implementation

## Next Steps

1. Implement opposition axis in Layer 2
2. Simplify Layer 4 decision logic
3. Test on SNLI
4. Compare with v5 baseline
5. Document findings

## References

- Invariant Laws: `core/law/invariant_laws.md`
- Reverse Physics Discovery: `experiments/nli_v5/REVERSE_PHYSICS_DISCOVERY.md`
- Divergence Law: `core/law/divergence_law.md`

