# Impulse Scale Explained

## What is `impulse_scale`?

`impulse_scale` is a **scaling factor** that controls how much "energy" each character or token injects into the Livnium geometry. It's the **strength** of the geometric perturbation caused by text.

---

## The Physics: Text → Geometry

In Nova, text is converted to geometry through **impulses**:

1. **Character → Impulse** (for `char_to_impulse`):
   ```python
   impulse = (ord(char) % 27) * impulse_scale
   ```
   - Each character produces a value 0-26
   - Multiplied by `impulse_scale` to get final impulse strength

2. **Token → Impulse** (for `token_hash`):
   ```python
   # Token is hashed to get (x, y, z) coordinates and base impulse
   coords, base_impulse = token_hash(token)  # base_impulse is -1.0 to 1.0
   impulse = base_impulse * impulse_scale    # Scaled by impulse_scale
   ```

3. **Impulse → Geometry**:
   - Impulse is added to the **Symbolic Weight (SW)** at cell `(x, y, z)`
   - Higher impulse = more energy in that cell
   - Geometry then collapses based on these energy perturbations

---

## What Does It Control?

### 1. **Magnitude of Geometric Perturbations**

- **Low `impulse_scale` (e.g., 0.01)**: Tiny perturbations
  - Text barely affects geometry
  - Signatures are very similar (hard to distinguish sentences)
  - Collapse is subtle

- **High `impulse_scale` (e.g., 1.0)**: Large perturbations
  - Text strongly affects geometry
  - Signatures are very different (easy to distinguish sentences)
  - Collapse is dramatic

- **Default `impulse_scale` (0.1)**: Moderate perturbations
  - Balanced: enough to distinguish meanings, not so much that geometry becomes chaotic

### 2. **Signature Differentiation**

The impulse scale directly affects how well the system can distinguish different sentences:

```
impulse_scale = 0.01:
  "Hello" → signature: [0.01, 0.02, 0.01, ...]
  "World" → signature: [0.01, 0.02, 0.01, ...]  (very similar!)

impulse_scale = 0.1:
  "Hello" → signature: [0.1, 0.2, 0.1, ...]
  "World" → signature: [0.15, 0.1, 0.2, ...]  (more distinct)

impulse_scale = 1.0:
  "Hello" → signature: [1.0, 2.0, 1.0, ...]
  "World" → signature: [1.5, 1.0, 2.0, ...]  (very distinct, maybe too much)
```

### 3. **Collapse Behavior**

- **Low impulse**: Geometry barely moves during collapse → shallow basins
- **High impulse**: Geometry moves dramatically → deep basins, but potentially unstable
- **Optimal impulse**: Smooth collapse into stable meaning basins

---

## Code Example

```python
from nova.core.text_to_geometry import TextToGeometry

# Low impulse: subtle perturbations
interface_low = TextToGeometry(
    lattice_size=5,
    impulse_scale=0.01  # Very small
)
sig1 = interface_low.get_meaning_signature("Hello world")
# Signature values are tiny: [0.01, 0.02, ...]

# Default impulse: balanced
interface_default = TextToGeometry(
    lattice_size=5,
    impulse_scale=0.1  # Default
)
sig2 = interface_default.get_meaning_signature("Hello world")
# Signature values are moderate: [0.1, 0.2, ...]

# High impulse: strong perturbations
interface_high = TextToGeometry(
    lattice_size=5,
    impulse_scale=1.0  # Very large
)
sig3 = interface_high.get_meaning_signature("Hello world")
# Signature values are large: [1.0, 2.0, ...]
```

---

## Why Default is 0.1?

The default `impulse_scale=0.1` is chosen because:

1. **Stability**: Not so large that geometry becomes chaotic
2. **Sensitivity**: Large enough to distinguish different meanings
3. **Collapse Quality**: Produces smooth, stable meaning basins
4. **Empirical**: Works well across different lattice sizes (3, 5, 7, ...)

---

## When to Adjust `impulse_scale`?

### Increase (e.g., 0.2 - 0.5) if:
- Sentences are too similar (clustering fails)
- Need more distinct signatures
- Working with very short sentences (need more signal)

### Decrease (e.g., 0.05 - 0.01) if:
- Geometry becomes unstable
- Signatures are too noisy
- Working with very long sentences (too much energy)

### Keep Default (0.1) if:
- Standard use case
- Balanced sentence lengths
- Stable training

---

## Relationship to Other Parameters

### `lattice_size`
- Larger lattice = more cells = can handle higher impulse
- Smaller lattice = fewer cells = need lower impulse to avoid saturation

### `collapse_steps`
- More collapse steps = can handle higher impulse (more time to stabilize)
- Fewer collapse steps = need lower impulse (less time to stabilize)

### `num_clusters`
- More clusters = can benefit from higher impulse (more distinct signatures)
- Fewer clusters = lower impulse may be better (less need for distinction)

---

## Mathematical View

The impulse scale is a **multiplicative factor** in the energy injection:

```
SW_new(x, y, z) = SW_old(x, y, z) + impulse_scale * base_impulse
```

Where:
- `SW` = Symbolic Weight (energy in cell)
- `base_impulse` = Raw impulse from character/token (typically -1.0 to 1.0)
- `impulse_scale` = Scaling factor (default 0.1)

**Total energy injected** for a sentence:
```
Total_Energy = sum(|impulse_scale * base_impulse_i| for all tokens i)
```

---

## Summary

**`impulse_scale`** controls:
- ✅ How much energy each character/token injects into geometry
- ✅ How distinct different sentences' signatures are
- ✅ How dramatic the collapse process is
- ✅ The stability vs. sensitivity trade-off

**Default: 0.1** (balanced for most use cases)

**Range**: Typically 0.01 (subtle) to 1.0 (strong), but 0.1 is optimal for most cases.

---

## Quick Reference

| `impulse_scale` | Effect | Use Case |
|----------------|--------|----------|
| 0.01 | Very subtle, signatures similar | Experimental, very stable systems |
| 0.05 | Subtle, some distinction | Long sentences, conservative |
| **0.1** | **Balanced (default)** | **Standard use** |
| 0.2 | Moderate, good distinction | Short sentences, need more signal |
| 0.5 | Strong, very distinct | Need maximum separation |
| 1.0 | Very strong, potentially unstable | Experimental only |

**Recommendation**: Start with 0.1, adjust only if you see specific issues (too similar signatures → increase, instability → decrease).

