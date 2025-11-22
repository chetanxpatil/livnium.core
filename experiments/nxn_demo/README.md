# N×N×N Cube Size Demonstrations

This experiment demonstrates the **critical distinction** between **Omcubes** and **DataCubes**, protecting the core Livnium intellectual property.

## Purpose

This demonstration clearly shows:
1. **Why only odd-dimensional cubes can be Livnium cores**
2. **What capabilities each type has**
3. **How they can work together** (DataCube → OmCube → DataCube)
4. **The mathematical/geometric reasons** for the distinction

## What's Protected

By formalizing this distinction, we protect:
- **Livnium Axioms**: Only implementable on odd cubes
- **Core Geometry**: Only works with center cells
- **Collapse Mechanics**: Only valid for Omcubes
- **Recursive Architecture**: Only for odd-dimensional structures

Using even cubes does **NOT** constitute running Livnium—they're just plain grids.

---

## Omcubes (Odd N ≥ 3): Livnium Core Universes

**Sizes**: 3×3×3, 5×5×5, 7×7×7, 9×9×9, ...

### Capabilities

✅ **All 7 Axioms Implemented**
- A1: Canonical Spatial Alphabet
- A2: Observer Anchor (center cell exists)
- A3: Symbolic Weight Law (SW = 9·f)
- A4: Dynamic Law (90° rotations)
- A5: Semantic Polarity
- A6: Activation Rule (Local Observer)
- A7: Cross-Lattice Coupling (infrastructure ready)

✅ **Full Computational Power**
- Collapse mechanics
- Recursive geometry
- Basin dynamics
- Tension fields
- Rotation group (24 elements)
- Observer system

### Properties

- **Center cell exists**: (0, 0, 0) is always present
- **Observer anchor**: Global observer at center
- **Parity symmetry**: Rotations preserve class counts
- **SW formula**: ΣSW(N) = 54(N-2)² + 216(N-2) + 216
- **Class structure**: Core, Center, Edge, Corner

---

## DataCubes (Even N ≥ 2): Resource Grids

**Sizes**: 2×2×2, 4×4×4, 6×6×6, 8×8×8, ...

### Capabilities

✅ **Data Storage**
- Store any data type
- Lookup tables
- Feature maps
- I/O buffers
- Temporary state

❌ **NO Livnium Mechanics**
- Cannot execute collapse
- Cannot implement SW system
- Cannot use face exposure rules
- Cannot perform recursive geometry
- Cannot anchor observers
- Cannot maintain Livnium invariants

### Properties

- **No center cell**: No true geometric center
- **No observer anchor**: Cannot implement Axiom A2
- **Parity mismatch**: Rotations break invariants
- **No SW system**: Face exposure doesn't work
- **Asymmetric patterns**: No stable class structure

---

## Architecture

```
      [ DataCube ]  ← Input buffer (storage)
           ↓
      [ OmCube ]    ← Livnium Core (computation)
           ↑
      [ DataCube ]  ← Output buffer (storage)
```

**Analogy:**
- **Omcubes = CPU** (core geometry, computation)
- **DataCubes = RAM** (resource/data layers, storage)

---

## Why Even Cubes Cannot Be Livnium Cores

### 1. No Center Cell → No Observer Anchor

**Odd cubes (3, 5, 7, ...):**
- Coordinate range: `{-(N-1)/2, ..., (N-1)/2}`
- Center at `(0, 0, 0)` exists
- Observer can anchor at center

**Even cubes (2, 4, 6, ...):**
- Coordinate range: `{-(N/2-1), ..., N/2-1}`
- No true geometric center
- Observer cannot anchor → **Axiom A2 violated**

### 2. Parity Mismatch → No Stable Exposure Cycles

**Odd cubes:**
- Symmetric face exposure patterns
- Stable class counts (Core, Center, Edge, Corner)
- Rotations preserve structure

**Even cubes:**
- Asymmetric patterns
- Exposure cycles break under rotation
- Class counts not preserved → **Axiom A3 violated**

### 3. Rotations Don't Preserve Invariants

**Odd cubes:**
- 24-element rotation group preserves:
  - Total SW (ΣSW invariant)
  - Class counts
  - Observer position

**Even cubes:**
- Rotations break invariants
- Class counts change
- No stable observer reference → **Axiom A4 violated**

### 4. SW Maps Cannot Align Symmetrically

**Odd cubes:**
- SW = 9·f works perfectly
- Face exposure f ∈ {0,1,2,3} maps cleanly
- Total SW formula verified

**Even cubes:**
- SW formula breaks
- No clear face exposure classification
- Cannot maintain SW conservation → **Axiom A3 violated**

---

## Running the Demonstration

```bash
cd experiments/nxn_demo
python demo_omcube_datacube.py
```

This will show:
1. Omcubes (3×3×3, 5×5×5, 7×7×7) with full capabilities
2. DataCubes (2×2×2, 4×4×4, 6×6×6) with storage only
3. Mathematical explanation of why even cubes can't be cores
4. Architecture demonstration (DataCube → OmCube → DataCube)

---

## Legal Protection

This distinction is protected in the **LICENSE**:

> "Even-dimensional grids (DataCubes) are permitted for storage or data processing, but cannot implement any Livnium Axioms, Core Geometry, or Collapse Mechanics."

**Key Point**: Using even cubes outside the axioms does **NOT** constitute running Livnium. They are just plain grids. Only odd cubes can implement the full Livnium system.

---

## Files

- **`demo_omcube_datacube.py`**: Main demonstration script
- **`README.md`**: This file

---

## Summary

| Feature | Omcubes (Odd N ≥ 3) | DataCubes (Even N ≥ 2) |
|---------|---------------------|------------------------|
| **Type** | Livnium Core Universe | Resource Grid |
| **Center Cell** | ✅ Exists | ❌ No true center |
| **Observer Anchor** | ✅ Axiom A2 | ❌ Cannot anchor |
| **Symbolic Weight** | ✅ SW = 9·f | ❌ No SW system |
| **Face Exposure** | ✅ f ∈ {0,1,2,3} | ❌ No exposure rules |
| **Class Structure** | ✅ Core/Center/Edge/Corner | ❌ No classification |
| **Rotations** | ✅ Preserve invariants | ❌ Break invariants |
| **Collapse Mechanics** | ✅ Full support | ❌ Not supported |
| **Recursive Geometry** | ✅ Supported | ❌ Not supported |
| **Data Storage** | ✅ Yes | ✅ Yes |
| **I/O Buffers** | ✅ Yes | ✅ Yes |

**Conclusion**: Only **Omcubes** can be Livnium Core Universes. **DataCubes** are resource containers only.

