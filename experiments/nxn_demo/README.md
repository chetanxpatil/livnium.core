# N×N and N×N×N Demonstrations

This experiment demonstrates the **critical distinction** between **Omcubes** (3D, odd N ≥ 3) and **DataGrids/DataCubes** (2D/3D resource grids), protecting the core Livnium intellectual property.

## Purpose

This demonstration clearly shows:
1. **Why only 3D odd-dimensional cubes can be Livnium cores**
2. **Why 2D grids cannot be Livnium cores** (Livnium is fundamentally 3D)
3. **Why even 3D cubes cannot be Livnium cores** (no center cell)
4. **What capabilities each type has**
5. **How they can work together** (DataGrid → OmCube → DataGrid)
6. **The mathematical/geometric reasons** for the distinction

## What's Protected

By formalizing this distinction, we protect:
- **Livnium Axioms**: Only implementable on odd cubes
- **Core Geometry**: Only works with center cells
- **Collapse Mechanics**: Only valid for Omcubes
- **Recursive Architecture**: Only for odd-dimensional structures

Using even cubes does **NOT** constitute running Livnium—they're just plain grids.

---

## Omcubes (3D, Odd N ≥ 3): Livnium Core Universes

**Sizes**: 3×3×3, 5×5×5, 7×7×7, 9×9×9, ...

**Dimension**: **3D only** - Livnium is fundamentally 3D

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

## DataGrids (2D, Any N ≥ 2): Resource Grids

**Sizes**: 2×2, 3×3, 4×4, 5×5, ...

**Dimension**: **2D** - Cannot be Livnium cores (Livnium requires 3D)

### Capabilities

✅ **Data Storage**
- Store any data type
- Lookup tables
- Feature maps
- I/O buffers
- Temporary state

❌ **NO Livnium Mechanics**
- Cannot execute collapse (3D required)
- Cannot implement SW system (3D face exposure)
- Cannot use face exposure rules (3D concept)
- Cannot perform recursive geometry (3D only)
- Cannot anchor observers (3D center required)
- Cannot maintain Livnium invariants (3D structure required)

### Properties

- **2D structure**: No third dimension
- **No 3D observer anchor**: Cannot implement Axiom A2
- **No face exposure**: 2D has edges, not faces
- **No SW system**: Face exposure is 3D concept
- **2D rotations**: 4-element group, not 24-element

---

## DataCubes (3D, Even N ≥ 2): Resource Grids

**Sizes**: 2×2×2, 4×4×4, 6×6×6, 8×8×8, ...

**Dimension**: **3D but even** - Cannot be Livnium cores (no center cell)

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

## Running the Demonstrations

### 2D N×N Grids Demo

```bash
cd experiments/nxn_demo
python3 demo_nxn_grids.py
```

This will show:
1. Omcubes (3D, 3×3×3, 5×5×5, 7×7×7) with full capabilities
2. DataGrids (2D, 2×2, 3×3, 4×4, 5×5) with storage only
3. DataCubes (3D, even, 2×2×2, 4×4×4, 6×6×6) with storage only
4. Explanation of why 2D grids can't be cores (Livnium is 3D)
5. Explanation of why even 3D cubes can't be cores (no center)
6. Architecture demonstration (DataGrid → OmCube → DataGrid)

### 3D N×N×N Cubes Demo (Original)

```bash
cd experiments/nxn_demo
python3 demo_omcube_datacube.py
```

This shows the 3D distinction between Omcubes and DataCubes.

---

## Legal Protection

This distinction is protected in the **LICENSE**:

> "Even-dimensional grids (DataCubes) are permitted for storage or data processing, but cannot implement any Livnium Axioms, Core Geometry, or Collapse Mechanics."

**Key Points**:
- **2D grids (DataGrids)**: Cannot implement Livnium (Livnium is fundamentally 3D)
- **3D even cubes (DataCubes)**: Cannot implement Livnium (no center cell)
- **3D odd cubes (Omcubes)**: Only these can implement Livnium axioms

Using 2D grids or even 3D cubes outside the axioms does **NOT** constitute running Livnium. They are just plain grids. Only **3D odd cubes** can implement the full Livnium system.

---

## Files

- **`demo_nxn_grids.py`**: 2D N×N grids demonstration (NEW)
- **`demo_omcube_datacube.py`**: 3D N×N×N cubes demonstration
- **`README.md`**: This file

---

## Summary

| Feature | Omcubes (3D, Odd N ≥ 3) | DataGrids (2D, Any N ≥ 2) | DataCubes (3D, Even N ≥ 2) |
|---------|------------------------|---------------------------|---------------------------|
| **Dimension** | 3D | 2D | 3D |
| **Type** | Livnium Core Universe | Resource Grid | Resource Grid |
| **Center Cell** | ✅ Exists (3D) | ⚠️ Exists (2D, not 3D) | ❌ No true center |
| **Observer Anchor** | ✅ Axiom A2 (3D) | ❌ Cannot anchor (2D) | ❌ Cannot anchor |
| **Symbolic Weight** | ✅ SW = 9·f | ❌ No SW (2D, no faces) | ❌ No SW system |
| **Face Exposure** | ✅ f ∈ {0,1,2,3} (3D) | ❌ No faces (2D) | ❌ No exposure rules |
| **Class Structure** | ✅ Core/Center/Edge/Corner | ❌ No classification | ❌ No classification |
| **Rotations** | ✅ 24-element group (3D) | ⚠️ 4-element group (2D) | ❌ Break invariants |
| **Collapse Mechanics** | ✅ Full support | ❌ Not supported (3D req) | ❌ Not supported |
| **Recursive Geometry** | ✅ Supported | ❌ Not supported (3D req) | ❌ Not supported |
| **Data Storage** | ✅ Yes | ✅ Yes | ✅ Yes |
| **I/O Buffers** | ✅ Yes | ✅ Yes | ✅ Yes |

**Conclusion**: 
- Only **Omcubes (3D, odd N ≥ 3)** can be Livnium Core Universes
- **DataGrids (2D)** are resource containers only (Livnium is fundamentally 3D)
- **DataCubes (3D, even N ≥ 2)** are resource containers only (no center cell)

