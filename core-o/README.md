# Livnium-O System — Stand-Alone Spherical Semantic Engine

Complete specification and implementation of the **Livnium-O System** — a pure spherical semantic engine independent of Livnium Core, Livnium-T, and Livnium-C.

**Stand-Alone:** Not dependent on other Livnium systems  
**Spherical:** Pure 3D sphere geometry with continuous surface  
**Independent:** Parallel system with its own mechanics  
**Complete:** Self-contained axiomatic foundation

---

## 📖 Main References

**👉 [LIVNIUM_O_LAWS.md](LIVNIUM_O_LAWS.md) - Complete canonical axiomatic specification**

All axioms, derived laws, and implementation principles are documented in the canonical specification.

---

## Quick Overview

Livnium-O implements a **spherical semantic engine** using:

- **1+N topology**: 1 central core sphere (Om, radius=1) + N neighbor spheres with arbitrary radii
- **Continuous exposure**: Based on solid angle coverage
- **Symbolic Weight**: SW_O = 9·f, ΣSW_O = 9N (canonical)
- **Rotation group**: Spherical rotations SO(3) (continuous)
- **Generalized kissing constraint**: Fundamental packing law
- **Conservation ledger**: Invariant quantities preserved

**Critical Distinction:** Livnium-O is **NOT a spherical lattice** like cubes have a lattice. It is a **1+N topological object** with spherical adjacency—the most general continuous universe.

---

## The Six Axioms

**Core Axioms:**

1. **O-A1**: Canonical Sphere Alphabet (1+N spherical structure)
2. **O-A2**: Observer Anchor & Frame (Om-Sphere)
3. **O-A3**: Exposure Law (Solid Angle System)
4. **O-A4**: Symbolic Weight Law (SW_O = k_O·f)
5. **O-A5**: Dynamic Law (Generalized Kissing Constraint)
6. **O-A6**: Connection & Activation Rule

**Derived Laws:**

- **O-D1**: Sphere Equilibrium Constant (K_O)
- **O-D2**: Exposure Density Law
- **O-D3**: Conservation Ledger
- **O-D4**: Perfect Reversibility Law
- **O-D5**: Base-(N+1) Encoding Law

See [LIVNIUM_O_LAWS.md](LIVNIUM_O_LAWS.md) for complete details on each axiom and law.

---

## The Generalized Kissing Constraint

**The Fundamental Law:**

\[
\boxed{\sum_{i=1}^{N} \left(1 - \sqrt{1 - \left(\frac{r_i}{1 + r_i}\right)^2}\right) \le 2}
\]

This is the **structural law** that determines which configurations of neighbor spheres can simultaneously kiss the core without overlapping.

**What it means:**

- Each neighbor sphere with radius \(r_i\) covers a spherical cap on the core
- The cap has solid angle proportional to \(1 - \sqrt{1 - \left(\frac{r_i}{1 + r_i}\right)^2}\)
- The sum of all caps' normalized solid angles must be ≤ 2
- This ensures neighbors can physically pack around the core

**For uniform radius \(r\):**

The maximum number of neighbors (generalized kissing number) is:

\[
n_{\max}(r) = \left\lfloor \frac{2}{1 - \sqrt{1 - \left(\frac{r}{1 + r}\right)^2}} \right\rfloor
\]

**Examples:**

- \(r = 1\): \(n_{\max} \approx 6\) (classical kissing number)
- \(r = 0.5\): \(n_{\max} \approx 12\)
- \(r = 0.1\): \(n_{\max} \approx 60\)

---

## Key Differences from Other Livnium Systems

| Feature | Livnium Core | Livnium-T | Livnium-C | Livnium-O |
|---------|--------------|-----------|-----------|-----------|
| **Structure** | 3×3×3 lattice (27 cells) | 5-node topology (1 core + 4 vertices) | 1+N circle (1 core + N ring) | 1+N sphere (1 core + N neighbors) |
| **Geometry** | Cubic (Cartesian) | Tetrahedral (topological) | Circular (2D periodic) | Spherical (3D continuous) |
| **Classes** | 4 classes | 2 classes | 2 classes | Continuous |
| **Exposure** | f ∈ {0,1,2,3} | f ∈ {0,3} | f ∈ {0,1} | f ∈ [0,1] continuous |
| **SW Formula** | SW = 9·f | SW = 9·f | SW = 9·f | SW = 9·f |
| **Total SW** | ΣSW = 486 | ΣSW = 108 | ΣSW = 9N | ΣSW = 9N |
| **Rotation Group** | Cubic (24) | Tetrahedral A₄ (12) | Cyclic C_N (N) | Spherical SO(3) (continuous) |
| **Complexity** | Higher | Minimal | Simplest | Most general |

**Livnium-O is NOT other Livnium systems.** It is a parallel, independent system.

---

## Structure

```
core-o/
├── README.md                # This file
├── LIVNIUM_O_LAWS.md        # Canonical geometric specification
├── __init__.py              # Package exports
├── demo.py                  # Classical demo
├── classical/               # Classical geometric system
│   ├── __init__.py
│   └── livnium_o_system.py
└── tests/                   # Test suite
    ├── __init__.py
    └── test_livnium_o.py
```

---

## Canonical Values

**For N neighbor spheres:**

- **Total elements**: 1 + N
- **Core**: 1 sphere, radius=1, f=0, SW=0
- **Neighbors**: N spheres, radii={r_i}, f=f_i, SW=9·f_i each
- **Total SW**: ΣSW_O = 9N (for uniform exposure)
- **Equilibrium Constant**: K_O = 9
- **Rotation Group**: SO(3) (continuous)
- **Encoding Base**: Base-(N+1)
- **Kissing Constraint**: \(\sum_i w_i \le 2\) where \(w_i = 1 - \sqrt{1 - \left(\frac{r_i}{1 + r_i}\right)^2}\)

---

## Verification Status

⏳ **Planned Tests:**

- **S1–S4**: Structure tests (sphere structure, tangency, exposure, coordinates)
- **R1–R3**: Rotation tests (bijection, orientation, tangency)
- **K1**: Kissing constraint test (valid/invalid configurations)
- **L1**: Ledger test (conservation invariants)

---

## Implementation Principles

1. **Polar Coordinates**: Use spherical coordinates for exact symmetry
2. **No Overlap**: Never allow neighbors to overlap—enforced by kissing constraint
3. **Exposure Tracking**: Track solid angle coverage at every step
4. **Rotation Group**: Implement spherical rotation group SO(3) (no reflections)
5. **Om Immovability**: Treat Om-core as immovable anchor
6. **Kissing Constraint**: Always verify \(\sum_i w_i \le 2\)

---

## Why Livnium-O?

**Spherical geometry offers:**

- **Most general structure**: 1+N elements with arbitrary radii
- **Continuous exposure**: Based on solid angle, not discrete classes
- **Clean algebra**: Perfect symmetry with simple formulas
- **Heterogeneous configurations**: Different neighbor radii allowed
- **Parallel system**: Independent from cubic, tetrahedral, and circular geometry
- **Canonical SW**: ΣSW_O = 9N (sphere equivalent of other systems)
- **Generalized kissing**: Works for arbitrary radii, not just uniform

**Use cases:**

- Semantic analysis requiring spherical/continuous structure
- Geometric reasoning with heterogeneous configurations
- Continuous and periodic phenomena
- Parallel semantic engines alongside other Livnium systems
- Generalized packing problems

---

## Status

✅ **Canonical Specification Complete**  
⏳ **Implementation In Progress**  
⏳ **Test Suite In Progress**

---

## References

- **Specification**: [LIVNIUM_O_LAWS.md](LIVNIUM_O_LAWS.md)
- **Livnium Core**: `../core/` (parallel system)
- **Livnium-T**: `../core-t/` (parallel system)
- **Livnium-C**: `../core-c/` (parallel system)
- **Documentation**: This file

---

**Version**: 1.0  
**Last Updated**: 2025-11-24  
**Status**: ✅ Specification Complete, Implementation Pending

