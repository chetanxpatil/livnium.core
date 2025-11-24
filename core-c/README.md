# Livnium-C System — Stand-Alone Circular Semantic Engine

Complete specification and implementation of the **Livnium-C System** — a pure circular semantic engine independent of Livnium Core and Livnium-T.

**Stand-Alone:** Not dependent on Livnium Core or Livnium-T  
**Circular:** Pure 2D ring geometry, no cubic or tetrahedral structures  
**Independent:** Parallel system with its own mechanics  
**Complete:** Self-contained axiomatic foundation

---

## 📖 Main References

**👉 [LIVNIUM_C_LAWS.md](LIVNIUM_C_LAWS.md) - Complete canonical axiomatic specification**

All axioms, derived laws, and implementation principles are documented in the canonical specification.

---

## Quick Overview

Livnium-C implements a **circular semantic engine** using:

- **1+N topology**: 1 central core (Om) + N ring nodes
- **Two-class system**: Core (f=0) and Ring (f=1) only
- **Symbolic Weight**: SW_C = 9·f, ΣSW_C = 9N (canonical)
- **Rotation group**: Cyclic rotations C_N (order N)
- **Conservation ledger**: Invariant quantities preserved

**Critical Distinction:** Livnium-C is **NOT a circular lattice** like cubes have a lattice. It is a **1+N topological object** with circular adjacency—the simplest periodic universe.

---

## The Six Axioms

**Core Axioms:**

1. **C-A1**: Canonical Circle Alphabet (1+N circular structure)
2. **C-A2**: Observer Anchor & Frame (Om-Core)
3. **C-A3**: Exposure Law (Circular Boundary Class)
4. **C-A4**: Symbolic Weight Law (SW_C = k_C·f)
5. **C-A5**: Dynamic Law (Cyclic Rotation Group)
6. **C-A6**: Connection & Activation Rule

**Derived Laws:**

- **C-D1**: Circle Equilibrium Constant (K_C)
- **C-D2**: Exposure Density Law
- **C-D3**: Conservation Ledger
- **C-D4**: Perfect Reversibility Law
- **C-D5**: Base-(N+1) Encoding Law

See [LIVNIUM_C_LAWS.md](LIVNIUM_C_LAWS.md) for complete details on each axiom and law.

---

## Key Differences from Livnium Core and Livnium-T

| Feature | Livnium Core | Livnium-T | Livnium-C |
|---------|--------------|-----------|-----------|
| **Structure** | 3×3×3 lattice (27 cells) | 5-node topology (1 core + 4 vertices) | 1+N circle (1 core + N ring) |
| **Geometry** | Cubic (Cartesian) | Tetrahedral (topological) | Circular (2D periodic) |
| **Classes** | 4 classes (Core, Center, Edge, Corner) | 2 classes (Core, Vertex) | 2 classes (Core, Ring) |
| **Exposure** | f ∈ {0,1,2,3} | f ∈ {0,3} only | f ∈ {0,1} only |
| **SW Formula** | SW = 9·f | SW = 9·f (same) | SW = 9·f (same) |
| **Total SW** | ΣSW = 486 | ΣSW = 108 | ΣSW = 9N |
| **Rotation Group** | Cubic (24 elements) | Tetrahedral (12 elements) | Cyclic C_N (N elements) |
| **Complexity** | Higher (4 classes, 27 cells) | Minimal (2 classes, 5 nodes) | Simplest (2 classes, 1+N nodes) |

**Livnium-C is NOT Livnium Core or Livnium-T.** It is a parallel, independent system.

---

## Structure

```
core-c/
├── README.md                # This file
├── LIVNIUM_C_LAWS.md        # Canonical geometric specification
├── __init__.py              # Package exports
├── demo.py                  # Classical demo
├── classical/               # Classical geometric system
│   ├── __init__.py
│   └── livnium_c_system.py
└── tests/                   # Test suite
    ├── __init__.py
    └── test_livnium_c.py
```

---

## Canonical Values

**For N ring nodes:**

- **Total nodes**: 1 + N
- **Core**: 1 node, f=0, SW=0
- **Ring**: N nodes, f=1, SW=9 each
- **Total SW**: ΣSW_C = 9N
- **Equilibrium Constant**: K_C = 9
- **Rotation Group**: C_N (order N)
- **Encoding Base**: Base-(N+1)

**Example (N=8):**

- Total nodes: 9
- Total SW: 72
- Rotation group: C_8 (8 rotations)
- Encoding base: Base-9

**Example (N=12):**

- Total nodes: 13
- Total SW: 108
- Rotation group: C_12 (12 rotations)
- Encoding base: Base-13

---

## Verification Status

⏳ **Planned Tests:**

- **S1–S4**: Structure tests (circle structure, adjacency, exposure, coordinates)
- **R1–R3**: Rotation tests (bijection, orientation, adjacency)
- **C1**: Connection test (core-to-ring coupling)
- **L1**: Ledger test (conservation invariants)

---

## Implementation Principles

1. **Polar Coordinates**: Use polar coordinates for exact symmetry
2. **No Overlap**: Never allow ring nodes to overlap—geometry must remain strict
3. **Exposure Tracking**: Track exposure class counts at every step
4. **Rotation Group**: Implement only cyclic rotation group C_N (no reflections)
5. **Om Immovability**: Treat Om-core as immovable anchor

---

## Why Livnium-C?

**Circular geometry offers:**

- **Simplest periodic structure**: 1+N nodes (1 core + N ring) vs 27 cells or 5 nodes
- **Two-class system**: Only Core (f=0) and Ring (f=1)
- **Clean algebra**: Perfect symmetry with simple formulas
- **Minimal complexity**: Simplest non-trivial periodic structure
- **Parallel system**: Independent from cubic and tetrahedral geometry
- **Canonical SW**: ΣSW_C = 9N (circle equivalent of 486 for cube and 108 for tetra)

**Use cases:**

- Semantic analysis requiring circular/periodic structure
- Geometric reasoning with cyclic geometry
- Phase and periodic phenomena
- Parallel semantic engines alongside Livnium Core and Livnium-T

---

## Status

✅ **Canonical Specification Complete**  
⏳ **Implementation In Progress**  
⏳ **Test Suite In Progress**

---

## References

- **Specification**: [LIVNIUM_C_LAWS.md](LIVNIUM_C_LAWS.md)
- **Livnium Core**: `../core/` (parallel system)
- **Livnium-T**: `../core-t/` (parallel system)
- **Documentation**: This file

---

**Version**: 1.0  
**Last Updated**: 2025-11-24  
**Status**: ✅ Specification Complete, Implementation Pending

