# Livnium-T System — Stand-Alone Tetrahedral Semantic Engine

Complete specification and implementation of the **Livnium-T System** — a pure tetrahedral semantic engine independent of Livnium Core.

**Stand-Alone:** Not dependent on Livnium Core  
**Tetrahedral:** Pure simplex geometry, no cubic structures  
**Independent:** Parallel system with its own mechanics  
**Complete:** Self-contained axiomatic foundation

---

## 📖 Main References

**👉 [LIVNIUM_T_LAWS.md](LIVNIUM_T_LAWS.md) - Complete canonical axiomatic specification**

**👉 [QUANTUM_T_LAWS.md](QUANTUM_T_LAWS.md) - Quantum layer canonical specification**

All axioms, derived laws, and implementation principles are documented in the canonical specifications.

---

## Quick Overview

Livnium-T implements a **tetrahedral semantic engine** using:

- **5-node topology**: 1 central core (Om) + 4 outer vertices (LOs)
- **Two-class system**: Core (f=0) and Vertex (f=3) only
- **Symbolic Weight**: SWₜ = 9·f, ΣSWₜ = 108 (canonical)
- **Rotation group**: Tetrahedral rotations (order 12)
- **Conservation ledger**: Invariant quantities preserved

**Critical Distinction:** Livnium-T is **NOT a tetrahedral lattice** like cubes have a lattice. It is a **5-node topological object** with simplex adjacency—the minimal universe.

---

## The Six Axioms

**Core Axioms:**

1. **T-A1**: Canonical Simplex Alphabet (5-simplex cluster)
2. **T-A2**: Observer Anchor & Frame (Om-Simplex)
3. **T-A3**: Exposure Law (Simplex Boundary Class)
4. **T-A4**: Symbolic Weight Law (SWₜ = kₜ·f)
5. **T-A5**: Dynamic Law (Tetrahedral Rotation Group)
6. **T-A6**: Connection & Activation Rule

**Derived Laws:**

- **T-D1**: Simplex Equilibrium Constant (Kₜ)
- **T-D2**: Exposure Density Law
- **T-D3**: Conservation Ledger

See [LIVNIUM_T_LAWS.md](LIVNIUM_T_LAWS.md) for complete details on each axiom and law.

---

## Key Differences from Livnium Core

| Feature | Livnium Core | Livnium-T |
|---------|--------------|-----------|
| **Structure** | 3×3×3 lattice (27 cells) | 5-node topology (1 core + 4 vertices) |
| **Geometry** | Cubic (Cartesian) | Tetrahedral (topological) |
| **Classes** | 4 classes (Core, Center, Edge, Corner) | 2 classes (Core, Vertex) |
| **Exposure** | f ∈ {0,1,2,3} | f ∈ {0,3} only |
| **SW Formula** | SW = 9·f | SW = 9·f (same) |
| **Total SW** | ΣSW = 486 | ΣSW = 108 |
| **Rotation Group** | Cubic (24 elements) | Tetrahedral (12 elements) |
| **Complexity** | Higher (4 classes, 27 cells) | Minimal (2 classes, 5 nodes) |

**Livnium-T is NOT Livnium Core.** It is a parallel, independent system.

---

## Structure

```
core-t/
├── README.md                # This file
├── LIVNIUM_T_LAWS.md        # Canonical geometric specification
├── QUANTUM_T_LAWS.md        # Canonical quantum specification
├── __init__.py              # Package exports
├── demo.py                  # Classical demo
├── quantum_demo.py          # Quantum demo
├── classical/               # Classical geometric system
│   ├── __init__.py
│   └── livnium_t_system.py
├── quantum/                 # Quantum layer
│   ├── __init__.py
│   ├── quantum_node.py
│   ├── quantum_gates.py
│   ├── quantum_system.py
│   ├── entanglement_manager.py
│   ├── measurement_engine.py
│   ├── geometry_quantum_coupling.py
│   └── README.md
└── tests/                   # Test suite
    ├── __init__.py
    └── test_livnium_t.py
```

---

## Verification Status

✅ **All Core Tests PASS:**

- **S1–S4**: Structure tests (simplex cluster, adjacency, exposure, barycentric) ✅
- **R1–R3**: Rotation tests (bijection, orientation, adjacency) ✅
- **C1**: Connection test (face-to-face coupling) ✅
- **L1**: Ledger test (conservation invariants) ✅

⏳ **Planned Tests:**

- **H1–H5**: Hierarchical and generalized structure tests

---

## Implementation Principles

1. **Barycentric Coordinates**: Use barycentric coordinates for exact symmetry
2. **No Overlap**: Never allow simplex overlap—geometry must remain strict
3. **Exposure Tracking**: Track exposure class counts at every step
4. **Rotation Group**: Implement only tetrahedral rotation group (no reflections)
5. **Om Immovability**: Treat Om-simplex as immovable anchor

---

## Why Livnium-T?

**Tetrahedral geometry offers:**

- **Minimal structure**: 5 nodes (1 core + 4 vertices) vs 27 cells
- **Two-class system**: Only Core (f=0) and Vertex (f=3)
- **Clean algebra**: Perfect symmetry with simple formulas
- **Minimal complexity**: Simplest non-trivial symmetric structure
- **Parallel system**: Independent from cubic geometry
- **Canonical SW**: ΣSW = 108 (tetrahedron equivalent of 486 for cube)

**Use cases:**

- Semantic analysis requiring tetrahedral structure
- Geometric reasoning with simplex geometry
- Hierarchical systems with natural recursion
- Parallel semantic engines alongside Livnium Core

---

## Status

✅ **Canonical Specification Complete**  
⏳ **Implementation In Progress**  
⏳ **Test Suite In Progress**

---

## References

- **Specification**: [LIVNIUM_T_LAWS.md](LIVNIUM_T_LAWS.md)
- **Livnium Core**: `../core/` (parallel system)
- **Documentation**: This file

---

**Version**: 1.0  
**Last Updated**: 2025-11-24  
**Status**: ✅ Specification Complete, Implementation Pending

