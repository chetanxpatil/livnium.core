# MPS Components Archived from Hierarchical System

## Why These Were Archived

These MPS (Matrix Product State) components were removed from `quantum/hierarchical/` to maintain **architectural purity** in LIVNIUM.

## LIVNIUM's Clean Architecture

LIVNIUM has **three distinct worlds**:

1. **`quantum/islands/`** - Quantum-inspired information processing
2. **`quantum/hierarchical/`** - Geometry-in-geometry architecture (quantum-inspired classical)
3. **`quantum/livnium_core/`** - **REAL** tensor networks (MPS, DMRG) for physics

## The Problem

MPS (Matrix Product States) is a **real tensor network method** used in quantum many-body physics. Having MPS inside the hierarchical system created confusion:

- ❌ Mixed "quantum-inspired" with "real physics"
- ❌ Duplicated functionality (MPS exists in both `hierarchical/` and `livnium_core/`)
- ❌ Incomplete implementation (two-qubit gates were placeholders)
- ❌ Misleading tests (appeared more complete than they were)

## What Was Archived

### 1. MPS Simulator
- **File**: `simulators/mps_hierarchical_simulator.py`
- **Reason**: MPS belongs in `livnium_core/` (real physics), not `hierarchical/` (geometry)

### 2. MPS Base Geometry
- **File**: `geometry/level0/mps_base_geometry.py`
- **Reason**: Tensor networks don't fit geometry-in-geometry architecture

### 3. MPS Geometry in Geometry
- **File**: `geometry/level1/mps_geometry_in_geometry.py`
- **Reason**: Contains incomplete two-qubit gate implementation (placeholder code)

### 4. MPS Test
- **File**: `tests/test_5000_qubits.py` (Test 3)
- **Reason**: Tested incomplete MPS functionality

## Result

After archiving:

✅ **Hierarchical** = Pure geometry-in-geometry architecture  
✅ **Livnium Core** = Real MPS/DMRG tensor networks  
✅ **Clear separation** = No confusion about what's real vs. quantum-inspired

## Where to Find Real MPS

For **real** MPS/DMRG tensor network physics simulation, see:

- `quantum/livnium_core/` - Complete DMRG/MPS implementation
- `quantum/livnium_core/livnium_core_1D.py` - 1D TFIM ground state solver

## Status

These files are preserved for reference but should **not** be used in production. They represent an architectural experiment that didn't fit LIVNIUM's clean separation of concerns.

---

*Archived to maintain architectural purity: hierarchical = geometry, livnium_core = real physics*

