# Pre-Core Systems Archive

This folder contains the **pre-core quantum systems** that existed before the unified `core/` system was built.

## Contents

### `quantum/` - Pre-Core Quantum Systems

This folder contains the original quantum systems that were built before the unified Livnium Core System:

1. **`quantum/hierarchical/`** - Hierarchical Geometry System
   - 3-level geometry-in-geometry architecture
   - Used for Ramsey number solving
   - Contains Grover's and Shor's algorithms
   - **Status**: To be integrated into `core/`

2. **`quantum/islands/`** - Islands System
   - Quantum-inspired islands architecture
   - 105-500+ qubit-analogous units
   - Feature-based semantic analysis
   - **Status**: To be integrated into `core/`

3. **`quantum/livnium_core/`** - Livnium Core 1D
   - 1D Transverse-Field Ising Model (TFIM) solver
   - DMRG/MPS tensor-network implementation
   - Real physics solver
   - **Status**: Reference implementation

4. **`quantum/shared/`** - Shared Utilities
   - Common quantum utilities
   - **Status**: May be integrated into `core/`

### `docs/` - Pre-Core Documentation

This folder contains documentation for the pre-core systems:

1. **`docs/hierarchical/`** - Hierarchical System Documentation
   - `GEOMETRY_SYSTEM_COMPLETE.md` - Complete geometry system docs
   - `OMCUBE_VS_LIVNIUM_CORE_REPORT.md` - System comparison report
   - **Status**: Reference documentation

2. **`docs/islands/`** - Islands System Documentation
   - Islands system documentation
   - **Status**: Reference documentation

3. **`docs/livnium_core/`** - Livnium Core 1D Documentation
   - 1D TFIM solver documentation
   - **Status**: Reference documentation

4. **`docs/project/`** - Project Documentation
   - General project documentation
   - **Status**: May be integrated into `core/` docs

5. **`docs/setup/`** - Setup Documentation
   - Setup and installation guides
   - **Status**: May be integrated into `core/` docs

6. **`docs/tests/`** - Test Documentation
   - Test results and reports
   - **Status**: Reference documentation

## Migration Status

**Date**: January 2025

**Status**: ⚠️ **ARCHIVED - DO NOT USE DIRECTLY**

These systems are archived because:
- New unified `core/` system is being built
- These systems need to be integrated into `core/`
- Quantum algorithms (Grover, Shor) exist here but not in `core/`

## Integration Plan

These systems should be integrated into `core/`:

1. **Quantum Algorithms** (`quantum/hierarchical/algorithms/`)
   - Grover's algorithm → `core/algorithms/grovers.py`
   - Shor's algorithm → `core/algorithms/shor.py`
   - VQE, QAOA → `core/algorithms/`

2. **Hierarchical Geometry** (`quantum/hierarchical/geometry/`)
   - Already has Layer 0 (Recursive Geometry) in `core/recursive/`
   - May need to integrate additional features

3. **Islands System** (`quantum/islands/`)
   - May be integrated as an alternative architecture
   - Or kept as reference implementation

## Important Notes

- **Do not use code from this archive directly** - Use `core/` instead
- **Imports have changed** - Old imports will not work
- **This is a backup** - Keep until integration is complete
- **Integration needed** - Quantum algorithms need to be moved to `core/`

## Related Documentation

- `core/MISSING_FEATURES.md` - Lists quantum algorithms as missing from core
- `core/ARCHITECTURE.md` - New unified architecture
- `docs/hierarchical/OMCUBE_VS_LIVNIUM_CORE_REPORT.md` - System comparison

