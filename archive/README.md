# Archive - Old Project Structure

This folder contains **archived code** from previous iterations of the project.

## Recent Archive (January 2025)

### `pre_core_systems/` - Pre-Core Quantum Systems
- **Location**: `archive/pre_core_systems/quantum/`
- **Contents**: Hierarchical, Islands, and Livnium Core 1D systems
- **Status**: Archived before integration into unified `core/` system
- **See**: `archive/pre_core_systems/README.md` for details

## Legacy Archive (November 2024)

This folder contains the **old project structure** that has been migrated to the new `quantum/` directory.

## Contents

### `quantum_2/` (migrated to `quantum/islands/`)
- Original quantum-inspired islands system
- 105-500+ qubit-analogous units
- Migrated to: `quantum/islands/`

### `quantum_computer/` (migrated to `quantum/hierarchical/`)
- Original hierarchical geometry system
- 3-level geometry-in-geometry architecture
- Migrated to: `quantum/hierarchical/`

### `livnium_core_demo/` (migrated to `quantum/livnium_core/`)
- Original 1D TFIM physics solver
- DMRG/MPS tensor-network implementation
- Migrated to: `quantum/livnium_core/`

### `quantum_computer_code.zip` (if present)
- Archived zip file of old structure

## Migration Date

November 2024

## Status

✅ **All code has been successfully migrated to the new `quantum/` structure**

These folders are kept as a backup/reference. You can safely delete them once you've verified the new structure works correctly.

## New Structure

All code is now in:
```
quantum/
├── islands/          # From quantum_2/
├── hierarchical/    # From quantum_computer/
├── livnium_core/    # From livnium_core_demo/
└── shared/          # Shared utilities
```

## Important Notes

- **Do not use code from this archive** - Use the new `quantum/` structure instead
- **Imports have changed** - Old imports will not work
- **This is a backup** - Keep until you're confident the migration is complete

## Archive Organization

For detailed information about archived broken/experimental components, see:
- `docs/archive/README_ARCHIVE.md` - Complete archive documentation

