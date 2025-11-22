# Memory System

Persistent memory cells that store and recall information across episodes.

## What is the Memory System?

**Memory that behaves like energy in a lattice.**

This system provides:
- **Per-cell memory capsules**: Each geometric cell has its own working + long-term memory
- **Global memory lattice**: A brain grid overlaying the geometric structure
- **Geometric coupling**: Memory strength tied to symbolic weight and face exposure
- **Decay rules**: Memories fade naturally based on geometry
- **Associative graph**: Memories link together forming connections
- **Cross-step persistence**: Information survives across episodes

## Contents

- **`memory_cell.py`**: Individual memory cell with MemoryCapsule (working + long-term memory)
- **`memory_lattice.py`**: Lattice of memory cells (1-to-1 overlay on geometric structure)
- **`memory_coupling.py`**: Coupling between memory cells and geometric properties

## Purpose

This module provides:
- **Persistent storage**: Information survives across episodes
- **Associative recall**: Memory cells can be queried by content
- **Geometric coupling**: Memory strength derived from symbolic weight and face exposure
- **Decay**: Old memories fade over time (geometric decay rules)
- **State transitions**: Working memory → long-term memory consolidation

Used by NLI system and other learning applications. The memory system integrates seamlessly with LivniumCoreSystem, using the same coordinate structure.

