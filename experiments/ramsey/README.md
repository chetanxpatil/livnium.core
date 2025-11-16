# Ramsey Number Solver

This directory contains the Ramsey number solver that uses 5000 omcubes (quantum-inspired classical search) to find 2-colorings of complete graphs that avoid monochromatic cliques.

## Files

- **`ramsey_number_solver.py`** - Main solver implementation
- **`ramsey_demo.py`** - Quick demo on small problems
- **`test_ramsey_solver.py`** - Test suite for known Ramsey numbers

## Problem

Find a 2-coloring of edges of K_n (complete graph on n vertices) that avoids:
- A red clique of size k
- A blue clique of size k

If such a coloring exists, we prove **R(k,k) > n**.

## Approach

- Use 5000 omcubes (geometric states) to represent different colorings
- Each omcube encodes a partial or complete edge coloring
- Use hierarchical system to explore search space
- Check constraints (no monochromatic k-cliques)
- Use symmetry reduction to avoid duplicate colorings

## Important: Quantum-Inspired Classical Search

**This is NOT quantum computing.** Each omcube is a CLASSICAL state (a RamseyGraph object), not a quantum superposition. We use 5000 omcubes = 5000 parallel classical universes exploring the search space together.

**What makes it "quantum-inspired":**
- Parallel exploration of many configurations simultaneously
- Geometric interference patterns (via coordinate evolution)
- Early collapse of invalid paths (constraint checking)
- Amplification of promising regions (elite archiving)
- Structured search through hierarchical geometry

This is **quantum-INSPIRED evolutionary search**, not quantum computing.

## Dependencies

The solver now uses the new **Livnium Core System**:
- `core/classical/livnium_core_system.py` - Main geometric system
- `core/recursive/recursive_geometry_engine.py` - Recursive geometry (optional)
- `core/config.py` - Configuration

Optional (from archive):
- `archive/pre_core_systems/quantum/hierarchical/monitoring/dual_cube_monitor.py` - Dual cube monitor (if available)

## Usage

### Quick Demo

```bash
python3 experiments/ramsey/ramsey_demo.py
```

### Full Solver

```bash
python3 experiments/ramsey/ramsey_number_solver.py --n 45 --k 5 --omcubes 5000
```

### With Dual Cube Monitor

```bash
python3 experiments/ramsey/ramsey_number_solver.py --n 45 --k 5 --omcubes 5000 --dual-monitor
```

### Test Suite

```bash
python3 experiments/ramsey/test_ramsey_solver.py
```

## Example: R(5,5) > 45

```bash
python3 experiments/ramsey/ramsey_number_solver.py --n 45 --k 5 --omcubes 5000
```

This attempts to find a 2-coloring of K_45 that avoids monochromatic K_5, which would prove R(5,5) > 45.

## Output

If successful, the solver will:
1. Print a success message: `✅ SUCCESS: R(k,k) > n`
2. Save the witness coloring to `ramsey_Rk_k_nN.txt`
3. Save metrics (if dual monitor enabled) to `logs/ramsey_dual_cube_metrics_Rk_k_nN.jsonl`

## Notes

- ✅ **Now uses the new `core/` system** - Fully integrated with Livnium Core System
- The solver uses `LivniumCoreSystem` for geometric operations
- Rotations are applied directly to the core system
- The dual cube monitor (if available) provides semantic guidance for the search
- Omcube states are mapped to lattice cells for efficient storage

