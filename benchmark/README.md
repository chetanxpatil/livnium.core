# Benchmark Suite

Standard benchmark problems solved using Livnium's geometric approach.

## Contents

- **`max_cut/`**: Max-Cut problem solver using GSET benchmark graphs
  - Implements tension-based optimization for graph partitioning
  - Compares against greedy baseline and literature values

- **`sat/`**: Boolean Satisfiability (SAT) solver
  - Uses Livnium's basin search for CNF formula satisfaction
  - Tests on SATLIB benchmark instances

- **`csp/`**: Constraint Satisfaction Problem solver
  - Handles N-Queens, Graph Coloring, Sudoku
  - Uses geometric constraint encoding

- **`sat_solver_livnium.py`**: Standalone SAT solver implementation

## Purpose

These benchmarks demonstrate Livnium's capabilities on standard optimization problems:
- **Tension minimization**: Problems mapped to energy landscapes
- **Basin search**: Multi-basin exploration for global optima
- **Geometric encoding**: Constraints represented as geometric structures

Each benchmark includes:
- Problem-specific solver
- Baseline comparison
- Basin distribution analysis
- Results visualization

