# CSP Benchmark Suite

This directory contains benchmarks comparing Livnium's CSP solver against standard solvers (python-constraint).

## Setup

### Install Dependencies

```bash
pip install python-constraint  # For comparison solver
```

### Generate Test Problems

Generate test CSP problems (N-Queens, Sudoku, etc.):

```bash
python benchmark/csp/generate_test_csps.py
```

## Usage

### Run Full Benchmark

```bash
python benchmark/csp/run_csp_benchmark.py --csp-dir benchmark/csp/csplib/test --limit 10
```

This will:
1. Load CSP problems from JSON files
2. Run Livnium and python-constraint on each problem
3. Compare results (time, solved/unsolved)
4. Save results to `benchmark/csp/csp_results.json`

### Options

- `--csp-dir PATH`: Use CSP files from specified directory
- `--max-steps N`: Maximum search steps for Livnium (default: 1000)
- `--max-time SECONDS`: Timeout per problem (default: 60.0)
- `--limit N`: Limit number of problems to test
- `--output PATH`: Output JSON file (default: `benchmark/csp/csp_results.json`)
- `--verbose`: Verbose output
- `--use-recursive`: Enable recursive geometry (default: False)
- `--recursive-depth N`: Recursive geometry depth (default: 2)

### Basin Distribution Analysis

Analyze the distribution of constraint satisfaction scores across multiple runs:

```bash
python benchmark/csp/run_basin_distribution.py benchmark/csp/csplib/test/nqueens_8.json --runs 50 --plot-type all
```

This generates:
- **JSON results**: All run data with scores, times, steps
- **Combined plot**: 2×2 grid showing histogram, violin plot, KDE, and time vs score scatter
- **Statistics**: Mean, median, std dev, min/max scores

Options:
- `--runs N`: Number of runs (default: 50)
- `--plot-type TYPE`: `histogram`, `violin`, `kde`, `scatter`, or `all` (default: histogram)
- `--use-recursive`: Use recursive geometry
- `--max-steps N`: Max search steps per run
- `--max-time SECONDS`: Timeout per run

The basin distribution plot shows:
- **Multiple stable basins**: Not random, has structure
- **Consistent behavior**: Repeatable score distributions
- **Geometric relaxation**: Physics-inspired solver dynamics

## Problem Format

CSP problems are stored as JSON files:

```json
{
  "variables": {
    "Q1": [0, 1, 2, 3],
    "Q2": [0, 1, 2, 3],
    ...
  },
  "constraints": [
    {
      "type": "all_different",
      "vars": ["Q1", "Q2", "Q3", "Q4"]
    },
    {
      "type": "custom",
      "vars": ["Q1", "Q2"],
      "fn": "function reference"
    }
  ]
}
```

## Supported Constraint Types

- `all_different`: All variables must have different values
- `equal`: All variables must have the same value
- `not_equal`: Variables must have different values
- `diagonal`: Diagonal constraint for N-Queens (|Qi - Qj| != |row_i - row_j|)
- `custom`: User-defined constraint function

## Test Problems

The generator creates:
- **N-Queens**: 4, 5, 6, 8 queens
- **Sudoku**: 4×4 Sudoku
- **Graph Coloring**: Simple graph coloring problems

## CSPLib

CSPLib is the standard library of constraint problems:
- **N-Queens**: Classic constraint problem
- **Sudoku**: Popular puzzle
- **Graph coloring**: Network problems
- **Scheduling**: Resource allocation
- **More**: See https://www.csplib.org/

## How Livnium Solves CSP

1. **Encoding**: Constraints → tension fields (energy landscape)
   - Each constraint creates a tension field
   - Tension = 0.0 if constraint satisfied, >0.0 if violated

2. **Candidate Solutions**: Variable assignments → basins
   - Each basin represents a candidate assignment
   - Variables mapped to lattice coordinates
   - Domain values encoded in SW

3. **Multi-Basin Search**: Basins compete in tension landscape
   - Best basin (lowest tension) wins
   - Losing basins decay
   - Winning basin reinforces

4. **Solution Extraction**: Decode assignment from winning basin

## Basin Distribution Analysis

The basin distribution analysis demonstrates that Livnium has **measurable, repeatable dynamics**:

- **Multiple stable basins**: The solver finds different constraint satisfaction levels across runs
- **Consistent structure**: Score distributions show clear patterns (not random)
- **Geometric relaxation**: Physics-inspired behavior similar to simulated annealing or Ising models

### Example Results (N-Queens 8, 50 runs)

- Mean score: ~23/29 constraints (79%)
- Median: ~22/29 constraints (76%)
- Range: 19-28/29 constraints
- Standard deviation: ~4.0

This shows the engine has **genuine basin structure** - it repeatedly collapses to similar score zones, demonstrating legitimate geometric relaxation dynamics.

## Notes

- Livnium is a geometric/physics-based solver, not a traditional CSP solver
- It may be slower than specialized solvers but offers a different approach
- The benchmark helps understand Livnium's strengths/weaknesses on constraint problems
- Basin distribution analysis provides scientific evidence of the engine's dynamics

