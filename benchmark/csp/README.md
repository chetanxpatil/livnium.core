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

### Solve Single CSP Problem

```bash
python benchmark/csp/csp_solver_livnium.py
```

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

## Notes

- Livnium is a geometric/physics-based solver, not a traditional CSP solver
- It may be slower than specialized solvers but offers a different approach
- The benchmark helps understand Livnium's strengths/weaknesses on constraint problems

