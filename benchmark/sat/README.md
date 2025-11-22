# SAT Benchmark Suite

This directory contains benchmarks comparing Livnium's SAT solver against standard solvers (PySAT/MiniSAT).

## Setup

### Install Dependencies

```bash
pip install python-sat  # For PySAT (wrapper around MiniSAT)
```

### Download SATLIB Benchmarks

The benchmark script can automatically download a sample of SATLIB benchmarks:

```bash
python benchmark/sat/run_sat_benchmark.py --download
```

Or manually download CNF files and place them in `benchmark/sat/satlib/`.

## Usage

### Run Full Benchmark

```bash
python benchmark/sat/run_sat_benchmark.py --download --limit 10
```

This will:
1. Download SATLIB sample (if `--download` is used)
2. Run Livnium and PySAT on each CNF file
3. Compare results (time, solved/unsolved)
4. Save results to `benchmark/sat/sat_results.json`

### Options

- `--download`: Download SATLIB benchmarks automatically
- `--cnf-dir PATH`: Use CNF files from specified directory
- `--max-steps N`: Maximum search steps for Livnium (default: 500)
- `--max-time SECONDS`: Timeout per problem (default: 60.0)
- `--limit N`: Limit number of problems to test
- `--output PATH`: Output JSON file (default: `benchmark/sat/sat_results.json`)
- `--verbose`: Verbose output

### Solve Single CNF File

```bash
python benchmark/sat/sat_solver_livnium.py path/to/file.cnf --verbose
```

## Results Format

The benchmark outputs a JSON file with:

```json
{
  "livnium": [
    {
      "file": "uf20-01.cnf",
      "solved": true,
      "satisfiable": true,
      "time": 2.345,
      "steps": 150,
      "num_satisfied_clauses": 91,
      "total_clauses": 91
    },
    ...
  ],
  "pysat": [
    {
      "file": "uf20-01.cnf",
      "solved": true,
      "satisfiable": true,
      "time": 0.012,
      ...
    },
    ...
  ],
  "summary": {
    "total_problems": 10,
    "livnium": {
      "solved": 8,
      "unsolved": 2,
      "avg_time": 3.456,
      ...
    },
    "pysat": {
      "solved": 10,
      "unsolved": 0,
      "avg_time": 0.023,
      ...
    }
  }
}
```

## SATLIB Benchmarks

SATLIB is a collection of SAT benchmark problems:
- **Random 3-SAT**: uf20-91, uf50-218, uf75-325, etc.
- **Graph coloring**: Various graph problems
- **Planning**: AI planning problems
- **More**: See https://www.cs.ubc.ca/~hoos/SATLIB/

The benchmark script currently downloads a small sample (uf20-91) for testing. You can expand this to include more problem sets.

## How Livnium Solves SAT

1. **Encoding**: CNF clauses → tension fields (energy landscape)
   - Each clause creates a tension field
   - Tension = 1.0 if clause unsatisfied, 0.0 if satisfied

2. **Candidate Solutions**: Variable assignments → basins
   - Each basin represents a candidate assignment
   - Variables mapped to lattice coordinates
   - SW < 10 = False, SW >= 10 = True

3. **Multi-Basin Search**: Basins compete in tension landscape
   - Best basin (lowest tension) wins
   - Losing basins decay
   - Winning basin reinforces

4. **Solution Extraction**: Decode assignment from winning basin

## Notes

- Livnium is a geometric/physics-based solver, not a traditional SAT solver
- It may be slower than specialized SAT solvers (MiniSAT, Glucose) but offers a different approach
- The benchmark helps understand Livnium's strengths/weaknesses on logical problems

