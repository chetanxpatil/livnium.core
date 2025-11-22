# Graph Coloring Benchmark Suite

This directory contains benchmarks comparing Livnium's graph coloring solver against standard solvers (NetworkX greedy coloring).

Graph coloring is **notoriously multimodal** - perfect for demonstrating Livnium's inward-collapse dynamics that can find stable "near-colorings" faster than classical greedy/SA methods that get stuck in trash basins.

## Setup

### Install Dependencies

```bash
pip install networkx  # For comparison solver
pip install matplotlib seaborn  # For basin distribution plots (optional)
```

### Generate Test Graphs

Generate synthetic test graphs:

```bash
python benchmark/graph_coloring/generate_test_graphs.py
```

This creates:
- Random graphs (20, 30, 40 vertices)
- Complete graphs (K_n)
- Bipartite graphs (2-colorable)

### Download DIMACS Instances

Download real DIMACS graph coloring instances:

```bash
python benchmark/graph_coloring/download_dimacs.py
```

**Note**: DIMACS instances may need to be downloaded manually from:
- https://mat.tepper.cmu.edu/COLOR/instances/
- https://github.com/mivia-lab/graph-coloring-instances

Target instances:
- `col-40-5`, `col-50-5` (small structured graphs)
- `DSJC125.1` (DIMACS standard)
- `flat300_20_0`, `flat300_26_0`, `flat300_28_0` (flat graphs)

## Usage

### Run Full Benchmark

```bash
python benchmark/graph_coloring/run_graph_coloring_benchmark.py --graph-dir benchmark/graph_coloring/dimacs/test
```

This will:
1. Load graph files (.col or .json)
2. Run Livnium and NetworkX on each graph
3. Compare results (time, valid colorings, violations)
4. Save results to `benchmark/graph_coloring/graph_coloring_results.json`

### Options

- `--graph-dir PATH`: Directory containing graph files (.col or .json)
- `--max-steps N`: Maximum search steps for Livnium (default: 2000)
- `--max-time SECONDS`: Timeout per graph (default: 120.0)
- `--limit N`: Limit number of graphs to test
- `--output PATH`: Output JSON file
- `--verbose`: Verbose output
- `--use-recursive`: Enable recursive geometry (default: False)
- `--recursive-depth N`: Recursive geometry depth (default: 2)

### Basin Distribution Analysis

Analyze the distribution of coloring quality across multiple runs:

```bash
python benchmark/graph_coloring/run_basin_distribution.py benchmark/graph_coloring/dimacs/test/random_40.col --runs 50 --plot-type all
```

## Graph Formats

### DIMACS Format (.col)

```
c comment
p edge <num_vertices> <num_edges>
e <u> <v>
e <u> <v>
...
```

### JSON Format

```json
{
  "vertices": [1, 2, 3, ...],
  "edges": [[1, 2], [2, 3], ...],
  "num_colors": 3
}
```

## Problem Definition

**Graph k-Coloring**: Assign colors to vertices such that no two adjacent vertices share the same color.

- **Variables**: Vertices
- **Domain**: Colors [0, 1, ..., k-1]
- **Constraints**: For each edge (u, v), color[u] ≠ color[v]

## How Livnium Solves Graph Coloring

1. **Encoding**: 
   - Vertices → lattice coordinates
   - Adjacency constraints → tension fields
   - Tension = 1.0 if adjacent vertices have same color, 0.0 otherwise

2. **Candidate Solutions**: 
   - Random colorings → basins
   - Each basin represents a candidate coloring

3. **Multi-Basin Search**: 
   - Basins compete in tension landscape
   - Best basin (lowest tension = fewest violations) wins
   - Inward collapse finds stable near-colorings

4. **Solution Extraction**: 
   - Decode coloring from winning basin
   - Count violations (adjacent vertices with same color)

## Why Graph Coloring is Perfect for Livnium

Graph coloring is **notoriously multimodal**:

- **Multiple stable basins**: Many valid colorings exist
- **Classical methods get stuck**: Greedy/SA often find local minima
- **Livnium's advantage**: Inward-collapse can find stable "near-colorings" faster
- **Geometric relaxation**: Physics-inspired dynamics avoid trash basins

## Test Problems

### Synthetic Graphs

- **Random graphs**: Various sizes, edge densities
- **Complete graphs**: K_n (requires n colors)
- **Bipartite graphs**: 2-colorable (easy baseline)

### DIMACS Instances

- **col-40-5, col-50-5**: Small structured graphs
- **DSJC125.1**: DIMACS standard benchmark
- **flat300_xx**: Flat graphs (challenging)

## Expected Results

- **Livnium**: Finds high-quality partial colorings (few violations)
- **NetworkX**: Finds valid colorings (may use more colors than optimal)

The goal is NOT to beat NetworkX on speed (it's highly optimized).

The goal is to show:
1. **Livnium behaves correctly** on standard graph coloring problems
2. **Inward-collapse dynamics** find stable basins
3. **Geometric relaxation** avoids getting stuck in trash basins
4. **Basin distribution** shows repeatable, measurable dynamics

## Notes

- Livnium is a geometric/physics-based solver, not a traditional graph coloring algorithm
- It may be slower than specialized solvers but offers a different approach
- The benchmark helps understand Livnium's strengths/weaknesses on multimodal constraint problems
- Graph coloring is ideal for demonstrating the engine's basin-finding capabilities

