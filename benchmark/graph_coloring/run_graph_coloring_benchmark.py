"""
Graph Coloring Benchmark: Compare Livnium vs NetworkX greedy coloring

Runs DIMACS-style graph coloring problems and compares results.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.graph_coloring.graph_coloring_solver_livnium import (
    solve_graph_coloring_livnium,
    GraphColoringProblem,
    parse_dimacs_graph
)


def load_graph_from_json(json_path: Path) -> GraphColoringProblem:
    """Load graph coloring problem from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return GraphColoringProblem(
        vertices=data['vertices'],
        edges=data['edges'],
        num_colors=data.get('num_colors', 3)
    )


def solve_with_networkx(problem: GraphColoringProblem, timeout: float = 120.0) -> Dict[str, Any]:
    """
    Solve using NetworkX greedy coloring.
    
    Returns:
        Dictionary with results
    """
    try:
        import networkx as nx
    except ImportError:
        return {
            'solved': False,
            'error': 'networkx not installed. Install with: pip install networkx'
        }
    
    try:
        start_time = time.time()
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(problem.vertices)
        G.add_edges_from(problem.edges)
        
        # Use greedy coloring
        coloring = nx.greedy_color(G, strategy='largest_first')
        
        # Convert to our format (1-indexed)
        coloring_dict = {v: c for v, c in coloring.items()}
        
        # Check validity
        is_valid, violations = problem.check_coloring(coloring_dict)
        
        elapsed = time.time() - start_time
        
        # Count colors used
        num_colors_used = len(set(coloring_dict.values()))
        
        return {
            'solved': True,
            'valid_coloring': is_valid,
            'coloring': coloring_dict,
            'num_violations': violations,
            'num_colored_vertices': len(coloring_dict),
            'num_colors_used': num_colors_used,
            'time': elapsed,
            'num_vertices': problem.num_vertices,
            'num_edges': problem.num_edges
        }
    except Exception as e:
        return {
            'solved': False,
            'error': str(e)
        }


def find_graph_files(directory: Path) -> List[Path]:
    """Find all graph files (DIMACS .col or JSON)."""
    graph_files = []
    
    for ext in ['.col', '.json']:
        graph_files.extend(directory.glob(f'*{ext}'))
    
    return sorted(graph_files)


def run_benchmark(
    graph_files: List[Path],
    max_steps: int = 2000,
    max_time: float = 120.0,
    verbose: bool = False,
    use_recursive: bool = False,
    recursive_depth: int = 2
) -> Dict[str, Any]:
    """
    Run benchmark on a set of graph files.
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'livnium': [],
        'networkx': [],
        'summary': {}
    }
    
    print(f"\n{'='*70}")
    print(f"Graph Coloring Benchmark: {len(graph_files)} graphs")
    print(f"{'='*70}\n")
    
    for i, graph_file in enumerate(graph_files, 1):
        print(f"[{i}/{len(graph_files)}] {graph_file.name}")
        
        # Load graph problem
        try:
            if graph_file.suffix == '.col':
                problem = parse_dimacs_graph(graph_file)
            else:
                problem = load_graph_from_json(graph_file)
            
            print(f"  Vertices: {problem.num_vertices}, Edges: {problem.num_edges}, Colors: {problem.num_colors}")
        except Exception as e:
            print(f"  Error loading: {e}")
            continue
        
        # Solve with Livnium
        print("  Livnium...", end=" ", flush=True)
        livnium_result = solve_graph_coloring_livnium(
            problem,
            max_steps=max_steps,
            max_time=max_time,
            verbose=verbose,
            use_recursive=use_recursive,
            recursive_depth=recursive_depth
        )
        livnium_result['file'] = graph_file.name
        results['livnium'].append(livnium_result)
        
        if livnium_result.get('valid_coloring'):
            print(f"✓ {livnium_result['time']:.3f}s, {livnium_result['num_violations']} violations, "
                  f"{livnium_result['num_colored_vertices']}/{livnium_result['num_vertices']} colored")
        else:
            print(f"✗ {livnium_result['time']:.3f}s, {livnium_result['num_violations']} violations")
        
        # Solve with NetworkX
        print("  NetworkX...", end=" ", flush=True)
        networkx_result = solve_with_networkx(problem, timeout=max_time)
        networkx_result['file'] = graph_file.name
        results['networkx'].append(networkx_result)
        
        if networkx_result.get('solved'):
            if networkx_result.get('valid_coloring'):
                print(f"✓ {networkx_result['time']:.3f}s, {networkx_result['num_colors_used']} colors used")
            else:
                print(f"✗ {networkx_result['time']:.3f}s, {networkx_result.get('num_violations', 0)} violations")
        else:
            print(f"✗ {networkx_result.get('error', 'Failed')}")
        
        print()
    
    # Calculate summary statistics
    livnium_times = [r['time'] for r in results['livnium'] if r.get('solved')]
    networkx_times = [r['time'] for r in results['networkx'] if r.get('solved')]
    
    livnium_valid = sum(1 for r in results['livnium'] if r.get('valid_coloring'))
    networkx_valid = sum(1 for r in results['networkx'] if r.get('valid_coloring'))
    
    results['summary'] = {
        'total_graphs': len(graph_files),
        'livnium': {
            'solved': len([r for r in results['livnium'] if r.get('solved')]),
            'valid_colorings': livnium_valid,
            'avg_time': sum(livnium_times) / len(livnium_times) if livnium_times else None,
            'min_time': min(livnium_times) if livnium_times else None,
            'max_time': max(livnium_times) if livnium_times else None
        },
        'networkx': {
            'solved': len([r for r in results['networkx'] if r.get('solved')]),
            'valid_colorings': networkx_valid,
            'avg_time': sum(networkx_times) / len(networkx_times) if networkx_times else None,
            'min_time': min(networkx_times) if networkx_times else None,
            'max_time': max(networkx_times) if networkx_times else None
        }
    }
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print benchmark summary."""
    summary = results['summary']
    
    print("="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\nTotal Graphs: {summary['total_graphs']}")
    
    print(f"\n{'Solver':<20} {'Solved':<10} {'Valid':<10} {'Avg Time':<12} {'Min Time':<12} {'Max Time':<12}")
    print("-" * 70)
    
    liv = summary['livnium']
    avg_time_liv = f"{liv['avg_time']:.3f}s" if liv['avg_time'] is not None else "N/A"
    min_time_liv = f"{liv['min_time']:.3f}s" if liv['min_time'] is not None else "N/A"
    max_time_liv = f"{liv['max_time']:.3f}s" if liv['max_time'] is not None else "N/A"
    
    print(f"{'Livnium':<20} {liv['solved']:<10} {liv['valid_colorings']:<10} "
          f"{avg_time_liv:<12} {min_time_liv:<12} {max_time_liv:<12}")
    
    nx = summary['networkx']
    avg_time_nx = f"{nx['avg_time']:.3f}s" if nx['avg_time'] is not None else "N/A"
    min_time_nx = f"{nx['min_time']:.3f}s" if nx['min_time'] is not None else "N/A"
    max_time_nx = f"{nx['max_time']:.3f}s" if nx['max_time'] is not None else "N/A"
    
    print(f"{'NetworkX':<20} {nx['solved']:<10} {nx['valid_colorings']:<10} "
          f"{avg_time_nx:<12} {min_time_nx:<12} {max_time_nx:<12}")
    
    print("\n" + "="*70)


def save_results(results: Dict[str, Any], output_path: Path):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Graph Coloring Benchmark: Livnium vs NetworkX')
    parser.add_argument('--graph-dir', type=str, help='Directory containing graph files (.col or .json)')
    parser.add_argument('--max-steps', type=int, default=2000, help='Max search steps for Livnium')
    parser.add_argument('--max-time', type=float, default=120.0, help='Max time per graph (seconds)')
    parser.add_argument('--limit', type=int, help='Limit number of graphs to test')
    parser.add_argument('--output', type=str, default='benchmark/graph_coloring/graph_coloring_results.json', 
                       help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--use-recursive', action='store_true', default=False, 
                       help='Use recursive geometry (default: False)')
    parser.add_argument('--recursive-depth', type=int, default=2, 
                       help='Recursive geometry depth (default: 2)')
    
    args = parser.parse_args()
    
    if not args.graph_dir:
        parser.error("--graph-dir is required")
    
    graph_dir = Path(args.graph_dir)
    if not graph_dir.exists():
        parser.error(f"Graph directory does not exist: {graph_dir}")
    
    # Find graph files
    graph_files = find_graph_files(graph_dir)
    
    if not graph_files:
        print(f"No graph files found in {graph_dir}")
        print("Generate test graphs first: python benchmark/graph_coloring/generate_test_graphs.py")
        print("Or download DIMACS graphs: python benchmark/graph_coloring/download_dimacs.py")
        return
    
    if args.limit:
        graph_files = graph_files[:args.limit]
    
    # Run benchmark
    results = run_benchmark(
        graph_files,
        max_steps=args.max_steps,
        max_time=args.max_time,
        verbose=args.verbose,
        use_recursive=args.use_recursive,
        recursive_depth=args.recursive_depth
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)


if __name__ == '__main__':
    main()

