"""
Basin Distribution Analysis for Livnium Graph Coloring Solver

Runs Livnium multiple times on the same graph and plots the distribution
of coloring quality (violations, colors used). This shows the repeatable
basin structure of the geometric relaxation engine on multimodal problems.
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse

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


def run_basin_sweep(
    graph_path: Path,
    runs: int = 50,
    max_steps: int = 2000,
    max_time: float = 120.0,
    use_recursive: bool = False,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Run Livnium multiple times on the same graph.
    
    Args:
        graph_path: Path to graph file (.col or .json)
        runs: Number of runs to perform
        max_steps: Max search steps per run
        max_time: Max time per run (seconds)
        use_recursive: Use recursive geometry
        verbose: Print progress
    
    Returns:
        List of result dictionaries with violations, colors, time, etc.
    """
    # Load graph problem once
    if graph_path.suffix == '.col':
        problem = parse_dimacs_graph(graph_path)
    else:
        problem = load_graph_from_json(graph_path)
    
    if verbose:
        print(f"Running basin sweep on {graph_path.name}")
        print(f"  Vertices: {problem.num_vertices}, Edges: {problem.num_edges}, Colors: {problem.num_colors}")
        print(f"  Runs: {runs}")
        print()
    
    results = []
    
    for i in range(runs):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Run {i+1}/{runs}...")
        
        start = time.time()
        
        # Solve with Livnium
        result = solve_graph_coloring_livnium(
            problem,
            max_steps=max_steps,
            max_time=max_time,
            verbose=False,
            use_recursive=use_recursive
        )
        
        elapsed = time.time() - start
        
        # Extract key metrics
        violations = result.get('num_violations', problem.num_edges)
        num_colored = result.get('num_colored_vertices', 0)
        valid = result.get('valid_coloring', False)
        
        # Calculate violation ratio
        violation_ratio = violations / problem.num_edges if problem.num_edges > 0 else 1.0
        
        results.append({
            "run": i + 1,
            "violations": violations,
            "total_edges": problem.num_edges,
            "violation_ratio": violation_ratio,
            "num_colored_vertices": num_colored,
            "num_vertices": problem.num_vertices,
            "valid_coloring": valid,
            "time": elapsed,
            "steps": result.get('steps', 0),
            "solved": result.get('solved', False)
        })
        
        if verbose and (i + 1) % 10 == 0:
            print(f"    Violations: {violations}/{problem.num_edges} ({violation_ratio:.2%})")
    
    if verbose:
        print(f"\nCompleted {runs} runs")
    
    return results


def plot_distribution(
    results: List[Dict[str, Any]],
    title: str,
    out_path: Path,
    plot_type: str = "histogram"
):
    """
    Plot distribution of coloring quality (violations).
    
    Args:
        results: List of result dictionaries
        title: Plot title
        out_path: Output path for plot
        plot_type: Type of plot ('histogram', 'violin', 'kde', 'scatter', or 'all')
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
        return
    
    violations = [r["violations"] for r in results]
    violation_ratios = [r["violation_ratio"] for r in results]
    times = [r["time"] for r in results]
    
    # If "all", create a 2x2 subplot figure
    if plot_type == "all":
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Graph Coloring Basin Distribution: {title}", 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Top-left: Histogram of violations
        ax1 = axes[0, 0]
        ax1.hist(violations, bins=min(20, len(set(violations))), color='skyblue', 
                edgecolor='black', alpha=0.7)
        mean_violations = np.mean(violations)
        median_violations = np.median(violations)
        ax1.axvline(mean_violations, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_violations:.1f}')
        ax1.axvline(median_violations, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_violations:.1f}')
        ax1.set_xlabel("Number of Violations", fontsize=11)
        ax1.set_ylabel("Frequency", fontsize=11)
        ax1.set_title("Violations Histogram", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Top-right: Violin plot
        ax2 = axes[0, 1]
        try:
            import seaborn as sns
            sns.violinplot(y=violations, ax=ax2, color='skyblue')
            ax2.set_ylabel("Number of Violations", fontsize=11)
            ax2.set_title("Violations Violin Plot", fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        except ImportError:
            ax2.text(0.5, 0.5, 'seaborn not installed', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Violin Plot (seaborn required)", fontsize=12)
        
        # Bottom-left: KDE plot
        ax3 = axes[1, 0]
        try:
            import seaborn as sns
            sns.kdeplot(violations, ax=ax3, fill=True, color='skyblue', alpha=0.7)
            ax3.set_xlabel("Number of Violations", fontsize=11)
            ax3.set_ylabel("Density", fontsize=11)
            ax3.set_title("Kernel Density Estimate", fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        except ImportError:
            ax3.text(0.5, 0.5, 'seaborn not installed', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("KDE Plot (seaborn required)", fontsize=12)
        
        # Bottom-right: Scatter plot (Time vs Violations)
        ax4 = axes[1, 1]
        ax4.scatter(times, violations, alpha=0.6, s=50, color='steelblue')
        ax4.set_xlabel("Time (seconds)", fontsize=11)
        ax4.set_ylabel("Number of Violations", fontsize=11)
        ax4.set_title("Time vs Violations", fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined plot to {out_path}")
        plt.close()
        return
    
    # Individual plots (for backward compatibility)
    if plot_type == "histogram":
        plt.figure(figsize=(10, 6))
        plt.hist(violations, bins=min(20, len(set(violations))), color='skyblue', 
                edgecolor='black', alpha=0.7)
        plt.title(f"Graph Coloring Basin Distribution: {title}\n(Violations)", 
                 fontsize=14, fontweight='bold')
        plt.xlabel("Number of Violations", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        mean_violations = np.mean(violations)
        median_violations = np.median(violations)
        plt.axvline(mean_violations, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_violations:.1f}')
        plt.axvline(median_violations, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_violations:.1f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram to {out_path}")
        plt.close()
    
    elif plot_type == "violin":
        try:
            import seaborn as sns
            plt.figure(figsize=(10, 6))
            sns.violinplot(y=violations, color='skyblue')
            plt.title(f"Graph Coloring Basin Distribution (Violin Plot): {title}", 
                     fontsize=14, fontweight='bold')
            plt.ylabel("Number of Violations", fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved violin plot to {out_path}")
            plt.close()
        except ImportError:
            print("WARNING: seaborn not installed. Skipping violin plot.")
    
    elif plot_type == "kde":
        try:
            import seaborn as sns
            plt.figure(figsize=(10, 6))
            sns.kdeplot(violations, fill=True, color='skyblue', alpha=0.7)
            plt.title(f"Graph Coloring Basin Distribution (KDE): {title}", 
                     fontsize=14, fontweight='bold')
            plt.xlabel("Number of Violations", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved KDE plot to {out_path}")
            plt.close()
        except ImportError:
            print("WARNING: seaborn not installed. Skipping KDE plot.")
    
    elif plot_type == "scatter":
        plt.figure(figsize=(10, 6))
        plt.scatter(times, violations, alpha=0.6, s=50, color='steelblue')
        plt.title(f"Time vs Violations: {title}", fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Number of Violations", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved scatter plot to {out_path}")
        plt.close()


def print_statistics(results: List[Dict[str, Any]]):
    """Print statistical summary of results."""
    import numpy as np
    
    violations = [r["violations"] for r in results]
    violation_ratios = [r["violation_ratio"] for r in results]
    times = [r["time"] for r in results]
    valid_count = sum(1 for r in results if r.get('valid_coloring', False))
    
    print("\n" + "=" * 70)
    print("BASIN DISTRIBUTION STATISTICS")
    print("=" * 70)
    print(f"Total runs: {len(results)}")
    print(f"Valid colorings: {valid_count} ({valid_count/len(results)*100:.1f}%)")
    print()
    print("Violations:")
    print(f"  Mean:   {np.mean(violations):.2f}")
    print(f"  Median: {np.median(violations):.2f}")
    print(f"  Std:    {np.std(violations):.2f}")
    print(f"  Min:    {np.min(violations):.0f}")
    print(f"  Max:    {np.max(violations):.0f}")
    print()
    print("Violation Ratio (violations/total_edges):")
    print(f"  Mean:   {np.mean(violation_ratios):.2%}")
    print(f"  Median: {np.median(violation_ratios):.2%}")
    print(f"  Std:    {np.std(violation_ratios):.2%}")
    print()
    print("Time (seconds):")
    print(f"  Mean:   {np.mean(times):.3f}s")
    print(f"  Median: {np.median(times):.3f}s")
    print(f"  Std:    {np.std(times):.3f}s")
    print(f"  Min:    {np.min(times):.3f}s")
    print(f"  Max:    {np.max(times):.3f}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Run basin distribution analysis for Livnium graph coloring solver'
    )
    parser.add_argument('graph_file', type=str, help='Path to graph file (.col or .json)')
    parser.add_argument('--runs', type=int, default=50, 
                       help='Number of runs (default: 50)')
    parser.add_argument('--max-steps', type=int, default=2000,
                       help='Max search steps per run (default: 2000)')
    parser.add_argument('--max-time', type=float, default=120.0,
                       help='Max time per run in seconds (default: 120.0)')
    parser.add_argument('--use-recursive', action='store_true',
                       help='Use recursive geometry')
    parser.add_argument('--output-json', type=str, 
                       help='Output JSON file (default: auto-generated)')
    parser.add_argument('--output-plot', type=str,
                       help='Output plot file (default: auto-generated)')
    parser.add_argument('--plot-type', type=str, 
                       choices=['histogram', 'violin', 'kde', 'scatter', 'all'],
                       default='histogram',
                       help='Type of plot to generate (default: histogram)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    graph_path = Path(args.graph_file)
    if not graph_path.exists():
        parser.error(f"Graph file does not exist: {graph_path}")
    
    # Auto-generate output paths if not provided
    if args.output_json:
        output_json = Path(args.output_json)
    else:
        output_json = Path(f"benchmark/graph_coloring/basin_results_{graph_path.stem}.json")
    
    if args.output_plot:
        output_plot = Path(args.output_plot)
    else:
        output_plot = Path(f"benchmark/graph_coloring/basin_distribution_{graph_path.stem}.png")
    
    # Run basin sweep
    print("=" * 70)
    print("LIVNIUM GRAPH COLORING BASIN DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()
    
    results = run_basin_sweep(
        graph_path=graph_path,
        runs=args.runs,
        max_steps=args.max_steps,
        max_time=args.max_time,
        use_recursive=args.use_recursive,
        verbose=args.verbose
    )
    
    # Save results
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump({
            'graph_file': str(graph_path),
            'runs': args.runs,
            'results': results
        }, f, indent=2)
    print(f"\nSaved results to {output_json}")
    
    # Print statistics
    print_statistics(results)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_distribution(
        results=results,
        title=graph_path.stem,
        out_path=output_plot,
        plot_type=args.plot_type
    )
    
    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

