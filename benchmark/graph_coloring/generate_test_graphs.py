"""
Generate test graph coloring instances.

Creates synthetic graphs for testing when DIMACS download fails.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple


def generate_random_graph(num_vertices: int, edge_probability: float = 0.3) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Generate a random graph.
    
    Args:
        num_vertices: Number of vertices
        edge_probability: Probability of edge between any two vertices
    
    Returns:
        (vertices, edges)
    """
    vertices = list(range(1, num_vertices + 1))
    edges = []
    
    for i in range(1, num_vertices + 1):
        for j in range(i + 1, num_vertices + 1):
            if random.random() < edge_probability:
                edges.append((i, j))
    
    return vertices, edges


def generate_complete_graph(num_vertices: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Generate a complete graph K_n."""
    vertices = list(range(1, num_vertices + 1))
    edges = []
    
    for i in range(1, num_vertices + 1):
        for j in range(i + 1, num_vertices + 1):
            edges.append((i, j))
    
    return vertices, edges


def generate_bipartite_graph(n1: int, n2: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Generate a complete bipartite graph K_{n1,n2}."""
    vertices = list(range(1, n1 + n2 + 1))
    edges = []
    
    # First partition: 1 to n1
    # Second partition: n1+1 to n1+n2
    for i in range(1, n1 + 1):
        for j in range(n1 + 1, n1 + n2 + 1):
            edges.append((i, j))
    
    return vertices, edges


def save_graph_to_json(vertices: List[int], edges: List[Tuple[int, int]], 
                       num_colors: int, output_path: Path):
    """Save graph to JSON format."""
    problem = {
        'vertices': vertices,
        'edges': edges,
        'num_colors': num_colors
    }
    
    with open(output_path, 'w') as f:
        json.dump(problem, f, indent=2)
    
    print(f"Saved: {output_path}")


def save_graph_to_dimacs(vertices: List[int], edges: List[Tuple[int, int]], 
                         output_path: Path):
    """Save graph to DIMACS format."""
    with open(output_path, 'w') as f:
        f.write(f"c Generated test graph\n")
        f.write(f"p edge {len(vertices)} {len(edges)}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")
    
    print(f"Saved: {output_path}")


def main():
    output_dir = Path("benchmark/graph_coloring/dimacs/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating test graph coloring instances...")
    print()
    
    # Small random graphs
    for n in [20, 30, 40]:
        vertices, edges = generate_random_graph(n, edge_probability=0.3)
        num_colors = max(3, int(n ** 0.5))  # Heuristic
        
        # Save as JSON
        json_path = output_dir / f"random_{n}.json"
        save_graph_to_json(vertices, edges, num_colors, json_path)
        
        # Save as DIMACS
        dimacs_path = output_dir / f"random_{n}.col"
        save_graph_to_dimacs(vertices, edges, dimacs_path)
    
    # Complete graphs (K_n requires n colors)
    for n in [5, 10]:
        vertices, edges = generate_complete_graph(n)
        num_colors = n
        
        json_path = output_dir / f"complete_{n}.json"
        save_graph_to_json(vertices, edges, num_colors, json_path)
        
        dimacs_path = output_dir / f"complete_{n}.col"
        save_graph_to_dimacs(vertices, edges, dimacs_path)
    
    # Bipartite graphs (2-colorable)
    for n1, n2 in [(10, 10), (15, 15)]:
        vertices, edges = generate_bipartite_graph(n1, n2)
        num_colors = 2
        
        json_path = output_dir / f"bipartite_{n1}_{n2}.json"
        save_graph_to_json(vertices, edges, num_colors, json_path)
        
        dimacs_path = output_dir / f"bipartite_{n1}_{n2}.col"
        save_graph_to_dimacs(vertices, edges, dimacs_path)
    
    print()
    print(f"Generated test graphs in {output_dir}")
    print("You can now run:")
    print(f"  python benchmark/graph_coloring/run_graph_coloring_benchmark.py --graph-dir {output_dir}")


if __name__ == '__main__':
    main()

