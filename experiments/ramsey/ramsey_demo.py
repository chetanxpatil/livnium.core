#!/usr/bin/env python3
"""
Quick Demo: Ramsey Number Solver

Demonstrates the solver on a small, verifiable problem.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# Add archive path for hierarchical system
archive_path = project_root / "archive" / "pre_core_systems"
sys.path.insert(0, str(archive_path))

from ramsey_number_solver import RamseyGraph, RamseySolver


def demo_small_problem():
    """Demo on a small problem we can verify manually."""
    
    print("=" * 70)
    print("RAMSEY NUMBER SOLVER - Quick Demo")
    print("=" * 70)
    
    # Test: R(3,3) = 6, so we know R(3,3) > 5 (can find coloring of K_5)
    n = 5
    k = 3
    
    print(f"\nProblem: Find 2-coloring of K_{n} avoiding monochromatic K_{k}")
    print(f"If found, this proves: R({k},{k}) > {n}")
    print(f"(We know R(3,3) = 6, so this should be possible)\n")
    
    # Create a simple test graph
    graph = RamseyGraph(n)
    
    # Manually create a valid coloring
    # Strategy: Color edges in a pattern that avoids triangles
    # One known construction: partition vertices into two sets, color edges within sets one color,
    # edges between sets another color
    
    # For K_5, try: vertices 0,1,2 in one group, 3,4 in another
    for u in range(n):
        for v in range(u + 1, n):
            # Within first group (0,1,2): red
            if u < 3 and v < 3:
                graph.set_edge_color(u, v, 0)  # Red
            # Within second group (3,4): red
            elif u >= 3 and v >= 3:
                graph.set_edge_color(u, v, 0)  # Red
            # Between groups: blue
            else:
                graph.set_edge_color(u, v, 1)  # Blue
    
    # Verify
    print("Created test coloring:")
    print(f"  Red edges (within groups): {sum(1 for c in graph.edge_coloring.values() if c == 0)}")
    print(f"  Blue edges (between groups): {sum(1 for c in graph.edge_coloring.values() if c == 1)}")
    
    # Check for monochromatic triangles
    has_clique, clique = graph.has_monochromatic_clique(k)
    
    if has_clique:
        print(f"\n  ❌ Found monochromatic {k}-clique: {clique}")
    else:
        print(f"\n  ✅ No monochromatic {k}-clique found!")
        print(f"  This proves R({k},{k}) > {n}")
    
    # Test coordinate encoding
    print(f"\nTesting coordinate encoding...")
    coords = graph.to_coordinates()
    print(f"  Coordinates: {coords}")
    
    # Test solver initialization (small scale)
    print(f"\nTesting solver with 100 omcubes (small scale)...")
    solver = RamseySolver(n=5, k=3, num_omcubes=100)
    solver.initialize_omcubes()
    
    print(f"\n✅ Demo complete!")
    print(f"\nTo run full solver with 5000 omcubes:")
    print(f"  python3 experiments/ramsey/ramsey_number_solver.py --n 45 --k 5")


if __name__ == '__main__':
    demo_small_problem()

