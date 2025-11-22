"""
Graph Coloring Solver using Livnium Core System

Solves k-coloring problems: assign colors to vertices such that
no two adjacent vertices have the same color.
"""

import time
import sys
from typing import List, Tuple, Optional, Dict, Any, Set
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import LivniumCoreConfig
from core.classical.livnium_core_system import LivniumCoreSystem
from core.recursive import RecursiveGeometryEngine
import importlib

# Import Universal Encoder (directory name has space)
encoder_module = importlib.import_module('core.Universal Encoder.problem_encoder')
UniversalProblemEncoder = encoder_module.UniversalProblemEncoder

from core.search.multi_basin_search import Basin, MultiBasinSearch


class GraphColoringProblem:
    """
    Represents a graph coloring problem.
    
    Graph: (V, E) where V = vertices, E = edges
    Goal: Assign colors to vertices such that no adjacent vertices share a color
    """
    
    def __init__(self, vertices: List[int], edges: List[Tuple[int, int]], num_colors: int):
        """
        Initialize graph coloring problem.
        
        Args:
            vertices: List of vertex IDs
            edges: List of (u, v) edge tuples
            num_colors: Number of colors available (k)
        """
        self.vertices = vertices
        self.edges = edges
        self.num_colors = num_colors
        self.num_vertices = len(vertices)
        self.num_edges = len(edges)
        
        # Build adjacency list
        self.adjacency: Dict[int, Set[int]] = {v: set() for v in vertices}
        for u, v in edges:
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)
    
    def check_coloring(self, coloring: Dict[int, int]) -> Tuple[bool, int]:
        """
        Check if a coloring is valid.
        
        Returns:
            (is_valid, num_violations)
        """
        violations = 0
        for u, v in self.edges:
            if u in coloring and v in coloring:
                if coloring[u] == coloring[v]:
                    violations += 1
        
        return violations == 0, violations
    
    def get_chromatic_number_lower_bound(self) -> int:
        """Get lower bound on chromatic number."""
        if not self.vertices:
            return 1
        
        # Clique size is a lower bound
        max_clique_size = self._find_max_clique_size()
        
        # Max degree + 1 is also a lower bound (greedy coloring upper bound)
        max_degree = max(len(self.adjacency[v]) for v in self.vertices)
        
        # Use the larger of the two
        return max(max_clique_size, 2)  # At least 2 colors
    
    def _find_max_clique_size(self) -> int:
        """Find size of maximum clique (simple heuristic)."""
        # Simple greedy approach: try to find large cliques
        max_size = 1
        
        for v in self.vertices:
            # Try to build clique starting from v
            clique = {v}
            candidates = self.adjacency[v].copy()
            
            while candidates:
                # Find vertex connected to all in clique
                next_v = None
                for candidate in candidates:
                    if all(candidate in self.adjacency[c] for c in clique):
                        next_v = candidate
                        break
                
                if next_v is None:
                    break
                
                clique.add(next_v)
                candidates = candidates & self.adjacency[next_v]
            
            max_size = max(max_size, len(clique))
        
        return max_size


def parse_dimacs_graph(dimacs_path: Path) -> GraphColoringProblem:
    """
    Parse DIMACS graph format.
    
    Format:
        c comment
        p edge <num_vertices> <num_edges>
        e <u> <v>
        ...
    """
    vertices = []
    edges = []
    
    with open(dimacs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            
            if line.startswith('p'):
                # Problem line: p edge <n> <m>
                parts = line.split()
                if len(parts) >= 4 and parts[1] == 'edge':
                    num_vertices = int(parts[2])
                    num_edges = int(parts[3])
                    vertices = list(range(1, num_vertices + 1))  # DIMACS is 1-indexed
                continue
            
            if line.startswith('e'):
                # Edge line: e <u> <v>
                parts = line.split()
                if len(parts) >= 3:
                    u = int(parts[1])
                    v = int(parts[2])
                    edges.append((u, v))
    
    # Estimate number of colors (use lower bound)
    problem = GraphColoringProblem(vertices, edges, num_colors=0)  # Will set later
    num_colors = problem.get_chromatic_number_lower_bound()
    problem.num_colors = num_colors
    
    return problem


def decode_coloring_from_basin(
    basin: Basin,
    system: LivniumCoreSystem,
    problem: GraphColoringProblem,
    variable_mappings: Dict[str, List[Tuple[int, int, int]]]
) -> Dict[int, int]:
    """
    Decode vertex coloring from winning basin.
    
    Color encoding: SW value maps to color index
    """
    coloring = {}
    
    for vertex_id in problem.vertices:
        var_key = f"vertex_{vertex_id}"
        if var_key in variable_mappings:
            coords = variable_mappings[var_key]
            if coords:
                cell = system.get_cell(coords[0])
                if cell:
                    # Map SW to color (0 to num_colors-1)
                    color_idx = int(cell.symbolic_weight) % problem.num_colors
                    coloring[vertex_id] = color_idx
    
    return coloring


def solve_graph_coloring_livnium(
    problem: GraphColoringProblem,
    max_steps: int = 2000,
    max_time: float = 120.0,
    verbose: bool = False,
    use_recursive: bool = False,
    recursive_depth: int = 2
) -> Dict[str, Any]:
    """
    Solve graph coloring problem using Livnium.
    
    Args:
        problem: GraphColoringProblem instance
        max_steps: Maximum search steps
        max_time: Maximum time in seconds
        verbose: Print progress
        use_recursive: Use RecursiveGeometryEngine
        recursive_depth: Recursion depth if using recursive geometry
    
    Returns:
        Dictionary with results
    """
    start_time = time.time()
    
    if verbose:
        print(f"Graph Coloring: {problem.num_vertices} vertices, {problem.num_edges} edges, {problem.num_colors} colors")
    
    # Estimate lattice size
    n_lattice = max(3, int((problem.num_vertices ** (1/3)) + 1))
    if n_lattice % 2 == 0:
        n_lattice += 1
    
    # For recursive, use smaller base lattice
    if use_recursive:
        base_lattice = min(5, n_lattice)
        if base_lattice % 2 == 0:
            base_lattice += 1
    else:
        base_lattice = n_lattice
    
    if verbose:
        if use_recursive:
            print(f"  Base lattice: {base_lattice}×{base_lattice}×{base_lattice}")
            print(f"  Recursive depth: {recursive_depth}")
        else:
            print(f"  Lattice size: {base_lattice}×{base_lattice}×{base_lattice}")
    
    # Create base Livnium system
    config = LivniumCoreConfig(
        lattice_size=base_lattice,
        enable_semantic_polarity=True
    )
    base_system = LivniumCoreSystem(config)
    
    # Optionally create recursive geometry engine
    recursive_engine = None
    if use_recursive:
        if verbose:
            print("  Building recursive geometry hierarchy...")
        recursive_engine = RecursiveGeometryEngine(
            base_geometry=base_system,
            max_depth=recursive_depth
        )
        system = base_system
        if verbose:
            total_cells = sum(len(level.geometry.lattice) for level in recursive_engine.levels.values())
            print(f"  ✓ Recursive geometry ready: {total_cells:,} total cells")
    else:
        system = base_system
    
    # Encode graph coloring problem
    encoder = UniversalProblemEncoder(system)
    
    # Convert to CSP-like format for encoding
    # Variables: vertices, Domain: [0, 1, ..., num_colors-1]
    variables = {f"vertex_{v}": list(range(problem.num_colors)) for v in problem.vertices}
    
    # Constraints: adjacent vertices must have different colors
    constraints = []
    for u, v in problem.edges:
        constraints.append({
            'type': 'not_equal',
            'vars': [f"vertex_{u}", f"vertex_{v}"]
        })
    
    problem_dict = {
        'type': 'constraint_satisfaction',
        'variables': variables,
        'constraints': constraints,
        'n_candidates': min(30, max(10, problem.num_vertices))
    }
    
    if verbose:
        print("Encoding graph coloring problem...")
    
    encoded = encoder.encode(problem_dict)
    
    if verbose:
        print(f"  Created {len(encoded.tension_fields)} tension fields")
        print(f"  Created {len(encoded.candidate_basins)} candidate basins")
    
    # Define correctness checker
    def check_correctness(basin: Basin, system: LivniumCoreSystem) -> bool:
        coloring = decode_coloring_from_basin(
            basin, system, problem, encoded.variable_mappings
        )
        is_valid, _ = problem.check_coloring(coloring)
        return is_valid
    
    # Create custom search with constraint tension
    class GraphColoringMultiBasinSearch(MultiBasinSearch):
        """Multi-basin search with graph coloring constraint tension."""
        
        def __init__(self, tension_fields, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tension_fields = tension_fields
        
        def update_all_basins(self, system):
            """Override to include constraint tension."""
            from core.search.native_dynamic_basin_search import get_geometry_signals
            
            for basin in self.basins:
                if not basin.is_alive:
                    continue
                
                curvature, base_tension, entropy = get_geometry_signals(
                    system, basin.active_coords
                )
                
                # Add constraint tension
                constraint_tension = 0.0
                for field in self.tension_fields:
                    constraint_tension += field.get_tension(system)
                
                basin.curvature = curvature
                basin.tension = base_tension + constraint_tension
                basin.entropy = entropy
                basin.update_score()
                basin.age += 1
            
            # Identify winner
            alive_basins = [b for b in self.basins if b.is_alive]
            if alive_basins:
                winner = max(alive_basins, key=lambda b: b.score)
                winner.is_winning = True
                for basin in alive_basins:
                    if basin.id != winner.id:
                        basin.is_winning = False
            
            self._apply_basin_dynamics(system)
            self._prune_dead_basins()
    
    # Solve with graph coloring-enhanced multi-basin search
    if verbose:
        print("Starting multi-basin search...")
    
    search = GraphColoringMultiBasinSearch(
        encoded.tension_fields,
        use_rotations=True
    )
    
    # Add candidate basins
    for coords in encoded.candidate_basins:
        search.add_basin(coords, system)
    
    if verbose:
        print(f"  Initialized {len(search.basins)} basins")
    
    # Iterative search loop
    import random
    from core.classical.livnium_core_system import RotationAxis
    
    best_coloring = None
    best_violations = float('inf')
    best_score = 0
    
    for step in range(max_steps):
        if time.time() - start_time > max_time:
            if verbose:
                print(f"  Timeout at step {step}")
            break
        
        search.update_all_basins(system)
        
        winner = search.get_winner()
        if winner:
            coloring = decode_coloring_from_basin(
                winner, system, problem, encoded.variable_mappings
            )
            is_valid, violations = problem.check_coloring(coloring)
            
            if violations < best_violations:
                best_violations = violations
                best_coloring = coloring
                best_score = len(coloring)
            
            if is_valid:
                if verbose:
                    print(f"  Valid coloring found at step {step+1}")
                stats = search.get_basin_stats()
                elapsed_time = time.time() - start_time
                break
        
        if step % 100 == 0 and verbose:
            print(f"  Step {step+1}: best violations = {best_violations}")
        
        # Apply rotations occasionally to explore solution space
        if step % 50 == 0 and step > 0:
            from core.classical.livnium_core_system import RotationAxis
            if random.random() < 0.1:
                axis = random.choice(list(RotationAxis))
                system.rotate(axis, 1)
    else:
        # Loop completed without finding solution
        stats = search.get_basin_stats()
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"  Search completed: best violations = {best_violations}")
    
    # Final result
    is_valid = best_violations == 0
    num_colored = len(best_coloring) if best_coloring else 0
    
    return {
        'solved': True,
        'valid_coloring': is_valid,
        'coloring': best_coloring,
        'num_violations': best_violations,
        'num_colored_vertices': num_colored,
        'num_vertices': problem.num_vertices,
        'num_edges': problem.num_edges,
        'num_colors': problem.num_colors,
        'time': elapsed_time,
        'steps': step + 1 if 'step' in locals() else max_steps,
        'basin_stats': stats
    }

