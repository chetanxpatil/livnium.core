"""
Max-Cut Solver using Livnium Core System

Max-Cut is the problem of partitioning a graph's vertices into two sets
such that the number of edges between the sets is maximized.

This maps perfectly to Livnium's tension minimization:
- Each edge creates a tension field
- Tension = 0 if vertices are on different sides (contributes to cut)
- Tension = 1 if vertices are on same side (doesn't contribute to cut)
- Total tension = number of "bad" edges
- Max-Cut = minimize tension
"""

import time
import sys
from typing import List, Tuple, Optional, Dict, Any, Set
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import LivniumCoreConfig
from core.classical.livnium_core_system import LivniumCoreSystem
from core.recursive import RecursiveGeometryEngine
import importlib

# Import Universal Encoder
encoder_module = importlib.import_module('core.encoder.problem_encoder')
UniversalProblemEncoder = encoder_module.UniversalProblemEncoder

from core.search.multi_basin_search import Basin, MultiBasinSearch


class MaxCutProblem:
    """
    Represents a Max-Cut problem.
    
    Graph with vertices and edges. Goal: partition vertices into two sets
    to maximize the number of edges crossing the partition.
    """
    
    def __init__(self, num_vertices: int, edges: List[Tuple[int, int]]):
        """
        Initialize Max-Cut problem.
        
        Args:
            num_vertices: Number of vertices (0-indexed: 0 to num_vertices-1)
            edges: List of (u, v) tuples representing edges
        """
        self.num_vertices = num_vertices
        self.edges = edges
        self.num_edges = len(edges)
        
        # Build adjacency list for faster lookups
        self.adjacency: Dict[int, Set[int]] = {i: set() for i in range(num_vertices)}
        for u, v in edges:
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)
    
    @classmethod
    def from_gset_file(cls, file_path: Path) -> 'MaxCutProblem':
        """
        Load Max-Cut problem from GSET format file.
        
        Format:
        - First line: num_vertices num_edges
        - Subsequent lines: u v (edge endpoints, 1-indexed)
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header = lines[0].strip().split()
        num_vertices = int(header[0])
        num_edges = int(header[1])
        
        # Parse edges (convert from 1-indexed to 0-indexed)
        edges = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                u = int(parts[0]) - 1  # Convert to 0-indexed
                v = int(parts[1]) - 1
                edges.append((u, v))
        
        return cls(num_vertices, edges)
    
    def compute_cut_size(self, partition: Dict[int, int]) -> int:
        """
        Compute the size of the cut for a given partition.
        
        Args:
            partition: Dictionary mapping vertex_id -> side (0 or 1, or -1 or +1)
        
        Returns:
            Number of edges crossing the cut
        """
        cut_size = 0
        for u, v in self.edges:
            # Edge crosses cut if vertices are on different sides
            side_u = partition.get(u, 0)
            side_v = partition.get(v, 0)
            
            # Normalize to 0/1 if using -1/+1 encoding
            if side_u < 0:
                side_u = 0
            elif side_u > 0:
                side_u = 1
            if side_v < 0:
                side_v = 0
            elif side_v > 0:
                side_v = 1
            
            if side_u != side_v:
                cut_size += 1
        
        return cut_size
    
    def compute_tension(self, partition: Dict[int, int]) -> float:
        """
        Compute total tension (number of edges NOT crossing the cut).
        
        Lower tension = larger cut = better solution.
        
        Args:
            partition: Dictionary mapping vertex_id -> side
        
        Returns:
            Total tension (number of "bad" edges)
        """
        bad_edges = 0
        for u, v in self.edges:
            side_u = partition.get(u, 0)
            side_v = partition.get(v, 0)
            
            # Normalize to 0/1
            if side_u < 0:
                side_u = 0
            elif side_u > 0:
                side_u = 1
            if side_v < 0:
                side_v = 0
            elif side_v > 0:
                side_v = 1
            
            # Tension = 1 if same side, 0 if different sides
            if side_u == side_v:
                bad_edges += 1
        
        return float(bad_edges)


def decode_partition_from_basin(
    basin: Basin,
    system: LivniumCoreSystem,
    problem: MaxCutProblem,
    variable_mappings: Dict[str, List[Tuple[int, int, int]]]
) -> Dict[int, int]:
    """
    Decode vertex partition from basin state.
    
    Returns:
        Dictionary mapping vertex_id -> side (0 or 1)
    """
    partition = {}
    
    for vertex_id in range(problem.num_vertices):
        var_name = f"vertex_{vertex_id}"
        if var_name not in variable_mappings:
            continue
        
        coords = variable_mappings[var_name]
        if not coords:
            continue
        
        cell = system.get_cell(coords[0])
        if not cell:
            continue
        
        # Decode side from SW
        # SW < 10 = side 0, SW >= 10 = side 1
        side = 1 if cell.symbolic_weight >= 10.0 else 0
        partition[vertex_id] = side
    
    return partition


def solve_max_cut_livnium(
    problem: MaxCutProblem,
    max_steps: int = 1000,
    max_time: float = 60.0,
    verbose: bool = False,
    use_recursive: bool = False,
    recursive_depth: int = 2
) -> Dict[str, Any]:
    """
    Solve a Max-Cut problem using Livnium.
    
    Args:
        problem: MaxCutProblem instance
        max_steps: Maximum search steps
        max_time: Maximum time in seconds
        verbose: Print progress
        use_recursive: Use RecursiveGeometryEngine for more capacity
        recursive_depth: Recursion depth if using recursive geometry
    
    Returns:
        Dictionary with results:
        - 'solved': bool
        - 'cut_size': int (number of edges in cut)
        - 'partition': Dict[int, int] (vertex_id -> side)
        - 'time': float
        - 'steps': int
        - 'tension': float
        - 'max_possible_cut': int (total edges)
    """
    start_time = time.time()
    
    if verbose:
        print(f"Max-Cut Problem: {problem.num_vertices} vertices, {problem.num_edges} edges")
    
    # Estimate lattice size needed
    n_lattice = max(3, int((problem.num_vertices ** (1/3)) + 1))
    if n_lattice % 2 == 0:
        n_lattice += 1  # Must be odd
    
    # For recursive, use smaller base lattice
    if use_recursive:
        base_lattice = min(5, n_lattice)
        if base_lattice % 2 == 0:
            base_lattice += 1
    else:
        base_lattice = n_lattice
    
    if verbose:
        if use_recursive:
            print(f"  Base lattice: {base_lattice}×{base_lattice}×{base_lattice} = {base_lattice**3} cells")
            print(f"  Recursive depth: {recursive_depth}")
        else:
            print(f"  Lattice size: {base_lattice}×{base_lattice}×{base_lattice} = {base_lattice**3} cells")
    
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
            print(f"  ✓ Recursive geometry ready: {total_cells:,} total cells across {recursive_depth + 1} levels")
    else:
        system = base_system
    
    # Encode Max-Cut problem
    encoder = UniversalProblemEncoder(system)
    
    # Convert edges to constraint format
    # Each edge (u, v) creates a constraint: u != v (different sides)
    constraints = []
    for u, v in problem.edges:
        constraints.append({
            'type': 'max_cut_edge',
            'vars': [f"vertex_{u}", f"vertex_{v}"],
            'edge': (u, v)
        })
    
    # Variables: each vertex can be 0 or 1 (side 0 or side 1)
    variables = {f"vertex_{i}": [0, 1] for i in range(problem.num_vertices)}
    
    problem_dict = {
        'type': 'max_cut',
        'variables': variables,
        'constraints': constraints,
        'num_vertices': problem.num_vertices,
        'edges': problem.edges,
        'n_candidates': min(20, max(5, problem.num_vertices))
    }
    
    if verbose:
        print("Encoding Max-Cut problem...")
    
    # For Max-Cut, we need custom encoding since it's not in UniversalProblemEncoder yet
    # We'll create tension fields manually
        constraint_encoder_module = importlib.import_module('core.encoder.constraint_encoder')
    ConstraintEncoder = constraint_encoder_module.ConstraintEncoder
    TensionField = constraint_encoder_module.TensionField
    
    constraint_encoder = ConstraintEncoder(system)
    tension_fields = []
    variable_mappings = {}
    candidate_basins = []
    
    # Map each vertex to a coordinate
    import random
    lattice_coords = list(system.lattice.keys())
    random.shuffle(lattice_coords)
    
    for i, vertex_id in enumerate(range(problem.num_vertices)):
        if i < len(lattice_coords):
            coord = lattice_coords[i]
            var_name = f"vertex_{vertex_id}"
            variable_mappings[var_name] = [coord]
            
            # Initialize cell with random side (0 or 1)
            cell = system.get_cell(coord)
            if cell:
                # SW < 10 = side 0, SW >= 10 = side 1
                cell.symbolic_weight = random.choice([5.0, 15.0])
    
    # Create tension field for each edge
    for u, v in problem.edges:
        var_u = f"vertex_{u}"
        var_v = f"vertex_{v}"
        
        if var_u not in variable_mappings or var_v not in variable_mappings:
            continue
        
        coords_u = variable_mappings[var_u]
        coords_v = variable_mappings[var_v]
        
        # Create tension function: tension = 0 if different sides, 1 if same side
        def create_edge_tension_fn(c_u, c_v):
            def compute_tension(system: LivniumCoreSystem) -> float:
                cell_u = system.get_cell(c_u[0]) if c_u else None
                cell_v = system.get_cell(c_v[0]) if c_v else None
                
                if not cell_u or not cell_v:
                    return 1.0  # Missing = violation
                
                # Decode sides
                side_u = 1 if cell_u.symbolic_weight >= 10.0 else 0
                side_v = 1 if cell_v.symbolic_weight >= 10.0 else 0
                
                # Tension = 0 if different, 1 if same
                if side_u == side_v:
                    return 1.0
                else:
                    return 0.0
            
            return compute_tension
        
        tension_fn = create_edge_tension_fn(coords_u, coords_v)
        
        field = TensionField(
            constraint_id=f"edge_{u}_{v}",
            involved_coords=coords_u + coords_v,
            compute_tension=tension_fn,
            compute_curvature=lambda sys: 1.0 / (1.0 + tension_fn(sys))
        )
        tension_fields.append(field)
    
    # Create candidate basins (random initial partitions)
    for _ in range(min(20, max(5, problem.num_vertices))):
        basin_coords = random.sample(lattice_coords[:problem.num_vertices], 
                                     min(3, problem.num_vertices))
        candidate_basins.append(basin_coords)
    
    if verbose:
        print(f"  Created {len(tension_fields)} tension fields (one per edge)")
        print(f"  Created {len(candidate_basins)} candidate basins")
    
    # Define correctness checker (not really applicable for Max-Cut optimization)
    def check_correctness(basin: Basin, system: LivniumCoreSystem) -> bool:
        # Max-Cut is optimization, not satisfaction
        # We'll just check if we have a valid partition
        partition = decode_partition_from_basin(
            basin, system, problem, variable_mappings
        )
        return len(partition) == problem.num_vertices
    
    # Create custom search with edge tension
    class MaxCutMultiBasinSearch(MultiBasinSearch):
        """Multi-basin search with Max-Cut edge tension fields."""
        
        def __init__(self, tension_fields, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tension_fields = tension_fields
        
        def update_all_basins(self, system):
            """Override to include edge tension."""
            from core.search.native_dynamic_basin_search import get_geometry_signals
            
            for basin in self.basins:
                if not basin.is_alive:
                    continue
                
                curvature, base_tension, entropy = get_geometry_signals(
                    system, basin.active_coords
                )
                
                # Add edge tension
                edge_tension = 0.0
                for field in self.tension_fields:
                    edge_tension += field.get_tension(system)
                
                basin.curvature = curvature
                basin.tension = base_tension + edge_tension
                basin.entropy = entropy
                basin.update_score()
                basin.age += 1
            
            # Identify winner (lowest tension = best cut)
            alive_basins = [b for b in self.basins if b.is_alive]
            if alive_basins:
                winner = min(alive_basins, key=lambda b: b.tension)
                winner.is_winning = True
                for basin in alive_basins:
                    if basin.id != winner.id:
                        basin.is_winning = False
            
            self._apply_basin_dynamics(system)
            self._prune_dead_basins()
    
    # Solve with Max-Cut-enhanced multi-basin search
    if verbose:
        print("Starting multi-basin search...")
    
    search = MaxCutMultiBasinSearch(
        tension_fields,
        use_rotations=True
    )
    
    # Add candidate basins
    for coords in candidate_basins:
        search.add_basin(coords, system)
    
    if verbose:
        print(f"  Initialized {len(search.basins)} basins")
    
    # Iterative search loop
    import random
    from core.classical.livnium_core_system import RotationAxis
    
    for step in range(max_steps):
        if time.time() - start_time > max_time:
            if verbose:
                print(f"  Timeout at step {step}")
            break
        
        search.update_all_basins(system)
        
        winner = search.get_winner()
        if winner and check_correctness(winner, system):
            if verbose:
                print(f"  Valid partition found at step {step+1}")
            stats = search.get_basin_stats()
            elapsed_time = time.time() - start_time
            break
        
        if step % 10 == 0:
            axis = random.choice(list(RotationAxis))
            system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
        
        stats = search.get_basin_stats()
        if stats['num_alive'] == 1:
            if verbose:
                print(f"  Converged at step {step+1}")
            winner = search.get_best_basin()
            elapsed_time = time.time() - start_time
            break
        
        if verbose and (step + 1) % 50 == 0:
            winner_check = search.get_best_basin()
            print(f"    Step {step+1}: {stats['num_alive']} basins, "
                  f"score: {stats['best_score']:.4f}, "
                  f"tension: {winner_check.tension if winner_check else 'N/A':.4f}")
    else:
        winner = search.get_best_basin()
        stats = search.get_basin_stats()
        if verbose:
            print(f"  Search completed: {stats['num_alive']} basins alive")
    
    elapsed_time = time.time() - start_time
    steps = step + 1 if 'step' in locals() else max_steps
    
    # Extract result
    result = {
        'solved': False,
        'cut_size': 0,
        'partition': None,
        'time': elapsed_time,
        'steps': steps,
        'tension': float('inf'),
        'max_possible_cut': problem.num_edges
    }
    
    if winner:
        partition = decode_partition_from_basin(
            winner, system, problem, variable_mappings
        )
        cut_size = problem.compute_cut_size(partition)
        tension = problem.compute_tension(partition)
        
        result['solved'] = True
        result['cut_size'] = cut_size
        result['partition'] = partition
        result['tension'] = tension
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Solve Max-Cut using Livnium')
    parser.add_argument('graph_file', help='Path to GSET graph file')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max search steps')
    parser.add_argument('--max-time', type=float, default=60.0, help='Max time (seconds)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--use-recursive', action='store_true', help='Use recursive geometry')
    parser.add_argument('--recursive-depth', type=int, default=2, help='Recursive depth')
    
    args = parser.parse_args()
    
    problem = MaxCutProblem.from_gset_file(Path(args.graph_file))
    
    result = solve_max_cut_livnium(
        problem,
        max_steps=args.max_steps,
        max_time=args.max_time,
        verbose=args.verbose,
        use_recursive=args.use_recursive,
        recursive_depth=args.recursive_depth
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Solved: {result['solved']}")
    print(f"Cut size: {result['cut_size']}/{result['max_possible_cut']} edges")
    print(f"Cut ratio: {result['cut_size']/result['max_possible_cut']:.2%}")
    print(f"Time: {result['time']:.3f}s")
    print(f"Steps: {result['steps']}")
    print(f"Tension: {result['tension']:.1f}")

