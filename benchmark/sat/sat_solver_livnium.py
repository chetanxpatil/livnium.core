"""
SAT Solver using Livnium Core System

Converts CNF formulas to tension fields and uses multi-basin search
to find satisfying assignments.
"""

import time
import sys
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import LivniumCoreConfig
from core.classical.livnium_core_system import LivniumCoreSystem
import importlib

# Import Universal Encoder
encoder_module = importlib.import_module('core.encoder.problem_encoder')
UniversalProblemEncoder = encoder_module.UniversalProblemEncoder

from core.search.multi_basin_search import solve_with_multi_basin, Basin, MultiBasinSearch


def parse_cnf_file(cnf_path: str) -> Tuple[int, List[List[int]]]:
    """
    Parse a CNF file in DIMACS format.
    
    Returns:
        (num_vars, clauses) where clauses is list of lists of literals
    """
    num_vars = 0
    clauses = []
    
    with open(cnf_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                # Header line: p cnf num_vars num_clauses
                parts = line.split()
                if len(parts) >= 4 and parts[1] == 'cnf':
                    num_vars = int(parts[2])
            else:
                # Clause line: literals ending with 0
                literals = [int(x) for x in line.split() if x != '0']
                if literals:
                    clauses.append(literals)
    
    return num_vars, clauses


def check_sat_assignment(
    assignment: Dict[int, bool],
    clauses: List[List[int]]
) -> Tuple[bool, int]:
    """
    Check if an assignment satisfies all clauses.
    
    Returns:
        (is_satisfying, num_satisfied_clauses)
    """
    num_satisfied = 0
    for clause in clauses:
        clause_satisfied = False
        for literal in clause:
            var_id = abs(literal)
            var_value = assignment.get(var_id, False)
            if (literal > 0 and var_value) or (literal < 0 and not var_value):
                clause_satisfied = True
                break
        if clause_satisfied:
            num_satisfied += 1
    
    is_satisfying = (num_satisfied == len(clauses))
    return is_satisfying, num_satisfied


def decode_assignment_from_basin(
    basin: Basin,
    system: LivniumCoreSystem,
    num_vars: int,
    variable_mappings: Dict[str, List[Tuple[int, int, int]]]
) -> Dict[int, bool]:
    """
    Decode variable assignment from basin state.
    
    Returns:
        Dictionary mapping variable_id -> bool
    """
    assignment = {}
    
    for var_id in range(1, num_vars + 1):
        if f"var_{var_id}" in variable_mappings:
            coords = variable_mappings[f"var_{var_id}"]
            if coords:
                cell = system.get_cell(coords[0])
                if cell:
                    # SW >= 10 = True, SW < 10 = False
                    assignment[var_id] = cell.symbolic_weight >= 10.0
                else:
                    assignment[var_id] = False
            else:
                assignment[var_id] = False
        else:
            assignment[var_id] = False
    
    return assignment


def solve_sat_livnium(
    cnf_path: str,
    max_steps: int = 500,
    max_time: float = 60.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Solve a SAT problem using Livnium.
    
    Args:
        cnf_path: Path to CNF file
        max_steps: Maximum search steps
        max_time: Maximum time in seconds
        verbose: Print progress
    
    Returns:
        Dictionary with results:
        - 'solved': bool
        - 'satisfiable': bool or None
        - 'assignment': Dict[int, bool] or None
        - 'time': float
        - 'steps': int
        - 'num_satisfied_clauses': int
    """
    start_time = time.time()
    
    # Parse CNF file
    if verbose:
        print(f"Parsing CNF file: {cnf_path}")
    
    num_vars, clauses = parse_cnf_file(cnf_path)
    
    if verbose:
        print(f"  Variables: {num_vars}, Clauses: {len(clauses)}")
    
    # Estimate lattice size needed
    # Need at least num_vars cells
    n_lattice = max(3, int((num_vars ** (1/3)) + 1))
    if n_lattice % 2 == 0:
        n_lattice += 1  # Must be odd
    
    if verbose:
        print(f"  Lattice size: {n_lattice}×{n_lattice}×{n_lattice} = {n_lattice**3} cells")
    
    # Create Livnium system
    config = LivniumCoreConfig(
        lattice_size=n_lattice,
        enable_semantic_polarity=True
    )
    system = LivniumCoreSystem(config)
    
    # Encode SAT problem
    encoder = UniversalProblemEncoder(system)
    
    problem = {
        'type': 'sat',
        'clauses': clauses,
        'num_vars': num_vars,
        'n_candidates': min(20, max(5, num_vars // 2))  # Adaptive candidate count
    }
    
    if verbose:
        print("Encoding SAT problem...")
    
    encoded = encoder.encode(problem)
    
    if verbose:
        print(f"  Created {len(encoded.tension_fields)} tension fields")
        print(f"  Created {len(encoded.candidate_basins)} candidate basins")
    
    # Define correctness checker
    def check_correctness(basin: Basin, system: LivniumCoreSystem) -> bool:
        assignment = decode_assignment_from_basin(
            basin, system, num_vars, encoded.variable_mappings
        )
        is_satisfying, num_satisfied = check_sat_assignment(assignment, clauses)
        return is_satisfying
    
    # Create custom tension function that uses constraint tension fields
    
    class SATMultiBasinSearch(MultiBasinSearch):
        """Multi-basin search with SAT constraint tension fields."""
        
        def __init__(self, tension_fields, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tension_fields = tension_fields
        
        def update_all_basins(self, system):
            """Override to include constraint tension."""
            # Update geometry signals for all basins
            for basin in self.basins:
                if not basin.is_alive:
                    continue
                
                # Get base geometry signals
                from core.search.native_dynamic_basin_search import get_geometry_signals
                curvature, base_tension, entropy = get_geometry_signals(
                    system, basin.active_coords
                )
                
                # Add constraint tension from SAT clauses
                constraint_tension = 0.0
                for field in self.tension_fields:
                    clause_tension = field.get_tension(system)
                    constraint_tension += clause_tension
                
                # Total tension = base + constraint
                basin.curvature = curvature
                basin.tension = base_tension + constraint_tension
                basin.entropy = entropy
                basin.update_score()
                basin.age += 1
            
            # Identify winner (highest score)
            alive_basins = [b for b in self.basins if b.is_alive]
            if alive_basins:
                winner = max(alive_basins, key=lambda b: b.score)
                winner.is_winning = True
                
                # Mark others as not winning
                for basin in alive_basins:
                    if basin.id != winner.id:
                        basin.is_winning = False
            
            # Apply basin dynamics
            self._apply_basin_dynamics(system)
            
            # Prune dead basins
            self._prune_dead_basins()
    
    # Solve with SAT-enhanced multi-basin search
    if verbose:
        print("Starting multi-basin search with constraint tension...")
    
    # Use custom search with tension fields
    search = SATMultiBasinSearch(
        encoded.tension_fields,
        use_rotations=True
    )
    
    # Add all candidate solutions as basins
    for coords in encoded.candidate_basins:
        search.add_basin(coords, system)
    
    if verbose:
        print(f"  Initialized {len(search.basins)} basins")
    
    # Iterative search loop
    import random
    from core.classical.livnium_core_system import RotationAxis
    
    for step in range(max_steps):
        # Check timeout
        if time.time() - start_time > max_time:
            if verbose:
                print(f"  Timeout at step {step}")
            break
        
        # Update all basins
        search.update_all_basins(system)
        
        # Check if we have a winner
        winner = search.get_winner()
        if winner and check_correctness:
            if check_correctness(winner, system):
                if verbose:
                    print(f"  Solution found at step {step+1}")
                stats = search.get_basin_stats()
                elapsed_time = time.time() - start_time
                break
        
        # Apply random rotations to explore
        if step % 10 == 0:
            axis = random.choice(list(RotationAxis))
            system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
        
        # Check for convergence
        stats = search.get_basin_stats()
        if stats['num_alive'] == 1:
            if verbose:
                print(f"  Converged to single basin at step {step+1}")
            winner = search.get_best_basin()
            elapsed_time = time.time() - start_time
            break
        
        if verbose and (step + 1) % 20 == 0:
            winner_check = search.get_best_basin()
            print(f"    Step {step+1}: {stats['num_alive']} basins alive, "
                  f"best score: {stats['best_score']:.4f}, "
                  f"tension: {winner_check.tension if winner_check else 'N/A':.4f}")
    else:
        # Loop completed without break
        winner = search.get_best_basin()
        stats = search.get_basin_stats()
        if verbose:
            print(f"  Search completed: {stats['num_alive']} basins alive")
    
    # Ensure elapsed_time is set
    elapsed_time = time.time() - start_time
    steps = step + 1 if 'step' in locals() else max_steps
    
    elapsed_time = time.time() - start_time
    
    # Check result
    result = {
        'solved': False,
        'satisfiable': None,
        'assignment': None,
        'time': elapsed_time,
        'steps': steps,
        'num_satisfied_clauses': 0,
        'total_clauses': len(clauses)
    }
    
    if winner:
        assignment = decode_assignment_from_basin(
            winner, system, num_vars, encoded.variable_mappings
        )
        is_satisfying, num_satisfied = check_sat_assignment(assignment, clauses)
        
        result['solved'] = True
        result['satisfiable'] = is_satisfying
        result['assignment'] = assignment
        result['num_satisfied_clauses'] = num_satisfied
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Solve SAT using Livnium')
    parser.add_argument('cnf_file', help='Path to CNF file')
    parser.add_argument('--max-steps', type=int, default=500, help='Max search steps')
    parser.add_argument('--max-time', type=float, default=60.0, help='Max time (seconds)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    result = solve_sat_livnium(
        args.cnf_file,
        max_steps=args.max_steps,
        max_time=args.max_time,
        verbose=args.verbose
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Solved: {result['solved']}")
    print(f"Satisfiable: {result['satisfiable']}")
    print(f"Time: {result['time']:.3f}s")
    print(f"Steps: {result['steps']}")
    print(f"Satisfied clauses: {result['num_satisfied_clauses']}/{result['total_clauses']}")
    
    if result['assignment']:
        print(f"\nAssignment (first 10 variables):")
        for var_id in sorted(result['assignment'].keys())[:10]:
            print(f"  x{var_id} = {result['assignment'][var_id]}")

