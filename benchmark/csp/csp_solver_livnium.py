"""
CSP Solver using Livnium Core System

Converts CSP problems to tension fields and uses multi-basin search
to find satisfying assignments.
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

# Import Universal Encoder
encoder_module = importlib.import_module('core.encoder.problem_encoder')
UniversalProblemEncoder = encoder_module.UniversalProblemEncoder

from core.search.multi_basin_search import Basin, MultiBasinSearch


class CSPProblem:
    """
    Represents a Constraint Satisfaction Problem.
    
    Variables: {var_name: domain}
    Constraints: List of constraint functions
    """
    
    def __init__(self, variables: Dict[str, List[Any]], constraints: List[Dict[str, Any]]):
        """
        Initialize CSP problem.
        
        Args:
            variables: Dictionary mapping variable names to their domains
            constraints: List of constraint dictionaries with:
                - 'type': 'all_different', 'equal', 'not_equal', 'custom'
                - 'vars': List of variable names involved
                - 'fn': Optional custom constraint function
        """
        self.variables = variables
        self.constraints = constraints
        self.var_names = list(variables.keys())
        self.num_vars = len(variables)
    
    def check_constraint(self, constraint: Dict[str, Any], assignment: Dict[str, Any]) -> bool:
        """
        Check if a constraint is satisfied by an assignment.
        
        Returns:
            True if constraint is satisfied, False otherwise
        """
        constraint_type = constraint.get('type', 'custom')
        vars_involved = constraint.get('vars', [])
        
        # Get values for variables in constraint
        values = []
        for var_name in vars_involved:
            if var_name not in assignment:
                return False  # Variable not assigned
            values.append(assignment[var_name])
        
        if constraint_type == 'all_different':
            # All variables must have different values
            return len(values) == len(set(values))
        
        elif constraint_type == 'equal':
            # All variables must be equal
            return len(set(values)) == 1
        
        elif constraint_type == 'not_equal':
            # All variables must be different (same as all_different for 2 vars)
            return len(values) == len(set(values))
        
        elif constraint_type == 'diagonal':
            # Diagonal constraint for N-Queens
            row_i = constraint.get('row_i')
            row_j = constraint.get('row_j')
            if row_i is not None and row_j is not None:
                qi_val = assignment.get(f"Q{row_i+1}")
                qj_val = assignment.get(f"Q{row_j+1}")
                if qi_val is not None and qj_val is not None:
                    return abs(qi_val - qj_val) != abs(row_i - row_j)
            return False
        
        elif constraint_type == 'custom':
            # Use custom function
            fn = constraint.get('fn')
            if fn:
                return fn(assignment)
            return True
        
        return True
    
    def check_all_constraints(self, assignment: Dict[str, Any]) -> Tuple[bool, int]:
        """
        Check if all constraints are satisfied.
        
        Returns:
            (all_satisfied, num_satisfied)
        """
        num_satisfied = 0
        for constraint in self.constraints:
            if self.check_constraint(constraint, assignment):
                num_satisfied += 1
        
        all_satisfied = (num_satisfied == len(self.constraints))
        return all_satisfied, num_satisfied


def decode_assignment_from_basin(
    basin: Basin,
    system: LivniumCoreSystem,
    csp: CSPProblem,
    variable_mappings: Dict[str, List[Tuple[int, int, int]]]
) -> Dict[str, Any]:
    """
    Decode variable assignment from basin state.
    
    Returns:
        Dictionary mapping variable_name -> value
    """
    assignment = {}
    
    for var_name in csp.var_names:
        if var_name not in variable_mappings:
            continue
        
        coords = variable_mappings[var_name]
        if not coords:
            continue
        
        cell = system.get_cell(coords[0])
        if not cell:
            continue
        
        # Decode value from SW
        # Map SW to domain value
        domain = csp.variables[var_name]
        if not domain:
            continue
        
        # Use SW to index into domain (modulo)
        sw_value = int(cell.symbolic_weight) % len(domain)
        assignment[var_name] = domain[sw_value]
    
    return assignment


def solve_csp_livnium(
    csp: CSPProblem,
    max_steps: int = 1000,
    max_time: float = 60.0,
    verbose: bool = False,
    use_recursive: bool = False,
    recursive_depth: int = 2
) -> Dict[str, Any]:
    """
    Solve a CSP problem using Livnium with optional recursive geometry.
    
    Args:
        csp: CSPProblem instance
        max_steps: Maximum search steps
        max_time: Maximum time in seconds
        verbose: Print progress
        use_recursive: Use RecursiveGeometryEngine for more capacity
        recursive_depth: Recursion depth if using recursive geometry
    
    Returns:
        Dictionary with results
    """
    start_time = time.time()
    
    if verbose:
        print(f"CSP Problem: {csp.num_vars} variables, {len(csp.constraints)} constraints")
    
    # Estimate lattice size needed
    n_lattice = max(3, int((csp.num_vars ** (1/3)) + 1))
    if n_lattice % 2 == 0:
        n_lattice += 1  # Must be odd
    
    # For recursive, use smaller base lattice
    if use_recursive:
        base_lattice = min(5, n_lattice)  # Use 5×5×5 base for recursive
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
        # Use base system for encoding (recursive levels available for expansion)
        system = base_system
        if verbose:
            total_cells = sum(len(level.geometry.lattice) for level in recursive_engine.levels.values())
            print(f"  ✓ Recursive geometry ready: {total_cells:,} total cells across {recursive_depth + 1} levels")
    else:
        system = base_system
    
    # Encode CSP problem
    encoder = UniversalProblemEncoder(system)
    
    problem = {
        'type': 'constraint_satisfaction',
        'variables': csp.variables,
        'constraints': csp.constraints,
        'n_candidates': min(20, max(5, csp.num_vars))
    }
    
    if verbose:
        print("Encoding CSP problem...")
    
    encoded = encoder.encode(problem)
    
    if verbose:
        print(f"  Created {len(encoded.tension_fields)} tension fields")
        print(f"  Created {len(encoded.candidate_basins)} candidate basins")
    
    # Define correctness checker
    def check_correctness(basin: Basin, system: LivniumCoreSystem) -> bool:
        assignment = decode_assignment_from_basin(
            basin, system, csp, encoded.variable_mappings
        )
        all_satisfied, _ = csp.check_all_constraints(assignment)
        return all_satisfied
    
    # Create custom search with constraint tension
    class CSPMultiBasinSearch(MultiBasinSearch):
        """Multi-basin search with CSP constraint tension fields."""
        
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
    
    # Solve with CSP-enhanced multi-basin search
    if verbose:
        print("Starting multi-basin search...")
    
    search = CSPMultiBasinSearch(
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
    
    for step in range(max_steps):
        if time.time() - start_time > max_time:
            if verbose:
                print(f"  Timeout at step {step}")
            break
        
        search.update_all_basins(system)
        
        winner = search.get_winner()
        if winner and check_correctness(winner, system):
            if verbose:
                print(f"  Solution found at step {step+1}")
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
    
    # Check result
    result = {
        'solved': False,
        'satisfiable': None,
        'assignment': None,
        'time': elapsed_time,
        'steps': steps,
        'num_satisfied_constraints': 0,
        'total_constraints': len(csp.constraints)
    }
    
    if winner:
        assignment = decode_assignment_from_basin(
            winner, system, csp, encoded.variable_mappings
        )
        all_satisfied, num_satisfied = csp.check_all_constraints(assignment)
        
        result['solved'] = True
        result['satisfiable'] = all_satisfied
        result['assignment'] = assignment
        result['num_satisfied_constraints'] = num_satisfied
    
    return result

