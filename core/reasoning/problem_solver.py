"""
Problem Solver: Task API for Livnium Core System

Provides high-level interface for solving problems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable

from .reasoning_engine import ReasoningEngine
from .search_engine import SearchStrategy
from ..classical.livnium_core_system import LivniumCoreSystem, RotationAxis


class ProblemSolver:
    """
    High-level problem solver for Livnium Core System.
    
    Provides task API: problems → lattice → answer
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize problem solver.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
        self.reasoning_engine = ReasoningEngine(core_system)
        self.solved_problems: List[Dict] = []
    
    def solve_rotation_problem(self,
                              target_state: Dict[str, Any],
                              max_rotations: int = 10) -> Optional[List[str]]:
        """
        Solve: Find sequence of rotations to reach target state.
        
        Args:
            target_state: Target state description
            max_rotations: Maximum number of rotations
            
        Returns:
            List of rotation actions or None
        """
        # Define problem
        initial_state = self._encode_lattice_state()
        
        def goal_test(state):
            return self._state_matches_target(state, target_state)
        
        def successors(state):
            # Generate successor states by applying rotations
            successors_list = []
            for axis in [RotationAxis.X, RotationAxis.Y, RotationAxis.Z]:
                for quarter_turns in [1, 2, 3]:
                    # Apply rotation
                    result = self.core_system.rotate(axis, quarter_turns)
                    new_state = self._encode_lattice_state()
                    action = f"rotate_{axis.name}_{quarter_turns}"
                    successors_list.append((action, new_state))
                    # Rotate back to restore
                    self.core_system.rotate(axis, 4 - quarter_turns)
            return successors_list
        
        def heuristic(state):
            # Distance to target (simplified)
            return self._state_distance(state, target_state)
        
        problem = {
            'name': 'rotation_problem',
            'initial_state': initial_state,
            'goal_test': goal_test,
            'successors': successors,
            'heuristic': heuristic,
        }
        
        solution = self.reasoning_engine.solve_problem(
            problem,
            search_strategy=SearchStrategy.A_STAR,
            max_depth=max_rotations
        )
        
        if solution and solution['solved']:
            self.solved_problems.append(solution)
            return solution['solution_path']
        
        return None
    
    def solve_constraint_satisfaction(self,
                                     constraints: List[Callable[[Any], bool]],
                                     max_iterations: int = 100) -> Optional[Dict[str, Any]]:
        """
        Solve constraint satisfaction problem.
        
        Args:
            constraints: List of constraint functions
            max_iterations: Maximum iterations
            
        Returns:
            Solution state or None
        """
        # Simple iterative constraint satisfaction
        current_state = self._encode_lattice_state()
        
        for iteration in range(max_iterations):
            # Check if all constraints satisfied
            if all(constraint(current_state) for constraint in constraints):
                return {
                    'solved': True,
                    'state': current_state,
                    'iterations': iteration + 1
                }
            
            # Try to satisfy constraints (simplified: random rotation)
            import random
            axis = random.choice([RotationAxis.X, RotationAxis.Y, RotationAxis.Z])
            self.core_system.rotate(axis, 1)
            current_state = self._encode_lattice_state()
        
        return {
            'solved': False,
            'iterations': max_iterations
        }
    
    def _encode_lattice_state(self) -> Dict[str, Any]:
        """Encode current lattice state."""
        state = {}
        for coords, cell in self.core_system.lattice.items():
            state[coords] = {
                'face_exposure': cell.face_exposure,
                'symbolic_weight': cell.symbolic_weight,
                'cell_class': cell.cell_class.value if cell.cell_class else None,
            }
        return state
    
    def _state_matches_target(self, state: Dict[str, Any], target: Dict[str, Any]) -> bool:
        """Check if state matches target."""
        # Simplified matching
        for key, value in target.items():
            if key not in state or state[key] != value:
                return False
        return True
    
    def _state_distance(self, state: Dict[str, Any], target: Dict[str, Any]) -> float:
        """Calculate distance between state and target."""
        # Simplified distance metric
        distance = 0.0
        for key in target.keys():
            if key in state:
                if isinstance(state[key], (int, float)) and isinstance(target[key], (int, float)):
                    distance += abs(state[key] - target[key])
        return distance
    
    def get_solver_statistics(self) -> Dict:
        """Get solver statistics."""
        return {
            'total_problems_solved': len(self.solved_problems),
            'successful_solves': sum(1 for p in self.solved_problems if p.get('solved', False)),
            'reasoning_statistics': self.reasoning_engine.get_reasoning_statistics(),
        }

