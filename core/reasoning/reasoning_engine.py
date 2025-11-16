"""
Reasoning Engine: High-Level Problem Solving

Combines search, rules, and symbolic reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable

from .search_engine import SearchEngine, SearchNode, SearchStrategy
from .rule_engine import RuleEngine
from ..classical.livnium_core_system import LivniumCoreSystem


class ReasoningEngine:
    """
    High-level reasoning engine that combines search and rules.
    
    Provides problem-solving capabilities for Livnium Core System.
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize reasoning engine.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
        self.rule_engine = RuleEngine(core_system)
        self.search_history: List[Dict] = []
    
    def solve_problem(self,
                     problem: Dict[str, Any],
                     search_strategy: SearchStrategy = SearchStrategy.A_STAR,
                     max_depth: int = 50) -> Optional[Dict[str, Any]]:
        """
        Solve a problem using search and reasoning.
        
        Args:
            problem: Problem definition with initial_state, goal_test, successors
            search_strategy: Search strategy to use
            max_depth: Maximum search depth
            
        Returns:
            Solution dictionary or None
        """
        # Extract problem components
        initial_state = problem.get('initial_state')
        goal_test = problem.get('goal_test')
        successors = problem.get('successors')
        heuristic = problem.get('heuristic')
        
        if not all([initial_state, goal_test, successors]):
            raise ValueError("Problem must have initial_state, goal_test, and successors")
        
        # Create search engine
        search_engine = SearchEngine(
            initial_state=initial_state,
            goal_test=goal_test,
            successors=successors,
            heuristic=heuristic
        )
        
        # Perform search
        solution_node = search_engine.search(
            strategy=search_strategy,
            max_depth=max_depth
        )
        
        # Record search
        stats = search_engine.get_search_statistics()
        self.search_history.append({
            'problem': problem.get('name', 'unknown'),
            'strategy': search_strategy.value,
            'solved': solution_node is not None,
            'statistics': stats
        })
        
        if solution_node:
            # Extract solution path
            path = solution_node.get_path()
            actions = [node.action for node in path if node.action]
            
            return {
                'solved': True,
                'solution_path': actions,
                'final_state': solution_node.state,
                'cost': solution_node.cost,
                'depth': solution_node.depth,
                'statistics': stats
            }
        else:
            return {
                'solved': False,
                'statistics': stats
            }
    
    def reason_about_lattice_state(self) -> Dict[str, Any]:
        """
        Apply reasoning to current lattice state.
        
        Returns:
            Reasoning results
        """
        # Collect state information
        states = []
        for coords in self.core_system.lattice.keys():
            cell = self.core_system.get_cell(coords)
            state = {
                'coordinates': coords,
                'face_exposure': cell.face_exposure if cell else None,
                'symbolic_weight': cell.symbolic_weight if cell else None,
                'cell_class': cell.cell_class.value if cell and cell.cell_class else None,
            }
            
            # Apply rules
            reasoned_state = self.rule_engine.reason_about_state(state)
            states.append(reasoned_state)
        
        return {
            'total_cells': len(states),
            'reasoned_states': states,
            'rule_statistics': self.rule_engine.get_rule_statistics(),
        }
    
    def get_reasoning_statistics(self) -> Dict:
        """Get reasoning engine statistics."""
        return {
            'total_searches': len(self.search_history),
            'successful_searches': sum(1 for h in self.search_history if h['solved']),
            'rule_statistics': self.rule_engine.get_rule_statistics(),
        }

