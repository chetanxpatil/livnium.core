"""
Multi-Basin Search: Competing Attractors in Simplex Space

This extends dynamic basin reinforcement to support multiple competing basins
(attractors) simultaneously. The best basin wins through geometric competition.

Adapted for Livnium-T's 5-node tetrahedral topology.

Key Concept:
- Multiple candidate solutions exist as competing basins
- Each basin has a score: curvature - tension
- Basins compete in shared geometry (SW fields)
- Losing basins decay, winning basin reinforces
- Natural selection through geometry physics

This enables solving general problems:
- SAT, constraint satisfaction, optimization
- Graph coloring, pathfinding, logic tasks
- All with unified geometric physics in simplex space
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import random

from ..classical.livnium_t_system import LivniumTSystem
from .native_dynamic_basin_search import (
    compute_local_curvature,
    compute_symbolic_tension,
    compute_noise_entropy,
    get_geometry_signals,
)
from .vertex_rotation_policy import should_allow_vertex_rotations


@dataclass
class Basin:
    """
    Represents a candidate solution as a geometric basin (attractor).
    
    Each basin has:
    - Active node IDs (nodes involved in this solution)
    - Score (curvature - tension) - higher is better
    - State (alive, decaying, winning)
    - Age (how long it's been active)
    """
    id: int
    active_node_ids: List[int]  # Node IDs (0-4) instead of coordinates
    score: float = 0.0
    curvature: float = 0.0
    tension: float = 0.0
    entropy: float = 0.0
    age: int = 0
    is_winning: bool = False
    is_alive: bool = True
    
    def update_score(self):
        """Update score: curvature - tension (higher is better)."""
        self.score = self.curvature - self.tension


class MultiBasinSearch:
    """
    Multi-basin search engine: multiple competing attractors in simplex space.
    
    This creates a natural energy landscape where:
    - Multiple candidate solutions compete
    - Best basin (highest score) wins
    - Losing basins decay and die
    - Winning basin reinforces and dominates
    """
    
    def __init__(
        self,
        base_alpha: float = 0.10,
        base_beta: float = 0.15,
        base_noise: float = 0.03,
        max_basins: int = 10,
        min_score_threshold: float = -1.0,
        use_rotations: bool = True
    ):
        """
        Initialize multi-basin search.
        
        Args:
            base_alpha: Base reinforcement strength
            base_beta: Base decay strength
            base_noise: Base decorrelation strength
            max_basins: Maximum number of basins to maintain
            min_score_threshold: Basins below this score are pruned
            use_rotations: If False, disable rotations (for tight constraints)
        """
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.base_noise = base_noise
        self.max_basins = max_basins
        self.min_score_threshold = min_score_threshold
        self.use_rotations = use_rotations
        
        self.basins: List[Basin] = []
        self.basin_counter = 0
    
    def add_basin(self, active_node_ids: List[int], system: LivniumTSystem) -> Basin:
        """
        Add a new candidate basin.
        
        Args:
            active_node_ids: List of node IDs (0-4) for this candidate
            system: LivniumTSystem
            
        Returns:
            Created Basin object
        """
        # Compute initial geometry signals
        curvature, tension, entropy = get_geometry_signals(system, active_node_ids)
        
        basin = Basin(
            id=self.basin_counter,
            active_node_ids=active_node_ids.copy(),
            curvature=curvature,
            tension=tension,
            entropy=entropy,
        )
        basin.update_score()
        
        self.basins.append(basin)
        self.basin_counter += 1
        
        # Prune if too many basins
        if len(self.basins) > self.max_basins:
            self._prune_basins()
        
        return basin
    
    def update_all_basins(self, system: LivniumTSystem) -> Dict[str, Any]:
        """
        Update all basins based on current geometry.
        
        Args:
            system: LivniumTSystem
            
        Returns:
            Statistics dictionary
        """
        # Update geometry signals for all basins
        for basin in self.basins:
            if not basin.is_alive:
                continue
            
            curvature, tension, entropy = get_geometry_signals(
                system, basin.active_node_ids
            )
            
            basin.curvature = curvature
            basin.tension = tension
            basin.entropy = entropy
            basin.update_score()
            basin.age += 1
        
        # Find winner (highest score)
        alive_basins = [b for b in self.basins if b.is_alive]
        if not alive_basins:
            return {'num_alive': 0, 'winner': None}
        
        winner = max(alive_basins, key=lambda b: b.score)
        
        # Mark winner
        for basin in self.basins:
            basin.is_winning = (basin.id == winner.id)
        
        # Reinforce winner, decay losers
        for basin in self.basins:
            if not basin.is_alive:
                continue
            
            if basin.is_winning:
                # Reinforce: strengthen attractor
                alpha = self.base_alpha * (1.0 + basin.curvature)
                for node_id in basin.active_node_ids:
                    node = system.get_node(node_id)
                    if node:
                        node.symbolic_weight += alpha
                        node.symbolic_weight = min(node.symbolic_weight, 50.0)
            else:
                # Decay: weaken attractor
                beta = self.base_beta * (1.0 + basin.tension)
                for node_id in basin.active_node_ids:
                    node = system.get_node(node_id)
                    if node:
                        node.symbolic_weight -= beta
                        node.symbolic_weight = max(node.symbolic_weight, 0.0)
                
                # Add decorrelation noise
                if self.base_noise > 0:
                    noise = self.base_noise * (1.0 + basin.entropy)
                    for node_id in basin.active_node_ids:
                        node = system.get_node(node_id)
                        if node:
                            noise_value = random.uniform(-noise, noise)
                            node.symbolic_weight += noise_value
                            node.symbolic_weight = max(node.symbolic_weight, 0.0)
                
                # Kill basins below threshold
                if basin.score < self.min_score_threshold:
                    basin.is_alive = False
        
        # Prune dead basins periodically
        if len([b for b in self.basins if b.is_alive]) > self.max_basins:
            self._prune_basins()
        
        return {
            'num_alive': len([b for b in self.basins if b.is_alive]),
            'winner': winner.id if winner else None,
            'winner_score': winner.score if winner else None,
        }
    
    def get_winner(self) -> Optional[Basin]:
        """Get the current winning basin."""
        alive_basins = [b for b in self.basins if b.is_alive and b.is_winning]
        if not alive_basins:
            return None
        return max(alive_basins, key=lambda b: b.score)
    
    def _prune_basins(self):
        """Remove dead basins and lowest-scoring basins if over limit."""
        # Remove dead basins
        self.basins = [b for b in self.basins if b.is_alive]
        
        # If still over limit, remove lowest-scoring
        if len(self.basins) > self.max_basins:
            self.basins.sort(key=lambda b: b.score, reverse=True)
            self.basins = self.basins[:self.max_basins]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        alive_basins = [b for b in self.basins if b.is_alive]
        
        if not alive_basins:
            return {
                'num_basins': 0,
                'num_alive': 0,
                'avg_score': 0.0,
                'avg_age': 0.0,
            }
        
        return {
            'num_basins': len(self.basins),
            'num_alive': len(alive_basins),
            'avg_score': float(np.mean([b.score for b in alive_basins])),
            'avg_age': float(np.mean([b.age for b in alive_basins])),
            'winner_id': self.get_winner().id if self.get_winner() else None,
        }


def solve_with_multi_basin(
    system: LivniumTSystem,
    candidate_node_ids_list: List[List[int]],
    max_steps: int = 100,
    convergence_threshold: float = 0.01,
    **kwargs
) -> Optional[Basin]:
    """
    Solve a problem using multi-basin search.
    
    Args:
        system: LivniumTSystem
        candidate_node_ids_list: List of candidate solutions (each is list of node IDs)
        max_steps: Maximum search steps
        convergence_threshold: Score change threshold for convergence
        **kwargs: Additional arguments for MultiBasinSearch
        
    Returns:
        Winning Basin if converged, None otherwise
    """
    search = MultiBasinSearch(**kwargs)
    
    # Add all candidates as basins
    for node_ids in candidate_node_ids_list:
        search.add_basin(node_ids, system)
    
    # Run search
    prev_score = None
    for step in range(max_steps):
        stats = search.update_all_basins(system)
        
        if stats['num_alive'] == 0:
            break
        
        winner = search.get_winner()
        if winner:
            # Check convergence
            if prev_score is not None:
                score_change = abs(winner.score - prev_score)
                if score_change < convergence_threshold:
                    return winner
            prev_score = winner.score
        
        # Single basin = converged
        if stats['num_alive'] == 1:
            return search.get_winner()
    
    return search.get_winner()


def create_candidate_basins(
    system: LivniumTSystem,
    num_candidates: int = 5,
    min_nodes: int = 2,
    max_nodes: int = 4
) -> List[List[int]]:
    """
    Create random candidate basins.
    
    Args:
        system: LivniumTSystem
        num_candidates: Number of candidates to create
        min_nodes: Minimum nodes per candidate
        max_nodes: Maximum nodes per candidate
        
    Returns:
        List of candidate node ID lists
    """
    candidates = []
    all_node_ids = list(range(5))  # 0-4
    
    for _ in range(num_candidates):
        num_nodes = random.randint(min_nodes, max_nodes)
        node_ids = random.sample(all_node_ids, num_nodes)
        candidates.append(node_ids)
    
    return candidates

