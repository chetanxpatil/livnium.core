"""
Multi-Basin Search: Competing Attractors in Geometric Space

This extends dynamic basin reinforcement to support multiple competing basins
(attractors) simultaneously. The best basin wins through geometric competition.

Key Concept:
- Multiple candidate solutions exist as competing basins
- Each basin has a score: curvature - tension
- Basins compete in shared geometry (SW fields)
- Losing basins decay, winning basin reinforces
- Natural selection through geometry physics

This enables solving general problems:
- SAT, constraint satisfaction, Ramsey, optimization
- Graph coloring, pathfinding, logic tasks
- All with unified geometric physics
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import random

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.search.native_dynamic_basin_search import (
    compute_local_curvature,
    compute_symbolic_tension,
    compute_noise_entropy,
    get_geometry_signals,
)
from core.search.corner_rotation_policy import should_allow_corner_rotations


@dataclass
class Basin:
    """
    Represents a candidate solution as a geometric basin (attractor).
    
    Each basin has:
    - Active coordinates (cells involved in this solution)
    - Score (curvature - tension) - higher is better
    - State (alive, decaying, winning)
    - Age (how long it's been active)
    """
    id: int
    active_coords: List[Tuple[int, int, int]]
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
    Multi-basin search engine: multiple competing attractors.
    
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
            use_rotations: If False, disable global rotations (for tight constraints like Ramsey)
        """
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.base_noise = base_noise
        self.max_basins = max_basins
        self.min_score_threshold = min_score_threshold
        self.use_rotations = use_rotations
        
        self.basins: List[Basin] = []
        self.basin_counter = 0
    
    def add_basin(
        self,
        active_coords: List[Tuple[int, int, int]],
        system: LivniumCoreSystem
    ) -> Basin:
        """
        Add a new basin (candidate solution).
        
        Args:
            active_coords: Coordinates involved in this solution
            system: LivniumCoreSystem to compute geometry signals
            
        Returns:
            Created Basin object
        """
        # Compute initial geometry signals
        curvature, tension, entropy = get_geometry_signals(system, active_coords)
        
        # Create basin
        basin = Basin(
            id=self.basin_counter,
            active_coords=active_coords,
            curvature=curvature,
            tension=tension,
            entropy=entropy
        )
        basin.update_score()
        
        self.basin_counter += 1
        self.basins.append(basin)
        
        # Prune if too many basins
        if len(self.basins) > self.max_basins:
            self._prune_basins()
        
        return basin
    
    def update_all_basins(self, system: LivniumCoreSystem) -> None:
        """
        Update all basins: compute scores, identify winner, apply dynamics.
        
        Args:
            system: LivniumCoreSystem
        """
        # Update geometry signals for all basins
        for basin in self.basins:
            if not basin.is_alive:
                continue
            
            # Recompute geometry signals
            curvature, tension, entropy = get_geometry_signals(
                system, basin.active_coords
            )
            basin.curvature = curvature
            basin.tension = tension
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
    
    def _apply_basin_dynamics(self, system: LivniumCoreSystem) -> None:
        """
        Apply dynamic basin updates: reinforce winners, decay losers.
        
        Args:
            system: LivniumCoreSystem
        """
        for basin in self.basins:
            if not basin.is_alive:
                continue
            
            # Compute dynamic parameters
            alpha = self.base_alpha * (1.0 + basin.curvature)
            beta = self.base_beta * (1.0 + basin.tension)
            noise = self.base_noise * (1.0 + basin.entropy)
            
            if basin.is_winning:
                # Reinforce winning basin (deepen well)
                for coords in basin.active_coords:
                    cell = system.get_cell(coords)
                    if cell:
                        cell.symbolic_weight += alpha
                        cell.symbolic_weight = min(cell.symbolic_weight, 200.0)
            else:
                # Decay losing basins (flatten well)
                for coords in basin.active_coords:
                    cell = system.get_cell(coords)
                    if cell:
                        cell.symbolic_weight *= (1.0 - beta)
                        if cell.symbolic_weight < 0:
                            cell.symbolic_weight = 0.0
                
                # Add noise to decorrelate losing basins
                if random.random() < noise * 10:
                    if self.use_rotations:
                        # Check if corner rotations are allowed (post-convergence refinement)
                        winner = self.get_winner()
                        convergence_stats = self.get_basin_stats()
                        allow_corners = should_allow_corner_rotations(
                            system,
                            winner.active_coords if winner else None,
                            basin_depth_threshold=0.5,
                            tension_epsilon=0.1,
                            convergence_stats=convergence_stats
                        )
                        
                        # Global rotation (can be destructive for tight constraints)
                        # Only apply if corners allowed OR if we're not in convergence phase
                        if allow_corners or convergence_stats.get('num_alive', 10) > 1:
                            axis = random.choice(list(RotationAxis))
                            system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
                    # If rotations disabled, noise only affects SW (local adjustments)
                
                # Mark as dead if score too low
                if basin.score < self.min_score_threshold:
                    basin.is_alive = False
    
    def _prune_basins(self) -> None:
        """Prune basins: keep only the best ones."""
        # Sort by score (descending)
        self.basins.sort(key=lambda b: b.score, reverse=True)
        
        # Keep only top max_basins
        if len(self.basins) > self.max_basins:
            # Mark excess basins as dead
            for basin in self.basins[self.max_basins:]:
                basin.is_alive = False
    
    def _prune_dead_basins(self) -> None:
        """Remove dead basins from the list."""
        self.basins = [b for b in self.basins if b.is_alive]
    
    def get_winner(self) -> Optional[Basin]:
        """
        Get the current winning basin.
        
        Returns:
            Winning Basin, or None if no basins exist
        """
        alive_basins = [b for b in self.basins if b.is_alive and b.is_winning]
        if alive_basins:
            return alive_basins[0]
        return None
    
    def get_best_basin(self) -> Optional[Basin]:
        """
        Get the basin with highest score.
        
        Returns:
            Best Basin, or None if no basins exist
        """
        alive_basins = [b for b in self.basins if b.is_alive]
        if alive_basins:
            return max(alive_basins, key=lambda b: b.score)
        return None
    
    def get_basin_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current basins.
        
        Returns:
            Dictionary with basin statistics
        """
        alive_basins = [b for b in self.basins if b.is_alive]
        
        if not alive_basins:
            return {
                'num_basins': 0,
                'num_alive': 0,
                'best_score': None,
                'avg_score': None,
                'winner_id': None
            }
        
        scores = [b.score for b in alive_basins]
        winner = self.get_winner()
        
        return {
            'num_basins': len(self.basins),
            'num_alive': len(alive_basins),
            'best_score': max(scores),
            'avg_score': np.mean(scores),
            'min_score': min(scores),
            'winner_id': winner.id if winner else None,
            'basin_scores': scores
        }


def solve_with_multi_basin(
    system: LivniumCoreSystem,
    candidate_solutions: List[List[Tuple[int, int, int]]],
    max_steps: int = 100,
    check_correctness: Optional[callable] = None,
    verbose: bool = False,
    use_rotations: bool = True
) -> Tuple[Optional[Basin], int, Dict[str, Any]]:
    """
    Solve a problem using multi-basin search.
    
    This is the high-level interface for general problem solving.
    
    Args:
        system: LivniumCoreSystem
        candidate_solutions: List of candidate solutions, each as list of coordinates
        max_steps: Maximum search steps
        check_correctness: Optional function to check if a basin is correct
        verbose: Print progress
        use_rotations: If False, disable global rotations (for tight constraints like Ramsey)
        
    Returns:
        (winning_basin, steps_taken, stats)
    """
    # Initialize multi-basin search
    search = MultiBasinSearch(use_rotations=use_rotations)
    
    # Add all candidate solutions as basins
    for coords in candidate_solutions:
        search.add_basin(coords, system)
    
    if verbose:
        print(f"  Initialized {len(search.basins)} basins")
    
    # Iterative search loop
    for step in range(max_steps):
        # Update all basins
        search.update_all_basins(system)
        
        # Check if we have a winner
        winner = search.get_winner()
        if winner and check_correctness:
            if check_correctness(winner, system):
                if verbose:
                    print(f"  Solution found at step {step+1}")
                return winner, step + 1, search.get_basin_stats()
        
        # Apply random rotations to explore (if enabled)
        if use_rotations and step % 10 == 0:
            # Check convergence state
            stats = search.get_basin_stats()
            winner = search.get_winner()
            
            # Check if corner rotations are allowed (post-convergence refinement)
            allow_corners = should_allow_corner_rotations(
                system,
                winner.active_coords if winner else None,
                basin_depth_threshold=0.5,
                tension_epsilon=0.1,
                convergence_stats=stats
            )
            
            # Only apply rotations if:
            # - Corners allowed (post-convergence), OR
            # - Multiple basins still alive (exploration phase)
            if allow_corners or stats.get('num_alive', 10) > 1:
                axis = random.choice(list(RotationAxis))
                system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
        
        # Check for convergence (single dominant basin)
        stats = search.get_basin_stats()
        if stats['num_alive'] == 1:
            if verbose:
                print(f"  Converged to single basin at step {step+1}")
            return search.get_best_basin(), step + 1, stats
        
        if verbose and (step + 1) % 20 == 0:
            print(f"    Step {step+1}: {stats['num_alive']} basins alive, "
                  f"best score: {stats['best_score']:.4f}")
    
    # Return best basin after max steps
    best = search.get_best_basin()
    if verbose:
        print(f"  Search completed: {stats['num_alive']} basins alive")
    
    return best, max_steps, search.get_basin_stats()


def create_candidate_basins(
    system: LivniumCoreSystem,
    n_candidates: int = 5,
    basin_size: int = 4
) -> List[List[Tuple[int, int, int]]]:
    """
    Create random candidate basins for testing.
    
    Args:
        system: LivniumCoreSystem
        n_candidates: Number of candidate basins to create
        basin_size: Number of coordinates per basin
        
    Returns:
        List of candidate solutions (each is a list of coordinates)
    """
    coords_list = list(system.lattice.keys())
    candidates = []
    
    for _ in range(n_candidates):
        # Random selection of coordinates
        selected = random.sample(coords_list, min(basin_size, len(coords_list)))
        candidates.append(selected)
    
    return candidates

