"""
Vertex Rotation Policy: Post-Convergence Refinement for Livnium-T

This module implements the physics rule that vertex rotations (high-exposure node rotations)
should only be allowed during the post-convergence refinement phase.

Adapted from corner rotation policy for cubic geometry.

Why:
- Vertices are max-exposure nodes (exposure f = 3, SW = 27)
- They have maximum geometric influence and basin pull
- Early/mid process: Vertex rotations destabilize the simplex
- End process: Vertex rotations fix final parity, global symmetry, SW distribution

Rule:
    if basin_depth > threshold and drift < epsilon:
        allow_vertex_rotations = True
    else:
        allow_vertex_rotations = False

This is similar to Rubik's cube physics: the last moves are almost always corner parity fixes.
In Livnium-T, vertices play the role of corners (max exposure).
"""

from typing import Optional, Dict, Any
from ..classical.livnium_t_system import LivniumTSystem, NodeClass
from .native_dynamic_basin_search import (
    compute_local_curvature,
    compute_symbolic_tension
)


def should_allow_vertex_rotations(
    system: LivniumTSystem,
    active_node_ids: Optional[list] = None,
    basin_depth_threshold: float = 0.5,
    tension_epsilon: float = 0.1,
    convergence_stats: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Determine if vertex rotations should be allowed.
    
    Vertex rotations are unlocked only when:
    - Basin depth (curvature) > threshold (system has strong attractor)
    - Tension (drift) < epsilon (system is stable)
    - Convergence detected (single basin or Moksha state)
    
    Args:
        system: LivniumTSystem
        active_node_ids: Optional list of active node IDs (for basin calculation)
        basin_depth_threshold: Minimum curvature to allow vertices (default: 0.5)
        tension_epsilon: Maximum tension to allow vertices (default: 0.1)
        convergence_stats: Optional basin stats dict with 'num_alive' key
        
    Returns:
        True if vertex rotations should be allowed, False otherwise
    """
    # If no active node IDs provided, check global convergence
    if active_node_ids is None:
        # Use convergence stats if provided
        if convergence_stats:
            num_alive = convergence_stats.get('num_alive', 10)
            # Single basin = converged
            if num_alive == 1:
                return True
        
        # Default: don't allow vertices without convergence signal
        return False
    
    # Compute geometry signals
    curvature = compute_local_curvature(system, active_node_ids)
    tension = compute_symbolic_tension(system, active_node_ids)
    
    # Check convergence conditions
    basin_deep_enough = curvature > basin_depth_threshold
    tension_low_enough = tension < tension_epsilon
    
    # Also check if we have convergence stats
    if convergence_stats:
        num_alive = convergence_stats.get('num_alive', 10)
        converged = (num_alive == 1)
    else:
        converged = False
    
    # Allow vertices if: deep basin AND low tension AND (converged OR both conditions met)
    return (basin_deep_enough and tension_low_enough) or converged


def rotation_affects_vertices(
    system: LivniumTSystem,
    rotation_id: int
) -> bool:
    """
    Check if a rotation affects vertices (max-exposure nodes).
    
    In Livnium-T, all rotations affect vertices (since Om never moves).
    But we can check if vertices are involved.
    
    Args:
        system: LivniumTSystem
        rotation_id: Rotation ID (0-11)
        
    Returns:
        True if rotation affects vertices, False otherwise
    """
    # In Livnium-T, all rotations affect vertices (Om is immovable)
    # Vertices are nodes 1-4 (f=3)
    # Any rotation will permute vertices
    return True  # All tetrahedral rotations affect vertices


def get_safe_rotation(
    system: LivniumTSystem,
    active_node_ids: Optional[list] = None,
    convergence_stats: Optional[Dict[str, Any]] = None,
    allow_vertices: Optional[bool] = None
) -> Optional[int]:
    """
    Get a safe rotation that respects vertex rotation policy.
    
    Args:
        system: LivniumTSystem
        active_node_ids: Optional list of active node IDs
        convergence_stats: Optional convergence statistics
        allow_vertices: Override vertex policy (if None, uses policy)
        
    Returns:
        Safe rotation ID (0-11) or None if no safe rotation
    """
    import random
    
    # Check if vertices should be allowed
    if allow_vertices is None:
        allow_vertices = should_allow_vertex_rotations(
            system, active_node_ids, convergence_stats=convergence_stats
        )
    
    if allow_vertices:
        # All rotations are safe (vertices unlocked)
        return random.randint(0, 11)
    else:
        # For now, we could restrict to identity or specific rotations
        # But in Livnium-T, all rotations affect vertices, so if locked, return None
        # Or return identity rotation (0)
        return 0  # Identity rotation (no-op)

