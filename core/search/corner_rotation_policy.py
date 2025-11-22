"""
Corner Rotation Policy: Post-Convergence Refinement

This module implements the physics rule that corner rotations (high-exposure cell rotations)
should only be allowed during the post-convergence refinement phase.

Why:
- Corners are max-exposure cells (face_exposure = 3, SW = 27)
- They have maximum geometric influence and basin pull
- Early/mid process: Corner flips destabilize the lattice
- End process: Corner flips fix final parity, global symmetry, SW distribution

Rule:
    if basin_depth > threshold and drift < epsilon:
        allow_corner_rotations = True
    else:
        allow_corner_rotations = False

This is the same physics as Rubik's cubes: the last moves are almost always corner parity fixes.
"""

from typing import Optional, Tuple, Dict, Any
from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis, CellClass
from core.search.native_dynamic_basin_search import (
    compute_local_curvature,
    compute_symbolic_tension
)


def should_allow_corner_rotations(
    system: LivniumCoreSystem,
    active_coords: Optional[list] = None,
    basin_depth_threshold: float = 0.5,
    tension_epsilon: float = 0.1,
    convergence_stats: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Determine if corner rotations should be allowed.
    
    Corner rotations are unlocked only when:
    - Basin depth (curvature) > threshold (system has strong attractor)
    - Tension (drift) < epsilon (system is stable)
    - Convergence detected (single basin or Moksha state)
    
    Args:
        system: LivniumCoreSystem
        active_coords: Optional list of active coordinates (for basin calculation)
        basin_depth_threshold: Minimum curvature to allow corners (default: 0.5)
        tension_epsilon: Maximum tension to allow corners (default: 0.1)
        convergence_stats: Optional basin stats dict with 'num_alive' key
        
    Returns:
        True if corner rotations should be allowed, False otherwise
    """
    # If no active coordinates provided, check global convergence
    if active_coords is None:
        # Use convergence stats if provided
        if convergence_stats:
            num_alive = convergence_stats.get('num_alive', 10)
            # Single basin = converged
            if num_alive == 1:
                return True
        
        # Default: don't allow corners without convergence signal
        return False
    
    # Compute geometry signals
    curvature = compute_local_curvature(system, active_coords)
    tension = compute_symbolic_tension(system, active_coords)
    
    # Check convergence conditions
    basin_deep_enough = curvature > basin_depth_threshold
    tension_low_enough = tension < tension_epsilon
    
    # Also check if we have convergence stats
    if convergence_stats:
        num_alive = convergence_stats.get('num_alive', 10)
        converged = (num_alive == 1)
    else:
        converged = False
    
    # Allow corners if: (basin deep AND tension low) OR (converged)
    return (basin_deep_enough and tension_low_enough) or converged


def rotation_affects_corners(
    system: LivniumCoreSystem,
    axis: RotationAxis,
    quarter_turns: int
) -> bool:
    """
    Check if a rotation will affect corner cells.
    
    All rotations affect corners in a 3x3x3 cube, but this function
    can be extended for larger cubes where some rotations might not affect corners.
    
    Args:
        system: LivniumCoreSystem
        axis: Rotation axis
        quarter_turns: Number of quarter turns
        
    Returns:
        True if rotation affects corners, False otherwise
    """
    # For 3x3x3, all rotations affect corners
    # For larger cubes, we could check if rotation plane contains corners
    # For now, assume all rotations can affect corners
    return True


def get_safe_rotation(
    system: LivniumCoreSystem,
    active_coords: Optional[list] = None,
    allow_corners: Optional[bool] = None,
    basin_depth_threshold: float = 0.5,
    tension_epsilon: float = 0.1,
    convergence_stats: Optional[Dict[str, Any]] = None
) -> Optional[Tuple[RotationAxis, int]]:
    """
    Get a safe rotation that respects corner rotation policy.
    
    If corners are not allowed, this will avoid rotations that primarily affect corners.
    For now, all rotations can affect corners, so this returns None when corners are locked.
    
    Args:
        system: LivniumCoreSystem
        active_coords: Optional list of active coordinates
        allow_corners: Override for corner policy (None = auto-detect)
        basin_depth_threshold: Minimum curvature threshold
        tension_epsilon: Maximum tension threshold
        convergence_stats: Optional basin stats
        
    Returns:
        (axis, quarter_turns) tuple, or None if no safe rotation available
    """
    import random
    
    # Auto-detect corner policy if not provided
    if allow_corners is None:
        allow_corners = should_allow_corner_rotations(
            system,
            active_coords,
            basin_depth_threshold,
            tension_epsilon,
            convergence_stats
        )
    
    # If corners allowed, any rotation is safe
    if allow_corners:
        axis = random.choice(list(RotationAxis))
        quarter_turns = random.choice([1, 2, 3])
        return (axis, quarter_turns)
    
    # If corners not allowed, we could filter rotations
    # For now, return None to indicate "no rotation recommended"
    # In practice, callers should check allow_corners first
    return None

