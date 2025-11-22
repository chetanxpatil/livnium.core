"""
Search Module: Dynamic Basin Reinforcement and Geometric Search

This module contains search algorithms and basin reinforcement mechanisms
for geometric problem solving.

Key Components:
- Dynamic Basin Reinforcement: Geometry-driven, self-tuning basin shaping
- Geometric Search: Search strategies that use geometry signals
"""

# Native dynamic basin search
from .native_dynamic_basin_search import (
    update_basin_dynamic,
    compute_local_curvature,
    compute_symbolic_tension,
    compute_noise_entropy,
    get_geometry_signals,
    apply_dynamic_basin,
    DynamicBasinSearch,
)

# Multi-basin search
from .multi_basin_search import (
    Basin,
    MultiBasinSearch,
    solve_with_multi_basin,
    create_candidate_basins,
)

# Corner rotation policy
from .corner_rotation_policy import (
    should_allow_corner_rotations,
    rotation_affects_corners,
    get_safe_rotation,
)

__all__ = [
    # Dynamic basin
    'update_basin_dynamic',
    'compute_local_curvature',
    'compute_symbolic_tension',
    'compute_noise_entropy',
    'get_geometry_signals',
    'apply_dynamic_basin',
    'DynamicBasinSearch',
    # Multi-basin
    'Basin',
    'MultiBasinSearch',
    'solve_with_multi_basin',
    'create_candidate_basins',
    # Corner rotation policy
    'should_allow_corner_rotations',
    'rotation_affects_corners',
    'get_safe_rotation',
]

