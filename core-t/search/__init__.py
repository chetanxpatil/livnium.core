"""
Search Module: Dynamic Basin & Multi-Basin Search for Livnium-T

This module contains search algorithms and basin reinforcement mechanisms
for geometric problem solving in tetrahedral/simplex geometry.

Key Components:
- Dynamic Basin Reinforcement: Geometry-driven, self-tuning basin shaping
- Multi-Basin Search: Competing attractors in simplex space
- Vertex Rotation Policy: Post-convergence refinement for vertices
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

# Vertex rotation policy
from .vertex_rotation_policy import (
    should_allow_vertex_rotations,
    rotation_affects_vertices,
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
    # Vertex rotation policy
    'should_allow_vertex_rotations',
    'rotation_affects_vertices',
    'get_safe_rotation',
]

