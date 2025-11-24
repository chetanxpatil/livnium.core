"""
Classical Livnium-T System

Tetrahedral geometric system with 5-node topology, symbolic weight, and A₄ rotation group.
"""

from .livnium_t_system import (
    LivniumTSystem,
    SimplexNode,
    Observer,
    NodeClass,
    TetrahedralRotationGroup,
)

__all__ = [
    'LivniumTSystem',
    'SimplexNode',
    'Observer',
    'NodeClass',
    'TetrahedralRotationGroup',
]

