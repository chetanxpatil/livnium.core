"""
Livnium-O System

Stand-alone spherical semantic engine independent of other Livnium systems.

Complete implementation of the Livnium-O System specification:
- 1+N topology (1 core + N neighbor spheres)
- Continuous exposure based on solid angle
- Symbolic Weight: SW = 9·f, ΣSW = 9N
- Spherical rotation group SO(3) (continuous)
- Generalized kissing constraint
- Perfect reversibility
"""

# Classical system
from .classical import (
    LivniumOSystem,
    SphereNode,
    Observer,
    NodeClass,
    SphericalRotationGroup,
    kissing_constraint_weight,
    check_kissing_constraint,
)

__all__ = [
    # Classical
    'LivniumOSystem',
    'SphereNode',
    'Observer',
    'NodeClass',
    'SphericalRotationGroup',
    'kissing_constraint_weight',
    'check_kissing_constraint',
]

