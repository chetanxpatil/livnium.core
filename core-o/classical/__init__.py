"""
Livnium-O Classical System

The classical geometric system for the spherical semantic engine.
"""

from .livnium_o_system import (
    LivniumOSystem,
    SphereNode,
    Observer,
    NodeClass,
    SphericalRotationGroup,
    kissing_constraint_weight,
    check_kissing_constraint,
)

__all__ = [
    'LivniumOSystem',
    'SphereNode',
    'Observer',
    'NodeClass',
    'SphericalRotationGroup',
    'kissing_constraint_weight',
    'check_kissing_constraint',
]

