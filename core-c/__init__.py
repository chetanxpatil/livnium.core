"""
Livnium-C System

Stand-alone circular semantic engine independent of Livnium Core and Livnium-T.

Complete implementation of the Livnium-C System specification:
- 1+N topology (1 core + N ring nodes)
- Two-class system (Core f=0, Ring f=1)
- Symbolic Weight: SW = 9·f, ΣSW = 9N
- Cyclic rotation group C_N (N elements)
- Perfect reversibility
"""

# Classical system
from .classical import (
    LivniumCSystem,
    CircleNode,
    Observer,
    NodeClass,
    CyclicRotationGroup,
)

__all__ = [
    # Classical
    'LivniumCSystem',
    'CircleNode',
    'Observer',
    'NodeClass',
    'CyclicRotationGroup',
]

