"""
Universal Encoder: Convert Any Problem to Geometric Patterns

This module provides universal encoding that converts any problem class
(parity, graph theory, logic, SNLI, Ramsey, QSAT, etc.) into geometric patterns.

Key Principle (Corrected):
- Constraints → Tension fields (energy landscape)
- Solutions → Basins (candidate attractors)
- Search minimizes tension by finding best basin

This is Layer 3 of the universal problem solver architecture.
"""

from .problem_encoder import (
    UniversalProblemEncoder,
    EncodedProblem,
)

from .constraint_encoder import (
    ConstraintEncoder,
    TensionField,
)

__all__ = [
    'UniversalProblemEncoder',
    'EncodedProblem',
    'ConstraintEncoder',
    'TensionField',
]

