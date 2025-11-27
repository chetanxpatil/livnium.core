"""
Recursive Geometry Engine: Layer 0 - The Structural Foundation

Geometry → Geometry → Geometry: The fractal engine that makes everything scalable.

This is the missing dimension - the recursive structural layer that:
- Subdivides geometry into smaller geometry
- Projects high-dimensional states downward
- Enforces conservation recursion
- Handles recursive entanglement
- Manages recursive observers
- Handles recursive motion/rotations
- Enables recursive problem solving
"""

from .recursive_geometry_engine import RecursiveGeometryEngine
from .geometry_subdivision import GeometrySubdivision
from .recursive_projection import RecursiveProjection
from .recursive_conservation import RecursiveConservation
from .moksha_engine import MokshaEngine, FixedPointState, ConvergenceState
from .inheritance import fabricate_child_universe

__all__ = [
    'RecursiveGeometryEngine',
    'GeometrySubdivision',
    'RecursiveProjection',
    'RecursiveConservation',
    'MokshaEngine',
    'FixedPointState',
    'ConvergenceState',
    'fabricate_child_universe',
]

# Optional: Recursive Hamiltonian (requires core-o)
try:
    from .recursive_hamiltonian import RecursiveHamiltonian
    __all__.append('RecursiveHamiltonian')
except ImportError:
    pass

