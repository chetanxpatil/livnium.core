"""
Livnium-T System

Stand-alone tetrahedral semantic engine independent of Livnium Core.

Complete implementation of the Livnium-T System specification:
- 5-node topology (1 core + 4 vertices)
- Two-class system (Core f=0, Vertex f=3)
- Symbolic Weight: SW = 9·f, ΣSW = 108
- Tetrahedral rotation group A₄ (12 elements)
- Perfect reversibility
- Quantum layer (optional)
"""

# Classical system
from .classical import (
    LivniumTSystem,
    SimplexNode,
    Observer,
    NodeClass,
    TetrahedralRotationGroup,
)

# Quantum layer (optional)
try:
    from .quantum import (
        QuantumSystem,
        QuantumNode,
        QuantumGates,
        GateType,
        EntanglementManager,
        EntangledPair,
        MeasurementEngine,
        MeasurementResult,
        MeasurementBasis,
        GeometryQuantumCoupling,
    )
    __all__ = [
        # Classical
        'LivniumTSystem',
        'SimplexNode',
        'Observer',
        'NodeClass',
        'TetrahedralRotationGroup',
        # Quantum
        'QuantumSystem',
        'QuantumNode',
        'QuantumGates',
        'GateType',
        'EntanglementManager',
        'EntangledPair',
        'MeasurementEngine',
        'MeasurementResult',
        'MeasurementBasis',
        'GeometryQuantumCoupling',
    ]
except ImportError:
    __all__ = [
        # Classical
        'LivniumTSystem',
        'SimplexNode',
        'Observer',
        'NodeClass',
        'TetrahedralRotationGroup',
    ]

