"""
Livnium Core System

Complete implementation of the Livnium Core System specification:
- N×N×N spatial lattice with Σ(N) symbols
- Symbolic Weight (SW = 9·f)
- Face exposure classification
- 90° rotation group
- Observer system
- Semantic polarity
- Invariants conservation
- Quantum layer (optional)
"""

# Core configuration
from .config import LivniumCoreConfig

# Classical system
from .classical import (
    LivniumCoreSystem,
    LatticeCell,
    Observer,
    RotationAxis,
    CellClass,
    RotationGroup,
)

# All layers (optional imports)
__all__ = [
    # Configuration
    'LivniumCoreConfig',
    # Classical
    'LivniumCoreSystem',
    'LatticeCell',
    'Observer',
    'RotationAxis',
    'CellClass',
    'RotationGroup',
]

# Quantum layer
try:
    from .quantum import (
        QuantumLattice, QuantumCell, QuantumGates, GateType,
        EntanglementManager, EntangledPair,
        MeasurementEngine, MeasurementResult,
        GeometryQuantumCoupling,
    )
    __all__.extend([
        'QuantumLattice', 'QuantumCell', 'QuantumGates', 'GateType',
        'EntanglementManager', 'EntangledPair',
        'MeasurementEngine', 'MeasurementResult',
        'GeometryQuantumCoupling',
    ])
except ImportError:
    pass

# Memory layer
try:
    from .memory import MemoryLattice, MemoryCell, MemoryCoupling
    __all__.extend(['MemoryLattice', 'MemoryCell', 'MemoryCoupling'])
except ImportError:
    pass

# Reasoning layer
try:
    from .reasoning import (
        ReasoningEngine, ProblemSolver,
        SearchEngine, SearchNode, SearchStrategy,
        RuleEngine, Rule, RuleSet,
    )
    __all__.extend([
        'ReasoningEngine', 'ProblemSolver',
        'SearchEngine', 'SearchNode', 'SearchStrategy',
        'RuleEngine', 'Rule', 'RuleSet',
    ])
except ImportError:
    pass

# Semantic layer
try:
    from .semantic import (
        SemanticProcessor, FeatureExtractor,
        MeaningGraph, SemanticNode,
        InferenceEngine,
    )
    __all__.extend([
        'SemanticProcessor', 'FeatureExtractor',
        'MeaningGraph', 'SemanticNode',
        'InferenceEngine',
    ])
except ImportError:
    pass

# Meta layer
try:
    from .meta import (
        MetaObserver, AnomalyDetector,
        CalibrationEngine, IntrospectionEngine,
    )
    __all__.extend([
        'MetaObserver', 'AnomalyDetector',
        'CalibrationEngine', 'IntrospectionEngine',
    ])
except ImportError:
    pass

# Runtime layer
try:
    from .runtime import (
        Orchestrator, TemporalEngine, Timestep,
        EpisodeManager, Episode,
    )
    __all__.extend([
        'Orchestrator', 'TemporalEngine', 'Timestep',
        'EpisodeManager', 'Episode',
    ])
except ImportError:
    pass

# Recursive Geometry layer (Layer 0)
try:
    from .recursive import (
        RecursiveGeometryEngine, GeometryLevel,
        GeometrySubdivision,
        RecursiveProjection,
        RecursiveConservation,
        MokshaEngine, FixedPointState, ConvergenceState,
    )
    __all__.extend([
        'RecursiveGeometryEngine', 'GeometryLevel',
        'GeometrySubdivision',
        'RecursiveProjection',
        'RecursiveConservation',
        'MokshaEngine', 'FixedPointState', 'ConvergenceState',
    ])
except ImportError:
    pass

