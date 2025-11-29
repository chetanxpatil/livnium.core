"""
Livnium NLI v8: Clean Architecture with Geometry-First Philosophy

Key Principles:
- Geometry is stable and invariant (the teacher, not the student)
- Collision-based fracture detection (negation = alignment tension)
- Semantic warp alignment (dynamic programming, no hardcoded rules)
- Pure physics + optimization

Architecture:
- Semantic Warp: Optimal alignment via DP
- Fracture Detection: Negation via collision analysis
- Geometry Layers: Resonance, Divergence, Curvature, Basins, Valley
- Decision: Final classification based on geometry zones
"""

from .core.encoder import ChainEncoder, ChainEncodedPair
from .core.classifier import LivniumV8Classifier, ClassificationResult
from .core.semantic_warp import SemanticWarp, WarpAlignment
from .core.fracture_dynamics import FractureDynamics, AlignmentFracture
from .core.geometry_teacher import GeometryTeacher, GeometryLabel

__all__ = [
    'ChainEncoder',
    'ChainEncodedPair',
    'LivniumV8Classifier',
    'ClassificationResult',
    'SemanticWarp',
    'WarpAlignment',
    'FractureDynamics',
    'AlignmentFracture',
    'GeometryTeacher',
    'GeometryLabel',
]

__version__ = '8.0.0'

