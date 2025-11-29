"""
Livnium NLI v5: Clean & Simplified Architecture

A streamlined, maintainable Natural Language Inference system that combines
the best ideas from previous versions with a cleaner structure.

Key improvements:
- Simplified 5-layer architecture (streamlined from v4's 7 layers)
- Fixed decision layer that properly predicts all 3 classes
- Clean separation of concerns
- Better code organization
- Comprehensive documentation
"""

from .encoder import ChainEncoder, ChainEncodedPair
from .classifier import LivniumV5Classifier, ClassificationResult
from .geometry_teacher import GeometryTeacher, GeometryLabel
from .fracture_detector import apply_structural_pressure, detect_negation_fracture
from .fracture_dynamics import FractureDynamics, AlignmentFracture
from .semantic_warp import SemanticWarp, WarpAlignment

__all__ = [
    'ChainEncoder',
    'ChainEncodedPair',
    'LivniumV5Classifier',
    'ClassificationResult',
    'GeometryTeacher',
    'GeometryLabel',
    'apply_structural_pressure',
    'detect_negation_fracture',
    'FractureDynamics',
    'AlignmentFracture',
    'SemanticWarp',
    'WarpAlignment',
]

