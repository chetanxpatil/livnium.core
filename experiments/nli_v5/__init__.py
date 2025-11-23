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

__all__ = [
    'ChainEncoder',
    'ChainEncodedPair',
    'LivniumV5Classifier',
    'ClassificationResult',
]

