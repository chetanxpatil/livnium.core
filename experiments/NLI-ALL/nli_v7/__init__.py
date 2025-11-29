"""
Livnium NLI v7: Geometry Shaping

Key principle: Train ONLY geometry (Layers 0-3), never Layer 4.
"""

from .classifier import LivniumV7Classifier
from .encoder import ChainEncoder, ChainEncodedPair
from .layers import (
    Layer0Resonance,
    Layer1Curvature,
    Layer2Opposition,
    Layer3Attraction,
    Layer4Decision
)

__all__ = [
    'LivniumV7Classifier',
    'ChainEncoder',
    'ChainEncodedPair',
    'Layer0Resonance',
    'Layer1Curvature',
    'Layer2Opposition',
    'Layer3Attraction',
    'Layer4Decision',
]

