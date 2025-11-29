"""
Livnium NLI v6: Corrected 3-Axis Manifold

Uses ONLY invariant signals discovered through reverse physics.
"""

from .classifier import LivniumV6Classifier
from .encoder import ChainEncoder, ChainEncodedPair
from .layers import (
    Layer0Resonance,
    Layer1Curvature,
    Layer2Opposition,
    Layer3Attraction,
    Layer4Decision
)

__all__ = [
    'LivniumV6Classifier',
    'ChainEncoder',
    'ChainEncodedPair',
    'Layer0Resonance',
    'Layer1Curvature',
    'Layer2Opposition',
    'Layer3Attraction',
    'Layer4Decision',
]

