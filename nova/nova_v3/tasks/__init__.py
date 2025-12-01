"""
Task Heads

Task-specific heads that attach to the core physics engine.
"""

from .snli.head_snli import SNLIHead
from .snli.encoding_snli import SNLIEncoder, GeometricSNLIEncoder, SanskritSNLIEncoder

__all__ = ['SNLIHead', 'SNLIEncoder', 'GeometricSNLIEncoder', 'SanskritSNLIEncoder']
