"""
Law Extraction Module: Auto-Discovery of Physical Laws

This module extracts physical laws from Livnium Core by observing:
- Invariants (values that remain constant)
- Conserved quantities (sum/mean/energy)
- Functional relationships (f(x) = y)

This enables Livnium to discover its own laws instead of having them hardcoded.

Versions:
- v1: Basic linear relationships and invariants
- v2: Nonlinear function discovery
- v3: Symbolic regression
- v4: Law stability + confidence scoring
- v5: Multi-layer law fusion
- v6: Basin-based law extraction
"""

from .law_extractor import LivniumLawExtractor
from .advanced_law_extractor import AdvancedLawExtractor, DiscoveredLaw

__all__ = [
    'LivniumLawExtractor',
    'AdvancedLawExtractor',
    'DiscoveredLaw',
]

