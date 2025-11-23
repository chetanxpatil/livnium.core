"""
Simple NLI System: Pure Geometric / Vector-Based Classification

This is a clean, simplified NLI system that:
- Uses only vectors and cosine similarity
- Learns word polarities from data
- Has zero dependency on Livnium physics (core/, omcubes, basins, collapse)

Livnium physics-based NLI lives separately in experiments/nli/
"""

from .native_chain import SimpleLexicon, WordVector, SentenceVector
from .inference_detectors import SimpleNLIClassifier
from .encoder import SimpleEncoder

__all__ = [
    'SimpleLexicon',
    'WordVector',
    'SentenceVector',
    'SimpleNLIClassifier',
    'SimpleEncoder',
]

