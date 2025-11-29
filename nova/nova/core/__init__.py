"""
Core Livnium Dialogue Components (Non-Recursive)

This module contains the core geometric processing pipeline:
- TextToGeometry: Text injection and collapse
- GeometricTokenLearner: Token hashing and clustering
- ClusterDecoder: Cluster-based text generation
- GeometricTransformer: Signature transformations
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    'TextToGeometry',
    'GeometricTokenLearner',
    'ClusterDecoder',
    'GeometricTransformer',
    'SentenceDecoder',
]

