"""
Semantic Layer: Meaning, Language, and Inference

Feature extraction, semantic embeddings, symbol-to-meaning graph, negation detection,
context propagation, entailment/contradiction mechanics, and causal link detection.
"""

from .semantic_processor import SemanticProcessor
from .feature_extractor import FeatureExtractor
from .meaning_graph import MeaningGraph, SemanticNode
from .inference_engine import InferenceEngine

__all__ = [
    'SemanticProcessor',
    'FeatureExtractor',
    'MeaningGraph',
    'SemanticNode',
    'InferenceEngine',
]

