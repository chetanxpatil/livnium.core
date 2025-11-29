"""
Livnium NLI v4: Layered Core Architecture

Geological architecture - each layer builds on the one below.
Gravity shapes everything. No manual tuning.

Layers:
  0. Pure Resonance (raw E/C chain resonance)
  1. Curvature (second-derivative of resonance)
  2. Basin (curvature → attraction wells)
  3. Valley (natural neutral from curvature overlap)
  4. Meta Routing (reads geometry, no override)
  5. Temporal Stability (tracks stability)
  6. Semantic Memory (polarity shaped by lower layers)
  7. Decision Layer (reads all, doesn't control)
"""

from .layer0_resonance import Layer0Resonance
from .layer1_curvature import Layer1Curvature
from .layer2_basin import Layer2Basin
from .layer3_valley import Layer3Valley
from .layer4_meta_routing import Layer4MetaRouting
from .layer5_temporal_stability import Layer5TemporalStability
from .layer6_semantic_memory import Layer6SemanticMemory
from .layer7_decision import Layer7Decision
from .layered_classifier import LayeredLivniumClassifier
from .auto_physics import AutoPhysicsEngine

__all__ = [
    'Layer0Resonance',
    'Layer1Curvature',
    'Layer2Basin',
    'Layer3Valley',
    'Layer4MetaRouting',
    'Layer5TemporalStability',
    'Layer6SemanticMemory',
    'Layer7Decision',
    'LayeredLivniumClassifier',
    'AutoPhysicsEngine',
]

