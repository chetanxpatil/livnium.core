"""
Meta Layer: Self-Reflection and Calibration

Reflection, introspection, reasoning about own states, anomaly detection,
self-alignment, invariance drift detection, and adaptive calibration.
"""

from .meta_observer import MetaObserver
from .anomaly_detector import AnomalyDetector
from .calibration_engine import CalibrationEngine
from .introspection import IntrospectionEngine

__all__ = [
    'MetaObserver',
    'AnomalyDetector',
    'CalibrationEngine',
    'IntrospectionEngine',
]

