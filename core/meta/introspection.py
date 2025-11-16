"""
Introspection Engine: Deep Self-Examination

Provides deep introspection capabilities for the system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .meta_observer import MetaObserver
from .anomaly_detector import AnomalyDetector
from ..classical.livnium_core_system import LivniumCoreSystem


class IntrospectionEngine:
    """
    Introspection engine for deep self-examination.
    
    Provides:
    - State analysis
    - Pattern recognition
    - Behavior prediction
    - System health assessment
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize introspection engine.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
        self.meta_observer = MetaObserver(core_system)
        self.anomaly_detector = AnomalyDetector(core_system)
    
    def introspect(self) -> Dict[str, Any]:
        """
        Perform full introspection.
        
        Returns:
            Complete introspection report
        """
        # Capture state
        state = self.meta_observer.capture_state()
        
        # Detect drift
        drift = self.meta_observer.detect_invariance_drift()
        
        # Reflect on behavior
        reflection = self.meta_observer.reflect_on_behavior()
        
        # Check alignment
        alignment = self.meta_observer.check_self_alignment()
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_all_anomalies()
        
        # Analyze patterns
        patterns = self._analyze_patterns()
        
        return {
            'state': {
                'total_sw': state.total_sw,
                'class_counts': state.class_counts,
                'rotation_count': state.rotation_count,
            },
            'drift': drift,
            'reflection': reflection,
            'alignment': alignment,
            'anomalies': {
                'count': len(anomalies),
                'types': [a.type for a in anomalies],
            },
            'patterns': patterns,
            'health_score': self._calculate_health_score(drift, alignment, anomalies),
        }
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in system behavior."""
        if len(self.meta_observer.state_history) < 3:
            return {'analysis': 'insufficient_data'}
        
        # Analyze SW trends
        sw_values = [s.total_sw for s in self.meta_observer.state_history]
        sw_trend = 'stable'
        if len(sw_values) >= 2:
            if sw_values[-1] > sw_values[0] * 1.01:
                sw_trend = 'increasing'
            elif sw_values[-1] < sw_values[0] * 0.99:
                sw_trend = 'decreasing'
        
        # Analyze rotation frequency
        rotation_counts = [s.rotation_count for s in self.meta_observer.state_history]
        rotation_rate = 0.0
        if len(rotation_counts) >= 2:
            time_span = self.meta_observer.state_history[-1].timestamp - self.meta_observer.state_history[0].timestamp
            if time_span > 0:
                rotation_rate = (rotation_counts[-1] - rotation_counts[0]) / time_span
        
        return {
            'sw_trend': sw_trend,
            'rotation_rate': float(rotation_rate),
            'state_history_size': len(self.meta_observer.state_history),
        }
    
    def _calculate_health_score(self, drift: Dict, alignment: Dict, anomalies: List) -> float:
        """Calculate overall system health score [0, 1]."""
        score = 1.0
        
        # Penalize drift
        if drift.get('drift_detected', False):
            score -= 0.3
        
        # Penalize misalignment
        if not alignment.get('aligned', True):
            score -= 0.3
        
        # Penalize anomalies
        if anomalies:
            score -= min(0.4, len(anomalies) * 0.1)
        
        return max(0.0, score)
    
    def predict_behavior(self, steps: int = 10) -> Dict[str, Any]:
        """
        Predict future behavior based on current patterns.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Prediction results
        """
        if len(self.meta_observer.state_history) < 2:
            return {'prediction': 'insufficient_data'}
        
        # Simple linear prediction
        sw_values = [s.total_sw for s in self.meta_observer.state_history[-5:]]
        if len(sw_values) >= 2:
            sw_trend = (sw_values[-1] - sw_values[0]) / len(sw_values)
            predicted_sw = sw_values[-1] + sw_trend * steps
        else:
            predicted_sw = sw_values[-1] if sw_values else 0.0
        
        return {
            'predicted_sw': float(predicted_sw),
            'prediction_steps': steps,
            'confidence': 0.7,  # Simplified confidence
        }

