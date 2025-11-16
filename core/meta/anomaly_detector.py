"""
Anomaly Detector: Detect Anomalies in System Behavior

Detects unusual patterns, violations, and anomalies.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..classical.livnium_core_system import LivniumCoreSystem


@dataclass
class Anomaly:
    """Anomaly detection result."""
    type: str
    severity: float  # 0-1
    description: str
    location: Optional[Tuple[int, int, int]] = None
    timestamp: float = 0.0


class AnomalyDetector:
    """
    Detects anomalies in Livnium Core System.
    
    Monitors for:
    - Invariant violations
    - Unusual state patterns
    - Unexpected transitions
    - Out-of-bounds values
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize anomaly detector.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
        self.detected_anomalies: List[Anomaly] = []
        self.baseline_stats: Optional[Dict] = None
    
    def detect_all_anomalies(self) -> List[Anomaly]:
        """
        Detect all anomalies in system.
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check invariants
        anomalies.extend(self._check_invariants())
        
        # Check state bounds
        anomalies.extend(self._check_state_bounds())
        
        # Check for unusual patterns
        anomalies.extend(self._check_unusual_patterns())
        
        self.detected_anomalies.extend(anomalies)
        return anomalies
    
    def _check_invariants(self) -> List[Anomaly]:
        """Check for invariant violations."""
        anomalies = []
        
        # Check total SW
        current_sw = self.core_system.get_total_symbolic_weight()
        expected_sw = self.core_system.get_expected_total_sw()
        sw_diff = abs(current_sw - expected_sw)
        
        if sw_diff > 1e-6:
            anomalies.append(Anomaly(
                type='invariant_violation',
                severity=min(1.0, sw_diff / expected_sw),
                description=f"Total SW mismatch: {current_sw} vs {expected_sw}",
                timestamp=__import__('time').time()
            ))
        
        # Check class counts
        current_counts = self.core_system.get_class_counts()
        expected_counts = self.core_system.get_expected_class_counts()
        
        for cls, expected_count in expected_counts.items():
            current_count = current_counts.get(cls, 0)
            if current_count != expected_count:
                anomalies.append(Anomaly(
                    type='class_count_violation',
                    severity=1.0,
                    description=f"{cls.name} count mismatch: {current_count} vs {expected_count}",
                    timestamp=__import__('time').time()
                ))
        
        return anomalies
    
    def _check_state_bounds(self) -> List[Anomaly]:
        """Check for out-of-bounds values."""
        anomalies = []
        
        for coords, cell in self.core_system.lattice.items():
            # Check face exposure bounds
            if cell.face_exposure < 0 or cell.face_exposure > 3:
                anomalies.append(Anomaly(
                    type='out_of_bounds',
                    severity=1.0,
                    description=f"Invalid face exposure: {cell.face_exposure} at {coords}",
                    location=coords,
                    timestamp=__import__('time').time()
                ))
            
            # Check SW bounds
            expected_sw = 9.0 * cell.face_exposure
            if abs(cell.symbolic_weight - expected_sw) > 0.1:
                anomalies.append(Anomaly(
                    type='sw_mismatch',
                    severity=0.5,
                    description=f"SW mismatch: {cell.symbolic_weight} vs {expected_sw} at {coords}",
                    location=coords,
                    timestamp=__import__('time').time()
                ))
        
        return anomalies
    
    def _check_unusual_patterns(self) -> List[Anomaly]:
        """Check for unusual patterns."""
        anomalies = []
        
        # Check for cells with same coordinates but different properties
        coord_to_cells = {}
        for coords, cell in self.core_system.lattice.items():
            if coords in coord_to_cells:
                anomalies.append(Anomaly(
                    type='duplicate_coordinates',
                    severity=1.0,
                    description=f"Duplicate coordinates: {coords}",
                    location=coords,
                    timestamp=__import__('time').time()
                ))
            coord_to_cells[coords] = cell
        
        return anomalies
    
    def get_anomaly_statistics(self) -> Dict:
        """Get anomaly detection statistics."""
        if not self.detected_anomalies:
            return {'total_anomalies': 0}
        
        type_counts = {}
        for anomaly in self.detected_anomalies:
            type_counts[anomaly.type] = type_counts.get(anomaly.type, 0) + 1
        
        avg_severity = np.mean([a.severity for a in self.detected_anomalies])
        
        return {
            'total_anomalies': len(self.detected_anomalies),
            'anomaly_types': type_counts,
            'average_severity': float(avg_severity),
        }

