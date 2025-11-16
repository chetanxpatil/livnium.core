"""
Calibration Engine: Adaptive Calibration and Self-Repair

Automatically calibrates and repairs system to maintain invariants.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .anomaly_detector import AnomalyDetector
from .meta_observer import MetaObserver
from ..classical.livnium_core_system import LivniumCoreSystem


class CalibrationEngine:
    """
    Calibration engine that maintains system health.
    
    Features:
    - Automatic calibration
    - Self-repair
    - Adaptive parameter adjustment
    - Invariant restoration
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize calibration engine.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
        self.anomaly_detector = AnomalyDetector(core_system)
        self.meta_observer = MetaObserver(core_system)
        self.calibration_history: List[Dict] = []
    
    def calibrate(self) -> Dict[str, Any]:
        """
        Perform calibration to restore invariants.
        
        Returns:
            Calibration results
        """
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_all_anomalies()
        
        # Check drift
        drift = self.meta_observer.detect_invariance_drift()
        
        calibration_actions = []
        
        # Fix SW if drifted
        if drift.get('drift_detected', False):
            current_sw = self.core_system.get_total_symbolic_weight()
            expected_sw = self.core_system.get_expected_total_sw()
            
            # Recalculate SW for all cells (should restore automatically)
            for coords, cell in self.core_system.lattice.items():
                # SW should be 9 * f, recalculate if wrong
                expected_cell_sw = 9.0 * cell.face_exposure
                if abs(cell.symbolic_weight - expected_cell_sw) > 0.1:
                    cell.symbolic_weight = expected_cell_sw
                    calibration_actions.append(f"Fixed SW at {coords}")
        
        # Fix class counts if drifted
        if drift.get('max_count_drift', 0) > 0:
            # Recalculate class counts (should be automatic, but verify)
            current_counts = self.core_system.get_class_counts()
            expected_counts = self.core_system.get_expected_class_counts()
            
            for cls, expected_count in expected_counts.items():
                current_count = current_counts.get(cls, 0)
                if current_count != expected_count:
                    calibration_actions.append(f"Class count mismatch: {cls.name}")
                    # Note: Class counts are structural, can't be fixed by recalculation
                    # This indicates a deeper issue
    
    def auto_repair(self) -> Dict[str, Any]:
        """
        Automatically repair detected issues.
        
        Returns:
            Repair results
        """
        anomalies = self.anomaly_detector.detect_all_anomalies()
        repairs = []
        
        for anomaly in anomalies:
            if anomaly.type == 'sw_mismatch' and anomaly.location:
                # Repair SW mismatch
                coords = anomaly.location
                cell = self.core_system.get_cell(coords)
                if cell:
                    cell.symbolic_weight = 9.0 * cell.face_exposure
                    repairs.append(f"Repaired SW at {coords}")
        
        return {
            'anomalies_found': len(anomalies),
            'repairs_performed': len(repairs),
            'repair_details': repairs,
        }
    
    def adaptive_calibration(self) -> Dict[str, Any]:
        """
        Perform adaptive calibration based on system behavior.
        
        Returns:
            Calibration results
        """
        # Reflect on behavior
        reflection = self.meta_observer.reflect_on_behavior()
        
        # Check alignment
        alignment = self.meta_observer.check_self_alignment()
        
        # Perform calibration if needed
        if not alignment['aligned']:
            self.calibrate()
        
        result = {
            'reflection': reflection,
            'alignment': alignment,
            'calibrated': not alignment['aligned'],
        }
        
        self.calibration_history.append(result)
        return result
    
    def get_calibration_statistics(self) -> Dict:
        """Get calibration statistics."""
        return {
            'total_calibrations': len(self.calibration_history),
            'anomaly_statistics': self.anomaly_detector.get_anomaly_statistics(),
            'meta_statistics': self.meta_observer.get_meta_statistics(),
        }

