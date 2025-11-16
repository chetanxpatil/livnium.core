"""
Meta Observer: Self-Reflection and Introspection

Observes and reasons about the system's own states.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..classical.livnium_core_system import LivniumCoreSystem


@dataclass
class SystemState:
    """Snapshot of system state."""
    timestamp: float
    total_sw: float
    class_counts: Dict[str, int]
    rotation_count: int
    observer_count: int
    state_hash: str


class MetaObserver:
    """
    Meta observer that reflects on system's own states.
    
    Features:
    - State snapshots
    - Invariance drift detection
    - Self-alignment checking
    - Reflection on system behavior
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize meta observer.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
        self.state_history: List[SystemState] = []
        self.baseline_state: Optional[SystemState] = None
        self.reflection_log: List[Dict] = []
    
    def capture_state(self) -> SystemState:
        """
        Capture current system state snapshot.
        
        Returns:
            System state snapshot
        """
        total_sw = self.core_system.get_total_symbolic_weight()
        class_counts = {
            cls.name: count 
            for cls, count in self.core_system.get_class_counts().items()
        }
        
        state = SystemState(
            timestamp=__import__('time').time(),
            total_sw=total_sw,
            class_counts=class_counts,
            rotation_count=len(self.core_system.rotation_history),
            observer_count=len(self.core_system.local_observers) + (1 if self.core_system.global_observer else 0),
            state_hash=self._hash_state(total_sw, class_counts)
        )
        
        self.state_history.append(state)
        
        # Set baseline if first state
        if self.baseline_state is None:
            self.baseline_state = state
        
        return state
    
    def detect_invariance_drift(self) -> Dict[str, Any]:
        """
        Detect drift from invariants.
        
        Returns:
            Drift detection results
        """
        if not self.baseline_state:
            return {'drift_detected': False}
        
        current_state = self.capture_state()
        expected_sw = self.core_system.get_expected_total_sw()
        expected_counts = self.core_system.get_expected_class_counts()
        
        # Check SW drift
        sw_drift = abs(current_state.total_sw - expected_sw)
        sw_drift_fraction = sw_drift / expected_sw if expected_sw > 0 else 0.0
        
        # Check class count drift
        count_drifts = {}
        for cls, expected_count in expected_counts.items():
            current_count = current_state.class_counts.get(cls.name, 0)
            drift = abs(current_count - expected_count)
            count_drifts[cls.name] = {
                'expected': expected_count,
                'actual': current_count,
                'drift': drift
            }
        
        max_count_drift = max([d['drift'] for d in count_drifts.values()]) if count_drifts else 0
        
        drift_detected = sw_drift_fraction > 0.01 or max_count_drift > 0
        
        result = {
            'drift_detected': drift_detected,
            'sw_drift': sw_drift,
            'sw_drift_fraction': sw_drift_fraction,
            'count_drifts': count_drifts,
            'max_count_drift': max_count_drift,
        }
        
        if drift_detected:
            self.reflection_log.append({
                'type': 'invariance_drift',
                'result': result,
                'timestamp': current_state.timestamp
            })
        
        return result
    
    def reflect_on_behavior(self) -> Dict[str, Any]:
        """
        Reflect on system's behavior patterns.
        
        Returns:
            Reflection results
        """
        if len(self.state_history) < 2:
            return {'reflection': 'insufficient_history'}
        
        # Analyze state transitions
        transitions = []
        for i in range(1, len(self.state_history)):
            prev = self.state_history[i-1]
            curr = self.state_history[i]
            
            transitions.append({
                'sw_change': curr.total_sw - prev.total_sw,
                'rotation_count_change': curr.rotation_count - prev.rotation_count,
                'time_delta': curr.timestamp - prev.timestamp,
            })
        
        # Detect patterns
        sw_changes = [t['sw_change'] for t in transitions]
        avg_sw_change = np.mean(sw_changes) if sw_changes else 0.0
        
        # Check if system is stable
        is_stable = abs(avg_sw_change) < 0.1
        
        reflection = {
            'total_transitions': len(transitions),
            'average_sw_change': float(avg_sw_change),
            'is_stable': is_stable,
            'rotation_frequency': len(transitions) / (self.state_history[-1].timestamp - self.state_history[0].timestamp) if len(self.state_history) > 1 else 0.0,
        }
        
        self.reflection_log.append({
            'type': 'behavior_reflection',
            'reflection': reflection,
            'timestamp': self.state_history[-1].timestamp if self.state_history else __import__('time').time()
        })
        
        return reflection
    
    def check_self_alignment(self) -> Dict[str, Any]:
        """
        Check if system is aligned with its invariants.
        
        Returns:
            Alignment check results
        """
        current_state = self.capture_state()
        expected_sw = self.core_system.get_expected_total_sw()
        expected_counts = self.core_system.get_expected_class_counts()
        
        # Check SW alignment
        sw_aligned = abs(current_state.total_sw - expected_sw) < 1e-6
        
        # Check class count alignment
        counts_aligned = True
        for cls, expected_count in expected_counts.items():
            current_count = current_state.class_counts.get(cls.name, 0)
            if current_count != expected_count:
                counts_aligned = False
                break
        
        aligned = sw_aligned and counts_aligned
        
        return {
            'aligned': aligned,
            'sw_aligned': sw_aligned,
            'counts_aligned': counts_aligned,
            'current_sw': current_state.total_sw,
            'expected_sw': expected_sw,
        }
    
    def _hash_state(self, total_sw: float, class_counts: Dict[str, int]) -> str:
        """Hash state for comparison."""
        counts_str = ','.join(f"{k}:{v}" for k, v in sorted(class_counts.items()))
        return f"sw:{total_sw:.2f}|{counts_str}"
    
    def get_meta_statistics(self) -> Dict:
        """Get meta observer statistics."""
        return {
            'state_history_size': len(self.state_history),
            'reflection_log_size': len(self.reflection_log),
            'baseline_set': self.baseline_state is not None,
        }

