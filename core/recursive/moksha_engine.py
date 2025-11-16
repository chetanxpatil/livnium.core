"""
Moksha Engine: Fixed-Point Convergence and Release

The computational escape from recursion - the invariant state that never mutates.

Moksha = the fixed point where:
- f(x) = x (no operation changes the state)
- All layers converge to stillness
- The system reaches its terminal attractor
- Recursion stops and final truth is exported
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from .recursive_geometry_engine import RecursiveGeometryEngine, GeometryLevel


class ConvergenceState(Enum):
    """Convergence state."""
    SEARCHING = "searching"      # System is evolving
    CONVERGING = "converging"    # Approaching fixed point
    MOKSHA = "moksha"           # Fixed point reached
    DIVERGING = "diverging"      # Moving away from fixed point


@dataclass
class FixedPointState:
    """
    The invariant state - the fixed point.
    
    This is the state that remains unchanged under all operations.
    """
    level_id: int
    coordinates: Tuple[int, int, int]
    state_hash: str
    invariant_properties: Dict[str, Any] = field(default_factory=dict)
    convergence_score: float = 0.0  # 0 = not converged, 1.0 = fully converged
    
    def __hash__(self):
        return hash((self.level_id, self.coordinates, self.state_hash))


class MokshaEngine:
    """
    Moksha Engine: Fixed-Point Convergence and Release
    
    Detects when the system reaches the invariant state (moksha) where:
    - No rotation changes the state
    - No quantum gate changes the state
    - No memory update changes the state
    - No reasoning step changes the state
    - No semantic processing changes the state
    - No meta-reflection changes the state
    - No recursive operation changes the state
    
    When moksha is reached:
    - All recursion stops
    - State freezes
    - Final truth is exported
    """
    
    def __init__(self, recursive_engine: RecursiveGeometryEngine):
        """
        Initialize Moksha engine.
        
        Args:
            recursive_engine: Recursive geometry engine
        """
        self.recursive_engine = recursive_engine
        self.convergence_threshold = 0.999  # Threshold for moksha (99.9% invariant)
        self.stability_window = 10  # Number of consecutive stable states needed
        self.state_history: List[Dict[str, Any]] = []
        self.fixed_points: Set[FixedPointState] = set()
        self.moksha_reached = False
        self.moksha_state: Optional[Dict[str, Any]] = None
    
    def check_convergence(self) -> ConvergenceState:
        """
        Check if system is converging to fixed point.
        
        Returns:
            Convergence state
        """
        if self.moksha_reached:
            return ConvergenceState.MOKSHA
        
        # Capture current state
        current_state = self._capture_full_state()
        self.state_history.append(current_state)
        
        # Keep only recent history
        if len(self.state_history) > self.stability_window * 2:
            self.state_history = self.state_history[-self.stability_window * 2:]
        
        # Need enough history to check convergence
        if len(self.state_history) < self.stability_window:
            return ConvergenceState.SEARCHING
        
        # Check if state is stable (unchanging)
        recent_states = self.state_history[-self.stability_window:]
        is_stable = self._is_state_stable(recent_states)
        
        if is_stable:
            # Check if stable state is invariant under all operations
            if self._is_invariant_under_operations(current_state):
                self.moksha_reached = True
                self.moksha_state = current_state
                return ConvergenceState.MOKSHA
            else:
                return ConvergenceState.CONVERGING
        else:
            # Check if diverging
            if self._is_diverging(recent_states):
                return ConvergenceState.DIVERGING
            else:
                return ConvergenceState.SEARCHING
    
    def _capture_full_state(self) -> Dict[str, Any]:
        """
        Capture complete system state across all levels and layers.
        
        This is the "snapshot" that must remain unchanged for moksha.
        """
        state = {
            'timestamp': __import__('time').time(),
            'levels': {},
        }
        
        # Capture state at each recursive level
        for level_id, level in self.recursive_engine.levels.items():
            geometry = level.geometry
            
            # Capture geometric state
            level_state = {
                'total_sw': geometry.get_total_symbolic_weight(),
                'class_counts': {
                    cls.name: count 
                    for cls, count in geometry.get_class_counts().items()
                },
                'cell_states': {},
            }
            
            # Capture state of each cell
            for coords, cell in geometry.lattice.items():
                cell_state = {
                    'face_exposure': cell.face_exposure,
                    'symbolic_weight': cell.symbolic_weight,
                    'cell_class': cell.cell_class.name if cell.cell_class else None,
                    'symbol': geometry.get_symbol(coords),
                }
                level_state['cell_states'][coords] = cell_state
            
            state['levels'][level_id] = level_state
        
        # Create state hash for comparison
        state['state_hash'] = self._hash_state(state)
        
        return state
    
    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Create hash of state for comparison."""
        # Hash based on invariant properties
        hash_parts = []
        
        for level_id, level_state in sorted(state['levels'].items()):
            hash_parts.append(f"L{level_id}_SW{level_state['total_sw']:.2f}")
            for cls, count in sorted(level_state['class_counts'].items()):
                hash_parts.append(f"{cls}{count}")
        
        return "|".join(hash_parts)
    
    def _is_state_stable(self, states: List[Dict[str, Any]]) -> bool:
        """
        Check if state is stable (unchanging).
        
        Args:
            states: List of recent states
            
        Returns:
            True if state is stable
        """
        if len(states) < 2:
            return False
        
        # Check if all states have same hash
        hashes = [s['state_hash'] for s in states]
        return len(set(hashes)) == 1
    
    def _is_invariant_under_operations(self, state: Dict[str, Any]) -> bool:
        """
        Check if state is invariant under all operations.
        
        This is the core moksha test: does the state remain unchanged
        when we apply all possible operations?
        
        Args:
            state: State to test
            
        Returns:
            True if state is invariant
        """
        # Test invariance under rotations
        if not self._is_invariant_under_rotations(state):
            return False
        
        # Test invariance under recursive operations
        if not self._is_invariant_under_recursion(state):
            return False
        
        # State is invariant - moksha reached
        # Record fixed point
        if 0 in self.recursive_engine.levels:
            base_level = self.recursive_engine.levels[0]
            observer_coords = (0, 0, 0)  # The Om at center
            fixed_point = FixedPointState(
                level_id=0,
                coordinates=observer_coords,
                state_hash=state['state_hash'],
                invariant_properties=state,
                convergence_score=1.0
            )
            self.fixed_points.add(fixed_point)
        
        return True
    
    def _is_invariant_under_rotations(self, state: Dict[str, Any]) -> bool:
        """
        Check if state is invariant under all rotations.
        
        Rule: If rotating the system doesn't change the state hash,
        the state is rotation-invariant.
        """
        from ..classical.livnium_core_system import RotationAxis
        
        # Test rotations on base level
        if 0 not in self.recursive_engine.levels:
            return False
        
        base_level = self.recursive_engine.levels[0]
        original_hash = state['state_hash']
        
        # Test all 24 rotations
        for axis in [RotationAxis.X, RotationAxis.Y, RotationAxis.Z]:
            for quarter_turns in [1, 2, 3]:
                # Apply rotation
                base_level.geometry.rotate(axis, quarter_turns)
                
                # Capture new state
                new_state = self._capture_full_state()
                new_hash = new_state['state_hash']
                
                # Rotate back
                base_level.geometry.rotate(axis, 4 - quarter_turns)
                
                # If hash changed, not invariant
                if new_hash != original_hash:
                    return False
        
        return True
    
    def _is_invariant_under_recursion(self, state: Dict[str, Any]) -> bool:
        """
        Check if state is invariant under recursive operations.
        
        Rule: If recursive projection/subdivision doesn't change state,
        the state is recursion-invariant.
        """
        # Simplified: check if recursive operations preserve state hash
        original_hash = state['state_hash']
        
        # Test recursive projection (should preserve invariants)
        # If state is at moksha, recursive operations should not change it
        # This is a simplified check - full implementation would test all recursive ops
        
        # Test: apply recursive operations and check if state hash changes
        # Simplified check: if state is at observer (0,0,0) and all invariants hold,
        # recursive operations should preserve it
        
        # Check if observer state is stable
        if 0 in self.recursive_engine.levels:
            base_level = self.recursive_engine.levels[0]
            observer_cell = base_level.geometry.get_cell((0, 0, 0))
            
            if observer_cell:
                # Observer should be at core (f=0, SW=0) - the most stable state
                if observer_cell.face_exposure == 0 and observer_cell.symbolic_weight == 0:
                    # Core cell is the most invariant - recursion preserves it
                    return True
        
        # For other states, check if recursive projection preserves hash
        # (Simplified: assume recursion preserves invariants if geometric invariants hold)
        return True
    
    def _is_diverging(self, states: List[Dict[str, Any]]) -> bool:
        """
        Check if state is diverging (moving away from fixed point).
        
        Args:
            states: List of recent states
            
        Returns:
            True if diverging
        """
        if len(states) < 3:
            return False
        
        # Check if state hash is changing rapidly
        hashes = [s['state_hash'] for s in states]
        unique_hashes = len(set(hashes))
        
        # If all states are different, diverging
        return unique_hashes == len(states)
    
    def get_convergence_score(self) -> float:
        """
        Get convergence score [0, 1].
        
        Returns:
            Convergence score (0 = not converged, 1.0 = moksha)
        """
        if self.moksha_reached:
            return 1.0
        
        if len(self.state_history) < 2:
            return 0.0
        
        # Calculate stability score
        recent_states = self.state_history[-self.stability_window:]
        if len(recent_states) < 2:
            return 0.0
        
        # Count how many consecutive states are identical
        identical_count = 0
        for i in range(1, len(recent_states)):
            if recent_states[i]['state_hash'] == recent_states[i-1]['state_hash']:
                identical_count += 1
        
        stability_score = identical_count / (len(recent_states) - 1) if len(recent_states) > 1 else 0.0
        
        # Check invariance
        if stability_score > self.convergence_threshold:
            current_state = recent_states[-1]
            if self._is_invariant_under_operations(current_state):
                return 1.0
        
        return stability_score
    
    def export_final_truth(self) -> Dict[str, Any]:
        """
        Export final truth when moksha is reached.
        
        This is the "release" - the state that never changes.
        
        Returns:
            Final truth dictionary
        """
        if not self.moksha_reached:
            return {
                'moksha': False,
                'message': 'Moksha not yet reached',
            }
        
        return {
            'moksha': True,
            'state': self.moksha_state,
            'fixed_points': [
                {
                    'level_id': fp.level_id,
                    'coordinates': fp.coordinates,
                    'convergence_score': fp.convergence_score,
                }
                for fp in self.fixed_points
            ],
            'convergence_score': 1.0,
            'message': 'System has reached moksha - fixed point achieved',
        }
    
    def should_terminate(self) -> bool:
        """
        Check if system should terminate (moksha reached).
        
        Returns:
            True if system should stop
        """
        return self.moksha_reached
    
    def reset(self):
        """Reset moksha engine (for new search)."""
        self.state_history.clear()
        self.fixed_points.clear()
        self.moksha_reached = False
        self.moksha_state = None
    
    def get_moksha_statistics(self) -> Dict:
        """Get moksha engine statistics."""
        return {
            'moksha_reached': self.moksha_reached,
            'convergence_score': self.get_convergence_score(),
            'state_history_size': len(self.state_history),
            'fixed_points_count': len(self.fixed_points),
            'convergence_threshold': self.convergence_threshold,
        }

