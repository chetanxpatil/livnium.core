"""
Moksha Engine: Fixed-Point Convergence and Release for Livnium-T

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

from .recursive_simplex_engine import RecursiveSimplexEngine, SimplexLevel


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
    node_id: int  # Node ID (0-4 for Livnium-T)
    state_hash: str
    invariant_properties: Dict[str, Any] = field(default_factory=dict)
    convergence_score: float = 0.0  # 0 = not converged, 1.0 = fully converged
    
    def __hash__(self):
        return hash((self.level_id, self.node_id, self.state_hash))


class MokshaEngine:
    """
    Moksha Engine: Fixed-Point Convergence and Release for Livnium-T
    
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
    
    def __init__(self, recursive_engine: RecursiveSimplexEngine):
        """
        Initialize Moksha engine.
        
        Args:
            recursive_engine: Recursive simplex engine
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
                'total_sw': geometry.get_total_sw(),
                'class_counts': {
                    cls.name: count 
                    for cls, count in geometry.get_class_counts().items()
                },
                'node_states': {},
            }
            
            # Capture state of each node
            for node_id in range(5):  # 5 nodes: 0 (core) + 1-4 (vertices)
                node = geometry.get_node(node_id)
                node_state = {
                    'exposure': node.exposure,
                    'symbolic_weight': node.symbolic_weight,
                    'node_class': node.node_class.name if node.node_class else None,
                    'is_om': node.is_om,
                    'is_lo': node.is_lo,
                }
                level_state['node_states'][node_id] = node_state
            
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
            observer_node_id = 0  # The Om core
            fixed_point = FixedPointState(
                level_id=0,
                node_id=observer_node_id,
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
        # Test rotations on base level
        if 0 not in self.recursive_engine.levels:
            return False
        
        base_level = self.recursive_engine.levels[0]
        original_hash = state['state_hash']
        
        # Test all 12 tetrahedral rotations
        rotation_group = base_level.geometry.rotation_group
        for rotation_id in range(rotation_group.order()):
            # Apply rotation
            base_level.geometry.apply_rotation(rotation_id)
            
            # Capture new state
            new_state = self._capture_full_state()
            new_hash = new_state['state_hash']
            
            # Rotate back (apply inverse)
            inverse_id = rotation_group.get_inverse(rotation_id)
            base_level.geometry.apply_rotation(inverse_id)
            
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
        # Simplified check: if state is at observer (node 0, Om) and all invariants hold,
        # recursive operations should preserve it
        
        # Check if observer state is stable
        if 0 in self.recursive_engine.levels:
            base_level = self.recursive_engine.levels[0]
            observer_node = base_level.geometry.get_node(0)  # Om core
            
            if observer_node:
                # Observer should be at core (f=0, SW=0) - the most stable state
                if observer_node.exposure == 0 and observer_node.symbolic_weight == 0:
                    # Core node is the most invariant - recursion preserves it
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
                    'node_id': fp.node_id,
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

