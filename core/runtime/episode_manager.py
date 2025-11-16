"""
Episode Manager: Run Episodes, Not Just Functions

Manages episodes with initialization, execution, and termination.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass

from .orchestrator import Orchestrator
from .temporal_engine import Timestep
from ..classical.livnium_core_system import LivniumCoreSystem


@dataclass
class Episode:
    """An episode of system execution."""
    episode_id: int
    initial_state: Dict[str, Any]
    final_state: Optional[Dict[str, Any]] = None
    timesteps: int = 0
    terminated: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = None


class EpisodeManager:
    """
    Manages episodes of system execution.
    
    Features:
    - Episode initialization
    - Episode execution
    - Episode termination
    - Episode history
    """
    
    def __init__(self, orchestrator: Orchestrator):
        """
        Initialize episode manager.
        
        Args:
            orchestrator: Orchestrator instance
        """
        self.orchestrator = orchestrator
        self.current_episode: Optional[Episode] = None
        self.episode_history: List[Episode] = []
        self.episode_counter = 0
    
    def start_episode(self, initial_state: Optional[Dict[str, Any]] = None) -> Episode:
        """
        Start a new episode.
        
        Args:
            initial_state: Optional initial state
            
        Returns:
            Started episode
        """
        self.episode_counter += 1
        
        if initial_state is None:
            initial_state = self._capture_system_state()
        
        self.current_episode = Episode(
            episode_id=self.episode_counter,
            initial_state=initial_state,
            metadata={}
        )
        
        return self.current_episode
    
    def run_episode(self, max_timesteps: int = 100,
                   termination_condition: Optional[Callable[[], bool]] = None) -> Episode:
        """
        Run episode until termination.
        
        Args:
            max_timesteps: Maximum timesteps
            termination_condition: Optional termination condition
            
        Returns:
            Completed episode
        """
        if not self.current_episode:
            self.start_episode()
        
        for timestep in range(max_timesteps):
            # Execute timestep
            result = self.orchestrator.step()
            self.current_episode.timesteps += 1
            
            # Check termination
            if termination_condition and termination_condition():
                self.current_episode.terminated = True
                break
        
        # Capture final state
        self.current_episode.final_state = self._capture_system_state()
        
        # Calculate reward (simplified)
        self.current_episode.reward = self._calculate_reward()
        
        # Store episode
        self.episode_history.append(self.current_episode)
        
        return self.current_episode
    
    def end_episode(self):
        """End current episode."""
        if self.current_episode:
            self.current_episode.final_state = self._capture_system_state()
            self.current_episode.terminated = True
            self.episode_history.append(self.current_episode)
            self.current_episode = None
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state."""
        return {
            'total_sw': self.orchestrator.core_system.get_total_symbolic_weight(),
            'class_counts': {
                cls.name: count 
                for cls, count in self.orchestrator.core_system.get_class_counts().items()
            },
            'timestep': self.orchestrator.temporal_engine.current_timestep,
        }
    
    def _calculate_reward(self) -> float:
        """Calculate episode reward."""
        if not self.current_episode:
            return 0.0
        
        # Simplified reward: based on system stability
        if self.current_episode.final_state:
            # Reward for maintaining invariants
            expected_sw = self.orchestrator.core_system.get_expected_total_sw()
            actual_sw = self.current_episode.final_state.get('total_sw', 0)
            sw_error = abs(actual_sw - expected_sw) / expected_sw if expected_sw > 0 else 1.0
            reward = 1.0 - sw_error
            return float(np.clip(reward, 0.0, 1.0))
        
        return 0.0
    
    def get_episode_statistics(self) -> Dict:
        """Get episode statistics."""
        if not self.episode_history:
            return {'total_episodes': 0}
        
        avg_timesteps = np.mean([e.timesteps for e in self.episode_history])
        avg_reward = np.mean([e.reward for e in self.episode_history])
        
        return {
            'total_episodes': len(self.episode_history),
            'current_episode': self.current_episode.episode_id if self.current_episode else None,
            'average_timesteps': float(avg_timesteps),
            'average_reward': float(avg_reward),
        }

