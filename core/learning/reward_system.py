import time
import json
import os
from typing import Dict, Any, Optional


class RewardSystem:
    """
    Reinforcement Learning Reward System for Livnium.
    
    Tracks agent actions and assigns rewards/penalties to guide learning.
    Updated [2025-11-11] to increase exploration cost and logging.
    """
    
    def __init__(self, log_file: str = "logs/rewards.log"):
        self.log_file = log_file
        self.total_reward = 0.0
        self.history = []
        
        # Hyperparameters (Tuned based on 2025-11-11 Analysis)
        self.exploration_cost = 0.05  # Penalty for random moves (Increased to reduce noise)
        self.alignment_reward = 2.0   # Reward for tension drop (Increased strongly)
        self.stability_reward = 0.1   # Reward for staying in a good basin
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def calculate_reward(self, 
                        action_type: str, 
                        old_tension: float, 
                        new_tension: float,
                        metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate reward for an action.
        
        Args:
            action_type: "mutate", "crossover", "explore", "stay"
            old_tension: Tension before action
            new_tension: Tension after action
            metadata: Extra info
        """
        reward = 0.0
        
        # 1. Tension Drop (Alignment) - The most important signal
        tension_drop = old_tension - new_tension
        if tension_drop > 0:
            # Super-linear reward for large drops
            reward += self.alignment_reward * (tension_drop * 10.0)
            if tension_drop > 0.1:
                reward += 5.0 # Bonus for massive breakthrough
        
        # 2. Exploration Cost
        if action_type in ["mutate", "explore"]:
            reward -= self.exploration_cost
            
        # 3. Stability (if maintaining low tension)
        if action_type == "stay" and new_tension < 0.1:
            reward += self.stability_reward * (1.0 - new_tension)
            
        # Update state
        self.total_reward += reward
        self.log_reward(action_type, reward, old_tension, new_tension, metadata)
        
        return reward

    def log_reward(self, action, reward, old_t, new_t, meta):
        """Log reward event to file."""
        event = {
            "timestamp": time.time(),
            "action": action,
            "reward": round(reward, 4),
            "tension_change": round(old_t - new_t, 4),
            "current_tension": round(new_t, 4),
            "metadata": meta or {}
        }
        self.history.append(event)
        
        # Write to file (append mode)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
            
    def get_stats(self):
        return {
            "total_reward": self.total_reward,
            "events_count": len(self.history)
        }

