"""
Native Dynamic Basin Search: Geometry-Driven Self-Tuning Basin Reinforcement for Livnium-T

This is the core implementation of dynamic basin reinforcement that uses
geometry signals (curvature, tension, entropy) to self-tune basin shaping
parameters instead of static hyperparameters.

Adapted for Livnium-T's 5-node tetrahedral topology.

Key Principle: Geometry decides the physics, not a parameter list.

Instead of:
    alpha = 0.10  # static
    beta = 0.15   # static
    noise = 0.03  # static

We use:
    alpha = f(curvature)  # geometry-driven
    beta = g(tension)     # geometry-driven
    noise = h(entropy)    # geometry-driven

This creates a self-regulating system that adapts to the geometry itself.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import random

from ..classical.livnium_t_system import LivniumTSystem


def compute_local_curvature(
    system: LivniumTSystem,
    active_node_ids: List[int]
) -> float:
    """
    Compute local curvature: how deep the basin is becoming.
    
    Curvature = variance in symbolic weights around active nodes.
    Higher variance = deeper basin = stronger attractor.
    
    Args:
        system: LivniumTSystem
        active_node_ids: List of active node IDs (0-4)
        
    Returns:
        Curvature value (0.0 to 1.0+)
    """
    if not active_node_ids:
        return 0.0
    
    # Get SW values for active nodes
    sw_values = []
    for node_id in active_node_ids:
        node = system.get_node(node_id)
        if node:
            sw_values.append(node.symbolic_weight)
    
    if len(sw_values) < 2:
        return 0.0
    
    # Curvature = normalized variance
    sw_array = np.array(sw_values)
    mean_sw = np.mean(sw_array)
    if mean_sw == 0:
        return 0.0
    
    variance = np.var(sw_array)
    # Normalize: variance relative to mean
    curvature = variance / (mean_sw + 1.0)  # +1 to avoid division by zero
    
    # Scale to reasonable range (0 to ~2)
    curvature = min(curvature, 2.0)
    
    return float(curvature)


def compute_symbolic_tension(
    system: LivniumTSystem,
    active_node_ids: List[int]
) -> float:
    """
    Compute symbolic tension: internal contradictions in SW.
    
    Tension = how much SW values conflict with each other.
    Higher tension = more contradictions = need more decay.
    
    Args:
        system: LivniumTSystem
        active_node_ids: List of active node IDs (0-4)
        
    Returns:
        Tension value (0.0 to 1.0+)
    """
    if not active_node_ids:
        return 0.0
    
    # Get SW values
    sw_values = []
    for node_id in active_node_ids:
        node = system.get_node(node_id)
        if node:
            sw_values.append(node.symbolic_weight)
    
    if len(sw_values) < 2:
        return 0.0
    
    sw_array = np.array(sw_values)
    
    # Tension = how spread out the values are
    # If all values are similar, tension is low
    # If values are very different, tension is high
    range_sw = np.max(sw_array) - np.min(sw_array)
    mean_sw = np.mean(sw_array)
    
    if mean_sw == 0:
        return 0.0
    
    # Normalized tension: range relative to mean
    tension = range_sw / (mean_sw + 1.0)
    
    # Scale to reasonable range (0 to ~2)
    tension = min(tension, 2.0)
    
    return float(tension)


def compute_noise_entropy(
    system: LivniumTSystem,
    active_node_ids: List[int]
) -> float:
    """
    Compute noise entropy: how noisy/disordered the state is.
    
    Entropy = variance in SW values relative to expected values.
    Higher entropy = more disorder = need more decorrelation.
    
    Args:
        system: LivniumTSystem
        active_node_ids: List of active node IDs (0-4)
        
    Returns:
        Entropy value (0.0 to 1.0+)
    """
    if not active_node_ids:
        return 0.0
    
    # Get SW values
    sw_values = []
    for node_id in active_node_ids:
        node = system.get_node(node_id)
        if node:
            sw_values.append(node.symbolic_weight)
    
    if len(sw_values) < 2:
        return 0.0
    
    sw_array = np.array(sw_values)
    
    # Entropy = variance in SW values
    # High variance = high entropy = disordered state
    variance = np.var(sw_array)
    mean_sw = np.mean(sw_array)
    
    if mean_sw == 0:
        return 0.0
    
    # Normalized entropy: variance relative to mean
    entropy = variance / (mean_sw + 1.0)
    
    # Scale to reasonable range (0 to ~2)
    entropy = min(entropy, 2.0)
    
    return float(entropy)


def get_geometry_signals(
    system: LivniumTSystem,
    active_node_ids: List[int]
) -> Tuple[float, float, float]:
    """
    Get all geometry signals at once.
    
    Args:
        system: LivniumTSystem
        active_node_ids: List of active node IDs (0-4)
        
    Returns:
        (curvature, tension, entropy) tuple
    """
    curvature = compute_local_curvature(system, active_node_ids)
    tension = compute_symbolic_tension(system, active_node_ids)
    entropy = compute_noise_entropy(system, active_node_ids)
    
    return (curvature, tension, entropy)


def update_basin_dynamic(
    system: LivniumTSystem,
    task: Any,
    is_correct: bool,
    base_alpha: float = 0.10,
    base_beta: float = 0.15,
    base_noise: float = 0.03
) -> None:
    """
    Update basin using dynamic (geometry-driven) parameters.
    
    This is the core dynamic basin reinforcement function for Livnium-T.
    
    Instead of static parameters, it computes:
    - alpha = base_alpha * (1.0 + curvature)  # Curvature amplifies reinforcement
    - beta = base_beta * (1.0 + tension)     # Tension amplifies decay
    - noise = base_noise * (1.0 + entropy)   # Entropy amplifies decorrelation
    
    Args:
        system: LivniumTSystem
        task: Task object (must have input_node_ids and output_node_id attributes)
        is_correct: Whether the task was solved correctly
        base_alpha: Base reinforcement strength (default 0.10)
        base_beta: Base decay strength (default 0.15)
        base_noise: Base decorrelation strength (default 0.03)
    """
    # Get active node IDs (input + output nodes)
    active_node_ids = []
    if hasattr(task, 'input_node_ids'):
        active_node_ids.extend(task.input_node_ids)
    if hasattr(task, 'output_node_id') and task.output_node_id is not None:
        active_node_ids.append(task.output_node_id)
    
    if not active_node_ids:
        return
    
    # Compute geometry signals
    curvature, tension, entropy = get_geometry_signals(system, active_node_ids)
    
    # Dynamic parameters: geometry-driven
    alpha = base_alpha * (1.0 + curvature)  # Curvature amplifies reinforcement
    beta = base_beta * (1.0 + tension)     # Tension amplifies decay
    noise = base_noise * (1.0 + entropy)   # Entropy amplifies decorrelation
    
    if is_correct:
        # Strengthen the attractor: deepen the well
        for node_id in active_node_ids:
            node = system.get_node(node_id)
            if node:
                # Deepen well: increase SW (stronger attractor)
                node.symbolic_weight += alpha
                # Clamp to reasonable bounds (max SW for vertex is 27, but allow some flexibility)
                node.symbolic_weight = min(node.symbolic_weight, 50.0)
    else:
        # Weaken the attractor: fill the well
        for node_id in active_node_ids:
            node = system.get_node(node_id)
            if node:
                # Fill well: decrease SW (weaker attractor)
                node.symbolic_weight -= beta
                # Clamp to minimum (0 for core, but vertices start at 27)
                if node.node_class.name == 'CORE':
                    node.symbolic_weight = max(node.symbolic_weight, 0.0)
                else:  # VERTEX
                    node.symbolic_weight = max(node.symbolic_weight, 0.0)
        
        # Add decorrelation noise
        if noise > 0:
            for node_id in active_node_ids:
                node = system.get_node(node_id)
                if node:
                    # Add small random perturbation
                    noise_value = random.uniform(-noise, noise)
                    node.symbolic_weight += noise_value
                    # Clamp again
                    node.symbolic_weight = max(node.symbolic_weight, 0.0)


def apply_dynamic_basin(
    system: LivniumTSystem,
    active_node_ids: List[int],
    is_correct: bool,
    base_alpha: float = 0.10,
    base_beta: float = 0.15,
    base_noise: float = 0.03
) -> Dict[str, float]:
    """
    Apply dynamic basin reinforcement directly.
    
    Args:
        system: LivniumTSystem
        active_node_ids: List of active node IDs (0-4)
        is_correct: Whether the solution is correct
        base_alpha: Base reinforcement strength
        base_beta: Base decay strength
        base_noise: Base decorrelation strength
        
    Returns:
        Dictionary with geometry signals and applied parameters
    """
    if not active_node_ids:
        return {
            'curvature': 0.0,
            'tension': 0.0,
            'entropy': 0.0,
            'alpha': base_alpha,
            'beta': base_beta,
            'noise': base_noise,
        }
    
    # Compute geometry signals
    curvature, tension, entropy = get_geometry_signals(system, active_node_ids)
    
    # Dynamic parameters
    alpha = base_alpha * (1.0 + curvature)
    beta = base_beta * (1.0 + tension)
    noise = base_noise * (1.0 + entropy)
    
    # Apply basin update
    if is_correct:
        for node_id in active_node_ids:
            node = system.get_node(node_id)
            if node:
                node.symbolic_weight += alpha
                node.symbolic_weight = min(node.symbolic_weight, 50.0)
    else:
        for node_id in active_node_ids:
            node = system.get_node(node_id)
            if node:
                node.symbolic_weight -= beta
                node.symbolic_weight = max(node.symbolic_weight, 0.0)
        
        if noise > 0:
            for node_id in active_node_ids:
                node = system.get_node(node_id)
                if node:
                    noise_value = random.uniform(-noise, noise)
                    node.symbolic_weight += noise_value
                    node.symbolic_weight = max(node.symbolic_weight, 0.0)
    
    return {
        'curvature': curvature,
        'tension': tension,
        'entropy': entropy,
        'alpha': alpha,
        'beta': beta,
        'noise': noise,
    }


class DynamicBasinSearch:
    """
    Dynamic basin search engine for Livnium-T.
    
    Manages geometry-driven basin reinforcement with self-tuning parameters.
    """
    
    def __init__(
        self,
        base_alpha: float = 0.10,
        base_beta: float = 0.15,
        base_noise: float = 0.03
    ):
        """
        Initialize dynamic basin search.
        
        Args:
            base_alpha: Base reinforcement strength
            base_beta: Base decay strength
            base_noise: Base decorrelation strength
        """
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.base_noise = base_noise
        self.history: List[Dict[str, Any]] = []
    
    def update(
        self,
        system: LivniumTSystem,
        active_node_ids: List[int],
        is_correct: bool
    ) -> Dict[str, float]:
        """
        Update basin with dynamic parameters.
        
        Args:
            system: LivniumTSystem
            active_node_ids: List of active node IDs (0-4)
            is_correct: Whether solution is correct
            
        Returns:
            Dictionary with geometry signals and parameters
        """
        result = apply_dynamic_basin(
            system,
            active_node_ids,
            is_correct,
            self.base_alpha,
            self.base_beta,
            self.base_noise
        )
        
        result['is_correct'] = is_correct
        result['active_node_ids'] = active_node_ids.copy()
        self.history.append(result)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        if not self.history:
            return {
                'total_updates': 0,
                'correct_count': 0,
                'avg_curvature': 0.0,
                'avg_tension': 0.0,
                'avg_entropy': 0.0,
            }
        
        correct_count = sum(1 for h in self.history if h.get('is_correct', False))
        avg_curvature = np.mean([h['curvature'] for h in self.history])
        avg_tension = np.mean([h['tension'] for h in self.history])
        avg_entropy = np.mean([h['entropy'] for h in self.history])
        
        return {
            'total_updates': len(self.history),
            'correct_count': correct_count,
            'correct_rate': correct_count / len(self.history) if self.history else 0.0,
            'avg_curvature': float(avg_curvature),
            'avg_tension': float(avg_tension),
            'avg_entropy': float(avg_entropy),
        }

