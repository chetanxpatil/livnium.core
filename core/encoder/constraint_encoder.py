"""
Constraint Encoder: Convert Constraints to Tension Fields

This module encodes constraints as tension fields (energy landscape),
NOT as basin shapes.

Key Principle:
- Constraints → Tension patches (energy landscape)
- Solutions → Basins (candidate attractors)
- Search minimizes tension by finding best basin
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass

from core.classical.livnium_core_system import LivniumCoreSystem


@dataclass
class TensionField:
    """
    Represents a tension field (energy landscape) from constraints.
    
    Tension fields define:
    - Which coordinates are involved
    - How to compute tension (violation measure)
    - How to compute curvature (satisfaction measure)
    """
    constraint_id: str
    involved_coords: List[Tuple[int, int, int]]
    compute_tension: Callable[[LivniumCoreSystem], float]
    compute_curvature: Callable[[LivniumCoreSystem], float]
    description: str = ""
    
    def get_tension(self, system: LivniumCoreSystem) -> float:
        """Compute current tension for this constraint."""
        return self.compute_tension(system)
    
    def get_curvature(self, system: LivniumCoreSystem) -> float:
        """Compute current curvature (satisfaction) for this constraint."""
        return self.compute_curvature(system)


class ConstraintEncoder:
    """
    Encodes constraints as tension fields.
    
    This is the correct architecture:
    - Constraints → Tension fields (energy landscape)
    - Solutions → Basins (handled by problem encoder)
    """
    
    def __init__(self, system: LivniumCoreSystem):
        """
        Initialize constraint encoder.
        
        Args:
            system: LivniumCoreSystem
        """
        self.system = system
        self.tension_fields: List[TensionField] = []
    
    def encode_equality_constraint(
        self,
        constraint_id: str,
        var1_coords: List[Tuple[int, int, int]],
        var2_coords: List[Tuple[int, int, int]],
        target_value: float = 0.0
    ) -> TensionField:
        """
        Encode equality constraint: var1 = var2 + target_value
        
        Creates tension field that measures violation.
        
        Args:
            constraint_id: Unique identifier
            var1_coords: Coordinates for variable 1
            var2_coords: Coordinates for variable 2
            target_value: Target difference
            
        Returns:
            TensionField
        """
        involved_coords = var1_coords + var2_coords
        
        def compute_tension(system: LivniumCoreSystem) -> float:
            # Get SW values (with safety check)
            vals1 = [system.get_cell(c).symbolic_weight for c in var1_coords if system.get_cell(c)]
            vals2 = [system.get_cell(c).symbolic_weight for c in var2_coords if system.get_cell(c)]
            
            # Safety check: return 0.0 if no valid cells
            if not vals1 or not vals2:
                return 0.0
            
            sw1 = np.mean(vals1)
            sw2 = np.mean(vals2)
            
            # Tension = violation magnitude
            violation = abs(sw1 - sw2 - target_value)
            return float(violation)
        
        def compute_curvature(system: LivniumCoreSystem) -> float:
            # Curvature = satisfaction (inverse of tension)
            tension = compute_tension(system)
            return 1.0 / (1.0 + tension)  # Higher when satisfied
        
        field = TensionField(
            constraint_id=constraint_id,
            involved_coords=involved_coords,
            compute_tension=compute_tension,
            compute_curvature=compute_curvature,
            description=f"Equality: var1 = var2 + {target_value}"
        )
        
        self.tension_fields.append(field)
        return field
    
    def encode_inequality_constraint(
        self,
        constraint_id: str,
        var1_coords: List[Tuple[int, int, int]],
        var2_coords: List[Tuple[int, int, int]],
        threshold: float
    ) -> TensionField:
        """
        Encode inequality constraint: var1 >= var2 + threshold
        
        Args:
            constraint_id: Unique identifier
            var1_coords: Coordinates for variable 1
            var2_coords: Coordinates for variable 2
            threshold: Threshold value
            
        Returns:
            TensionField
        """
        involved_coords = var1_coords + var2_coords
        
        def compute_tension(system: LivniumCoreSystem) -> float:
            # Get SW values (with safety check)
            vals1 = [system.get_cell(c).symbolic_weight for c in var1_coords if system.get_cell(c)]
            vals2 = [system.get_cell(c).symbolic_weight for c in var2_coords if system.get_cell(c)]
            
            # Safety check: return 0.0 if no valid cells
            if not vals1 or not vals2:
                return 0.0
            
            sw1 = np.mean(vals1)
            sw2 = np.mean(vals2)
            
            # Tension = violation (if var1 < var2 + threshold)
            violation = max(0.0, (sw2 + threshold) - sw1)
            return float(violation)
        
        def compute_curvature(system: LivniumCoreSystem) -> float:
            tension = compute_tension(system)
            return 1.0 / (1.0 + tension)
        
        field = TensionField(
            constraint_id=constraint_id,
            involved_coords=involved_coords,
            compute_tension=compute_tension,
            compute_curvature=compute_curvature,
            description=f"Inequality: var1 >= var2 + {threshold}"
        )
        
        self.tension_fields.append(field)
        return field
    
    def encode_custom_constraint(
        self,
        constraint_id: str,
        involved_coords: List[Tuple[int, int, int]],
        tension_fn: Callable[[LivniumCoreSystem], float],
        description: str = ""
    ) -> TensionField:
        """
        Encode custom constraint with user-defined tension function.
        
        Args:
            constraint_id: Unique identifier
            involved_coords: Coordinates involved in constraint
            tension_fn: Function that computes tension from system state
            description: Human-readable description
            
        Returns:
            TensionField
        """
        def compute_curvature(system: LivniumCoreSystem) -> float:
            tension = tension_fn(system)
            return 1.0 / (1.0 + tension)
        
        field = TensionField(
            constraint_id=constraint_id,
            involved_coords=involved_coords,
            compute_tension=tension_fn,
            compute_curvature=compute_curvature,
            description=description or f"Custom constraint: {constraint_id}"
        )
        
        self.tension_fields.append(field)
        return field
    
    def get_total_tension(self, system: LivniumCoreSystem) -> float:
        """
        Get total tension across all constraints.
        
        Args:
            system: LivniumCoreSystem
            
        Returns:
            Total tension (sum of all constraint tensions)
        """
        total = 0.0
        for field in self.tension_fields:
            total += field.get_tension(system)
        return total
    
    def get_constraint_tensions(self, system: LivniumCoreSystem) -> Dict[str, float]:
        """
        Get tension for each constraint.
        
        Args:
            system: LivniumCoreSystem
            
        Returns:
            Dictionary mapping constraint_id → tension
        """
        return {
            field.constraint_id: field.get_tension(system)
            for field in self.tension_fields
        }

