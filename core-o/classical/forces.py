"""
Geometric Potential: The Laws of Physics for Livnium

Derives forces purely from geometric gradients.
Implements the Unified Principle: Minimizing Geometric Stress.

No "Gravity Formula" - only Geometric Potential, from which gravity,
repulsion, and pressure emerge automatically.
"""

import numpy as np
from typing import Tuple


class GeometricPotential:
    """
    The 'Laws of Physics' for Livnium.
    
    Derives forces purely from geometric gradients.
    Implements soft potentials to avoid gradient traps.
    """
    
    def __init__(
        self,
        radius: float = 1.0,
        k_repulsion: float = 500.0,
        k_gravity: float = 2.0,
        sw_target: float = 12.0
    ):
        """
        Initialize geometric potential laws.
        
        Args:
            radius: Sphere radius (R)
            k_repulsion: Stiffness of spheres (repulsion strength)
            k_gravity: Strength of the "Inward Fall" (density attraction)
            sw_target: Ideal density (target SW value, e.g., 12 neighbors)
        """
        self.R = radius
        self.D = 2.0 * radius  # Diameter (hard core)
        self.R_cut = 3.0 * self.D  # Gravity range (3 diameters)
        
        # Tuning Constants
        self.k_rep = k_repulsion    # Stiffness of spheres
        self.k_grav = k_gravity     # Strength of the "Inward Fall"
        self.sw_target = sw_target  # Ideal density (e.g., 12 neighbors)
    
    def _kernel_smooth_step(self, r: float) -> Tuple[float, float]:
        """
        A smooth differentiable kernel for SW calculation.
        
        Returns: (value, derivative)
        - 1.0 when touching (r = D)
        - 0.0 at R_cut
        - Smoothly differentiable everywhere (C² continuous)
        
        This ensures gradients are always finite.
        
        Args:
            r: Distance between sphere centers
            
        Returns:
            Tuple of (kernel_value, derivative_d_value_d_r)
        """
        if r >= self.R_cut:
            return 0.0, 0.0
        
        if r < self.D:
            # Inside the hard core, influence saturates at 1.0
            # But we keep derivative distinct to avoid singularities.
            # For simplicity, treat r < D same as surface for SW purposes.
            r = self.D
        
        # Normalized distance x in [0, 1]
        # x = 0 at r=D (surface), x = 1 at r=R_cut
        dist_range = self.R_cut - self.D
        x = (r - self.D) / dist_range
        
        # Cubic spline (smooth falloff): (1 - x)^3
        val = (1.0 - x)**3
        
        # Derivative d(val)/dr = d(val)/dx * dx/dr
        # d(val)/dx = -3(1-x)^2
        # dx/dr = 1 / dist_range
        deriv = -3.0 * (1.0 - x)**2 / dist_range
        
        return val, deriv
    
    def compute_forces(
        self,
        positions: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Computes Potentials and Forces for the N-body system.
        
        Implements:
        1. Repulsion (Hard Constraint) - prevents overlap
        2. Geometric Gravity (Inward Fall) - density gradient attraction
        
        Args:
            positions: Array of shape (N, 3) with sphere positions
            
        Returns:
            Tuple of (forces, potential_energy, current_sw)
            - forces: Array of shape (N, 3) with force vectors
            - potential_energy: Total potential energy
            - current_sw: Array of shape (N,) with current SW values
        """
        N = len(positions)
        forces = np.zeros((N, 3))
        potential_energy = 0.0
        
        # Array to store current SW for each sphere
        current_sw = np.zeros(N)
        
        # Pre-compute distances and vectors (Naive O(N^2) for clarity)
        # In production, use Neighbor Lists (Verlet lists)
        for i in range(N):
            for j in range(i + 1, N):  # Exploiting symmetry i-j vs j-i
                r_vec = positions[i] - positions[j]
                dist = np.linalg.norm(r_vec)
                
                if dist == 0:
                    continue
                
                r_hat = r_vec / dist  # Direction j -> i
                
                # --- 1. Repulsion (Hard Constraint) ---
                if dist < self.D:
                    overlap = self.D - dist
                    # V = 0.5 * k * overlap^2 (harmonic repulsion)
                    potential_energy += 0.5 * self.k_rep * overlap**2
                    
                    # F = - grad V = k * overlap * r_hat
                    f_push = self.k_rep * overlap * r_hat
                    forces[i] += f_push
                    forces[j] -= f_push
                
                # --- 2. Calculate SW Density (The "Metric") ---
                # We interpret "Exposure" as "Connection Strength" to neighbors
                # SW = Sum of kernels
                val, deriv = self._kernel_smooth_step(dist)
                
                if val > 0:
                    current_sw[i] += val
                    current_sw[j] += val
        
        # --- 3. Geometric Gravity (The Inward Fall) ---
        # Potential U = 0.5 * k_grav * sum( (SW_i - Target)^2 )
        # Force_i = - sum_k ( dU/dSW_k * dSW_k/dr_i )
        
        for i in range(N):
            # Energy of sphere i due to density mismatch
            diff = current_sw[i] - self.sw_target
            potential_energy += 0.5 * self.k_grav * diff**2
        
        # Gradient Pass
        for i in range(N):
            for j in range(i + 1, N):
                r_vec = positions[i] - positions[j]
                dist = np.linalg.norm(r_vec)
                
                if dist >= self.R_cut:
                    continue
                
                r_hat = r_vec / dist
                
                _, deriv = self._kernel_smooth_step(dist)
                # deriv is d(Kernel)/dr
                
                # Chain Rule Application:
                # Interaction between i and j affects BOTH SW_i and SW_j.
                # Force on i comes from:
                # 1. i trying to fix its own SW (SW_i)
                # 2. i trying to fix j's SW (SW_j) - Newton's 3rd Law
                
                diff_i = current_sw[i] - self.sw_target
                diff_j = current_sw[j] - self.sw_target
                
                # F_gravity = - (k * diff_i * deriv + k * diff_j * deriv) * r_hat
                # If diff is negative (under-dense), we want attraction.
                # deriv is negative (kernel decreases with distance).
                # So if diff < 0 and deriv < 0, product is positive.
                # We need attraction (force opposite to r_hat).
                
                f_grav_mag = self.k_grav * (diff_i + diff_j) * deriv
                
                # Apply force
                f_vec = f_grav_mag * r_hat  # deriv is negative, so this pulls inward
                
                forces[i] -= f_vec
                forces[j] += f_vec
        
        return forces, potential_energy, current_sw

