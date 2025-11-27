"""
Livnium Hamiltonian Core: The Engine

Manages State (q, p), Time, and Thermodynamics.
Implements symplectic integrator with Langevin thermal bath.

Doesn't know about spheres or SW - just integrates momentum and applies thermostat.
"""

import numpy as np
from typing import Dict, Optional
from .forces import GeometricPotential


class LivniumHamiltonian:
    """
    The Engine.
    
    Manages State (q, p), Time, and Thermodynamics.
    Implements the Unified Principle: Minimizing Geometric Stress.
    """
    
    def __init__(
        self,
        n_spheres: int,
        temp: float = 0.1,
        friction: float = 0.05,
        dt: float = 0.01,
        radius: float = 1.0,
        k_repulsion: float = 500.0,
        k_gravity: float = 2.0,
        sw_target: float = 12.0,
        mass: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None
    ):
        """
        Initialize Hamiltonian engine.
        
        Args:
            n_spheres: Number of spheres in the system
            temp: Temperature (T) for thermal bath
            friction: Friction coefficient (Gamma) for Langevin dynamics
            dt: Time step
            radius: Sphere radius
            k_repulsion: Repulsion strength
            k_gravity: Gravity strength
            sw_target: Target SW density
            mass: Optional mass array (default: all ones)
            positions: Optional initial positions (default: random)
        """
        # Configuration
        self.N = n_spheres
        self.dt = dt
        
        # Thermodynamics
        self.temperature = temp  # T
        self.friction = friction  # Gamma
        
        # The Laws
        self.laws = GeometricPotential(
            radius=radius,
            k_repulsion=k_repulsion,
            k_gravity=k_gravity,
            sw_target=sw_target
        )
        
        # State Vector
        if positions is not None:
            self.q = positions.copy()
        else:
            # Random initial positions
            self.q = np.random.randn(self.N, 3) * 2.0
        
        self.p = np.zeros((self.N, 3))  # Momenta (start at rest)
        
        if mass is not None:
            self.mass = mass.copy()
        else:
            self.mass = np.ones(self.N)  # Inertia (can make dynamic later)
        
        # Metrics
        self.time = 0.0
        self.energy_log = []
    
    def step(self) -> Dict:
        """
        Velocity Verlet Integration with Langevin Thermostat.
        
        Preserves Symplectic structure (energy conservation when friction=0).
        Implements the Unified Principle: Minimizing Geometric Stress.
        
        Returns:
            Dictionary with:
            - time: Current time
            - total_energy: Total energy (kinetic + potential)
            - kinetic_energy: Kinetic energy
            - potential_energy: Potential energy
            - avg_sw: Average SW density
            - positions: Current positions
        """
        dt = self.dt
        
        # 1. First Half-Kick (Hamiltonian)
        forces, pot_energy, sw = self.laws.compute_forces(self.q)
        self.p += 0.5 * forces * dt
        
        # 2. Drift (Kinematic)
        self.q += (self.p / self.mass[:, None]) * dt
        
        # 3. Second Half-Kick (Hamiltonian)
        forces_new, pot_energy_new, sw_new = self.laws.compute_forces(self.q)
        self.p += 0.5 * forces_new * dt
        
        # Use updated values
        pot_energy = pot_energy_new
        sw = sw_new
        
        # 4. Thermal Bath (Langevin Dynamics)
        # Adds Entropy (friction) and Fluctuations (noise)
        if self.friction > 0:
            sigma = np.sqrt(2.0 * self.friction * self.temperature)
            noise = np.random.randn(self.N, 3)
            
            # Update momentum
            self.p -= self.friction * self.p * dt  # Drag
            self.p += sigma * noise * np.sqrt(dt)  # Kick
        
        self.time += dt
        
        # 5. Logging (The "Observer")
        kinetic_energy = 0.5 * np.sum(self.p**2 / self.mass[:, None])
        total_energy = kinetic_energy + pot_energy
        
        self.energy_log.append({
            'time': self.time,
            'total_energy': total_energy,
            'kinetic_energy': kinetic_energy,
            'potential_energy': pot_energy,
            'avg_sw': np.mean(sw)
        })
        
        return {
            'time': self.time,
            'total_energy': total_energy,
            'kinetic_energy': kinetic_energy,
            'potential_energy': pot_energy,
            'avg_sw': np.mean(sw),
            'sw': sw.copy(),
            'positions': self.q.copy()
        }
    
    def get_energy_history(self) -> Dict:
        """
        Get energy history for analysis.
        
        Returns:
            Dictionary with arrays of time, energies, and SW
        """
        if not self.energy_log:
            return {
                'time': np.array([]),
                'total_energy': np.array([]),
                'kinetic_energy': np.array([]),
                'potential_energy': np.array([]),
                'avg_sw': np.array([])
            }
        
        return {
            'time': np.array([e['time'] for e in self.energy_log]),
            'total_energy': np.array([e['total_energy'] for e in self.energy_log]),
            'kinetic_energy': np.array([e['kinetic_energy'] for e in self.energy_log]),
            'potential_energy': np.array([e['potential_energy'] for e in self.energy_log]),
            'avg_sw': np.array([e['avg_sw'] for e in self.energy_log])
        }

