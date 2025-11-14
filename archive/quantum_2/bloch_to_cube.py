"""
Bloch Sphere → Cube Coordinates Mapping

Maps quantum Bloch sphere coordinates (θ, φ) to Livnium cube coordinates (x, y, z).
Unifies quantum and geometric reasoning.
"""

import numpy as np
from typing import Tuple
from quantum.kernel import LivniumQubit


def bloch_to_cube(theta: float, phi: float, grid_size: int = 3) -> Tuple[int, int, int]:
    """
    Map Bloch sphere coordinates to cube coordinates.
    
    Args:
        theta: Polar angle [0, π] (from +Z axis)
        phi: Azimuthal angle [0, 2π] (around Z axis)
        grid_size: Size of cube grid (default 3 for 3×3×3)
        
    Returns:
        (x, y, z) cube coordinates
    """
    # Normalize angles
    theta_norm = theta / np.pi  # [0, 1]
    phi_norm = phi / (2 * np.pi)  # [0, 1]
    
    # Map to cube coordinates
    # x: based on theta (polar angle)
    x = int(theta_norm * grid_size) % grid_size
    
    # y: based on phi (azimuthal angle)
    y = int(phi_norm * grid_size) % grid_size
    
    # z: combination of both (creates 3D structure)
    z = int((theta_norm + phi_norm) / 2 * grid_size) % grid_size
    
    return (x, y, z)


def cube_to_bloch(x: int, y: int, z: int, grid_size: int = 3) -> Tuple[float, float]:
    """
    Map cube coordinates back to Bloch sphere coordinates.
    
    Args:
        x, y, z: Cube coordinates
        grid_size: Size of cube grid
        
    Returns:
        (theta, phi) Bloch sphere coordinates
    """
    # Normalize cube coordinates
    x_norm = x / grid_size  # [0, 1]
    y_norm = y / grid_size  # [0, 1]
    z_norm = z / grid_size  # [0, 1]
    
    # Map back to Bloch sphere
    theta = x_norm * np.pi  # [0, π]
    phi = y_norm * 2 * np.pi  # [0, 2π]
    
    return (theta, phi)


def qubit_to_cube_position(qubit: LivniumQubit, grid_size: int = 3) -> Tuple[int, int, int]:
    """
    Get cube position from qubit's Bloch sphere coordinates.
    
    Args:
        qubit: LivniumQubit instance
        grid_size: Size of cube grid
        
    Returns:
        (x, y, z) cube coordinates
    """
    theta, phi = qubit.get_bloch()
    return bloch_to_cube(theta, phi, grid_size)


def create_qubit_at_cube_position(x: int, y: int, z: int, grid_size: int = 3) -> LivniumQubit:
    """
    Create a qubit positioned at specific cube coordinates.
    
    Args:
        x, y, z: Cube coordinates
        grid_size: Size of cube grid
        
    Returns:
        LivniumQubit with Bloch coordinates matching cube position
    """
    theta, phi = cube_to_bloch(x, y, z, grid_size)
    
    # Create qubit state from Bloch coordinates
    # |ψ> = cos(θ/2)|0> + e^(iφ)sin(θ/2)|1>
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    initial_state = np.array([alpha + 0j, beta + 0j], dtype=np.complex128)
    
    qubit = LivniumQubit(
        position=(x, y, z),
        f=1,
        initial_state=initial_state
    )
    
    return qubit


def map_feature_to_cube(feature_value: float, grid_size: int = 3) -> Tuple[int, int, int]:
    """
    Map a feature value directly to cube coordinates.
    
    Uses feature value to determine Bloch sphere position, then maps to cube.
    
    Args:
        feature_value: Feature value [0, 1]
        grid_size: Size of cube grid
        
    Returns:
        (x, y, z) cube coordinates
    """
    # Map feature value to Bloch sphere
    # Higher value → higher theta (more |1>)
    theta = feature_value * np.pi  # [0, π]
    phi = feature_value * 2 * np.pi  # [0, 2π] (or use 0 for simplicity)
    
    return bloch_to_cube(theta, phi, grid_size)


if __name__ == "__main__":
    print("=" * 70)
    print("BLOCH SPHERE → CUBE COORDINATES MAPPING")
    print("=" * 70)
    
    # Test mapping
    print("\n1. Bloch → Cube Mapping:")
    test_cases = [
        (0.0, 0.0),           # |0> state
        (np.pi, 0.0),         # |1> state
        (np.pi/2, np.pi/2),   # Superposition
        (np.pi/3, np.pi),     # Arbitrary
    ]
    
    for theta, phi in test_cases:
        x, y, z = bloch_to_cube(theta, phi, grid_size=3)
        print(f"   (θ={theta:.3f}, φ={phi:.3f}) → Cube({x}, {y}, {z})")
    
    # Test reverse mapping
    print("\n2. Cube → Bloch Mapping:")
    cube_positions = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 2, 1)]
    for x, y, z in cube_positions:
        theta, phi = cube_to_bloch(x, y, z, grid_size=3)
        print(f"   Cube({x}, {y}, {z}) → (θ={theta:.3f}, φ={phi:.3f})")
    
    # Test qubit → cube
    print("\n3. Qubit → Cube Position:")
    qubit = LivniumQubit((0, 0, 0), f=1)
    qubit.hadamard()  # Put in superposition
    theta, phi = qubit.get_bloch()
    x, y, z = qubit_to_cube_position(qubit, grid_size=3)
    print(f"   Qubit state: {qubit.state_string()}")
    print(f"   Bloch: (θ={theta:.3f}, φ={phi:.3f})")
    print(f"   Cube position: ({x}, {y}, {z})")
    
    # Test feature → cube
    print("\n4. Feature Value → Cube Position:")
    feature_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    for val in feature_values:
        x, y, z = map_feature_to_cube(val, grid_size=3)
        print(f"   Feature={val:.2f} → Cube({x}, {y}, {z})")
    
    print("\n" + "=" * 70)
    print("✅ BLOCH SPHERE ↔ CUBE MAPPING: Quantum and geometric unified!")
    print("=" * 70)

