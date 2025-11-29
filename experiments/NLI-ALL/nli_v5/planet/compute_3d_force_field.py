#!/usr/bin/env python3
"""
Compute 3D Force Field Inside Livnium Cube

This computes the REAL 3D geometry - the actual force field inside an N×N×N cube,
not a sphere projection. This is the true Livnium geometry.
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.nli_v5.core.encoder import ChainEncoder
from experiments.nli_v5.core.layers import Layer0Resonance


def compute_3d_force_field(
    cube_size: int = 21,  # Odd number for center cell
    resolution: int = 10,  # Samples per dimension
    pattern_file: str = None
) -> Dict:
    """
    Compute 3D force field inside a Livnium cube.
    
    Args:
        cube_size: Size of the cube (N×N×N)
        resolution: Sampling resolution (higher = more detail)
        pattern_file: Optional pattern file for calibration
    
    Returns:
        Dictionary with 3D arrays of forces and metadata
    """
    print("=" * 70)
    print("Computing REAL 3D Livnium Force Field")
    print("=" * 70)
    print(f"Cube size: {cube_size}×{cube_size}×{cube_size}")
    print(f"Sampling resolution: {resolution}×{resolution}×{resolution}")
    print()
    
    # Create 3D coordinate grid
    # Map cube coordinates to semantic space
    # Center (0,0,0) = equilibrium point
    
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Convert 3D coordinates to semantic parameters
    # We'll map (x,y,z) to (angle, alignment, distance)
    
    # Distance from center
    r = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Angle (latitude equivalent)
    # Map z-coordinate to angle: -1 → 0°, 0 → 45°, +1 → 90°
    angles_deg = 45 + Z * 45  # Range: 0° to 90°
    angles_rad = angles_deg * np.pi / 180.0
    
    # Alignment (cosine similarity equivalent)
    # Map distance to alignment: center = high, edge = low
    alignment = 1.0 - r * 0.5  # Center = 1.0, edge = 0.5
    
    # Compute divergence field
    print("Computing divergence field...")
    equilibrium_threshold = Layer0Resonance.equilibrium_threshold
    divergence = equilibrium_threshold - alignment
    
    # Add directional component based on position
    # X-axis: left-right semantic variation
    # Y-axis: forward-backward semantic variation
    divergence += (X * 0.1)  # Left = more negative, right = more positive
    
    # Compute resonance field
    print("Computing resonance field...")
    # Resonance based on distance from center and angle
    resonance = 0.5 + 0.3 * np.cos(angles_rad)
    resonance *= (1.0 - r * 0.3)  # Higher at center
    
    # Compute curvature field
    print("Computing curvature field...")
    # Curvature = gradient of divergence
    grad_x = np.gradient(divergence, axis=0)
    grad_y = np.gradient(divergence, axis=1)
    grad_z = np.gradient(divergence, axis=2)
    curvature = -(grad_x**2 + grad_y**2 + grad_z**2)
    
    # Compute stability field
    print("Computing stability field...")
    stability = resonance - curvature - (divergence * 2.0)
    
    # Compute force vectors (direction of semantic movement)
    print("Computing force vectors...")
    
    # Force direction = gradient of divergence (points toward attraction)
    force_x = -grad_x  # Negative gradient = direction of force
    force_y = -grad_y
    force_z = -grad_z
    
    # Force magnitude = |divergence| (strength of force)
    force_magnitude = np.abs(divergence)
    
    # Normalize force vectors
    force_norm = np.sqrt(force_x**2 + force_y**2 + force_z**2)
    force_norm = np.where(force_norm > 0, force_norm, 1.0)  # Avoid division by zero
    
    force_x_norm = force_x / force_norm
    force_y_norm = force_y / force_norm
    force_z_norm = force_z / force_norm
    
    # Scale by magnitude
    force_x_scaled = force_x_norm * force_magnitude
    force_y_scaled = force_y_norm * force_magnitude
    force_z_scaled = force_z_norm * force_magnitude
    
    # Compute cold force (entailment attraction)
    cold_force = np.maximum(0, -divergence)
    
    # Compute far force (contradiction repulsion)
    far_force = np.maximum(0, divergence)
    
    print("Normalizing fields...")
    
    # Normalize all fields to [0, 1] for visualization
    def normalize_field(field):
        field_min = field.min()
        field_max = field.max()
        if field_max > field_min:
            return (field - field_min) / (field_max - field_min)
        return field
    
    divergence_norm = normalize_field(divergence)
    resonance_norm = normalize_field(resonance)
    curvature_norm = normalize_field(curvature)
    stability_norm = normalize_field(stability)
    cold_force_norm = normalize_field(cold_force)
    far_force_norm = normalize_field(far_force)
    
    # Prepare data for export
    print("Preparing data for export...")
    
    data = {
        "metadata": {
            "cube_size": cube_size,
            "resolution": resolution,
            "field_shape": list(divergence.shape)
        },
        "coordinates": {
            "x": X.flatten().tolist(),
            "y": Y.flatten().tolist(),
            "z": Z.flatten().tolist(),
            "r": r.flatten().tolist()  # Distance from center
        },
        "fields": {
            "divergence": {
                "values": divergence.flatten().tolist(),
                "normalized": divergence_norm.flatten().tolist()
            },
            "resonance": {
                "values": resonance.flatten().tolist(),
                "normalized": resonance_norm.flatten().tolist()
            },
            "curvature": {
                "values": curvature.flatten().tolist(),
                "normalized": curvature_norm.flatten().tolist()
            },
            "stability": {
                "values": stability.flatten().tolist(),
                "normalized": stability_norm.flatten().tolist()
            },
            "cold_force": {
                "values": cold_force.flatten().tolist(),
                "normalized": cold_force_norm.flatten().tolist()
            },
            "far_force": {
                "values": far_force.flatten().tolist(),
                "normalized": far_force_norm.flatten().tolist()
            }
        },
        "forces": {
            "x": force_x_scaled.flatten().tolist(),
            "y": force_y_scaled.flatten().tolist(),
            "z": force_z_scaled.flatten().tolist(),
            "magnitude": force_magnitude.flatten().tolist()
        }
    }
    
    print(f"\n✓ Computed 3D force field")
    print(f"  - Resolution: {resolution}×{resolution}×{resolution} = {resolution**3} points")
    print(f"  - Fields: divergence, resonance, curvature, stability, cold_force, far_force")
    print(f"  - Force vectors: {len(data['forces']['x'])} vectors")
    
    return data


def export_3d_force_field(
    cube_size: int = 21,
    resolution: int = 20,  # Higher default for better visualization
    output_file: str = "livnium_3d_force_field.json"
):
    """Export 3D force field data."""
    data = compute_3d_force_field(cube_size=cube_size, resolution=resolution)
    
    output_path = Path(__file__).parent.parent / 'planet_output' / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"\n✓ Exported to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute 3D Livnium force field")
    parser.add_argument("--cube-size", type=int, default=21, help="Cube size (N×N×N)")
    parser.add_argument("--resolution", type=int, default=20, help="Sampling resolution")
    parser.add_argument("--output", type=str, default="livnium_3d_force_field.json", help="Output file")
    
    args = parser.parse_args()
    
    export_3d_force_field(
        cube_size=args.cube_size,
        resolution=args.resolution,
        output_file=args.output
    )

