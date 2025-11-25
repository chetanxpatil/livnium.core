#!/usr/bin/env python3
"""
Example: Using Livnium Force in Phase 6

This demonstrates the minimal Livnium implementation - just a simple
geometric bias function that Cursor can understand and use.
"""

import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from livnium_force import LivniumForce, create_default_livnium


def example_basic_usage():
    """Basic example: create and use Livnium force."""
    print("=" * 60)
    print("Example 1: Basic Livnium Force Usage")
    print("=" * 60)
    
    # Create a default Livnium force (8D vector, scale=0.01)
    livnium = create_default_livnium(n_components=8, force_scale=0.01)
    
    # Create a sample 8D state
    y_t = np.random.randn(8)
    print(f"\nInput state y_t: {y_t}")
    print(f"  Shape: {y_t.shape}")
    
    # Apply Livnium force
    bias = livnium.apply_livnium_force(y_t)
    print(f"\nLivnium bias: {bias}")
    print(f"  Shape: {bias.shape}")
    print(f"  Magnitude: {np.linalg.norm(bias):.6f}")
    
    # Show the effect
    y_t_modified = y_t + bias
    print(f"\nModified state: y_t + Livnium = {y_t_modified}")
    print(f"  Change magnitude: {np.linalg.norm(bias):.6f}")
    print(f"  Relative change: {np.linalg.norm(bias) / np.linalg.norm(y_t) * 100:.2f}%")


def example_matrix_vs_vector():
    """Compare matrix-based vs vector-based Livnium force."""
    print("\n" + "=" * 60)
    print("Example 2: Matrix vs Vector Force Types")
    print("=" * 60)
    
    y_t = np.random.randn(8)
    
    # Vector force (constant bias)
    livnium_vector = LivniumForce(
        force_scale=0.01,
        force_type='vector',
        n_components=8
    )
    bias_vector = livnium_vector.apply_livnium_force(y_t)
    
    # Matrix force (state-dependent)
    livnium_matrix = LivniumForce(
        force_scale=0.01,
        force_type='matrix',
        n_components=8
    )
    bias_matrix = livnium_matrix.apply_livnium_force(y_t)
    
    print(f"\nInput state: {y_t}")
    print(f"\nVector force (constant):")
    print(f"  Bias: {bias_vector}")
    print(f"  Magnitude: {np.linalg.norm(bias_vector):.6f}")
    
    print(f"\nMatrix force (state-dependent):")
    print(f"  Bias: {bias_matrix}")
    print(f"  Magnitude: {np.linalg.norm(bias_matrix):.6f}")
    
    print(f"\nDifference:")
    print(f"  Matrix bias depends on y_t (matrix @ y_t)")
    print(f"  Vector bias is constant (same for all y_t)")


def example_integration():
    """Show how Livnium integrates into dynamics."""
    print("\n" + "=" * 60)
    print("Example 3: Integration into Dynamics Step")
    print("=" * 60)
    
    livnium = create_default_livnium(n_components=8, force_scale=0.01)
    y_t = np.random.randn(8)
    
    # Simulate one step of dynamics
    print(f"\nStep 1: Current state y_t = {y_t}")
    
    # Existing dynamics (simplified - in real code this is the polynomial model)
    y_tp1_dynamics = y_t * 0.95  # Example: damped dynamics
    print(f"Step 2: After dynamics: y_tp1 = {y_tp1_dynamics}")
    
    # Add noise (simplified)
    noise = np.random.randn(8) * 0.1
    y_tp1_noise = y_tp1_dynamics + noise
    print(f"Step 3: After noise: y_tp1 = {y_tp1_noise}")
    
    # Add Livnium force (Phase 6)
    livnium_bias = livnium.apply_livnium_force(y_t)
    y_tp1_final = y_tp1_noise + livnium_bias
    print(f"Step 4: After Livnium: y_tp1 = {y_tp1_final}")
    
    print(f"\nLivnium contribution:")
    print(f"  Bias: {livnium_bias}")
    print(f"  Relative to dynamics: {np.linalg.norm(livnium_bias) / np.linalg.norm(y_tp1_dynamics) * 100:.2f}%")
    print(f"\nThis is the minimal Livnium - just a small steering force!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Livnium Force Examples (Phase 6 - Minimal Version)")
    print("=" * 60)
    print("\nThis demonstrates the minimal Livnium implementation:")
    print("  - Simple function: apply_livnium_force(y_t) -> 8D bias")
    print("  - No complex geometry, no recursive systems, no quantum")
    print("  - Just a small geometric influence operator")
    print("=" * 60)
    
    example_basic_usage()
    example_matrix_vs_vector()
    example_integration()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nLivnium (Phase 6) is:")
    print("  ✓ A simple function that returns an 8D bias vector")
    print("  ✓ Added on top of existing dynamics: y_tp1 += Livnium(y_t)")
    print("  ✓ Small scale (default 0.01 = 1% influence)")
    print("  ✓ Can be vector (constant) or matrix (state-dependent)")
    print("\nThis is NOT the full Livnium system - it's a minimal proxy")
    print("designed for Cursor to understand and use in Rule 30 Shadow.")
    print("=" * 60 + "\n")

