"""
Livnium-O System Demo

Demonstrates the spherical semantic engine with basic operations,
including the generalized kissing constraint.
"""

import numpy as np
from .classical import (
    LivniumOSystem,
    NodeClass,
    kissing_constraint_weight,
    check_kissing_constraint,
    SphericalRotationGroup,
)


def main():
    """Run the demo."""
    print("=" * 70)
    print("Livnium-O System Demo")
    print("=" * 70)
    print()
    
    # Demonstrate kissing constraint formula
    print("1. Generalized Kissing Constraint Formula")
    print("-" * 70)
    print()
    print("The fundamental law:")
    print("  sum_i [1 - sqrt(1 - (r_i/(1+r_i))^2)] <= 2")
    print()
    
    # Test with uniform radius
    print("Testing with uniform radius r=1.0:")
    radius = 1.0
    weight = kissing_constraint_weight(radius)
    print(f"  Weight per neighbor: {weight:.6f}")
    max_neighbors = int(2.0 / weight)
    print(f"  Maximum neighbors: {max_neighbors}")
    print(f"  (Classical kissing number for r=1 is 6)")
    print()
    
    # Test with smaller radius
    print("Testing with smaller radius r=0.5:")
    radius_small = 0.5
    weight_small = kissing_constraint_weight(radius_small)
    print(f"  Weight per neighbor: {weight_small:.6f}")
    max_neighbors_small = int(2.0 / weight_small)
    print(f"  Maximum neighbors: {max_neighbors_small}")
    print()
    
    # Test with mixed radii
    print("Testing with mixed radii [1.0, 0.5, 0.3, 0.2, 0.1]:")
    mixed_radii = [1.0, 0.5, 0.3, 0.2, 0.1]
    weights = [kissing_constraint_weight(r) for r in mixed_radii]
    total_weight = sum(weights)
    is_valid, check_weight = check_kissing_constraint(mixed_radii)
    print(f"  Weights: {[f'{w:.6f}' for w in weights]}")
    print(f"  Total weight: {total_weight:.6f}")
    print(f"  Valid: {is_valid} (must be <= 2.0)")
    print()
    
    # Create system with valid configuration
    print("2. Creating Livnium-O System")
    print("-" * 70)
    print()
    
    # Use 6 neighbors with radius 1.0 (classical kissing number)
    neighbor_radii = [1.0] * 6
    is_valid, total_weight = check_kissing_constraint(neighbor_radii)
    print(f"Configuration: 6 neighbors with radius 1.0")
    print(f"  Kissing constraint weight: {total_weight:.6f}")
    print(f"  Valid: {is_valid}")
    print()
    
    try:
        system = LivniumOSystem(neighbor_radii=neighbor_radii, core_radius=1.0)
        print(f"System created: {system}")
        print()
        
        # Display ledger
        print("Conservation Ledger:")
        ledger = system.get_ledger()
        for key, value in ledger.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print()
        
        # Display nodes
        print("Nodes:")
        core = system.get_core()
        print(f"  Core (Om): node_id={core.node_id}, radius={core.radius}, "
              f"f={core.exposure}, SW={core.symbolic_weight}")
        
        neighbor_nodes = system.get_neighbor_nodes()
        print(f"  Neighbor nodes ({len(neighbor_nodes)}):")
        for node in neighbor_nodes[:3]:  # Show first 3
            pos = node.position
            dist = np.sqrt(sum(x*x for x in pos))
            print(f"    Node {node.node_id}: radius={node.radius:.3f}, "
                  f"f={node.exposure:.6f}, SW={node.symbolic_weight:.6f}, "
                  f"distance={dist:.3f}")
        if len(neighbor_nodes) > 3:
            print(f"    ... ({len(neighbor_nodes) - 3} more)")
        print()
        
        # Test rotation
        print("3. Testing Rotation")
        print("-" * 70)
        print()
        
        rotation_group = system.rotation_group
        # Create rotation around z-axis by 45 degrees
        axis = np.array([0, 0, 1])
        angle = np.pi / 4
        rotation_matrix = rotation_group.rotation_matrix_axis_angle(axis, angle)
        
        print("  Applying rotation around z-axis by 45 degrees...")
        rotated_system = system.rotate(rotation_matrix)
        
        # Show first neighbor position change
        original_pos = system.get_neighbor_nodes()[0].position
        rotated_pos = rotated_system.get_neighbor_nodes()[0].position
        print(f"  Original position: ({original_pos[0]:.3f}, {original_pos[1]:.3f}, {original_pos[2]:.3f})")
        print(f"  Rotated position:  ({rotated_pos[0]:.3f}, {rotated_pos[1]:.3f}, {rotated_pos[2]:.3f})")
        print()
        
        # Verify ledger is preserved
        print("  Verifying ledger preservation...")
        original_ledger = system.get_ledger()
        rotated_ledger = rotated_system.get_ledger()
        
        print(f"  Original SW: {original_ledger['total_sw']:.6f}")
        print(f"  Rotated SW: {rotated_ledger['total_sw']:.6f}")
        print(f"  Match: {abs(original_ledger['total_sw'] - rotated_ledger['total_sw']) < 1e-10}")
        print()
        
        # Test inverse rotation
        print("  Testing inverse rotation...")
        inverse_matrix = rotation_group.get_inverse(rotation_matrix)
        restored_system = rotated_system.rotate(inverse_matrix)
        
        # Check if restored
        original_positions = [n.position for n in system.get_neighbor_nodes()]
        restored_positions = [n.position for n in restored_system.get_neighbor_nodes()]
        positions_match = all(
            np.allclose(np.array(p1), np.array(p2), rtol=1e-10)
            for p1, p2 in zip(original_positions, restored_positions)
        )
        print(f"  Positions restored: {positions_match}")
        print()
        
    except ValueError as e:
        print(f"Error creating system: {e}")
        print()
    
    # Test invalid configuration
    print("4. Testing Invalid Configuration")
    print("-" * 70)
    print()
    
    # Try to create system with too many neighbors
    invalid_radii = [1.0] * 10  # 10 neighbors with radius 1.0
    is_valid, total_weight = check_kissing_constraint(invalid_radii)
    print(f"Configuration: 10 neighbors with radius 1.0")
    print(f"  Kissing constraint weight: {total_weight:.6f}")
    print(f"  Valid: {is_valid}")
    
    if not is_valid:
        print("  This configuration violates the kissing constraint!")
        print("  Cannot create system with this configuration.")
    print()
    
    # Test with heterogeneous radii
    print("5. Testing Heterogeneous Configuration")
    print("-" * 70)
    print()
    
    heterogeneous_radii = [1.0, 0.8, 0.6, 0.4, 0.2]
    is_valid, total_weight = check_kissing_constraint(heterogeneous_radii)
    print(f"Configuration: radii {heterogeneous_radii}")
    print(f"  Kissing constraint weight: {total_weight:.6f}")
    print(f"  Valid: {is_valid}")
    
    if is_valid:
        try:
            system2 = LivniumOSystem(neighbor_radii=heterogeneous_radii, core_radius=1.0)
            print(f"  System created: {system2}")
            print(f"  Total SW: {system2.total_symbolic_weight:.6f}")
        except ValueError as e:
            print(f"  Error: {e}")
    print()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

