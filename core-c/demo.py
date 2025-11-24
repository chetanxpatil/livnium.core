"""
Livnium-C System Demo

Demonstrates the circular semantic engine with basic operations.
"""

from .classical import LivniumCSystem, NodeClass


def main():
    """Run the demo."""
    print("=" * 60)
    print("Livnium-C System Demo")
    print("=" * 60)
    print()
    
    # Create system with N=8 ring nodes
    print("Creating Livnium-C system with N=8 ring nodes...")
    system = LivniumCSystem(n_ring=8, radius=1.0)
    print(f"System: {system}")
    print()
    
    # Display ledger
    print("Conservation Ledger:")
    ledger = system.get_ledger()
    for key, value in ledger.items():
        print(f"  {key}: {value}")
    print()
    
    # Display nodes
    print("Nodes:")
    core = system.get_core()
    print(f"  Core (Om): node_id={core.node_id}, f={core.exposure}, SW={core.symbolic_weight}")
    
    ring_nodes = system.get_ring_nodes()
    print(f"  Ring nodes ({len(ring_nodes)}):")
    for node in ring_nodes[:4]:  # Show first 4
        pos = node.get_position()
        print(f"    Node {node.node_id}: f={node.exposure}, SW={node.symbolic_weight}, "
              f"angle={node.angle:.3f}, pos=({pos[0]:.3f}, {pos[1]:.3f})")
    if len(ring_nodes) > 4:
        print(f"    ... ({len(ring_nodes) - 4} more)")
    print()
    
    # Test rotation
    print("Testing rotation...")
    print(f"  Original ring node angles:")
    for node in ring_nodes[:4]:
        print(f"    Node {node.node_id}: angle={node.angle:.3f}")
    
    # Apply rotation by 1 step
    rotated_system = system.rotate(rotation_id=1)
    rotated_ring = rotated_system.get_ring_nodes()
    print(f"  After rotation by 1 step:")
    for node in rotated_ring[:4]:
        print(f"    Node {node.node_id}: angle={node.angle:.3f}")
    print()
    
    # Verify ledger is preserved
    print("Verifying ledger preservation...")
    original_ledger = system.get_ledger()
    rotated_ledger = rotated_system.get_ledger()
    
    print(f"  Original SW: {original_ledger['total_sw']}")
    print(f"  Rotated SW: {rotated_ledger['total_sw']}")
    print(f"  Match: {abs(original_ledger['total_sw'] - rotated_ledger['total_sw']) < 1e-10}")
    print()
    
    # Test inverse rotation
    print("Testing inverse rotation...")
    inverse_id = system.rotation_group.get_inverse(1)
    print(f"  Inverse of rotation 1 is rotation {inverse_id}")
    
    # Apply inverse
    restored_system = rotated_system.rotate(rotation_id=inverse_id)
    restored_ring = restored_system.get_ring_nodes()
    print(f"  After inverse rotation:")
    for node in restored_ring[:4]:
        print(f"    Node {node.node_id}: angle={node.angle:.3f}")
    
    # Check if restored
    original_angles = [node.angle for node in ring_nodes]
    restored_angles = [node.angle for node in restored_ring]
    angles_match = all(abs(a - b) < 1e-10 for a, b in zip(original_angles, restored_angles))
    print(f"  Angles restored: {angles_match}")
    print()
    
    # Test with different N
    print("Testing with N=12...")
    system12 = LivniumCSystem(n_ring=12, radius=1.0)
    print(f"  System: {system12}")
    print(f"  Total SW: {system12.total_symbolic_weight}")
    print(f"  Encoding base: {system12.encoding_base}")
    print()
    
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

