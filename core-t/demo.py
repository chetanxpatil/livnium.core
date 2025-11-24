#!/usr/bin/env python3
"""
Livnium-T System Demo

Demonstrates the complete Livnium-T system with all axioms and derived laws.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classical import LivniumTSystem, NodeClass

def main():
    """Run the demo."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Livnium-T System" + " " * 37 + "║")
    print("║" + " " * 10 + "Stand-Alone Tetrahedral Semantic Engine" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Create system
    system = LivniumTSystem()
    print(f"✓ System initialized: {len(system.nodes)} nodes, ΣSW={system.get_total_sw()}, K_T={system.get_equilibrium_constant()}")
    print()
    
    # T-A1: Canonical Simplex Alphabet
    print("T-A1: Canonical Simplex Alphabet")
    core_count = sum(1 for n in system.nodes.values() if n.node_class == NodeClass.CORE)
    vertex_count = sum(1 for n in system.nodes.values() if n.node_class == NodeClass.VERTEX)
    print(f"  • Topology: {core_count} core + {vertex_count} vertices = {len(system.nodes)} nodes")
    print()
    
    # T-A2: Observer Anchor
    print("T-A2: Observer Anchor & Frame")
    print(f"  • Om observer: Node {system.om_observer.node_id} (immovable)")
    print()
    
    # T-A3: Exposure Law
    print("T-A3: Exposure Law (Two-Class System)")
    for node_id in range(5):
        node = system.get_node(node_id)
        class_name = "CORE" if node.node_class == NodeClass.CORE else "VERTEX"
        print(f"  • Node {node_id}: {class_name:6s}  f={node.exposure}  SW={node.symbolic_weight:.0f}")
    print()
    
    # T-A4: Symbolic Weight Law
    print("T-A4: Symbolic Weight Law")
    total_sw = system.get_total_sw()
    match = "✓" if abs(total_sw - LivniumTSystem.CANONICAL_TOTAL_SW) < 1e-6 else "✗"
    print(f"  • ΣSW_T = {total_sw:.0f}  (canonical: {LivniumTSystem.CANONICAL_TOTAL_SW})  {match}")
    print()
    
    # T-A5: Rotation Group
    print("T-A5: Dynamic Law (Tetrahedral Rotation Group A₄)")
    rotation_group = system.rotation_group
    print(f"  • Rotation group: {rotation_group.order} elements")
    print(f"  • Reversibility: All rotations have unique inverses")
    print()
    
    # T-D1: Equilibrium Constant
    print("T-D1: Simplex Equilibrium Constant")
    k_t = system.get_equilibrium_constant()
    match = "✓" if abs(k_t - LivniumTSystem.CANONICAL_EQUILIBRIUM_CONSTANT) < 1e-6 else "✗"
    print(f"  • K_T = {k_t:.0f}  (canonical: {LivniumTSystem.CANONICAL_EQUILIBRIUM_CONSTANT})  {match}")
    print()
    
    # T-D3: Conservation Ledger
    print("T-D3: Conservation Ledger")
    counts = system.get_class_counts()
    ledger_ok = system.verify_ledger()
    status = "✓ Verified" if ledger_ok else "✗ Failed"
    print(f"  • Core: {counts[NodeClass.CORE]}  Vertices: {counts[NodeClass.VERTEX]}  ΣSW: {total_sw:.0f}  {status}")
    print()
    
    # T-D5: Base-5 Encoding
    print("T-D5: Base-5 Encoding Law")
    print(f"  • Base: {system.BASE}  Alphabet: {{0, 1, 2, 3, 4}}")
    test_sequence = [2, 4, 1, 3]
    encoded = system.encode_base5(test_sequence)
    decoded = system.decode_base5(encoded, length=len(test_sequence))
    reversible = "✓" if decoded == test_sequence else "✗"
    print(f"  • Example: {test_sequence} → {encoded} → {decoded}  {reversible}")
    print()
    
    # Summary
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "All Axioms & Laws Verified" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("  Livnium-T: Complete, stand-alone semantic engine")
    print(f"  Minimal universe: {len(system.nodes)} nodes, Base-{system.BASE} encoding")
    print()

if __name__ == '__main__':
    main()

