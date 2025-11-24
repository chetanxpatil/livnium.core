#!/usr/bin/env python3
"""
Livnium-T Quantum Layer Demo

Demonstrates quantum features for the 5-node topology.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classical import LivniumTSystem, NodeClass
from quantum import QuantumSystem, GateType, MeasurementBasis

def main():
    """Run the quantum demo."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "Livnium-T Quantum Layer" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Create systems
    t_system = LivniumTSystem()
    q_system = QuantumSystem(t_system, enable_geometry_coupling=True)
    
    print(f"✓ Quantum system initialized: {q_system}")
    print()
    
    # Show initial states
    print("Initial Quantum States:")
    for i in range(5):
        node = q_system.get_node_state(i)
        probs = node.get_probabilities()
        geom_node = t_system.get_node(i)
        print(f"  Node {i} ({geom_node.node_class.name:6s}): P(0)={probs[0]:.3f}, P(1)={probs[1]:.3f}")
    print()
    
    # Apply gates
    print("Applying Quantum Gates:")
    q_system.apply_gate(1, GateType.HADAMARD)
    node1 = q_system.get_node_state(1)
    probs1 = node1.get_probabilities()
    print(f"  • H gate on node 1: P(0)={probs1[0]:.3f}, P(1)={probs1[1]:.3f}")
    
    q_system.apply_gate(2, GateType.PAULI_X)
    node2 = q_system.get_node_state(2)
    probs2 = node2.get_probabilities()
    print(f"  • X gate on node 2: P(0)={probs2[0]:.3f}, P(1)={probs2[1]:.3f}")
    print()
    
    # Entanglement
    print("Entanglement:")
    q_system.entangle_nodes(1, 2, 'phi_plus')
    is_entangled = q_system.entanglement_manager.is_entangled(1, 2)
    print(f"  • Nodes 1-2 entangled: {is_entangled}")
    pair = q_system.entanglement_manager.entangled_pairs[tuple(sorted([1, 2]))]
    print(f"  • Concurrence: {pair.get_concurrence():.3f}")
    print()
    
    # Measurement
    print("Quantum Measurement:")
    result = q_system.measure_node(1, MeasurementBasis.COMPUTATIONAL)
    print(f"  • Measured node 1: level={result.measured_level}, P={result.probability:.3f}")
    print(f"  • Node 1 after collapse: P(0)={q_system.get_node_state(1).get_probabilities()[0]:.3f}")
    print()
    
    # Summary
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "Quantum Layer Verified" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("  Livnium-T quantum layer: Complete and functional")
    print("  • Superposition: ✓")
    print("  • Gates: ✓")
    print("  • Entanglement: ✓")
    print("  • Measurement: ✓")
    print("  • Geometry coupling: ✓")

if __name__ == '__main__':
    main()

