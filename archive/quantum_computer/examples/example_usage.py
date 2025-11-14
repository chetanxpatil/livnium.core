"""
Example: Using Hierarchical Geometry Quantum Computer

Demonstrates geometry > geometry in geometry system.
"""

import numpy as np

from quantum_computer.core.quantum_processor import QuantumProcessor
from quantum_computer.simulators.hierarchical_simulator import HierarchicalQuantumSimulator


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Hierarchical Geometry Quantum Computer")
    print("Principle: Geometry > Geometry in Geometry")
    print("=" * 60)
    
    # Create processor
    processor = QuantumProcessor(base_dimension=3)
    
    # Create qubits in base geometry (Level 0)
    print("\n[Level 0] Creating qubits in base geometry...")
    q0 = processor.create_qubit((0.0, 0.0, 0.0), amplitude=1.0+0j)
    q1 = processor.create_qubit((1.0, 0.0, 0.0), amplitude=1.0+0j)
    print(f"Created qubits: {q0}, {q1}")
    
    # Apply Level 1 operations (geometry in geometry)
    print("\n[Level 1] Applying meta-geometric operations...")
    processor.apply_hadamard(q0)
    print("Applied Hadamard using geometry-in-geometry")
    
    # Apply Level 2 operations (geometry in geometry in geometry)
    print("\n[Level 2] Applying meta-meta-geometric operations...")
    processor.apply_cnot(q0, q1)
    print("Applied CNOT using geometry-in-geometry-in-geometry")
    
    # Get system info
    info = processor.get_system_info()
    print(f"\n=== System Info ===")
    print(f"Qubits: {info['num_qubits']}")
    print(f"Principle: {info['principle']}")
    print(f"Geometry levels: {info['geometry_structure']['hierarchical_levels']}")


def example_simulator():
    """Simulator example."""
    print("\n" + "=" * 60)
    print("Hierarchical Quantum Simulator")
    print("=" * 60)
    
    # Create simulator
    simulator = HierarchicalQuantumSimulator(base_dimension=3)
    
    # Build circuit
    print("\nBuilding quantum circuit...")
    q0 = simulator.add_qubit((0.0, 0.0, 0.0))
    q1 = simulator.add_qubit((1.0, 0.0, 0.0))
    
    simulator.hadamard(q0)
    simulator.cnot(q0, q1)
    
    # Run simulation
    print("\nRunning simulation (100 shots)...")
    results = simulator.run(num_shots=100)
    
    print(f"\n=== Results ===")
    print(f"Shots: {results['shots']}")
    print(f"Unique outcomes: {len(results['results'])}")
    for outcome, count in results['results'].items():
        print(f"  {outcome}: {count} ({count/results['shots']*100:.1f}%)")
    
    # Circuit info
    circuit_info = simulator.get_circuit_info()
    print(f"\n=== Circuit Info ===")
    print(f"Qubits: {circuit_info['num_qubits']}")
    print(f"Gates: {circuit_info['num_gates']}")
    print(f"Geometry structure: {circuit_info['geometry_structure']['principle']}")


def example_hierarchical_structure():
    """Show hierarchical structure."""
    print("\n" + "=" * 60)
    print("Hierarchical Structure")
    print("=" * 60)
    
    from quantum_computer.geometry.level2.geometry_in_geometry_in_geometry import HierarchicalGeometrySystem
    
    system = HierarchicalGeometrySystem(base_dimension=3)
    
    # Level 0: Base geometry
    print("\n[Level 0] Base Geometry")
    system.add_base_state((0.0, 0.0, 0.0))
    print("  Added state to base geometry")
    
    # Level 1: Geometry in geometry
    print("\n[Level 1] Geometry in Geometry")
    system.add_meta_operation('rotation', angle=np.pi/4, axis=0)
    print("  Added meta-geometric operation")
    
    # Level 2: Geometry in geometry in geometry
    print("\n[Level 2] Geometry in Geometry in Geometry")
    system.add_meta_meta_operation('scale_operations', scale=1.5)
    print("  Added meta-meta-geometric operation")
    
    # Get full structure
    structure = system.get_full_structure()
    print(f"\n=== Full Structure ===")
    print(f"Levels: {structure['hierarchical_levels']}")
    print(f"Principle: {structure['principle']}")
    print(f"\nLevel 0: {structure['level_0']['type']}")
    print(f"Level 1: {structure['level_1']['type']}")
    print(f"Level 2: {structure['level_2']['type']}")


if __name__ == '__main__':
    example_basic_usage()
    example_simulator()
    example_hierarchical_structure()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey Concept: Geometry > Geometry in Geometry")
    print("- Level 0: Base geometry (foundation)")
    print("- Level 1: Geometry operating on geometry")
    print("- Level 2: Geometry operating on geometry operating on geometry")

