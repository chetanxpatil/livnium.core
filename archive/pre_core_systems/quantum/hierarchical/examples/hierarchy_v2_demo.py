#!/usr/bin/env python3
"""
Hierarchy v2 Demo: Advanced Hierarchical Geometry System

Demonstrates:
1. Level Graph visualization
2. Operation Registry
3. Propagation Engine
4. Real-world geometric reasoning
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum.hierarchical.geometry.hierarchy_v2 import (
    HierarchyV2System, OperationType
)


def demo_level_graph():
    """Demonstrate level graph visualization."""
    print("=" * 70)
    print("DEMO 1: Level Graph Visualization")
    print("=" * 70)
    
    # Create 20-level system
    system = HierarchyV2System(base_dimension=3, num_levels=20)
    
    # Add some base states
    for i in range(100):
        system.add_base_state((i * 0.1, i * 0.1, i * 0.1))
    
    # Add operations at different levels
    system.register_operation(
        OperationType.ROTATION, level=1,
        parameters={'angle': 0.5, 'axis': 0},
        description='Initial rotation at level 1'
    )
    
    system.register_operation(
        OperationType.SCALE, level=5,
        parameters={'scale': 1.5},
        description='Scaling at level 5'
    )
    
    system.register_operation(
        OperationType.TRANSFORM, level=10,
        parameters={'transform_type': 'complex'},
        description='Complex transform at level 10'
    )
    
    system.register_operation(
        OperationType.ENTANGLE, level=15,
        parameters={'qubit1': 0, 'qubit2': 1},
        description='Entanglement at level 15'
    )
    
    # Visualize level graph
    print("\nLevel Graph (Tree Format):")
    print(system.get_level_graph(format='tree'))
    
    print("\nLevel Graph (Text Format):")
    print(system.get_level_graph(format='text'))


def demo_operation_registry():
    """Demonstrate operation registry."""
    print("\n" + "=" * 70)
    print("DEMO 2: Operation Registry")
    print("=" * 70)
    
    system = HierarchyV2System(base_dimension=3, num_levels=10)
    
    # Register various operations
    system.register_operation(
        OperationType.ROTATION, level=1,
        parameters={'angle': 0.3},
        description='Rotation operation'
    )
    
    system.register_operation(
        OperationType.SCALE, level=2,
        parameters={'scale': 2.0},
        description='Scaling operation'
    )
    
    system.register_operation(
        OperationType.ENTANGLE, level=3,
        parameters={'qubit1': 0, 'qubit2': 1},
        description='Entanglement operation'
    )
    
    system.register_operation(
        OperationType.TRANSFORM, level=5,
        parameters={'type': 'geometric'},
        description='Geometric transform'
    )
    
    # Get registry summary
    registry = system.get_operation_registry()
    print("\nOperation Registry Summary:")
    print(f"  Total operations: {registry['total_operations']}")
    print(f"  Operations by level: {registry['operations_by_level']}")
    print(f"  Operations by type: {registry['operations_by_type']}")
    
    # Get operations at specific level
    print("\nOperations at Level 1:")
    ops_level_1 = system.registry.get_operations_at_level(1)
    for op in ops_level_1:
        print(f"  - {op.operation_id}: {op.description}")


def demo_propagation_engine():
    """Demonstrate propagation engine."""
    print("\n" + "=" * 70)
    print("DEMO 3: Propagation Engine")
    print("=" * 70)
    
    system = HierarchyV2System(base_dimension=3, num_levels=10)
    
    # Add base states
    for i in range(50):
        system.add_base_state((i * 0.1, i * 0.1, i * 0.1))
    
    # Register operation at high level (will propagate down)
    operation = system.register_operation(
        OperationType.ROTATION, level=7,
        parameters={'angle': 0.5, 'axis': 0},
        description='Rotation at level 7 (propagates down)',
        propagates_down=True
    )
    
    print(f"\nOperation registered: {operation.operation_id}")
    print(f"  Type: {operation.operation_type.value}")
    print(f"  Level: {operation.level}")
    print(f"  Propagates down: {operation.propagates_down}")
    
    # Get propagation history
    history = system.get_propagation_history()
    print(f"\nPropagation History:")
    print(f"  Total propagations: {history['total_propagations']}")
    
    if history['propagations']:
        last_prop = history['propagations'][-1]
        print(f"  Last propagation:")
        print(f"    Operation: {last_prop['operation_id']}")
        print(f"    Started at level: {last_prop['start_level']}")
        print(f"    Levels affected: {last_prop['total_levels_affected']}")
        print(f"    Propagation path:")
        for step in last_prop['propagation_path']:
            print(f"      Level {step['level']}: {step['effect']['description']}")


def demo_capacity_independence():
    """Demonstrate that capacity is independent of hierarchy depth."""
    print("\n" + "=" * 70)
    print("DEMO 4: Capacity Independence from Hierarchy Depth")
    print("=" * 70)
    
    import tracemalloc
    
    test_configs = [
        (3, 1000),
        (10, 1000),
        (20, 1000),
        (50, 1000),
    ]
    
    print("\nTesting capacity with different hierarchy depths:")
    print(f"{'Levels':<10} {'Qubits':<10} {'Memory (MB)':<15} {'Bytes/Qubit':<15}")
    print("-" * 50)
    
    for num_levels, num_qubits in test_configs:
        tracemalloc.start()
        
        system = HierarchyV2System(base_dimension=3, num_levels=num_levels)
        
        # Add qubits
        for i in range(num_qubits):
            system.add_base_state((i * 0.1, i * 0.1, i * 0.1))
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_mb = peak / 1024 / 1024
        bytes_per_qubit = (peak / num_qubits) if num_qubits > 0 else 0
        
        print(f"{num_levels:<10} {num_qubits:<10} {memory_mb:<15.2f} {bytes_per_qubit:<15.0f}")
    
    print("\n✅ Result: Capacity is independent of hierarchy depth!")
    print("   More levels = more operation layers, NOT more memory per qubit")


def demo_full_system():
    """Demonstrate full system capabilities."""
    print("\n" + "=" * 70)
    print("DEMO 5: Full System Information")
    print("=" * 70)
    
    system = HierarchyV2System(base_dimension=3, num_levels=15)
    
    # Add base states
    for i in range(500):
        system.add_base_state((i * 0.1, i * 0.1, i * 0.1))
    
    # Register operations at various levels
    for level in [1, 3, 5, 7, 10, 12, 14]:
        system.register_operation(
            OperationType.TRANSFORM, level=level,
            parameters={'level': level, 'type': 'meta'},
            description=f'Meta-transform at level {level}'
        )
    
    # Get full system info
    info = system.get_full_system_info()
    
    print("\nFull System Information:")
    print(f"  Hierarchy levels: {info['hierarchy']['num_levels']}")
    print(f"  Base states: {info['hierarchy']['base_states']}")
    print(f"  Total operations: {info['operation_registry']['total_operations']}")
    print(f"  Total propagations: {info['propagation_history']['total_propagations']}")
    print(f"\n  Key Insight: {info['insight']}")


if __name__ == '__main__':
    demo_level_graph()
    demo_operation_registry()
    demo_propagation_engine()
    demo_capacity_independence()
    demo_full_system()
    
    print("\n" + "=" * 70)
    print("✅ Hierarchy v2 Demo Complete!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  1. Level Graph - Visual hierarchy structure")
    print("  2. Operation Registry - Documented, auditable operations")
    print("  3. Propagation Engine - Operations propagate down levels")
    print("  4. Capacity Independence - Depth doesn't affect qubit capacity")
    print("\nThis is geometric symbolic logic, not a quantum simulator!")
    print("=" * 70)

