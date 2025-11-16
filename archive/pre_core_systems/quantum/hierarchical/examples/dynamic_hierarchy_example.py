#!/usr/bin/env python3
"""
Example: Dynamic Hierarchical Geometry System

Demonstrates how to use the dynamic hierarchy with different numbers of levels.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum.hierarchical.geometry.dynamic_hierarchical_geometry import DynamicHierarchicalGeometrySystem


def example_3_levels():
    """Example with 3 levels (like current system)."""
    print("=" * 70)
    print("Example: 3 Levels (Base + 2 Meta Levels)")
    print("=" * 70)
    
    system = DynamicHierarchicalGeometrySystem(base_dimension=3, num_levels=3)
    
    # Add base states
    state1 = system.add_base_state((0.1, 0.2, 0.3))
    state2 = system.add_base_state((0.4, 0.5, 0.6))
    
    # Add Level 1 operation
    system.add_meta_operation('rotation', angle=0.5, axis=0)
    
    # Add Level 2 operation
    system.add_meta_meta_operation('entangle', qubit1=0, qubit2=1)
    
    # Get structure
    structure = system.get_full_structure()
    print(f"\nStructure:")
    print(f"  Levels: {structure['num_levels']}")
    print(f"  Principle: {structure['principle']}")
    print(f"  Base states: {structure['level_0']['num_states']}")
    print(f"  Level 1 operations: {structure['meta_levels'][0]['num_operations']}")
    print(f"  Level 2 operations: {structure['meta_levels'][1]['num_operations']}")


def example_5_levels():
    """Example with 5 levels."""
    print("\n" + "=" * 70)
    print("Example: 5 Levels (Base + 4 Meta Levels)")
    print("=" * 70)
    
    system = DynamicHierarchicalGeometrySystem(base_dimension=3, num_levels=5)
    
    # Add base states
    for i in range(5):
        system.add_base_state((i * 0.1, i * 0.2, i * 0.3))
    
    # Add operations at different levels
    system.add_operation(level=1, operation_type='rotation', angle=0.1)
    system.add_operation(level=2, operation_type='scale', scale=1.5)
    system.add_operation(level=3, operation_type='transform', param1='value1')
    system.add_operation(level=4, operation_type='entangle', qubit1=0, qubit2=1)
    
    # Get structure
    structure = system.get_full_structure()
    print(f"\nStructure:")
    print(f"  Levels: {structure['num_levels']}")
    print(f"  Principle: {structure['principle']}")
    print(f"  Base states: {structure['level_0']['num_states']}")
    
    for i, meta_level in enumerate(structure['meta_levels'], 1):
        print(f"  Level {i} operations: {meta_level['num_operations']}")


def example_6_levels():
    """Example with 6 levels."""
    print("\n" + "=" * 70)
    print("Example: 6 Levels (Base + 5 Meta Levels)")
    print("=" * 70)
    
    system = DynamicHierarchicalGeometrySystem(base_dimension=3, num_levels=6)
    
    # Add base states
    for i in range(10):
        system.add_base_state((i * 0.1, i * 0.1, i * 0.1))
    
    # Add operations at all levels
    for level in range(1, 6):
        system.add_operation(
            level=level,
            operation_type='transform',
            level_num=level,
            data=f'level_{level}_data'
        )
    
    # Get structure
    structure = system.get_full_structure()
    print(f"\nStructure:")
    print(f"  Levels: {structure['num_levels']}")
    print(f"  Principle: {structure['principle']}")
    print(f"  Base states: {structure['level_0']['num_states']}")
    
    for i, meta_level in enumerate(structure['meta_levels'], 1):
        print(f"  Level {i} operations: {meta_level['num_operations']}")


def example_comparison():
    """Compare different numbers of levels."""
    print("\n" + "=" * 70)
    print("Comparison: Different Numbers of Levels")
    print("=" * 70)
    
    for num_levels in [1, 2, 3, 4, 5, 6]:
        system = DynamicHierarchicalGeometrySystem(
            base_dimension=3,
            num_levels=num_levels
        )
        
        # Add some base states
        for i in range(3):
            system.add_base_state((i * 0.1, i * 0.1, i * 0.1))
        
        # Add operations at available levels
        for level in range(1, num_levels):
            system.add_operation(level=level, operation_type='test', level_num=level)
        
        structure = system.get_full_structure()
        print(f"\n{num_levels} levels:")
        print(f"  Principle: {structure['principle']}")
        print(f"  Total operations: {sum(m['num_operations'] for m in structure['meta_levels'])}")


if __name__ == '__main__':
    example_3_levels()
    example_5_levels()
    example_6_levels()
    example_comparison()
    
    print("\n" + "=" * 70)
    print("✅ Dynamic hierarchy examples complete!")
    print("=" * 70)
    print("\nYou can now use any number of levels:")
    print("  - 1 level: Just base geometry")
    print("  - 2 levels: Base + 1 meta")
    print("  - 3 levels: Base + 2 meta (current system)")
    print("  - 4, 5, 6... levels: As many as you need!")
    print("=" * 70)

