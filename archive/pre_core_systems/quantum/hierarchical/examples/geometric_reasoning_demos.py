#!/usr/bin/env python3
"""
Real-World Geometric Reasoning Demos

Demonstrates LIVNIUM's geometric symbolic logic capabilities:
1. Collapse Detection - Detect unstable geometric states
2. Anomaly Detection - Find geometric anomalies
3. Stability Analysis - Stable vs unstable states
4. Multi-Level Structural Reasoning - Deep hierarchical analysis
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum.hierarchical.geometry.hierarchy_v2 import (
    HierarchyV2System, OperationType
)


def demo_collapse_detection():
    """
    Demo: Collapse Detection
    
    Detects when geometric states become unstable and "collapse"
    to simpler configurations.
    """
    print("=" * 70)
    print("DEMO: Collapse Detection")
    print("=" * 70)
    
    system = HierarchyV2System(base_dimension=3, num_levels=10)
    
    # Create geometric structure
    print("\n1. Creating geometric structure...")
    for i in range(100):
        # Create states in a pattern
        x = np.sin(i * 0.1) * 0.5
        y = np.cos(i * 0.1) * 0.5
        z = i * 0.01
        system.add_base_state((x, y, z))
    
    print(f"   Created {len(system.base_geometry.states)} base states")
    
    # Apply operations that might cause collapse
    print("\n2. Applying potentially destabilizing operations...")
    
    # High-level transform that might collapse structure
    system.register_operation(
        OperationType.TRANSFORM, level=8,
        parameters={'intensity': 10.0, 'type': 'destabilizing'},
        description='High-intensity transform (potential collapse)',
        propagates_down=True
    )
    
    # Check for collapse indicators
    print("\n3. Analyzing for collapse indicators...")
    
    base_states = system.base_geometry.states
    collapse_indicators = []
    
    # Check for coordinate clustering (collapse to single point)
    coordinates = [state.coordinates for state in base_states]
    coord_array = np.array(coordinates)
    
    # Calculate variance (low variance = collapse)
    variance = np.var(coord_array, axis=0)
    total_variance = np.sum(variance)
    
    if total_variance < 0.01:
        collapse_indicators.append("Low variance - possible collapse")
    
    # Check for amplitude concentration
    amplitudes = [abs(state.amplitude) for state in base_states]
    max_amplitude = max(amplitudes) if amplitudes else 0
    avg_amplitude = np.mean(amplitudes) if amplitudes else 0
    
    if max_amplitude > avg_amplitude * 10:
        collapse_indicators.append("Amplitude concentration - possible collapse")
    
    print(f"   Variance: {total_variance:.6f}")
    print(f"   Max amplitude: {max_amplitude:.4f}")
    print(f"   Average amplitude: {avg_amplitude:.4f}")
    
    if collapse_indicators:
        print(f"\n   ⚠️  COLLAPSE DETECTED:")
        for indicator in collapse_indicators:
            print(f"      - {indicator}")
    else:
        print(f"\n   ✅ Structure is stable (no collapse detected)")


def demo_anomaly_detection():
    """
    Demo: Anomaly Detection
    
    Finds geometric states that deviate from expected patterns.
    """
    print("\n" + "=" * 70)
    print("DEMO: Anomaly Detection")
    print("=" * 70)
    
    system = HierarchyV2System(base_dimension=3, num_levels=8)
    
    # Create normal pattern
    print("\n1. Creating normal geometric pattern...")
    normal_states = []
    for i in range(50):
        # Normal pattern: smooth progression
        x = i * 0.1
        y = i * 0.1
        z = i * 0.1
        state = system.add_base_state((x, y, z))
        normal_states.append((x, y, z))
    
    # Add anomalies (outliers)
    print("2. Injecting anomalies...")
    anomalies = [
        (10.0, 10.0, 10.0),  # Far outlier
        (-5.0, -5.0, -5.0),  # Negative outlier
        (0.0, 0.0, 100.0),   # Z-axis outlier
    ]
    
    for anomaly_coords in anomalies:
        system.add_base_state(anomaly_coords)
        print(f"   Added anomaly at {anomaly_coords}")
    
    # Detect anomalies using multi-level analysis
    print("\n3. Detecting anomalies using hierarchical analysis...")
    
    all_coords = [state.coordinates for state in system.base_geometry.states]
    coord_array = np.array(all_coords)
    
    # Calculate statistics
    mean_coords = np.mean(coord_array, axis=0)
    std_coords = np.std(coord_array, axis=0)
    
    print(f"   Mean coordinates: {mean_coords}")
    print(f"   Std deviation: {std_coords}")
    
    # Find outliers (beyond 3 standard deviations)
    threshold = 3.0
    detected_anomalies = []
    
    for i, coords in enumerate(all_coords):
        z_scores = np.abs((np.array(coords) - mean_coords) / (std_coords + 1e-10))
        max_z_score = np.max(z_scores)
        
        if max_z_score > threshold:
            detected_anomalies.append({
                'index': i,
                'coordinates': coords,
                'z_score': max_z_score
            })
    
    print(f"\n   Detected {len(detected_anomalies)} anomalies:")
    for anomaly in detected_anomalies:
        print(f"      Index {anomaly['index']}: {anomaly['coordinates']} (z-score: {anomaly['z_score']:.2f})")
    
    # Use meta-level operations to analyze anomalies
    print("\n4. Applying meta-level analysis...")
    system.register_operation(
        OperationType.FILTER, level=5,
        parameters={'filter_type': 'anomaly', 'threshold': threshold},
        description='Filter anomalies at meta-level 5',
        propagates_down=True
    )
    
    print("   ✅ Anomaly detection complete using hierarchical reasoning")


def demo_stability_analysis():
    """
    Demo: Stability Analysis
    
    Analyzes which geometric states are stable vs unstable.
    """
    print("\n" + "=" * 70)
    print("DEMO: Stability Analysis")
    print("=" * 70)
    
    system = HierarchyV2System(base_dimension=3, num_levels=12)
    
    # Create mix of stable and unstable states
    print("\n1. Creating geometric states...")
    
    stable_states = []
    unstable_states = []
    
    # Stable states: low energy, well-distributed
    for i in range(30):
        # Stable: smooth, predictable pattern
        x = np.sin(i * 0.2) * 0.3
        y = np.cos(i * 0.2) * 0.3
        z = i * 0.05
        state = system.add_base_state((x, y, z))
        stable_states.append(state)
    
    # Unstable states: high energy, concentrated
    for i in range(10):
        # Unstable: concentrated, high amplitude
        x = 0.0 + np.random.normal(0, 0.01)
        y = 0.0 + np.random.normal(0, 0.01)
        z = 0.0 + np.random.normal(0, 0.01)
        state = system.add_base_state((x, y, z), amplitude=5.0+0j)
        unstable_states.append(state)
    
    print(f"   Created {len(stable_states)} stable states")
    print(f"   Created {len(unstable_states)} unstable states")
    
    # Analyze stability using multi-level reasoning
    print("\n2. Analyzing stability using hierarchical levels...")
    
    # Level 1: Basic stability check
    stable_count = 0
    unstable_count = 0
    
    for state in system.base_geometry.states:
        # Stability metric: amplitude and coordinate spread
        amplitude = abs(state.amplitude)
        coords = np.array(state.coordinates)
        distance_from_origin = np.linalg.norm(coords)
        
        # Stable: moderate amplitude, distributed coordinates
        # Unstable: high amplitude, concentrated coordinates
        if amplitude > 3.0 and distance_from_origin < 0.1:
            unstable_count += 1
        else:
            stable_count += 1
    
    print(f"   Level 1 analysis:")
    print(f"      Stable: {stable_count}")
    print(f"      Unstable: {unstable_count}")
    
    # Level 5: Meta-stability analysis
    print("\n3. Meta-level stability analysis (Level 5)...")
    system.register_operation(
        OperationType.AGGREGATE, level=5,
        parameters={'analysis_type': 'stability', 'levels': [1, 2, 3, 4]},
        description='Aggregate stability analysis across multiple levels',
        propagates_down=True
    )
    
    # Level 10: Deep stability reasoning
    print("4. Deep stability reasoning (Level 10)...")
    system.register_operation(
        OperationType.COMPOSE, level=10,
        parameters={'compose_type': 'stability_chain'},
        description='Compose stability analysis across all levels',
        propagates_down=True
    )
    
    print("   ✅ Multi-level stability analysis complete")
    
    # Summary
    print(f"\n   Summary:")
    print(f"      Total states: {len(system.base_geometry.states)}")
    print(f"      Stable: {stable_count} ({100*stable_count/len(system.base_geometry.states):.1f}%)")
    print(f"      Unstable: {unstable_count} ({100*unstable_count/len(system.base_geometry.states):.1f}%)")


def demo_multi_level_reasoning():
    """
    Demo: Multi-Level Structural Reasoning
    
    Demonstrates deep hierarchical reasoning across many levels.
    """
    print("\n" + "=" * 70)
    print("DEMO: Multi-Level Structural Reasoning")
    print("=" * 70)
    
    # Create deep hierarchy
    system = HierarchyV2System(base_dimension=3, num_levels=20)
    
    print(f"\n1. Created {system.num_levels}-level hierarchy")
    
    # Add base structure
    print("2. Building base geometric structure...")
    for i in range(200):
        # Complex geometric pattern
        t = i * 0.05
        x = np.sin(t) * np.cos(t * 2)
        y = np.cos(t) * np.sin(t * 2)
        z = t * 0.1
        system.add_base_state((x, y, z))
    
    print(f"   Created {len(system.base_geometry.states)} base states")
    
    # Apply operations at multiple levels for complex reasoning
    print("\n3. Applying multi-level operations for deep reasoning...")
    
    operation_levels = [1, 3, 5, 7, 10, 12, 15, 18]
    
    for level in operation_levels:
        system.register_operation(
            OperationType.TRANSFORM, level=level,
            parameters={
                'level': level,
                'reasoning_depth': level,
                'type': f'level_{level}_reasoning'
            },
            description=f'Structural reasoning at level {level}',
            propagates_down=True
        )
        print(f"   Level {level}: Applied reasoning operation")
    
    # Get propagation analysis
    print("\n4. Analyzing propagation effects...")
    history = system.get_propagation_history()
    
    print(f"   Total operations: {len(history['propagations'])}")
    print(f"   Total propagations: {sum(p['total_levels_affected'] for p in history['propagations'])}")
    
    # Show level graph
    print("\n5. Level Graph Structure:")
    graph_text = system.get_level_graph(format='text')
    # Show first few levels
    lines = graph_text.split('\n')[:15]
    for line in lines:
        print(f"   {line}")
    print("   ...")
    
    # System summary
    print("\n6. System Summary:")
    info = system.get_full_system_info()
    print(f"   Hierarchy levels: {info['hierarchy']['num_levels']}")
    print(f"   Base states: {info['hierarchy']['base_states']}")
    print(f"   Total operations: {info['operation_registry']['total_operations']}")
    print(f"   Key insight: {info['insight']}")


if __name__ == '__main__':
    demo_collapse_detection()
    demo_anomaly_detection()
    demo_stability_analysis()
    demo_multi_level_reasoning()
    
    print("\n" + "=" * 70)
    print("✅ Real-World Geometric Reasoning Demos Complete!")
    print("=" * 70)
    print("\nThese demos show LIVNIUM's capabilities:")
    print("  - Collapse Detection: Find unstable geometric states")
    print("  - Anomaly Detection: Identify geometric outliers")
    print("  - Stability Analysis: Classify stable vs unstable")
    print("  - Multi-Level Reasoning: Deep hierarchical analysis")
    print("\nThis is geometric symbolic logic - not quantum simulation!")
    print("=" * 70)

