# Meta-Learning System

Self-observation and adaptation mechanisms.

## What is the Meta-Layer?

**The system that watches itself and adapts.**

This layer provides:
- **Self-observation**: System monitors its own behavior (rotation history, state drift, alignment)
- **Anomaly detection**: Identifies unusual patterns (SW violations, face exposure bounds, duplicate coordinates)
- **Calibration**: Automatically adjusts parameters (repairs SW mismatches, corrects class-count drift)
- **Introspection**: System analyzes its own state and behavior patterns

All interactions are **read-based** (no structural mutations) - the meta-layer observes and calibrates, but never corrupts the lattice structure.

## Contents

- **`meta_observer.py`**: Observes system behavior, tracks state history, detects drift
- **`introspection.py`**: Self-analysis capabilities, pattern recognition
- **`anomaly_detector.py`**: Detects unusual patterns (SW violations, bounds issues)
- **`calibration_engine.py`**: Calibrates system parameters, auto-repairs mismatches

## Purpose

This module provides:
- **Self-observation**: System monitors its own behavior (invariance drift, reflection, alignment)
- **Anomaly detection**: Identifies unusual patterns (SW correctness, face exposure bounds, coordinate duplicates)
- **Calibration**: Automatically adjusts parameters (recalculates SW, repairs mismatches)
- **Introspection**: System can analyze its own state and behavior patterns

## Integration

The meta-layer integrates cleanly with:
- **LivniumCoreSystem**: Uses valid APIs (get_total_symbolic_weight, get_class_counts, etc.)
- **Read-only observation**: Never mutates forbidden structure
- **Safe calibration**: Only modifies allowed fields (e.g., cell.symbolic_weight)

Enables adaptive behavior and self-improvement without breaking the geometric structure.

## Future Directions

Potential expansions for the meta-layer:

- **Integrate with Reward + Memory Layer**: Connect meta-observation with reward signals and memory persistence
- **Attach "haircut" pruning logic**: Use meta-detection to identify and prune low-value structures
- **Add meta-level reward shaping**: Use introspection to dynamically adjust reward signals
- **Implement true recursive self-modeling**: System models its own behavior recursively across hierarchy levels

These expansions would enable deeper self-awareness and adaptive optimization.

