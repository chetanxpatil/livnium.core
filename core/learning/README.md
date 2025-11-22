# Learning System

Reward-based learning mechanisms.

## What is "Learning"?

**"Learning"** here refers to **reinforcement through geometric feedback**, not traditional gradient descent:

- **No neural networks**: No backpropagation, no weight updates
- **Reward-only**: Positive reinforcement only (no punishment)
- **Geometric feedback**: Rewards deepen correct "basins" (attractors) in the energy landscape
- **Physics-based**: Learning emerges from geometric structure, not statistical optimization

This is fundamentally different from machine learning - it's more like **shaping a landscape** where correct solutions become deeper valleys that the system naturally falls into.

## Contents

- **`reward_system.py`**: Reward calculation and distribution

## Purpose

This module provides:
- **Reward calculation**: Computes rewards based on outcomes (e.g., correct NLI classification)
- **Reward distribution**: Propagates rewards through the geometric structure
- **Basin reinforcement**: Deepens correct attractors in the energy landscape
- **Reward-only learning**: Positive reinforcement only (no punishment)

## How It Works

1. **Outcome evaluation**: System produces a result (e.g., classification)
2. **Reward calculation**: If correct, compute reward signal
3. **Geometric propagation**: Reward flows through connected geometric structures
4. **Basin deepening**: Correct patterns become stronger attractors

Used by training pipelines (e.g., NLI training) to reinforce correct behaviors through geometric feedback rather than gradient descent.

