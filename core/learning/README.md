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

## Philosophy: Reward to Let Structure Emerge

**Reward the good. Don't punish wrong answers.**

- **Reward correct behaviors**: Deepen the basins that work (tension drops get large rewards)
- **Let incorrect patterns fade naturally**: Bad outcomes get no reward (just absence of positive signal)
- **Small exploration cost**: A tiny operational cost (0.05) for trying new actions, not punishment for being wrong
- **Trust the geometry**: The structure will emerge from positive reinforcement alone
- **No forced correction**: Don't fight the system - guide it gently

**What the code does:**
- ✅ Tension drops → Large positive rewards (2.0 × tension_drop × 10)
- ✅ Staying in good basins → Stability rewards (0.1)
- ⚠️ Exploration actions → Small cost (0.05) - operational, not punitive
- ✅ Tension increases → No reward (just 0, or small exploration cost)

This approach lets the geometric structure naturally organize itself around successful patterns, creating stable attractors without destructive interference.

Used by training pipelines (e.g., NLI training) to reinforce correct behaviors through geometric feedback rather than gradient descent.

## Future Directions

Potential improvements and deeper analysis:

- **Stability of reward curves**: Analyze long-term reward behavior and convergence patterns
- **Long-run drift**: Monitor reward signals over extended training periods
- **Reward saturation/explosion**: Ensure rewards remain bounded and effective
- **Basin deepening rate**: Tune how quickly correct attractors strengthen
- **Geometric attractor shaper**: Optimize reward system to properly shape the energy landscape for Livnium

These improvements would enhance the learning system's ability to shape geometric attractors effectively.

