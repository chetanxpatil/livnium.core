# Quick Start Guide

## What Was Created

A clean, frozen architecture for Livnium Core v1.0:

- ✅ **Layer 0: Core Physics** (FROZEN) - Pure physics engine
- ✅ **Layer 1: Encoding & Heads** - Task-specific components
- ✅ **Layer 2: Training Scripts** - Data loading and training

## Structure

```
nova_v2/
├── core/              # Layer 0: Physics (FROZEN)
│   ├── vector_state.py
│   ├── physics_laws.py
│   └── vector_collapse_engine.py
├── text/              # Layer 1: Encoding
│   └── encoder.py
├── tasks/             # Layer 1: Task Heads
│   └── snli/
│       ├── encoding_snli.py
│       └── head_snli.py
├── training/          # Layer 2: Training
│   └── train_snli_vector.py
├── chat/              # Layer 2: Testing
│   └── test_snli_vector.py
└── utils/             # Utilities
    └── vocab.py
```

## Installation

Make sure you have PyTorch installed:

```bash
pip install torch numpy tqdm
```

## Training

Train SNLI model:

```bash
cd nova/nova_v2

python3 training/train_snli_vector.py \
  --snli-train data/snli/snli_1.0_train.jsonl \
  --snli-dev data/snli/snli_1.0_dev.jsonl \
  --max-samples 10000 \
  --dim 256 \
  --num-layers 6 \
  --batch-size 32 \
  --epochs 10 \
  --lr 1e-3 \
  --output-dir model/snli_v1
```

## Testing

Test trained model:

```bash
cd nova/nova_v2

python3 chat/test_snli_vector.py \
  --model-dir model/snli_v1 \
  --snli-test data/snli/snli_1.0_test.jsonl \
  --max-samples 1000
```

## Key Features

1. **Vector-based** - No 3D cells, no hash collisions
2. **Clean architecture** - 3 layers, clear separation
3. **Frozen core** - No more redesigns
4. **Task-agnostic core** - Same physics for all tasks

## Core Law

**Divergence Law**: `divergence = 0.38 - alignment`

- `divergence < 0` → entailment
- `divergence ≈ 0` → neutral
- `divergence > 0` → contradiction

## What's Different from nova/

- ❌ Removed: 3D lattice, hash collisions, cell-based geometry
- ✅ Added: Vector-based state, clean architecture
- ✅ Kept: Divergence law, OM/LO separation, collapse dynamics

## Next Steps

1. Train on full SNLI dataset
2. Tune hyperparameters (dim, num_layers, lr)
3. Add watchdogs (read from trace)
4. Add dialogue head (same core, different head)

## Notes

- This is the **last big conceptual rebuild**
- Next changes should be **tuning**, not **ontology changes**
- The core is **frozen** - no more redesigns

