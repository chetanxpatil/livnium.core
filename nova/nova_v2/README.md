# Nova v2: Livnium Core v1.0 (Vector-Based)

**Clean, frozen architecture. No more redesigns.**

## Architecture

### Layer 0: Core Physics (Frozen)
- `core/physics_laws.py` - Core laws (divergence = 0.38 - alignment)
- `core/vector_collapse_engine.py` - Collapse dynamics

**No tokens, no labels, no tasks. Just physics.**

### Layer 1: Encoding & Heads
- `text/encoder.py` - Task-agnostic text encoding
- `tasks/snli/encoding_snli.py` - SNLI-specific encoding (OM/LO)
- `tasks/snli/head_snli.py` - SNLI classification head

### Layer 2: Training Scripts
- `training/train_snli_vector.py` - SNLI training
- `chat/test_snli_vector.py` - SNLI testing

## Core Principles

1. **Livnium Core = physics engine (no labels, no tasks)**
2. **Everything else = heads attached on top**
3. **Same core for SNLI, dialogue, Ramsey, etc.**
4. **Vector-based (no 3D cells, no hash collisions)**

## Quick Start

### 1. Train SNLI

```bash
cd nova/nova_v2
python3 training/train_snli_vector.py \
  --snli-train data/snli/snli_1.0_train.jsonl \
  --snli-dev data/snli/snli_1.0_dev.jsonl \
  --dim 256 \
  --batch-size 32 \
  --epochs 5 \
  --output-dir model/snli_v1
```

### 2. Test SNLI

```bash
cd nova/nova_v2
python3 chat/test_snli_vector.py \
  --model-dir model/snli_v1 \
  --snli-test data/snli/snli_1.0_test.jsonl 
```

## What Changed from nova/

- ✅ **Removed**: 3D lattice, hash collisions, cell-based geometry
- ✅ **Added**: Vector-based state, clean architecture
- ✅ **Kept**: Divergence law (0.38 - alignment), OM/LO separation, collapse dynamics

## What Stays "Livnium"

- Geometry-first reasoning
- Divergence law (`0.38 - alignment`)
- OM/LO directions
- Basins of attraction
- Collapse
- Conservation-ish behavior
- Watchdogs (can read from trace)

## Future Tasks

To add a new task (e.g., dialogue):

1. Create `tasks/dialogue/encoding_dialogue.py` (builds h0 from context)
2. Create `tasks/dialogue/head_dialogue.py` (outputs next token distribution)
3. Create `training/train_dialogue_vector.py` (uses same core)

**No changes to Layer 0. Ever.**

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

## Notes

- This is the **last big conceptual rebuild**
- Next changes should be **tuning**, not **ontology changes**
- The core is **frozen** - no more redesigns
