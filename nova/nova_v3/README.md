# Nova v3: Livnium Core (Multi-Basin Vector-Based)

**Multi-basin collapse with neutral-aware head.**

## Architecture

### Layer 0: Core Physics
- `core/physics_laws.py` - Core laws (divergence = 0.38 - alignment)
- `core/vector_collapse_engine.py` - Multi-anchor collapse dynamics (E/C/N basins)

**No tokens, no labels, no tasks. Just physics.**

### Layer 1: Encoding & Heads
- `text/encoder.py` - Task-agnostic text encoding
- `tasks/snli/encoding_snli.py` - SNLI-specific encoding (OM/LO)
- `tasks/snli/head_snli.py` - SNLI classification head with neutral anchor + MLP

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
cd nova/nova_v3
python3 training/train_snli_vector.py \
  --snli-train data/snli/snli_1.0_train.jsonl \
  --snli-dev data/snli/snli_1.0_dev.jsonl \
  --dim 256 \
  --batch-size 32 \
  --epochs 5 \
  --strength-entail 0.1 \
  --strength-contra 0.1 \
  --strength-neutral 0.05 \
  --neutral-weight 1.2 \
  --label-smoothing 0.05 \
  --encoder-type geom \
  --output-dir model/snli_v1
```

### 2. Test SNLI

```bash
cd nova/nova_v3
python3 chat/test_snli_vector.py \
  --model-dir model/snli_v1 \
  --snli-test data/snli/snli_1.0_test.jsonl 
```

### Encoders

- `geom` (default): deterministic geometric encoder (no embedding tables). Converts tokens → base-27 signatures → geometric features → projected vectors with fixed norm, optional transformer token interaction + attention pooling.
- `legacy`: mean-pooled learned embeddings for backward compatibility.

Geom knobs (all optional):
- `--geom-disable-transformer` to turn off the token interaction layer
- `--geom-disable-attn-pool` to use masked mean instead of attention pooling
- `--geom-nhead`, `--geom-num-layers`, `--geom-ff-mult`, `--geom-dropout`, `--geom-token-norm-cap` for finer control

## What Changed from nova_v2

- ✅ **Added**: Multi-anchor collapse (entail/contradict/neutral strengths)
- ✅ **Added**: Neutral-aware head (neutral anchor + small MLP)
- ✅ **Added**: Training knobs (class weight, smoothing, neutral oversample)
- ✅ **Kept**: Divergence law (0.38 - alignment), OM/LO encoding, vector state

## What Stays "Livnium"

- Geometry-first reasoning
- Divergence law (`0.38 - alignment`)
- OM/LO directions
- Basins of attraction (now three anchors)
- Collapse traces
- Conservation-ish behavior

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
