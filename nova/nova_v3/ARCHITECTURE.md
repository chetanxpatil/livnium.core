# Nova v2 Architecture: Livnium Core v1.0

## Overview

**This is the frozen architecture. No more redesigns.**

The system is organized into 3 clean layers:

1. **Layer 0: Core Physics** - Pure physics engine (no tokens, no labels)
2. **Layer 1: Encoding & Heads** - Task-specific encoding and classification
3. **Layer 2: Training Scripts** - Data loading and training loops

## Layer 0: Core Physics (FROZEN)

**Location**: `nova_v2/core/`

**Files**:
- `vector_state.py` - State representation (single vector h ∈ ℝ^D)
- `physics_laws.py` - Core laws (alignment, divergence, tension)
- `vector_collapse_engine.py` - Collapse dynamics

**What it does**:
- Defines vector state `h`
- Implements OM/LO construction rules
- Computes alignment, divergence (0.38 - alignment), tension
- Evolves state through L collapse steps
- Logs trace (alignment_t, divergence_t, tension_t)

**What it does NOT know**:
- "entailment", "neutral", "contradiction"
- "tokens", "English"
- Any task-specific concepts

**Key Law**: `divergence = 0.38 - alignment`

## Layer 1: Encoding & Heads

### Text Encoding

**Location**: `nova_v2/text/`

**Files**:
- `encoder.py` - Task-agnostic text encoder

**What it does**:
- Converts tokens → embeddings → sentence vector
- Simple average pooling

### Task Heads

**Location**: `nova_v2/tasks/`

**SNLI Head** (`tasks/snli/`):
- `encoding_snli.py` - Builds initial state h0 from premise/hypothesis
- `head_snli.py` - Classifies h_final → logits (E, N, C)

**Future Heads**:
- `tasks/dialogue/` - Dialogue encoding and generation head
- `tasks/ramsey/` - Ramsey-specific head

## Layer 2: Training Scripts

**Location**: `nova_v2/training/` and `nova_v2/chat/`

**Files**:
- `training/train_snli_vector.py` - SNLI training
- `chat/test_snli_vector.py` - SNLI testing

**What they do**:
- Load data
- Encode text → initial state h0
- Run collapse → h_final, trace
- Apply head → logits
- Compute loss & optimize
- (Future: run watchdogs on trace)

## Data Flow

### Training Flow

```
SNLI Data
  ↓
Vocabulary Builder
  ↓
Tokenize (premise, hypothesis)
  ↓
SNLIEncoder.build_initial_state()
  → h0 (initial state)
  → v_p (OM vector)
  → v_h (LO vector)
  ↓
VectorCollapseEngine.collapse(h0)
  → h_final (collapsed state)
  → trace (alignment, divergence, tension)
  ↓
SNLIHead(h_final)
  → logits (E, N, C)
  ↓
CrossEntropyLoss(logits, gold_label)
  → loss
  ↓
Backward & Optimize
```

### Physics Computation

```
OM (v_p) and LO (v_h) vectors
  ↓
alignment = cosine_similarity(OM, LO)
  ↓
divergence = 0.38 - alignment
  ↓
tension = |divergence|
```

## Key Principles

1. **Livnium Core = physics engine (no labels, no tasks)**
2. **Everything else = heads attached on top**
3. **Same core for SNLI, dialogue, Ramsey, etc.**
4. **Vector-based (no 3D cells, no hash collisions)**

## What Changed from nova/

### Removed
- ❌ 3D lattice with cells
- ❌ hash(token) → (x, y, z)
- ❌ Token collisions (92%+)
- ❌ Direct SW per cell as signature

### Added
- ✅ Vector-based state `h`
- ✅ Tokens → embeddings → vectors
- ✅ Clean 3-layer architecture
- ✅ Frozen core (no more redesigns)

### Kept
- ✅ Divergence law (0.38 - alignment)
- ✅ OM/LO separation
- ✅ Collapse dynamics
- ✅ Trace logging
- ✅ Conservation-ish behavior

## Adding a New Task

To add a new task (e.g., dialogue):

1. **Create encoding** (`tasks/dialogue/encoding_dialogue.py`):
   ```python
   def build_initial_state(self, context, query):
       # Build h0 from context and query
       return h0, v_context, v_query
   ```

2. **Create head** (`tasks/dialogue/head_dialogue.py`):
   ```python
   def forward(self, h_final):
       # Output next token distribution
       return logits
   ```

3. **Create training script** (`training/train_dialogue_vector.py`):
   ```python
   # Use same VectorCollapseEngine
   # Use same physics laws
   # Just different encoding and head
   ```

**No changes to Layer 0. Ever.**

## File Structure

```
nova_v2/
├── core/                    # Layer 0: Physics (FROZEN)
│   ├── __init__.py
│   ├── vector_state.py
│   ├── physics_laws.py
│   └── vector_collapse_engine.py
├── text/                    # Layer 1: Encoding
│   ├── __init__.py
│   └── encoder.py
├── tasks/                   # Layer 1: Task Heads
│   ├── __init__.py
│   └── snli/
│       ├── __init__.py
│       ├── encoding_snli.py
│       └── head_snli.py
├── training/               # Layer 2: Training
│   ├── __init__.py
│   └── train_snli_vector.py
├── chat/                   # Layer 2: Testing
│   ├── __init__.py
│   └── test_snli_vector.py
├── utils/                  # Utilities
│   ├── __init__.py
│   └── vocab.py
├── data/                  # Data
│   └── snli/
│       ├── snli_1.0_train.jsonl
│       ├── snli_1.0_dev.jsonl
│       └── snli_1.0_test.jsonl
├── README.md
└── ARCHITECTURE.md
```

## Next Steps

1. ✅ **Core is frozen** - no more redesigns
2. ✅ **Architecture is clean** - 3 layers, clear separation
3. 🔄 **Tune hyperparameters** - dim, num_layers, lr, etc.
4. 🔄 **Add watchdogs** - read from trace, not cells
5. 🔄 **Add dialogue head** - same core, different head

## Notes

- This is the **last big conceptual rebuild**
- Next changes should be **tuning**, not **ontology changes**
- The core is **frozen** - no more redesigns
- Watchdogs can read from `trace` (alignment, divergence, tension)
- Future tasks just need new encoding + head, same core

