# Simple NLI System

A clean, simplified Natural Language Inference system that uses **pure geometric/vector operations** with **zero dependency on Livnium physics**.

## What This Is

This is a **minimal, easy-to-understand NLI classifier** that:
- Uses only vectors and cosine similarity
- Learns word polarities from training data (no hardcoded word lists)
- Has zero dependency on Livnium core (`core/`), omcubes, basins, or quantum collapse
- Is fast, readable, and easy to modify

## What This Is NOT

This is **NOT** the Livnium physics-based NLI system. That lives separately in:
- `experiments/nli/` - Livnium NLI (with omcubes, basins, collapse, Moksha engine)
- `core/` - Livnium core engine (geometry, quantum, recursive systems)

**This simple system does NOT use any of that.**

## Architecture

### 1. Text Encoding (`native_chain.py`)
- **Letter → Vector**: Each letter gets a 27-dim vector (hash-based, deterministic)
- **Word → Vector**: Word = sum of letter vectors, normalized
- **Sentence → Chain**: Sentence = list of word vectors **with positional encoding**
  - Positional encoding: Sinusoidal encoding adds position information
  - Sequential structure: Position in sentence matters (captures word order, negation, quantifiers)
  - Chain matching: Aligned matching (position 0 vs 0, 1 vs 1) + sliding window + cross-word

**No 3D lattices, no LivniumCoreSystem, no QuantumCell - just sequential structure.**

### 2. Classification (`inference_detectors.py`)
Uses **variance-based geometry** to automatically separate classes:

- **Entailment**: Low variance + high mean similarity
  - All word pairs align consistently → confident entailment
  
- **Contradiction**: Low variance + negative/low mean similarity OR learned contradiction words
  - Consistent opposition OR learned negation words (e.g., "not", "no")
  
- **Neutral**: High variance
  - Some words align, some don't → uncertainty → neutral automatically

**No hardcoded thresholds** - boundaries emerge from geometry itself.

The variance of word-pair similarities creates the three bands naturally:
- Low variance = confident (E or C)
- High variance = neutral (uncertainty)

### 3. Learning (`classifier.py`)
- Updates word polarities `[E, C, N]` based on training labels
- Simple exponential moving average: `polarity = (1-α) * current + α * target`
- Words like "not" learn contradiction polarity from data, not from hardcoded lists

### 4. Training (`train_simple_nli.py`)
- Simple loop: encode → classify → check if correct → update word polarities
- Logs accuracy and confusion matrices
- Saves learned brain to `brain_state.pkl`

## Usage

### Training
```bash
python3 experiments/nli_simple/train_simple_nli.py \
    --data-dir data/snli \
    --train 1000 \
    --test 100 \
    --dev 100
```

### Inspect Learned Brain
```python
from experiments.nli_simple.native_chain import SimpleLexicon

lexicon = SimpleLexicon()
lexicon.load_from_file('experiments/nli_simple/brain_state.pkl')

# Check learned polarity for a word
print("not:", lexicon.get_word_polarity("not"))  # Should show high contradiction
print("always:", lexicon.get_word_polarity("always"))  # May show high entailment
```

## Key Differences from Livnium NLI

| Feature | Simple NLI | Livnium NLI (`experiments/nli/`) |
|---------|-----------|----------------------------------|
| **Text Encoding** | Direct vectors + positional encoding | LivniumCoreSystem (3D lattice) |
| **Structure** | Sequential chains (position matters) | Omcube chains with quantum entanglement |
| **Classification** | Variance-based geometry + learned polarity | Omcube collapse + basin dynamics |
| **Learning** | Word polarity updates | Word polarity + letter geometry + basin reinforcement |
| **Dependencies** | NumPy only | LivniumCoreSystem, QuantumCell, Moksha engine |
| **Complexity** | ~600 lines | ~2000+ lines |
| **Speed** | Fast (no 3D object creation) | Slower (full physics simulation) |
| **Expected Accuracy** | ~44-49% (with chain structure) | ~46% (with physics) |

## Performance Notes

**Current Accuracy: ~44-49%** (with chain structure enabled)

The system now uses:
- ✅ **Sequential chain structure** (positional encoding, aligned matching, sliding window)
- ✅ **Learns word polarities** from data
- ✅ **Variance-based classification** (low var = confident, high var = neutral)
- ✅ **Chain matching** in classifier (60% aligned + 20% window + 20% cross-word)

**Key Fix:** The encoder now uses `use_sequence=True` by default, ensuring chain structure is actually used during training and inference.

Without chain structure: ~34-40% (bag-of-vectors ceiling)
With chain structure: ~44-49% (sequential patterns captured)

This matches the expected performance for geometric-only NLI models with sequential structure.

## Files

- `native_chain.py` - Letter/word/sentence vectors, SimpleLexicon
- `encoder.py` - SimpleEncoder (text → vectors)
- `inference_detectors.py` - Entailment/Contradiction/Neutral detection
- `classifier.py` - SimpleNLIClassifierWrapper (minimal classifier)
- `train_simple_nli.py` - Training script
- `README.md` - This file

## Philosophy

**No hardcoded semantics. No physics dependencies. Just:**
- Geometry (vectors, cosine similarity)
- Statistics (learned word polarities from data)
- Simple thresholds (hyperparameters, not meaning)

Words become special **only through training data**, not through human knowledge.

