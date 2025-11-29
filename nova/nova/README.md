# Nova Livnium Dialogue Engine (Non-Recursive)

Clean, maintainable geometric dialogue system using pure Livnium physics.

## Structure

```
nova/
├── config/          # Configuration files (YAML)
├── data/            # Raw datasets (CSV)
├── training/         # Training scripts
├── core/            # Core geometry components (non-recursive)
├── model/           # Trained models (JSON, PKL)
├── chat/            # Chat interface and reply generation
├── analysis/        # Analysis and visualization tools
├── utils/           # Utility functions
└── archive/         # Archived experimental code
```

## Quick Start

### 1. Prepare Data

If you have the emotion-emotion_69k.csv file:

```bash
python3 nova/training/convert_emotion_csv.py
```

Or prepare EmpatheticDialogues dataset:

```bash
python3 nova/training/prepare_empathetic.py
```

### 2. Train Model

#### Option A: Dialogue Training (Full Conversation)

Train the geometric dialogue model:

```bash
python3 nova/training/train_text_to_geometry.py \
  --csv nova/data/empathetic_train.csv \
  --dataset nova \
  --max-dialogues 10000 \
  --lattice-size 5 \
  --collapse-steps 12 \
  --num-clusters 3000 \
  --impulse-scale 0.1 \
  --output-dir nova/model
```

#### Option B: SNLI Phase 1 Training (E/C/N Classification)

Train Phase 1 SNLI using **existing Nova training pipeline** with **divergence law** and **symmetry breaking** (outputs ONLY: "entailment", "contradiction", or "neutral"):

```bash
python3 nova/training/train_snli_phase1.py \
  --snli-train nova/data/snli/snli_1.0_train.jsonl \
  --max-samples 10000 \
  --lattice-size 5 \
  --collapse-steps 12 \
  --num-clusters 2000 \
  --impulse-scale 0.1 \
  --output-dir nova/model/snli_phase1_divergence
```

**What This Does:**
- Extracts signatures with **divergence primitive** (alignment, divergence, fracture)
- Applies **symmetry breaking** for angular variation (SNLI only)
- Computes divergence statistics per label
- Auto-tunes thresholds based on training data
- Saves divergence classifier with tuned parameters

**Phase 1 Discipline:**
- Uses the **same Nova model structure** (TextToGeometry, GeometricTokenLearner, ClusterDecoder)
- Trains on SNLI data where labels (E/C/N) are part of the vocabulary
- Output must be EXACTLY one word: "entailment", "contradiction", or "neutral"
- Reward: Correct label
- Punish: Wrong label OR any extra words

**Training Output:**
- `geometric_clusters.json` - Cluster token distributions
- `geometric_clusters.pkl` - KMeans model
- `learned_patterns.json` - Markov grammar
- `divergence_classifier.pkl` - Physics-based classifier with tuned thresholds

---

### Testing SNLI Phase 1

#### Test Mode A: Cluster + Grammar (Default)

Tests using the unsupervised cluster+grammar pipeline:

```bash
python3 nova/chat/test_snli_phase1.py \
  --model-dir nova/model/snli_phase1_divergence \
  --max-samples 100
```

#### Test Mode B: Pure Physics (Divergence Law)

Tests using **direct physics** (divergence law), bypassing cluster+grammar:

```bash
python3 nova/chat/test_snli_phase1.py \
  --model-dir nova/model/snli_phase1_divergence \
  --physics \
  --max-samples 100
```

**Expected Results:**
- **Mode A (Cluster+Grammar)**: ~25-30% accuracy (unsupervised pipeline)
- **Mode B (Pure Physics)**: ~40-55% accuracy (direct divergence law)

If Mode B performs better, the divergence law is correct and the issue is in the unsupervised pipeline.

#### Test on Dev Set

```bash
# Cluster + Grammar mode
python3 nova/chat/test_snli_phase1.py \
  --model-dir nova/model/snli_phase1_divergence \
  --dev \
  --max-samples 1000

# Pure Physics mode
python3 nova/chat/test_snli_phase1.py \
  --model-dir nova/model/snli_phase1_divergence \
  --dev \
  --physics \
  --max-samples 1000
```

**Test Parameters:**
- `--model-dir`: Path to trained model directory
- `--test-file`: Path to custom SNLI JSONL file (optional)
- `--dev`: Test on dev set (`nova/data/snli/snli_1.0_dev.jsonl`)
- `--physics`: Use pure physics mode (divergence law directly)
- `--max-samples`: Maximum number of test samples

**Note:** Phase 1 uses the same model files as regular Nova training (`geometric_clusters.json`, `learned_patterns.json`). The decoder is set to `phase1_mode=True` to enforce E/C/N-only output. Symmetry breaking is automatically enabled for SNLI (not used by law extractor).

**Training Parameters:**
- `--csv`: Path to CSV dataset file
- `--dataset`: Dataset type (`nova`, `cornell`, `empathetic`)
- `--max-dialogues`: Maximum dialogues to process (default: all)
- `--lattice-size`: Geometry cube size (3, 5, 7, 9... must be odd)
- `--collapse-steps`: Number of collapse iterations (12-20 recommended)
- `--num-clusters`: K-Means cluster count (1000-3000 recommended)
- `--impulse-scale`: Character impulse strength (0.1 recommended)
- `--output-dir`: Output directory for models (default: `nova/model`)

### 3. Run Chat Demo

Start interactive chat:

```bash
python3 nova/chat/chat_demo.py \
  --lattice-size 5 \
  --collapse-steps 12 \
  --impulse-scale 0.1 \
  --temperature 0.7 \
  --repetition-penalty 0.1 \
  --context-alpha 0.4 \
  --context-beta 0.6
```

**Chat Parameters:**
- `--lattice-size`: Must match training lattice size (auto-detected from model)
- `--collapse-steps`: Collapse iterations (default: 12)
- `--impulse-scale`: Impulse strength (default: 0.1)
- `--temperature`: Sampling temperature 0.0-1.0 (default: 0.7, higher = more random)
- `--repetition-penalty`: Penalty for repeating words 0.0-1.0 (default: 0.1)
- `--context-alpha`: Current query weight (default: 0.4)
- `--context-beta`: Context history weight (default: 0.6)

### 4. Quick Training Script

Use the quick training script:

```bash
bash nova/training/quick_train.sh [max_dialogues] [collapse_steps] [output_dir]
```

Example:
```bash
bash nova/training/quick_train.sh 1000 12 nova/model
```

## Core Pipeline

1. **Text → Geometry**: `core/text_to_geometry.py`
   - Tokenize → Hash → Inject impulses → Collapse → Signature

2. **Clustering**: `core/geometric_token_learner.py`
   - K-Means clustering on signatures
   - Cluster → Token distribution mapping

3. **Generation**: `core/cluster_decoder.py`
   - Signature → Cluster ID
   - Cluster → Vocabulary
   - Markov grammar → Word ordering

4. **Reply**: `chat/reply_generator.py`
   - Context blending (alpha/beta)
   - Temperature sampling
   - Repetition penalty

## Configuration

Edit configuration files to adjust parameters:

- `config/model_config.yaml` - Model and training parameters
- `config/decoding_config.yaml` - Generation and decoding parameters

## Model Files

After training, model files are saved to `nova/model/`:

- `learned_patterns.json` - Markov grammar patterns
- `geometric_clusters.json` - Cluster token distributions
- `geometric_clusters.pkl` - K-Means model

## Example Workflows

### Workflow A: Dialogue Training

```bash
# 1. Convert emotion CSV to training format
python3 nova/training/convert_emotion_csv.py

# 2. Train model (10k dialogues, 5x5x5 lattice, 3000 clusters)
python3 nova/training/train_text_to_geometry.py \
  --csv nova/data/empathetic_train.csv \
  --dataset nova \
  --max-dialogues 10000 \
  --lattice-size 5 \
  --collapse-steps 12 \
  --num-clusters 3000 \
  --impulse-scale 0.1

# 3. Start chat (auto-detects lattice size from model)
python3 nova/chat/chat_demo.py \
  --collapse-steps 12 \
  --impulse-scale 0.1 \
  --temperature 0.7
```

### Workflow B: SNLI Phase 1 Training & Testing

```bash
# 1. Train SNLI Phase 1 with divergence law
python3 nova/training/train_snli_phase1.py \
  --snli-train nova/data/snli/snli_1.0_train.jsonl \
  --max-samples 10000 \
  --lattice-size 5 \
  --collapse-steps 12 \
  --num-clusters 2000 \
  --impulse-scale 0.1 \
  --output-dir nova/model/snli_phase1_divergence

# 2. Test with Cluster+Grammar mode (default)
python3 nova/chat/test_snli_phase1.py \
  --model-dir nova/model/snli_phase1_divergence \
  --dev \
  --max-samples 100

# 3. Test with Pure Physics mode (divergence law directly)
python3 nova/chat/test_snli_phase1.py \
  --model-dir nova/model/snli_phase1_divergence \
  --dev \
  --physics \
  --max-samples 100

# Compare results:
# - Mode A (Cluster+Grammar): ~25-30% accuracy
# - Mode B (Pure Physics): ~40-55% accuracy (if divergence law is working)
```

## Troubleshooting

**Error: "No module named 'nova'"**
- Make sure you're running from the project root directory
- Check that `nova/` folder exists in the project root

**Error: "X has 27 features, but KMeans is expecting 125 features"**
- Lattice size mismatch between training and chat
- Use `--lattice-size` to match your trained model, or let chat_demo.py auto-detect

**Model not found errors**
- Ensure you've trained a model first
- Check that `nova/model/geometric_clusters.json` exists

**SNLI Phase 1: Low accuracy (~25%)**
- This is expected for Mode A (Cluster+Grammar)
- Try Mode B (Pure Physics): `--physics` flag
- If Mode B is better, the divergence law is working but the unsupervised pipeline needs improvement

**SNLI Phase 1: "Divergence classifier not found"**
- Ensure you trained with the divergence-enabled script
- Check that `divergence_classifier.pkl` exists in model directory
- Re-run training to generate divergence statistics and classifier

## Archive

Experimental and recursive code is archived in `archive/`:
- `archive/recursive/` - Recursive geometry experiments
- `archive/experimental/` - Other experimental code

## License

See project root LICENSE file.
