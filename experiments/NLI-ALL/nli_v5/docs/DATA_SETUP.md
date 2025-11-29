# Data Setup Guide

## SNLI Dataset

The geometry-first training script needs SNLI (Stanford Natural Language Inference) data.

### Option 1: Download SNLI Data

1. **Download SNLI 1.0:**
   - URL: https://nlp.stanford.edu/projects/snli/
   - Download: `snli_1.0.zip`

2. **Extract and place files:**
   ```bash
   # Create data directory
   mkdir -p experiments/nli_simple/data
   
   # Extract snli_1.0_train.jsonl, snli_1.0_test.jsonl, snli_1.0_dev.jsonl
   # Place them in: experiments/nli_simple/data/
   ```

3. **Verify:**
   ```bash
   ls experiments/nli_simple/data/
   # Should show:
   # snli_1.0_train.jsonl
   # snli_1.0_test.jsonl
   # snli_1.0_dev.jsonl
   ```

### Option 2: Use Existing Patterns

If you have existing pattern files, the script will try to load examples from them:

```bash
# Script will check: experiments/nli_v5/patterns/patterns.json
# And extract premise/hypothesis from pattern metadata if available
```

### Option 3: Synthetic Examples (Demo Only)

If no data is found, the script creates synthetic examples for demonstration:

- Simple entailment/contradiction/neutral examples
- Good for testing the geometry-first approach
- Not suitable for real training

## Quick Test Without Data

To test the geometry-first approach without SNLI data:

```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/training/train_geometry_first.py --train 50
```

This will use synthetic examples and show you how geometry-first classification works.

