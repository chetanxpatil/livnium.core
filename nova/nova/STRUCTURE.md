# Clean Livnium Dialogue Structure

## ✅ Completed Restructure

The `nova` folder has been reorganized into a clean, maintainable structure:

```
nova/
├── config/              # Configuration files
│   ├── model_config.yaml
│   └── decoding_config.yaml
│
├── data/                # Raw datasets
│   ├── empathetic_train.csv
│   ├── daily_test.csv
│   └── ...
│
├── training/            # Training scripts
│   ├── train_text_to_geometry.py
│   ├── prepare_empathetic.py
│   └── convert_emotion_csv.py
│
├── core/                # Core geometry components (NON-RECURSIVE)
│   ├── text_to_geometry.py
│   ├── geometric_token_learner.py
│   ├── cluster_decoder.py
│   ├── geometric_transformer.py
│   └── sentence_decoder.py
│
├── model/               # Trained models
│   ├── learned_patterns.json
│   ├── geometric_clusters.json
│   ├── geometric_clusters.pkl
│   └── version.txt
│
├── chat/                # Chat interface
│   ├── chat_demo.py
│   └── reply_generator.py
│
├── archive/             # Archived experimental code
│   ├── recursive/
│   │   └── recursive_text_to_geometry.py
│   └── experimental/
│
└── README.md
```

## ✅ Changes Made

1. **Removed all recursive dependencies** from main pipeline
2. **Organized files** into logical folders
3. **Updated all imports** to match new structure
4. **Created config files** for easy parameter adjustment
5. **Archived experimental code** to `archive/`
6. **Cleaned training script** - removed `--use-recursive` and all recursive code paths

## ✅ Training Command (Non-Recursive)

```bash
python3 nova/training/train_text_to_geometry.py \
  --csv nova/data/empathetic_train.csv \
  --dataset nova \
  --max-dialogues 10000 \
  --lattice-size 5 \
  --collapse-steps 12 \
  --num-clusters 3000 \
  --impulse-scale 0.1
```

## ✅ Chat Command

```bash
python3 nova/chat/chat_demo.py \
  --lattice-size 5 \
  --collapse-steps 12 \
  --impulse-scale 0.1
```

## ✅ What's Clean Now

- **No recursive code** in main pipeline
- **Clear separation** of concerns (training/core/chat)
- **Consistent imports** throughout
- **Config-driven** parameters
- **Maintainable structure** for future development

