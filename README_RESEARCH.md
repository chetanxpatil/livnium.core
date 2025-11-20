# Livnium: A geometric alternative to neural language models

Livnium is an experimental system for representing language using 3D geometric structures instead of embeddings or transformers.

## Core Idea

The system encodes language at the letter level using 3×3×3 geometric structures called "omcubes":

- Letters → 3×3×3 geometric structures (`LetterOmcube`)
- Words → Chains of letter omcubes (`WordChain`)
- Sentences → Chains of word chains (`SentenceChain`)
- Meaning → Emerges from geometric interactions

This creates natural morphological similarity (e.g., "run" and "running" share geometric structure) and compositional semantics without neural networks.

## Architecture

The system uses a Matrix Product State (MPS) architecture where:

1. Each letter is encoded as a 3×3×3 quantum-inspired geometric structure
2. Letters are chained together to form words
3. Words are chained together to form sentences
4. Geometric similarity and quantum-inspired collapse determine semantic relationships

Key components:
- `native_chain.py`: MPS architecture with letter-by-letter encoding
- `inference_detectors.py`: Native logic for entailment/contradiction detection
- `omcube.py`: Quantum-inspired collapse mechanism for 3-way classification
- `train_moksha_nli.py`: Training pipeline with geometric feedback

## Applications

**Natural Language Inference (NLI)**: Classifies premise-hypothesis pairs as Entailment, Contradiction, or Neutral using pure geometric reasoning.

**Ramsey Number Solving**: Geometric basin search for graph problems.

**Quantum Many-Body Physics**: Real DMRG/MPS tensor network methods for physics problems.

## Why This Approach

Current neural language models are black boxes that require massive datasets and computational resources. Livnium offers:

- **Interpretability**: Every decision is traceable through geometric structures
- **Data efficiency**: Learns from structure, not just statistics
- **Lightweight**: Runs on CPU, no GPU required
- **Compositional semantics**: Meaning emerges from atomic letter-level units

## Quick Start

```bash
git clone https://github.com/chetanxpatil/livnium.core.git
cd livnium.core
python3 -m venv .venv
source .venv/bin/activate
pip install numpy

# Train NLI system
python3 experiments/nli/train_moksha_nli.py --clean --train 20000 --test 2000 --dev 2000
```

## Requirements

- Python 3.7+
- numpy
- Optional: numba for JIT compilation

## License

This project is licensed under the Livnium License v1.1 (Fortress Grade) - a proprietary research license. See [LICENSE](LICENSE) for details.

**Permitted**: Personal, non-commercial, research, and educational use  
**Prohibited**: Commercial use, redistribution, AI training, public hosting

For commercial licensing inquiries: chetan12patil@gmail.com

## Documentation

- Core architecture: `core/README.md`
- NLI system: `experiments/nli/DIAGNOSTIC_REPORT.md`
- Project structure: See main [README.md](README.md) for full documentation

## Research Status

This is experimental research software. The system demonstrates functional NLI classification and geometric reasoning, but is not production-ready.

## Contact

Chetan Patil  
Email: chetan12patil@gmail.com

---

*"Information is geometry. Understanding is structure. Intelligence is composition."*

