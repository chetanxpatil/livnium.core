# LIVNIUM
## Quantum-Inspired Geometric Computing & Native Language Understanding

**A breakthrough computational architecture combining quantum-inspired geometry, native language processing, and tensor network physics.**

---

## ⚠️ IMPORTANT: Personal/Research Use Only

**This software is provided for PERSONAL, NON-COMMERCIAL, RESEARCH, and EDUCATIONAL PURPOSES ONLY.**

- ✅ **Permitted**: Personal study, research, education, non-commercial use
- ❌ **Prohibited**: Commercial use, redistribution, derivative works, AI training, public hosting
- 🔒 **Commercial Rights**: Reserved exclusively by the Owner

**For commercial licensing inquiries**, please contact: chetan12patil@gmail.com

See [LICENSE](LICENSE) for full terms and conditions.

---

## 🌟 What is LIVNIUM?

LIVNIUM is a **pure native computing system** that processes language and information through geometric quantum-inspired structures—**without transformers, embeddings, or neural networks**. It represents a fundamental departure from conventional deep learning approaches.

### Core Innovation: The Livnium Phoneme Layer

**Letter-by-letter chained omcubes** form the atomic foundation:

- **Letters** → Individual 3×3×3 quantum geometries (`LetterOmcube`)
- **Words** → Chains of entangled letters (`WordChain`)
- **Sentences** → Chains of word-chains (`SentenceChain`)

This creates **natural morphological similarity**, **compositional meaning**, and **stable memory** through shared letter-level learning—like DNA built from nucleotide chains.

---

## 🚀 Key Features

### ✅ Pure Native Architecture
- **Zero Transformers**: No BERT, GPT, or any neural language models
- **Zero Embeddings**: No pre-trained word vectors or sentence transformers
- **Zero External Dependencies**: Pure geometric physics and native logic
- **100% Interpretable**: Every decision is traceable through geometric structures

### ✅ Quantum-Inspired Geometry
- **3×3×3 Omcubes**: Each letter/word encoded as a quantum-inspired geometric structure
- **Matrix Product States (MPS)**: Sentence-level entanglement through chained omcubes
- **Quantum Collapse**: 3-way decision making (Entailment/Contradiction/Neutral)
- **Basin Reinforcement**: Physics-based learning through geometric feedback

### ✅ Research-Grade Systems
- **Livnium Core**: Real tensor network physics (DMRG/MPS) for quantum many-body problems
- **Islands System**: Quantum-inspired information processing (105-500+ qubit-analogues)
- **Hierarchical System**: Geometry-in-geometry architecture (5000+ qubit-analogue capacity)
- **NLI System**: Natural Language Inference using pure geometric reasoning

---

## 📐 Architecture Overview

### The Livnium Phoneme Layer (Letter-by-Letter Encoding)

```
Letter → LetterOmcube (3×3×3 geometry)
  ↓
Word → WordChain (chained LetterOmcubes)
  ↓
Sentence → SentenceChain (chained WordChains)
  ↓
Meaning → Emergent from geometric interactions
```

**Why This Works:**
- **Morphological Understanding**: "run" and "running" share letters → geometric overlap
- **Stable Memory**: Letter-level learning shared across entire language
- **Compositional Semantics**: Word meaning emerges from letter chains
- **Zero Magic**: Everything is reversible, hash-based, structural

### System Components

1. **`native_chain.py`**: Core MPS architecture
   - `LetterOmcube`: Atomic letter geometry
   - `WordChain`: Letter entanglement
   - `SentenceChain`: Word-level chains
   - `GlobalLexicon`: Persistent letter-level memory

2. **`inference_detectors.py`**: Native logic engine
   - Lexical overlap detection
   - Negation detection
   - Semantic gap analysis
   - Double negative handling

3. **`omcube.py`**: Quantum collapse engine
   - 3-way classification (E/C/N)
   - Basin reinforcement learning
   - Cross-omcube coupling
   - Geometric feedback

4. **`train_moksha_nli.py`**: Complete training pipeline
   - Native Chain encoding
   - Quantum collapse classification
   - Moksha convergence detection
   - Reward-only learning

---

## 🔬 Research Applications

### Natural Language Inference (NLI)
- **Task**: Classify premise-hypothesis pairs as Entailment, Contradiction, or Neutral
- **Approach**: Pure geometric reasoning with zero neural networks
- **Status**: Functional 3-way collapse with physics-based learning

### Ramsey Number Solving
- **Task**: Find maximum clique-free graphs
- **Approach**: Geometric basin search with dynamic tension
- **Status**: Operational with checkpoint system

### Quantum Many-Body Physics
- **Task**: Solve 1D Transverse Field Ising Model ground states
- **Approach**: Real DMRG/MPS tensor network methods
- **Status**: Production-ready physics solver

---

## 💡 Why This Matters

### The Problem with Current AI
- **Black Boxes**: Neural networks are uninterpretable
- **Data Dependency**: Requires massive training datasets
- **Computational Cost**: Expensive GPU clusters
- **No True Understanding**: Pattern matching, not reasoning

### The LIVNIUM Approach
- **Transparent**: Every decision is geometrically traceable
- **Data Efficient**: Learns from structure, not just statistics
- **Lightweight**: Runs on CPU, no GPU required
- **True Compositionality**: Meaning emerges from atomic units

### Research Significance
This represents a **fundamental alternative** to transformer-based AI:
- **Geometric Computing**: Information as geometry, not vectors
- **Native Logic**: Built-in reasoning, not learned patterns
- **Physics-Based Learning**: Reinforcement through geometric feedback
- **Compositional Semantics**: Meaning from structure, not statistics

---

## 🛠️ Quick Start

### Installation

```bash
git clone https://github.com/chetanxpatil/livnium.core.git
cd livnium.core
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
```

### Run NLI Training

```bash
# Clean start (removes all caches)
python3 experiments/nli/train_moksha_nli.py --clean --train 20000 --test 2000 --dev 2000
```

### Test Golden Label Collapse

```bash
# Verify 3-way collapse mechanism
python3 experiments/nli/test_golden_label_collapse.py --clean
```

### Run Ramsey Solver

```bash
python3 experiments/ramsey/run_ramsey_experiment.py
```

---

## 📚 Documentation

- **Architecture**: `core/README.md` - Core system architecture
- **NLI System**: `experiments/nli/DIAGNOSTIC_REPORT.md` - Complete diagnostic
- **Ramsey Solver**: `experiments/ramsey/README.md` - Ramsey number solving
- **Universal Encoder**: `core/Universal Encoder/README.md` - Constraint encoding

---

## 🏗️ Project Structure

```
livnium.core/
├── core/                          # Core Livnium systems
│   ├── classical/                 # Classical geometry engine
│   ├── quantum/                   # Quantum layer
│   ├── Universal Encoder/         # Constraint problem encoding
│   └── search/                    # Multi-basin search
│
├── experiments/
│   ├── nli/                       # Natural Language Inference
│   │   ├── native_chain.py        # Letter-by-letter MPS architecture
│   │   ├── omcube.py              # Quantum collapse engine
│   │   ├── inference_detectors.py # Native logic
│   │   └── train_moksha_nli.py    # Training pipeline
│   │
│   └── ramsey/                    # Ramsey number solving
│       └── ramsey_dynamic_search.py
│
└── archive/                       # Historical implementations
```

---

## 🔬 Key Principles

### 1. **Geometric Computing**
Information is encoded as **3D geometric structures**, not high-dimensional vectors. This enables:
- Visual interpretability
- Structural reasoning
- Compositional semantics

### 2. **Native Logic**
Built-in reasoning capabilities through:
- Lexical overlap detection
- Negation handling
- Semantic gap analysis
- Double negative resolution

### 3. **Physics-Based Learning**
Learning through **geometric feedback**, not gradient descent:
- Basin reinforcement (deepening correct attractors)
- Natural decay (forgetting incorrect patterns)
- Reward-only learning (no punishment)

### 4. **Compositional Architecture**
Meaning emerges from structure:
- Letters → Words → Sentences
- Atomic units → Complex structures
- Local interactions → Global understanding

---

## ⚠️ Important Notes

### Experimental Research Software
**This is experimental research software.** It is:
- ✅ Suitable for research and education
- ✅ Designed for understanding novel computational approaches
- ❌ NOT production-ready
- ❌ NOT guaranteed to be error-free
- ❌ NOT suitable for commercial deployment without licensing

### Quantum-Inspired vs. Real Quantum
- **Livnium Core**: Uses real tensor network physics (MPS/DMRG)
- **Islands/Hierarchical**: Quantum-inspired classical systems (NOT physical quantum computers)
- **NLI System**: Pure geometric computing with quantum-inspired collapse

---

## 📋 Requirements

- **Python**: 3.7+
- **Core Dependencies**: `numpy`
- **Optional**: For faster performance, `numba` (JIT compilation)

---

## 📄 License

This project is licensed under the **Livnium License v1.1 (Fortress Grade)** - a proprietary research license.

### Quick Summary
- ✅ **Permitted**: Personal, non-commercial, research, and educational use
- ❌ **Prohibited**: Commercial use, redistribution, derivative works, AI training, public hosting, reverse engineering
- 🔒 **Commercial Rights**: Reserved exclusively by the Owner
- 🛡️ **Fortress Grade**: Includes protections against AI model training, data extraction, and public hosting

For full license terms, see [LICENSE](LICENSE) or [LICENSE.md](LICENSE.md).

**For commercial licensing inquiries**, please contact: chetan12patil@gmail.com

---

## 🤝 Contributing

This is a research project. Contributions are welcome for:
- Bug fixes
- Documentation improvements
- Research discussions

**Note**: All contributions become the exclusive property of the Owner per the License terms.

---

## 📧 Contact

**Chetan Patil**  
Email: chetan12patil@gmail.com

For research collaborations, commercial licensing, or technical inquiries.

---

## 🌟 Vision

LIVNIUM represents a **fundamental rethinking** of how computers can understand language and information:

- **From Vectors to Geometry**: Information as spatial structures
- **From Statistics to Structure**: Meaning from composition, not correlation
- **From Black Boxes to Transparency**: Every decision is traceable
- **From Data to Physics**: Learning through geometric feedback

This is not just another AI system—it's a **new computational paradigm**.

---

*"Information is geometry. Understanding is structure. Intelligence is composition."*

**LIVNIUM: Where quantum-inspired geometry meets native language understanding.**
