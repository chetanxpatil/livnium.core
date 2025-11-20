# LIVNIUM
## Quantum-Inspired Geometric Computing & Native Language Understanding

**A breakthrough computational architecture combining quantum-inspired geometry, native language processing, and tensor network physics.**

---

## IMPORTANT: Personal/Research Use Only

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

## 📖 Research Background & Theoretical Foundations

### What "Qubit-Analogues" Means

**Important**: This is **NOT** a physical quantum computer. The term "qubit-analogue" refers to classical simulation of quantum-like states using geometric structures.

- **Real quantum computing**: Uses physical qubits with superposition and entanglement
- **Livnium approach**: Uses 3×3×3 geometric structures that can represent quantum-like states classically
- **Capacity claims**: A 5×5×5 base lattice with 2 levels of recursion = 94,625 cells, each capable of storing quantum-like state information
- **Why "analogue"**: These are classical geometric structures that mimic quantum behavior, not actual qubits

**Verification**: See `core/tests/test_qubit_capacity.py` for capacity measurements.

### What "Self-Healing Geometry" Means

The term refers to **tension-based convergence** in geometric search:

- **Tension fields**: Geometric constraints create tension when violated
- **Self-correction**: The system naturally moves toward lower-tension states
- **No external optimization**: Convergence emerges from geometric physics, not gradient descent

**Implementation**: See `core/RAMSEY_READY_PATCHES.md` for the tension-based system.

### Scientific Foundations

**Matrix Product States (MPS)**:
- Standard method in quantum many-body physics (see Schuch et al., 2013; Orús, 2014)
- Used here for sentence-level representation: words as tensors, sentences as chains
- **Reference**: Schuch, N., et al. "Matrix product states, projected entangled pair states, and variational renormalization group methods for quantum spin systems." *Advances in Physics* 62.4 (2013): 277-356.

**Density Matrix Renormalization Group (DMRG)**:
- Real tensor network method for quantum ground states
- Implemented in `core/quantum/` for solving 1D Transverse Field Ising Model
- **Reference**: White, S. R. "Density matrix formulation for quantum renormalization groups." *Physical Review Letters* 69.19 (1992): 2863.

**Tensor Networks**:
- Mathematical framework for representing high-dimensional quantum states
- Used throughout Livnium for geometric encoding
- **Reference**: Orús, R. "A practical introduction to tensor networks: Matrix product states and projected entangled pair states." *Annals of Physics* 349 (2014): 117-158.

### What This Project Is (And Isn't)

**This IS**:
- An experimental research system exploring geometric alternatives to neural networks
- A classical simulation system that uses quantum-inspired structures
- Open-source research code for academic investigation
- A proof-of-concept for geometric language representation

**This IS NOT**:
- A published peer-reviewed paper (this is code-first research)
- A production-ready system
- A physical quantum computer
- A replacement for transformer models (yet)

### Verification & Reproducibility

**To verify claims**:

1. **Qubit capacity**: Run `python3 core/tests/test_qubit_capacity.py`
2. **NLI performance**: Run `python3 experiments/nli/train_moksha_nli.py --clean`
3. **Ramsey solver**: Run `python3 experiments/ramsey/run_ramsey_experiment.py`
4. **Code inspection**: All code is available for review in the repository

**Experimental status**: This is active research. Results are preliminary and subject to change.

### Research Methodology

- **Letter-by-letter encoding**: Novel approach, not from literature (experimental)
- **Geometric NLI**: Experimental alternative to neural NLI systems
- **Basin reinforcement**: Physics-inspired learning mechanism (experimental)
- **MPS for language**: Adaptation of tensor networks to NLP (experimental)

**Note**: Some components are based on established physics (MPS, DMRG), while others (letter-level encoding, geometric NLI) are novel experimental approaches.

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
- ❌ NOT a published peer-reviewed paper (code-first research)

### Quantum-Inspired vs. Real Quantum
- **Livnium Core**: Uses real tensor network physics (MPS/DMRG) - these are established methods
- **Islands/Hierarchical**: Quantum-inspired classical systems (NOT physical quantum computers)
- **NLI System**: Pure geometric computing with quantum-inspired collapse
- **"Qubit-analogues"**: Classical geometric structures that simulate quantum-like behavior, not physical qubits

### Transparency Statement
This repository contains:
- **Established methods**: MPS, DMRG tensor networks (well-documented in physics literature)
- **Experimental approaches**: Letter-by-letter encoding, geometric NLI (novel, unproven)
- **Verifiable code**: All implementations are open for inspection
- **Test suites**: Capacity and functionality tests included

**For skeptics**: We encourage code review, reproduction of results, and critical evaluation. This is research in progress, not a finished product.

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
