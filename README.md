# LIVNIUM
## Quantum-Inspired Geometric Computing & Native Language Understanding

**A computational architecture combining quantum-inspired geometry, native language processing, and tensor network physics.**

---

## ⚠️ License: Source Available, Non-Commercial

**This software is provided for PERSONAL, NON-COMMERCIAL, RESEARCH, and EDUCATIONAL PURPOSES ONLY.**

- ✅ **Permitted**: Personal study, research, education, non-commercial use, contributions (with credit)
- ❌ **Prohibited**: Commercial use, selling, hosting as a service, redistribution without permission
- 🔒 **Commercial Rights**: Reserved exclusively by the Owner

**Non-Commercial use includes only personal, academic, or research use by individuals. Use within any company or organization for internal purposes (including research, development, or evaluation) is considered commercial and requires prior written permission from the Owner.**

See [LICENSE](LICENSE) for full terms. For commercial licensing inquiries: chetan12patil@gmail.com

---

## 🌟 What is LIVNIUM?

LIVNIUM is a **pure native computing system** that processes language and information through geometric quantum-inspired structures—**without transformers, embeddings, or neural networks**.

### Core Innovation: Geometric Quantum-Inspired Architecture

LIVNIUM uses **3D geometric structures** (omcubes) as the fundamental unit of computation:
- **Information** → Geometric patterns in 3D space
- **Meaning** → Emerges from spatial relationships
- **Learning** → Physics-based basin reinforcement
- **Reasoning** → Geometric logic, not neural networks

---

## 🚀 Key Features

### Pure Native Architecture
- **Zero Transformers**: No BERT, GPT, or neural language models
- **Zero Embeddings**: No pre-trained word vectors
- **100% Interpretable**: Every decision is traceable through geometric structures

### Quantum-Inspired Geometry
- **3×3×3 Omcubes**: Each letter/word encoded as a quantum-inspired geometric structure
- **Matrix Product States (MPS)**: Sentence-level entanglement through chained omcubes
- **Quantum Collapse**: 3-way decision making (Entailment/Contradiction/Neutral)
- **Basin Reinforcement**: Physics-based learning through geometric feedback

### Complete System Architecture
- **Classical Layer**: N×N×N geometric lattice with symbolic weights
- **Quantum Layer**: True quantum mechanics with tensor products (`TrueQuantumRegister`)
- **Recursive Geometry**: 2.5M+ qubit-analogue capacity through fractal compression
- **Runtime System**: Temporal cognition engine (hierarchical timesteps, episodes)
- **Semantic Layer**: Bridge between geometry and meaning
- **Memory System**: Persistent memory that behaves like energy in a lattice
- **Learning System**: Reward-based basin reinforcement (no gradients)
- **Meta Layer**: Self-observation and adaptation
- **Reasoning Engine**: Native logic and rule-based reasoning
- **Search Module**: Multi-basin competition with corner rotation policy
- **Universal Encoder**: Converts any problem → tension fields + basins

---

## 📐 Architecture Overview

### System Layers

```
Layer 4: Multi-Basin Competition  ← Complete
    ↓
Layer 3: Universal Encoding       ← Complete
    ↓
Layer 2: Task Encoding             ← Future
    ↓
Layer 1: Dynamic Basin Physics     ← Complete
    ↓
Layer 0: LivniumCoreSystem         ← Base
```

### Core Components

1. **Classical Layer** (`core/classical/`)
   - N×N×N lattice (any odd N ≥ 3)
   - Symbolic Weight Law (SW = 9·f)
   - 24-element rotation group
   - Observer system (global + local)
   - Semantic polarity

2. **Quantum Layer** (`core/quantum/`)
   - True quantum mechanics (`TrueQuantumRegister`)
   - Tensor product states
   - Multi-qubit entanglement
   - Meta-interference optimization
   - Geometry-quantum coupling

3. **Recursive Layer** (`core/recursive/`)
   - Fractal geometry engine
   - 2.5M+ qubit capacity
   - Moksha convergence (fixed-point truth)
   - Multi-scale physics

4. **Runtime Layer** (`core/runtime/`)
   - Temporal engine (hierarchical timesteps)
   - Episode manager (experience buffer)
   - Orchestrator (executive function)

5. **Semantic Layer** (`core/semantic/`)
   - Feature extraction
   - Meaning graphs
   - Geometric logic (entailment/contradiction)
   - Bridge between geometry and meaning

6. **Memory Layer** (`core/memory/`)
   - Per-cell memory capsules
   - Global memory lattice
   - Geometric coupling
   - Associative recall

7. **Learning Layer** (`core/learning/`)
   - Reward-based reinforcement
   - Basin deepening
   - No gradients, no neural nets

8. **Meta Layer** (`core/meta/`)
   - Self-observation
   - Anomaly detection
   - Auto-calibration
   - Introspection

9. **Reasoning Layer** (`core/reasoning/`)
   - Rule-based inference
   - Search strategies (A*, beam, greedy)
   - Problem decomposition
   - Native logic

10. **Search Module** (`core/search/`)
    - Dynamic basin reinforcement
    - Multi-basin competition
    - Corner rotation policy (post-convergence refinement)

11. **Universal Encoder** (`core/encoder/`)
    - Problem → tension fields + basins
    - Constraint encoding
    - Graph coloring, SAT, Ramsey, CSP support

12. **Embedding Module** (`core/embedding/`)
    - Geometric key embedding
    - Locality-preserving mappings
    - Gray code interleaving

---

## 🔬 Research Applications

### Natural Language Inference (NLI)
- **Task**: Classify premise-hypothesis pairs as Entailment, Contradiction, or Neutral
- **Approach**: Pure geometric reasoning with zero neural networks
- **Location**: `experiments/nli/`

### Quantum Many-Body Physics
- **Task**: Solve 1D Transverse Field Ising Model ground states
- **Approach**: Real DMRG/MPS tensor network methods

### Quantum-Inspired Cryptanalysis
- **Task**: Explore AES-128 key space using quantum superposition
- **Approach**: Recursive geometry + quantum layer + tension fields
- **Capacity**: 2.5M logical qubits, 125+ entangled qubits simultaneously
- **Location**: `experiments/quantum_core/`

### Quantum Protocol Tests
- **Task**: True quantum teleportation and Bell's inequality
- **Approach**: `TrueQuantumRegister` with tensor products
- **Location**: `experiments/quantum_teleportation/`

### Ramsey Number Solving
- **Task**: Find maximum clique-free graphs
- **Approach**: Geometric basin search with dynamic tension
- **Location**: `experiments/ramsey/`

### Benchmark Problems
- **Max-Cut**: GSET benchmark graphs
- **SAT**: Boolean satisfiability (CNF formulas)
- **CSP**: Constraint satisfaction (N-Queens, Graph Coloring, Sudoku)
- **Location**: `benchmark/`

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
python3 experiments/nli/train_moksha_nli.py --clean --train 20000 --test 2000 --dev 2000
```

### Test Quantum Capabilities

```bash
# Test recursive qubit capacity (2.5M qubits)
python3 experiments/quantum_core/test_recursive_qubit_capacity.py

# Test quantum teleportation
python3 experiments/quantum_teleportation/test_quantum_teleportation.py

# Test Bell's inequality violation
python3 experiments/quantum_teleportation/test_bell_inequality.py
```

### Run AES Cryptanalysis

```bash
# Recursive collapse (geometric manifold search)
python3 experiments/quantum_core/aes128_recursive_collapse.py

# Quantum topology mapper
python3 experiments/quantum_core/aes128_quantum_topology_mapper.py
```

### Run Benchmarks

```bash
# Max-Cut solver
python3 benchmark/max_cut/max_cut_solver_livnium.py --graph gset_14

# SAT solver
python3 benchmark/sat/sat_solver_livnium.py --cnf problem.cnf

# CSP solver
python3 benchmark/csp/csp_solver_livnium.py --problem nqueens --size 8
```

---

## 📚 Documentation

### Core System
- **Architecture**: `core/README.md` - Core system architecture
- **Classical Layer**: `core/classical/README.md` - Geometric foundation
- **Quantum Layer**: `core/quantum/README.md` - Quantum mechanics implementation
- **Recursive Layer**: `core/recursive/README.md` - Fractal geometry engine
- **Runtime Layer**: `core/runtime/README.md` - Temporal cognition engine
- **Semantic Layer**: `core/semantic/README.md` - Geometry → meaning bridge
- **Memory System**: `core/memory/README.md` - Persistent memory
- **Learning System**: `core/learning/README.md` - Reward-based learning
- **Meta Layer**: `core/meta/README.md` - Self-observation
- **Reasoning Engine**: `core/reasoning/README.md` - Native logic
- **Search Module**: `core/search/README.md` - Multi-basin search
- **Universal Encoder**: `core/encoder/README.md` - Problem encoding
- **Embedding**: `core/embedding/README.md` - Geometric key embedding

### Experiments
- **NLI System**: `experiments/nli/README.md` - Natural Language Inference
- **Quantum Experiments**: `experiments/quantum_core/README.md`
- **Quantum Teleportation**: `experiments/quantum_teleportation/README.md`
- **Ramsey Problems**: `experiments/ramsey/README.md`
- **Crypto Experiments**: `experiments/crypto/README.md`

### Benchmarks
- **Benchmark Suite**: `benchmark/README.md`
- **Max-Cut**: `benchmark/max_cut/README.md`
- **SAT**: `benchmark/sat/README.md`
- **CSP**: `benchmark/csp/README.md`

---

## 🏗️ Project Structure

```
livnium.core/
├── core/                          # Core Livnium systems
│   ├── classical/                 # Classical geometry engine (N×N×N lattice)
│   ├── quantum/                   # Quantum layer (TrueQuantumRegister)
│   ├── recursive/                 # Recursive geometry (fractal, Moksha)
│   ├── runtime/                   # Temporal engine (episodes, orchestration)
│   ├── semantic/                  # Semantic processing (geometry → meaning)
│   ├── memory/                    # Memory system (persistent, associative)
│   ├── learning/                  # Learning system (reward-based)
│   ├── meta/                      # Meta layer (self-observation)
│   ├── reasoning/                 # Reasoning engine (rules, search)
│   ├── embedding/                 # Geometric key embedding
│   ├── encoder/                   # Universal problem encoder
│   ├── search/                    # Multi-basin search
│   └── tests/                     # Test suite
│
├── experiments/
│   ├── nli/                       # Natural Language Inference
│   ├── quantum_teleportation/     # Quantum protocol tests
│   ├── quantum_core/              # Quantum-enhanced experiments
│   ├── crypto/                    # Cryptanalysis experiments
│   └── ramsey/                    # Ramsey number solving
│
└── benchmark/                     # Benchmark suites
    ├── max_cut/                   # Max-Cut problem solver
    ├── sat/                       # SAT solver
    └── csp/                       # CSP solver
```

---

## 🔬 Key Principles

### 1. Geometric Computing
Information is encoded as **3D geometric structures**, not high-dimensional vectors.

### 2. Native Logic
Built-in reasoning through geometric logic, not neural networks:
- Entailment = SW(premise) > SW(conclusion)
- Contradiction = opposite classes + <2 distance
- Causal link = SW differential + spatial proximity

### 3. Physics-Based Learning
Learning through **geometric feedback**, not gradient descent:
- Basin reinforcement (deepening correct attractors)
- Natural decay (forgetting incorrect patterns)
- Reward-only (no punishment)

### 4. Compositional Architecture
Meaning emerges from structure: Letters → Words → Sentences

### 5. Temporal Cognition
Hierarchical timesteps (MACRO, MICRO, QUANTUM, MEMORY, SEMANTIC) create cognitive rhythm

### 6. Recursive Universe
Every cell can spawn a smaller universe, creating exponential capacity

---

## ⚠️ Important Notes

### Experimental Research Software
**This is experimental research software.** It is:
- ✅ Suitable for research and education
- ❌ NOT production-ready
- ❌ NOT suitable for commercial deployment without licensing

### Quantum-Inspired vs. Real Quantum
- **Livnium Core**: Uses real tensor network physics (MPS/DMRG)
- **Quantum Layer**: Implements true quantum mechanics with tensor products
- **"Qubit-analogues"**: Classical geometric structures that simulate quantum-like behavior

---

## 📋 Requirements

- **Python**: 3.7+
- **Core Dependencies**: `numpy`
- **Optional**: `numba` for JIT compilation

---

## 📄 License

This project is licensed under the **LIVNIUM License (MIT + Commons Clause)** - Source Available, Non-Commercial.

- ✅ **Permitted**: Personal, non-commercial, research, and educational use; contributions (with credit)
- ❌ **Prohibited**: Commercial use, selling, hosting as a service, redistribution without permission
- 🔒 **Commercial Rights**: Reserved exclusively by the Owner

**Non-Commercial use includes only personal, academic, or research use by individuals. Use within any company or organization for internal purposes (including research, development, or evaluation) is considered commercial and requires prior written permission from the Owner.**

For full license terms, see [LICENSE](LICENSE) or [LICENSE.md](LICENSE.md).

**For commercial licensing inquiries**: chetan12patil@gmail.com

---

## 🤝 Contributing

Contributions are welcome for bug fixes, documentation improvements, and research discussions.

**Note**: All contributions become part of the Livnium project per the License terms.

### Coding Standards: Keep It Simple

**We prioritize clarity and understanding over complexity.**

- ✅ **Use simple, descriptive names** that clearly communicate purpose
- ✅ **Make names resonate** - they should make sense in context
- ✅ **Avoid unnecessarily complicated terminology** when simpler alternatives exist
- ✅ **Prefer readability** - code should be self-documenting through good naming

**Philosophy**: If someone can't understand what a function, class, or variable does from its name alone, it needs a better name. We want the codebase to be accessible and intuitive, not intimidating.

Examples:
- ✅ `measure_tension()` instead of `compute_geometric_energy_landscape_metric()`
- ✅ `get_neighbors()` instead of `retrieve_adjacent_cell_coordinates()`
- ✅ `refine_candidate()` instead of `apply_stochastic_optimization_heuristic()`

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
- **From Static to Temporal**: Time-aware cognitive rhythms
- **From Flat to Recursive**: Fractal universe with exponential capacity

*"Information is geometry. Understanding is structure. Intelligence is composition."*

**LIVNIUM: Where quantum-inspired geometry meets native language understanding.**
