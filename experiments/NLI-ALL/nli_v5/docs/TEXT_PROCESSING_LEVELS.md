# Text Processing Levels in Livnium

## The Answer: **All Three Levels, Hierarchically**

Livnium processes text at **three levels simultaneously**:

1. **Letter → Vector** (foundation)
2. **Word → Vector** (aggregation)
3. **Sentence → Chain of Word Vectors** (sequence)

---

## Level 1: Letter → Vector

### How It Works

```python
def letter_to_vector(letter: str, vector_size: int = 27) -> np.ndarray:
    """Generate a fixed-size vector for a letter using hash-based seeding."""
    letter_hash = hash(letter.lower()) % (2**32)
    np.random.seed(letter_hash)
    vector = np.random.uniform(-1.0, 1.0, vector_size)
    return vector
```

**Process:**
- Each letter gets a deterministic vector (hash-based)
- Vector size: 27 dimensions (default)
- Same letter → same vector (deterministic)

**Example:**
```
'a' → [0.23, -0.45, 0.67, ...]  (27D vector)
'b' → [-0.12, 0.89, -0.34, ...] (27D vector)
```

---

## Level 2: Word → Vector

### How It Works

```python
class WordVector:
    def __init__(self, word: str, vector_size: int = 27):
        self.letters = list(word.lower())
        # Create letter vectors
        letter_vectors = [letter_to_vector(letter, vector_size) for letter in self.letters]
        # Aggregate: sum and normalize
        self.vector = np.sum(letter_vectors, axis=0)
        self.vector = self.vector / np.linalg.norm(self.vector)
```

**Process:**
- Word is split into letters
- Each letter → letter vector
- Letter vectors are **summed** and **normalized**
- Result: Single word vector

**Example:**
```
"cat" → ['c', 'a', 't']
       → [c_vec, a_vec, t_vec]
       → sum([c_vec, a_vec, t_vec]) → normalized → word_vector
```

**Key Point:** Word vector = **aggregated letter vectors**, not learned embeddings.

---

## Level 3: Sentence → Chain of Word Vectors

### How It Works

```python
class SentenceVector:
    def __init__(self, sentence: str, vector_size: int = 27):
        # Tokenize (split by spaces)
        self.tokens = sentence.lower().split()
        # Create word vectors
        self.word_vectors = [WordVector(word, vector_size) for word in self.tokens]
        # Store position information
        self.positions = list(range(len(self.word_vectors)))
```

**Process:**
- Sentence is tokenized (split by spaces)
- Each word → WordVector
- Sentence = **ordered sequence** of word vectors
- Position matters (sequential matching)

**Example:**
```
"A cat runs" → ["a", "cat", "runs"]
              → [WordVector("a"), WordVector("cat"), WordVector("runs")]
              → SentenceVector with 3 word vectors + positions
```

---

## The Complete Pipeline

### Input: Text Pair
```
Premise: "A cat runs"
Hypothesis: "A cat is running"
```

### Step 1: Letter Level
```
'a' → letter_vector_a
'c' → letter_vector_c
'a' → letter_vector_a  (same as first 'a')
't' → letter_vector_t
...
```

### Step 2: Word Level
```
"a" → sum([letter_vector_a]) → word_vector_a
"cat" → sum([letter_vector_c, letter_vector_a, letter_vector_t]) → word_vector_cat
"runs" → sum([letter_vector_r, letter_vector_u, letter_vector_n, letter_vector_s]) → word_vector_runs
```

### Step 3: Sentence Level
```
Premise: [word_vector_a, word_vector_cat, word_vector_runs]
Hypothesis: [word_vector_a, word_vector_cat, word_vector_is, word_vector_running]
```

### Step 4: Geometry Computation
```
- Compare word vectors position-by-position
- Compute divergence (angle between vectors)
- Compute resonance (geometric similarity)
- Apply physics layers
```

---

## Why This Hierarchy?

### Letter Level (Foundation)
- **Deterministic**: Same letter → same vector
- **No learning**: Pure geometry, no embeddings
- **Compositional**: Letters compose into words

### Word Level (Aggregation)
- **Compositional**: Words = sum of letters
- **Position-independent**: Word vector doesn't depend on position in sentence
- **Geometric**: Pure vector math, no neural networks

### Sentence Level (Sequence)
- **Position-aware**: Order matters (sequential matching)
- **Chain structure**: Words form a chain
- **Resonance**: Position-by-position comparison

---

## Key Differences from Neural Models

### Neural Models (BERT, GPT, etc.)
- **Word-level**: Pre-trained word embeddings
- **No letter-level**: Letters don't matter
- **Learned**: Embeddings learned from data

### Livnium
- **Letter-level foundation**: Letters → vectors deterministically
- **Word-level aggregation**: Words = sum of letters
- **No pre-training**: No learned embeddings
- **Pure geometry**: Everything is geometric computation

---

## The Geometry Connection

### How Letters Create Geometry

Each letter vector is a **point in 27D space**.

When you sum letter vectors:
- You're **composing geometric shapes**
- Word vector = **center of mass** of letter vectors
- Sentence = **trajectory** through word vector space

### How This Creates Semantic Physics

1. **Letter vectors** → Geometric foundation
2. **Word vectors** → Aggregated geometry
3. **Sentence chains** → Sequential geometry
4. **Pair comparison** → Divergence/resonance fields
5. **Physics layers** → Meaning from geometry

---

## Summary

**Livnium processes text at ALL THREE levels:**

1. ✅ **Letter-by-letter** → Deterministic letter vectors
2. ✅ **Word-by-word** → Aggregated word vectors (sum of letters)
3. ✅ **Sentence-by-sentence** → Chain of word vectors (position-aware)

**The hierarchy:**
```
Letters → Words → Sentences → Geometry → Meaning
```

**No learned embeddings.**
**No pre-training.**
**Pure geometric composition.**

This is why geometry is stable — it's built from deterministic letter vectors, not learned patterns.

