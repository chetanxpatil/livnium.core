# Token Hash Collision Analysis: How Our System Actually Works

## ChatGPT's Claim

**Claim**: 92.4% collision rate because:
- 5×5×5 = 125 cells
- SNLI has 1645 distinct words
- 1645 / 125 ≈ 13 words per cell
- This causes OM ≈ LO, constant divergence, physics failure

## How Our System Actually Works

### 1. Token Hashing (The Collision Source)

**Location**: `nova/core/geometric_token_learner.py::token_hash()`

```python
def token_hash(self, token: str) -> Tuple[Tuple[int, int, int], float]:
    h = hashlib.md5(token.encode('utf-8')).hexdigest()
    x = int(h[0:4], 16) % self.lattice_size  # [0, 4] for lattice_size=5
    y = int(h[4:8], 16) % self.lattice_size  # [0, 4] for lattice_size=5
    z = int(h[8:12], 16) % self.lattice_size # [0, 4] for lattice_size=5
    val = int(h[12:14], 16)
    impulse = (val / 127.5) - 1.0
    return ((x, y, z), impulse)
```

**Math Check**:
- For `lattice_size=5`: coordinates are in [0, 4] for each dimension
- Total possible coordinate tuples: 5×5×5 = **125**
- If SNLI has 1645 distinct words → **collision rate ≈ 92.4%** ✓ (ChatGPT is correct)

### 2. But Signatures Are NOT Just Hash Coordinates

**Critical Point**: The signature is NOT the hash coordinates. It's the **SW distribution across ALL 125 cells after collapse**.

**Process**:
1. Token → Hash → (x, y, z, impulse)
2. Multiple tokens in a sentence → Multiple impulses at different coordinates
3. **Collapse process** → SW redistributes across ALL 125 cells
4. Final signature = SW value at each of the 125 cells

**Key Insight**: Even if tokens hash to the same coordinates, the **sentence-level signature** can still be different because:
- Different sentences have different token sequences
- Different impulse values (from hash)
- Collapse creates different SW distributions

### 3. Why Collisions Still Matter

**The Problem**: If many words hash to the same coordinate:
- "woman", "ocean", "artist", "black" → all hash to (2, 3, 1)
- They all add impulse to the SAME cell
- The system can't distinguish them at the token level

**But**: At the **sentence level**, different sentences still create different signatures because:
- Different word orders
- Different sentence lengths
- Different combinations of impulses

**However**: If collision rate is 92.4%, then:
- Most words are indistinguishable at the token level
- Sentence signatures become less distinct
- OM and LO signatures become more similar

### 4. The Real Question: Is Collision the Root Cause?

**To Answer This, We Need**:
1. Actual collision rate measurement (run Section 7 of diagnostic)
2. OM/LO signature similarity measurement (run Section 1 & 3)
3. Check if signatures are actually identical or just similar

**Hypothesis Testing**:
- **If** collision rate ≈ 92% **AND** OM/LO cosine similarity ≈ 0.99998 → **Collision is the problem**
- **If** collision rate ≈ 92% **BUT** OM/LO cosine similarity varies (0.6-0.95) → **Collision is NOT the problem**

### 5. Why Increasing Lattice Size Would Help

**If we increase to 15×15×15 = 3375 cells**:
- Collision rate drops: 1645 / 3375 ≈ 48.7% (still high but better)
- More coordinate space → more token separation
- But signatures are still 3375-dimensional (not just coordinates)

**If we increase to 21×21×21 = 9261 cells**:
- Collision rate: 1645 / 9261 ≈ 17.8% (much better)
- Most tokens get unique coordinates
- Signatures become 9261-dimensional

**Trade-off**:
- Larger lattice = more memory, slower computation
- But better token separation = more distinct signatures

### 6. Alternative: The Signature Is Already 125-Dimensional

**Current System**:
- Lattice: 5×5×5 = 125 cells
- Signature: 125-dimensional vector (SW at each cell)
- Each dimension can hold different values

**The Collision Issue**:
- Hash coordinates: Only 125 possible values
- But signature values: Continuous (SW can be any float)
- So even with collisions, signatures can still differ

**The Real Test**:
- Are OM and LO signatures actually identical? (cosine ≈ 1.0)
- Or are they just similar? (cosine ≈ 0.95-0.99)
- If cosine ≈ 0.95-0.99, collisions are NOT the root cause
- If cosine ≈ 1.0, collisions ARE the root cause

### 7. What We Need to Measure

**Run the diagnostic script** to get actual data:

```bash
python3 nova/debug/snli_physics_report.py
```

**Key Metrics**:
1. **Section 1**: OM vs LO cosine similarity distribution
   - If mean ≈ 0.99998 → collisions are the problem
   - If mean ≈ 0.85-0.95 → collisions are NOT the problem

2. **Section 7**: Actual collision rate
   - Verify if it's really 92.4%
   - See which words collide

3. **Section 3**: Alignment/divergence distributions
   - If all values are constant → collisions are the problem
   - If values vary → collisions are NOT the problem

### 8. Conclusion

**ChatGPT's Analysis**:
- ✅ Math is correct: 1645 words / 125 cells ≈ 13 words per cell
- ✅ Collision rate would be high (~92%)
- ❓ But is this the ROOT CAUSE of OM ≈ LO?

**Our System's Reality**:
- Signatures are 125-dimensional SW distributions (not just hash coordinates)
- Even with collisions, different sentences can create different signatures
- The collapse process redistributes SW across all cells

**The Verdict**:
- **We need to run the diagnostic to know for sure**
- If OM/LO cosine similarity ≈ 0.99998 → ChatGPT is right, increase lattice size
- If OM/LO cosine similarity varies (0.6-0.95) → ChatGPT is wrong, problem is elsewhere

**Next Step**: Run `python3 nova/debug/snli_physics_report.py` and check Section 1 & 3 results.

