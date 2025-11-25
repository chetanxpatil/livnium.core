# Theorem Hunting: From Livnium Physics to Math Proof

This directory contains tools to turn Livnium's geometric discoveries into formal mathematical theorems.

## Goal

Transform the discovered divergence invariant `λ = -0.572222233` from:
- **Evidence** (numerical experiments) 
- → **Exact Formula** (rational arithmetic, combinatorial expression)
- → **Formal Proof** (algebraic derivation)

## Tools

### 1. Rational Divergence (`rational_divergence.py`)

Computes divergence using exact rational arithmetic instead of floating point.

**Goal**: Find the exact rational form of `-0.572222233`

```python
from experiments.rule30.rational_divergence import find_exact_invariant_value

sequences = [generate_center_column_direct(n) for n in [1000, 10000, 100000]]
result = find_exact_invariant_value(sequences)
print(f"Exact form: {result['invariant_as_fraction']}")
# e.g., "-515/900" instead of "-0.572222233"
```

### 2. Pattern Analysis (`pattern_analysis.py`)

Analyzes how divergence relates to pattern frequencies.

**Goal**: Find `D(s) = Σ α_p · freq_p(s)` where `freq_p(s)` is frequency of pattern `p`.

```python
from experiments.rule30.pattern_analysis import find_invariant_pattern_combination

rule30_seqs = [generate_center_column_direct(n) for n in [1000, 10000]]
result = find_invariant_pattern_combination(rule30_seqs)
print(f"Formula: D(s) = {result['formula']}")
# e.g., "D(s) = 0.5*freq('111') - 0.3*freq('000') + ..."
```

### 3. Invariant Hunter (`invariant_hunter.py`)

Searches for exact algebraic invariants by enumerating candidates.

**Goal**: Find the simplest linear combination that is exactly conserved.

```bash
python3 experiments/rule30/invariant_hunter.py \
    --steps 1000 10000 \
    --target -0.572222233 \
    --max-coeff 5
```

### 4. Divergence Structure Analysis (`analyze_divergence_structure.py`)

Comprehensive analysis combining all approaches.

**Goal**: Discover the exact formula and verify it's invariant.

```bash
python3 experiments/rule30/analyze_divergence_structure.py \
    --rule30-steps 1000 5000 10000 \
    --random-count 50 \
    --pattern-length 3
```

## Workflow

### Step 1: Find Exact Rational Form

```bash
python3 -c "
from experiments.rule30.rational_divergence import find_exact_invariant_value
from experiments.rule30.rule30_optimized import generate_center_column_direct

seqs = [generate_center_column_direct(n, show_progress=False) for n in [1000, 10000, 100000]]
result = find_exact_invariant_value(seqs)
print(f'Exact: {result[\"invariant_as_fraction\"]}')
print(f'Decimal: {result[\"invariant_decimal\"]:.9f}')
"
```

### Step 2: Discover Pattern Formula

```bash
python3 experiments/rule30/analyze_divergence_structure.py \
    --rule30-steps 1000 5000 10000 \
    --pattern-length 3
```

### Step 3: Hunt for Exact Invariant

```bash
python3 experiments/rule30/invariant_hunter.py \
    --steps 1000 10000 \
    --max-coeff 3
```

### Step 4: Verify with Rational Arithmetic

Once you have a candidate formula, verify it gives exact rational results:

```python
from fractions import Fraction
from experiments.rule30.pattern_analysis import compute_pattern_frequencies

# Test formula with exact arithmetic
seq = generate_center_column_direct(1000)
freqs = compute_pattern_frequencies(seq)

# Compute using exact formula (replace with discovered coefficients)
exact_value = Fraction(0)  # Start with discovered formula
for pattern, coeff in discovered_coefficients.items():
    exact_value += Fraction(coeff) * Fraction(freqs[pattern])

print(f"Exact invariant: {exact_value}")
```

## Next Steps After Discovery

Once you have the exact formula `D(s) = Σ α_p · freq_p(s)`:

1. **Derive Pattern Evolution**: How do pattern frequencies evolve under Rule 30?
   - Express `freq_p(t+1)` in terms of `freq_q(t)` for patterns `q` at time `t`
   - Use Rule 30's local update rule

2. **Prove Conservation**: Show algebraically that:
   ```
   Σ α_p · freq_p(t+1) = Σ α_p · freq_p(t)
   ```
   for all `t`

3. **Formalize**: Write the proof in standard mathematical notation

## Example: What Success Looks Like

**Discovered Formula:**
```
D(s) = (1/2) * freq('111') - (1/3) * freq('000') + (1/6) * freq('101')
```

**Proof Sketch:**
1. Rule 30 update rule: `new_cell = left XOR center OR (center AND NOT right)`
2. Pattern frequency evolution: `freq_p(t+1) = f(freq_q(t))` for some function `f`
3. Show: `(1/2)*freq_111(t+1) - (1/3)*freq_000(t+1) + (1/6)*freq_101(t+1) = (1/2)*freq_111(t) - (1/3)*freq_000(t) + (1/6)*freq_101(t)`
4. Conclude: `D(c_t) = constant` for Rule 30 center column `c_t`

## Notes

- Livnium provides the **telescope** (discovers the invariant)
- These tools provide the **microscope** (finds the exact formula)
- You provide the **proof** (algebraic derivation)

The tools can't write the final proof, but they give you everything needed to write it.

