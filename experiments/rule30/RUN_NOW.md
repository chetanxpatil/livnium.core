# What to Run Now

## Immediate Next Steps

### 1. Get Exact Invariants (with sympy if available)

```bash
python3 experiments/rule30/test_divergence_v3_invariant.py \
    --num-rows 300 \
    --row-length 200 \
    --exact \
    --test-steps 20
```

**If sympy not installed:**
```bash
pip install sympy
```

**What this does:**
- Finds all 4 invariants
- Identifies trivial vs non-trivial
- Shows exact rational weights (if sympy available)
- Tests invariance

---

### 2. Extract Weights and Verify Exhaustively

After STEP 1, copy the weights from one invariant and run:

```bash
python3 experiments/rule30/bruteforce_verify_invariant.py \
    --N 8 10 12 \
    --max-steps 20 \
    --weights "w000,w001,w010,w011,w100,w101,w110,w111"
```

**Example (replace with actual weights from STEP 1):**
```bash
python3 experiments/rule30/bruteforce_verify_invariant.py \
    --N 8 10 \
    --max-steps 10 \
    --weights "0,1,-1,0,0,-1,1,0"
```

---

### 3. Analyze Results

After verification:
- If all verified → You have a solid result!
- If counterexamples → Need to investigate
- Document what's proven vs conjectured

---

## Current Status

✅ System working
✅ 4 invariants found
✅ Need exact weights (STEP 2)
✅ Need exhaustive verification (STEP 3)

**Run STEP 1 first, then proceed based on results.**

