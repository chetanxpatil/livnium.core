# Action Plan: From Nullspace to Solid Result

## Current Status
✅ V3 system frozen and working
✅ Found 4-dimensional nullspace
✅ All invariants preserved exactly (~1e-17 error)

## Next Steps (In Order)

### STEP 1: Get Exact Rational Weights
**Command:**
```bash
python3 experiments/rule30/test_divergence_v3_invariant.py \
    --num-rows 300 \
    --row-length 200 \
    --exact \
    --test-steps 20
```

**What it does:**
- Uses sympy for exact rational arithmetic
- Extracts exact integer/rational weights
- Identifies trivial vs non-trivial invariants
- Shows all 4 invariants clearly

**Expected output:**
- Exact formulas like `D3(s) = (3*freq('000')) - (5*freq('001')) + ...`
- Identification of which are trivial (normalization, symmetry)
- Which are non-trivial (potentially interesting)

---

### STEP 2: Bruteforce Verification
**Command:**
```bash
# First, extract weights from STEP 1, then:
python3 experiments/rule30/bruteforce_verify_invariant.py \
    --N 8 10 12 \
    --max-steps 20 \
    --weights "w000,w001,w010,w011,w100,w101,w110,w111"
```

**What it does:**
- Tests invariant for ALL 2^N rows
- Verifies exact preservation over evolution steps
- Reports any counterexamples

**Expected output:**
- "✓✓✓ ALL ROWS VERIFIED!" for each N
- Or counterexamples if invariant doesn't hold exactly

---

### STEP 3: Simplify Invariants
**Manual step:** Analyze the exact weights to find:
- Symmetries (e.g., weight(001) = weight(100))
- Grouped patterns
- Human-readable form

---

### STEP 4: Document Final Result
**Write:** Clear statement of what's proven vs conjectured

---

## Quick Start (Run These Now)

```bash
# 1. Get exact invariants
python3 experiments/rule30/test_divergence_v3_invariant.py --exact

# 2. Copy weights from output, then verify
python3 experiments/rule30/bruteforce_verify_invariant.py \
    --N 8 10 \
    --weights "your,weights,here"
```

