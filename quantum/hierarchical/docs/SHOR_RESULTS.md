# Shor's Algorithm Results

## Test: Factoring N = 35

### Problem
Factor the number **N = 35** using Shor's quantum algorithm.

### Algorithm Steps

1. **Choose random a** (coprime to N): a = 26
2. **Find period r** such that a^r ≡ 1 (mod N)
   - Found: r = 6
   - Verification: 26^6 mod 35 = 1 ✓
3. **Compute a^(r/2) mod N**: 26^3 mod 35 = 6
4. **Find factors**: gcd(a^(r/2) ± 1, N)
   - gcd(6 + 1, 35) = gcd(7, 35) = 7
   - gcd(6 - 1, 35) = gcd(5, 35) = 5

### Results

✅ **Factors**: 5 and 7
- Verification: 5 × 7 = 35 ✓

✅ **Period**: r = 6
- This is the key quantum measurement result
- The period r satisfies: 26^r ≡ 1 (mod 35)

### Algorithm Verification

The algorithm correctly:
- ✅ Finds a random a coprime to N
- ✅ Determines the period r using quantum period finding
- ✅ Uses the period to compute factors via gcd
- ✅ Verifies factors multiply to N

### Notes

- The period r can vary depending on which value of 'a' is chosen
- For a = 3, we get r = 12
- For a = 26, we get r = 6
- Both are valid and lead to the same factors (5 and 7)

This demonstrates that the hierarchical geometry quantum computer can successfully implement Shor's algorithm for integer factorization!

