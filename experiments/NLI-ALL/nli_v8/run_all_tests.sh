#!/bin/bash
# Run all v8 tests in sequence

set -e

echo "======================================================================"
echo "LIVNIUM V8: COMPLETE TEST SUITE"
echo "======================================================================"
echo ""

cd "$(dirname "$0")/../.."

# Test 1: Normal training
echo "======================================================================"
echo "TEST 1: Normal Training (10k train, 1k test)"
echo "======================================================================"
python3 experiments/nli_v8/training/train_v8.py --train 10000 --test 1000
echo ""

# Test 2: Golden label (should be 100%)
echo "======================================================================"
echo "TEST 2: Golden Label Mode (should be 100%)"
echo "======================================================================"
python3 experiments/nli_v8/training/train_v8.py --train 10000 --test 100 --golden-label
echo ""

# Test 3: Tension fix verification
echo "======================================================================"
echo "TEST 3: Tension-Preserving Fixes Verification"
echo "======================================================================"
cd experiments/nli_v8
python3 test_tension_fix.py
cd ../..
echo ""

# Test 4: Law compliance
echo "======================================================================"
echo "TEST 4: Law Compliance Test"
echo "======================================================================"
cd experiments/nli_v8
python3 test_law_compliance.py
cd ../..
echo ""

echo "======================================================================"
echo "ALL TESTS COMPLETE"
echo "======================================================================"

