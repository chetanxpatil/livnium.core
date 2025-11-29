"""
Analyze alignment values to understand divergence behavior.
"""

import json
import numpy as np
import os

pattern_file = 'experiments/nli_v5/patterns/patterns_fixed.json'

if not os.path.exists(pattern_file):
    print(f"Pattern file not found: {pattern_file}")
    exit(1)

with open(pattern_file, 'r') as f:
    data = json.load(f)

print("=" * 80)
print("ALIGNMENT ANALYSIS")
print("=" * 80)
print()

for label in ['entailment', 'contradiction', 'neutral']:
    if label not in data.get('patterns', {}):
        continue
    
    patterns = data['patterns'][label]
    
    # Compute alignment from divergence (reverse: alignment = 0.5 - divergence)
    alignments = [0.5 - p.get('divergence', 0.0) for p in patterns]
    divergences = [p.get('divergence', 0.0) for p in patterns]
    
    mean_align = np.mean(alignments)
    mean_div = np.mean(divergences)
    
    print(f"{label.upper()}:")
    print(f"  Mean alignment: {mean_align:.4f}")
    print(f"  Mean divergence: {mean_div:.4f}")
    print(f"  Alignment range: [{np.min(alignments):.4f}, {np.max(alignments):.4f}]")
    print(f"  Divergence range: [{np.min(divergences):.4f}, {np.max(divergences):.4f}]")
    
    # Count by alignment threshold
    high_align = sum(1 for a in alignments if a > 0.5)
    low_align = sum(1 for a in alignments if a < 0.5)
    
    print(f"  High alignment (>0.5): {high_align}/{len(alignments)} ({100*high_align/len(alignments):.1f}%)")
    print(f"  Low alignment (<0.5): {low_align}/{len(alignments)} ({100*low_align/len(alignments):.1f}%)")
    
    # Expected vs actual
    if label == 'entailment':
        expected = "high alignment (>0.5) → negative divergence"
        actual_div_sign = "positive" if mean_div > 0 else "negative"
        print(f"  Expected: {expected}")
        print(f"  Actual divergence: {actual_div_sign}")
        if mean_div > 0:
            print(f"  ⚠️  PROBLEM: Entailment has positive divergence (should be negative)")
    elif label == 'contradiction':
        expected = "low alignment (<0.5) → positive divergence"
        actual_div_sign = "positive" if mean_div > 0 else "negative"
        print(f"  Expected: {expected}")
        print(f"  Actual divergence: {actual_div_sign}")
        if mean_div < 0:
            print(f"  ⚠️  PROBLEM: Contradiction has negative divergence (should be positive)")
    
    print()

