"""
Per-Example Law Test: Compare Same Examples Across Normal and Inverted Modes

This test properly verifies that divergence signs are preserved by comparing
the SAME examples in normal vs inverted label modes, not group averages.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.nli_v5.core.encoder import ChainEncoder
from experiments.nli_v5.core.classifier import LivniumV5Classifier


def load_snli_data(file_path: str, max_examples: int = 1000):
    """Load SNLI data with example tracking."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if len(examples) >= max_examples:
                break
            data = json.loads(line.strip())
            if data['gold_label'] in ['entailment', 'contradiction', 'neutral']:
                examples.append({
                    'index': idx,
                    'sentence1': data['sentence1'],
                    'sentence2': data['sentence2'],
                    'gold_label': data['gold_label']
                })
    return examples


def compute_divergence_for_examples(examples: List[Dict], invert_labels: bool = False) -> Dict[int, Dict]:
    """
    Compute divergence for each example.
    
    Returns:
        Dict mapping example index to {'divergence': float, 'resonance': float, 'gold_label': str, 'inverted_label': str}
    """
    encoder = ChainEncoder()
    results = {}
    
    for example in examples:
        idx = example['index']
        premise = example['sentence1']
        hypothesis = example['sentence2']
        gold_label = example['gold_label']
        
        # Encode
        pair = encoder.encode_pair(premise, hypothesis)
        
        # Determine label to use
        if invert_labels:
            label_inverter = {
                'entailment': 'contradiction',
                'contradiction': 'entailment',
                'neutral': 'neutral'
            }
            inverted_label = label_inverter.get(gold_label, gold_label)
            classifier = LivniumV5Classifier(pair, debug_mode=False, golden_label_hint=inverted_label, reverse_physics_mode=True)
            used_label = inverted_label
        else:
            classifier = LivniumV5Classifier(pair)
            used_label = gold_label
        
        # Classify
        result = classifier.classify()
        
        results[idx] = {
            'divergence': result.layer_states.get('divergence', 0.0),
            'resonance': result.layer_states.get('resonance', 0.0),
            'gold_label': gold_label,
            'used_label': used_label,
            'premise': premise,
            'hypothesis': hypothesis
        }
    
    return results


def test_divergence_sign_preservation(max_examples: int = 1000):
    """
    Test that divergence signs are preserved for the SAME examples across normal/inverted modes.
    """
    print("="*80)
    print("PER-EXAMPLE DIVERGENCE SIGN PRESERVATION TEST")
    print("="*80)
    print(f"\nTesting {max_examples} examples from SNLI training set...")
    print("Comparing SAME examples in normal vs inverted label modes.\n")
    
    # Load data
    data_file = os.path.join(os.path.dirname(__file__), '../nli/data/snli_1.0_train.jsonl')
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    examples = load_snli_data(data_file, max_examples=max_examples)
    print(f"✓ Loaded {len(examples)} examples\n")
    
    # Compute divergence in normal mode
    print("Computing divergence in NORMAL mode...")
    normal_results = compute_divergence_for_examples(examples, invert_labels=False)
    print(f"✓ Processed {len(normal_results)} examples\n")
    
    # Compute divergence in inverted mode
    print("Computing divergence in INVERTED mode...")
    inverted_results = compute_divergence_for_examples(examples, invert_labels=True)
    print(f"✓ Processed {len(inverted_results)} examples\n")
    
    # Compare same examples
    print("="*80)
    print("COMPARING SAME EXAMPLES")
    print("="*80)
    
    violations = []
    sign_preserved_count = 0
    sign_changed_count = 0
    
    # Group by original gold label
    by_label = defaultdict(list)
    
    for idx in normal_results:
        if idx not in inverted_results:
            continue
        
        normal = normal_results[idx]
        inverted = inverted_results[idx]
        
        normal_sign = np.sign(normal['divergence'])
        inverted_sign = np.sign(inverted['divergence'])
        
        # Track by original gold label
        gold_label = normal['gold_label']
        by_label[gold_label].append({
            'idx': idx,
            'normal_div': normal['divergence'],
            'inverted_div': inverted['divergence'],
            'normal_sign': normal_sign,
            'inverted_sign': inverted_sign,
            'preserved': normal_sign == inverted_sign
        })
        
        if normal_sign == inverted_sign:
            sign_preserved_count += 1
        else:
            sign_changed_count += 1
            violations.append({
                'idx': idx,
                'gold_label': gold_label,
                'normal_div': normal['divergence'],
                'inverted_div': inverted['divergence'],
                'normal_label': normal['used_label'],
                'inverted_label': inverted['used_label']
            })
    
    # Report by label
    print("\nResults by Original Gold Label:")
    print("-" * 80)
    
    for label in ['entailment', 'contradiction', 'neutral']:
        if label not in by_label:
            continue
        
        examples_for_label = by_label[label]
        preserved = sum(1 for e in examples_for_label if e['preserved'])
        total = len(examples_for_label)
        
        if total > 0:
            normal_mean = np.mean([e['normal_div'] for e in examples_for_label])
            inverted_mean = np.mean([e['inverted_div'] for e in examples_for_label])
            normal_sign = np.sign(normal_mean)
            inverted_sign = np.sign(inverted_mean)
            
            print(f"\n{label.upper()} (n={total}):")
            print(f"  Normal mean divergence:   {normal_mean:.6f} (sign: {normal_sign:+.0f})")
            print(f"  Inverted mean divergence:  {inverted_mean:.6f} (sign: {inverted_sign:+.0f})")
            print(f"  Sign preserved: {preserved}/{total} ({100*preserved/total:.1f}%)")
            print(f"  Status: {'✅ SIGN PRESERVED' if normal_sign == inverted_sign else '❌ SIGN CHANGED'}")
    
    # Overall summary
    total_compared = sign_preserved_count + sign_changed_count
    preservation_rate = sign_preserved_count / total_compared if total_compared > 0 else 0
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total examples compared: {total_compared}")
    print(f"Sign preserved: {sign_preserved_count} ({100*preservation_rate:.1f}%)")
    print(f"Sign changed: {sign_changed_count} ({100*(1-preservation_rate):.1f}%)")
    
    if violations:
        print(f"\n⚠️  {len(violations)} violations found (showing first 5):")
        for v in violations[:5]:
            print(f"  Example {v['idx']} ({v['gold_label']}):")
            print(f"    Normal:   div={v['normal_div']:.6f} (label: {v['normal_label']})")
            print(f"    Inverted: div={v['inverted_div']:.6f} (label: {v['inverted_label']})")
    
    print("\n" + "="*80)
    if preservation_rate >= 0.95:  # 95% threshold
        print("✅ TEST PASSED: Divergence signs are preserved (≥95%)")
    else:
        print(f"❌ TEST FAILED: Only {100*preservation_rate:.1f}% of examples preserve sign")
    print("="*80)
    
    return {
        'preservation_rate': preservation_rate,
        'total_compared': total_compared,
        'violations': len(violations)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test divergence sign preservation per-example')
    parser.add_argument('--max-examples', type=int, default=1000,
                        help='Maximum number of examples to test (default: 1000)')
    
    args = parser.parse_args()
    
    test_divergence_sign_preservation(max_examples=args.max_examples)

