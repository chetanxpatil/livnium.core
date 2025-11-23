"""
Quick script to test rule engine integration.

This shows how to enable the rule engine in Layer 7.
"""

from experiments.nli_v4.rule_engine import RuleEngine
from experiments.nli_v4.layer7_decision import Layer7Decision

# Create rule engine (hand-tuned or from discovered rules)
# Option 1: Hand-tuned rules (clean, interpretable)
rule_engine = RuleEngine.from_hand_tuned_rules()

# Option 2: From discovered rules (if you have discovered_rules.json)
# rule_engine = RuleEngine.from_discovered_rules('discovered_rules.json')

# Option 3: Load from saved rules
# rule_engine = RuleEngine()
# rule_engine.load_rules('saved_rules.json')

# Print rules
rule_engine.print_rules()

# Create Layer 7 with rule engine enabled
layer7_with_rules = Layer7Decision(use_rule_engine=True, rule_engine=rule_engine)

print("\n✓ Rule engine integrated into Layer 7")
print("  Set use_rule_engine=True in LayeredLivniumClassifier to enable")

