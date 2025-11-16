"""
Rule Engine: Symbolic Reasoning and Rule Application

Implements rule-based reasoning for Livnium Core System.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Rule:
    """
    A reasoning rule.
    
    Rule format: IF condition THEN action
    """
    name: str
    condition: Callable[[Any], bool]  # Condition function
    action: Callable[[Any], Any]      # Action function
    priority: int = 0                 # Higher priority = applied first
    enabled: bool = True
    
    def apply(self, state: Any) -> Optional[Any]:
        """Apply rule if condition is met."""
        if self.enabled and self.condition(state):
            return self.action(state)
        return None


@dataclass
class RuleSet:
    """Collection of rules."""
    rules: List[Rule] = field(default_factory=list)
    
    def add_rule(self, rule: Rule):
        """Add rule to set."""
        self.rules.append(rule)
        # Sort by priority (higher first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def apply_rules(self, state: Any, max_applications: int = 10) -> Tuple[Any, List[str]]:
        """
        Apply rules to state until no more apply.
        
        Args:
            state: Current state
            max_applications: Maximum number of rule applications
            
        Returns:
            Tuple of (final_state, list of applied rule names)
        """
        current_state = state
        applied_rules = []
        
        for _ in range(max_applications):
            applied = False
            for rule in self.rules:
                result = rule.apply(current_state)
                if result is not None:
                    current_state = result
                    applied_rules.append(rule.name)
                    applied = True
                    break  # Apply one rule per iteration
            
            if not applied:
                break
        
        return current_state, applied_rules


class RuleEngine:
    """
    Rule engine for Livnium Core System.
    
    Applies symbolic reasoning rules based on geometric and quantum properties.
    """
    
    def __init__(self, core_system):
        """
        Initialize rule engine.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
        self.rule_set = RuleSet()
        self._initialize_livnium_rules()
    
    def _initialize_livnium_rules(self):
        """Initialize Livnium-specific rules."""
        
        # Rule: If cell is corner and has high SW, increase importance
        def corner_high_sw_condition(state):
            coords = state.get('coordinates')
            if coords:
                cell = self.core_system.get_cell(coords)
                return cell and cell.cell_class.value == 3 and cell.symbolic_weight >= 27
            return False
        
        def corner_high_sw_action(state):
            state['importance'] = state.get('importance', 0.5) + 0.2
            return state
        
        self.rule_set.add_rule(Rule(
            name="corner_high_sw",
            condition=corner_high_sw_condition,
            action=corner_high_sw_action,
            priority=5
        ))
        
        # Rule: If cell is core and has low SW, decrease importance
        def core_low_sw_condition(state):
            coords = state.get('coordinates')
            if coords:
                cell = self.core_system.get_cell(coords)
                return cell and cell.cell_class.value == 0 and cell.symbolic_weight == 0
            return False
        
        def core_low_sw_action(state):
            state['importance'] = max(0.0, state.get('importance', 0.5) - 0.1)
            return state
        
        self.rule_set.add_rule(Rule(
            name="core_low_sw",
            condition=core_low_sw_condition,
            action=core_low_sw_action,
            priority=3
        ))
    
    def add_rule(self, rule: Rule):
        """Add custom rule."""
        self.rule_set.add_rule(rule)
    
    def reason_about_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply reasoning rules to state.
        
        Args:
            state: State dictionary
            
        Returns:
            Modified state after rule application
        """
        final_state, applied_rules = self.rule_set.apply_rules(state)
        final_state['applied_rules'] = applied_rules
        return final_state
    
    def get_rule_statistics(self) -> Dict:
        """Get rule engine statistics."""
        return {
            'total_rules': len(self.rule_set.rules),
            'enabled_rules': sum(1 for r in self.rule_set.rules if r.enabled),
        }

