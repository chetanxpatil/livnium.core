"""
Advanced Law Extractor: v2-v6 Features

Implements:
- v2: Nonlinear function discovery
- v3: Symbolic regression
- v4: Law stability + confidence scoring
- v5: Multi-layer law fusion
- v6: Basin-based law extraction
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class DiscoveredLaw:
    """Represents a discovered physical law."""
    name: str
    formula: str
    function: Callable
    confidence: float
    stability: float
    history: List[float] = field(default_factory=list)
    first_seen: int = 0
    last_seen: int = 0


class AdvancedLawExtractor:
    """
    Advanced law extractor with v2-v6 features.
    
    Features:
    - v2: Nonlinear function discovery (polynomial, exponential, power laws)
    - v3: Symbolic regression (basic symbolic expressions)
    - v4: Law stability + confidence scoring
    - v5: Multi-layer law fusion
    - v6: Basin-based law extraction
    """
    
    def __init__(
        self,
        invariant_threshold: float = 1e-6,
        relationship_error_threshold: float = 1e-3,
        min_confidence: float = 0.7,
        stability_window: int = 10
    ):
        """
        Initialize advanced law extractor.
        
        Args:
            invariant_threshold: Maximum std dev for invariant
            relationship_error_threshold: Maximum error for relationship
            min_confidence: Minimum confidence to accept a law
            stability_window: Number of timesteps for stability calculation
        """
        self.history: List[Dict[str, float]] = []
        self.invariant_threshold = invariant_threshold
        self.relationship_error_threshold = relationship_error_threshold
        self.min_confidence = min_confidence
        self.stability_window = stability_window
        
        # v4: Law tracking
        self.discovered_laws: Dict[str, DiscoveredLaw] = {}
        self.law_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        
        # v5: Multi-layer tracking
        self.layer_laws: Dict[int, Dict[str, DiscoveredLaw]] = defaultdict(dict)
        
        # v6: Basin-specific laws
        self.basin_laws: Dict[str, Dict[str, DiscoveredLaw]] = defaultdict(dict)
    
    def record_state(self, state: Dict[str, float], layer: int = 0, basin_id: Optional[str] = None):
        """
        Record state with layer and basin information.
        
        Args:
            state: Physics state dictionary
            layer: Recursion layer (0 = base, 1+ = recursive)
            basin_id: Optional basin identifier for basin-specific laws
        """
        self.history.append(state.copy())
        timestep = len(self.history)
        
        # Track for multi-layer fusion (v5)
        if layer > 0:
            if layer not in self.layer_laws:
                self.layer_laws[layer] = {}
        
        # Track for basin extraction (v6)
        if basin_id:
            if basin_id not in self.basin_laws:
                self.basin_laws[basin_id] = {}
    
    # v2: Nonlinear Function Discovery
    def detect_nonlinear_relationships(self) -> Dict[str, Tuple[str, Callable, float]]:
        """
        Detect nonlinear relationships: polynomial, exponential, power laws.
        
        Returns:
            Dictionary mapping relationship names to (formula_str, function, error)
        """
        if len(self.history) < 5:
            return {}
        
        relationships = {}
        keys = list(self.history[0].keys())
        
        for i, xkey in enumerate(keys):
            for j, ykey in enumerate(keys):
                if i == j:
                    continue
                
                x = np.array([h[xkey] for h in self.history if xkey in h and ykey in h])
                y = np.array([h[ykey] for h in self.history if xkey in h and ykey in h])
                
                if len(x) < 5:
                    continue
                
                # Try different nonlinear forms
                best_error = float('inf')
                best_formula = None
                best_func = None
                
                # 1. Polynomial (y = a*x^2 + b*x + c)
                try:
                    # Check for valid data
                    if len(x) >= 3 and np.std(x) > 1e-10:
                        coeffs = np.polyfit(x, y, 2)
                        y_pred = np.polyval(coeffs, x)
                        error = np.mean(np.abs(y_pred - y))
                        if error < best_error and not np.isnan(error) and not np.isinf(error):
                            best_error = error
                            a, b, c = coeffs
                            best_formula = f"{ykey} = {a:.6f}*{xkey}^2 + {b:.6f}*{xkey} + {c:.6f}"
                            best_func = lambda x_val: np.polyval(coeffs, x_val)
                except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
                    pass
                
                # 2. Power law (y = a * x^b)
                try:
                    # Log transform: log(y) = log(a) + b*log(x)
                    x_pos = x[x > 0]
                    y_pos = y[x > 0]
                    if len(x_pos) > 3 and np.std(x_pos) > 1e-10:
                        log_x = np.log(x_pos)
                        log_y = np.log(y_pos)
                        if np.std(log_x) > 1e-10:
                            b, log_a = np.polyfit(log_x, log_y, 1)
                            a = np.exp(log_a)
                            y_pred = a * (x_pos ** b)
                            error = np.mean(np.abs(y_pred - y_pos))
                            if error < best_error and not np.isnan(error) and not np.isinf(error):
                                best_error = error
                                best_formula = f"{ykey} = {a:.6f} * {xkey}^{b:.6f}"
                                best_func = lambda x_val: a * (x_val ** b) if x_val > 0 else 0
                except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
                    pass
                
                # 3. Exponential (y = a * exp(b*x))
                try:
                    y_pos = y[y > 0]
                    x_exp = x[y > 0]
                    if len(y_pos) > 3 and np.std(x_exp) > 1e-10:
                        log_y = np.log(y_pos)
                        b, log_a = np.polyfit(x_exp, log_y, 1)
                        a = np.exp(log_a)
                        y_pred = a * np.exp(b * x_exp)
                        error = np.mean(np.abs(y_pred - y_pos))
                        if error < best_error and not np.isnan(error) and not np.isinf(error):
                            best_error = error
                            best_formula = f"{ykey} = {a:.6f} * exp({b:.6f}*{xkey})"
                            best_func = lambda x_val: a * np.exp(b * x_val)
                except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
                    pass
                
                # 4. Logarithmic (y = a * log(x) + b)
                try:
                    x_pos = x[x > 0]
                    y_log = y[x > 0]
                    if len(x_pos) > 3 and np.std(x_pos) > 1e-10:
                        log_x = np.log(x_pos)
                        if np.std(log_x) > 1e-10:
                            a, b = np.polyfit(log_x, y_log, 1)
                            y_pred = a * log_x + b
                            error = np.mean(np.abs(y_pred - y_log))
                            if error < best_error and not np.isnan(error) and not np.isinf(error):
                                best_error = error
                                best_formula = f"{ykey} = {a:.6f} * log({xkey}) + {b:.6f}"
                                best_func = lambda x_val: a * np.log(x_val) + b if x_val > 0 else 0
                except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
                    pass
                
                # Accept if error is low enough
                if best_error < self.relationship_error_threshold and best_formula:
                    relationships[f"{ykey}_vs_{xkey}"] = (best_formula, best_func, best_error)
        
        return relationships
    
    # v3: Symbolic Regression (Basic)
    def detect_symbolic_expressions(self) -> Dict[str, Tuple[str, Callable, float]]:
        """
        Basic symbolic regression: try common symbolic forms.
        
        Returns:
            Dictionary mapping relationship names to (formula_str, function, error)
        """
        if len(self.history) < 5:
            return {}
        
        relationships = {}
        keys = list(self.history[0].keys())
        
        # Common symbolic forms to try
        symbolic_forms = [
            # y = a*x + b (linear)
            lambda x, y: self._fit_linear(x, y, "{y} = {a}*{x} + {b}"),
            # y = a*x^2 + b (quadratic)
            lambda x, y: self._fit_polynomial(x, y, 2, "{y} = {a}*{x}^2 + {b}*{x} + {c}"),
            # y = a/x + b (inverse)
            lambda x, y: self._fit_inverse(x, y, "{y} = {a}/{x} + {b}"),
            # y = a*sqrt(x) + b (square root)
            lambda x, y: self._fit_sqrt(x, y, "{y} = {a}*sqrt({x}) + {b}"),
        ]
        
        for i, xkey in enumerate(keys):
            for j, ykey in enumerate(keys):
                if i == j:
                    continue
                
                x = np.array([h[xkey] for h in self.history if xkey in h and ykey in h])
                y = np.array([h[ykey] for h in self.history if xkey in h and ykey in h])
                
                if len(x) < 5:
                    continue
                
                best_error = float('inf')
                best_result = None
                
                for form_func in symbolic_forms:
                    try:
                        result = form_func(x, y)
                        if result and result[2] < best_error:
                            best_error = result[2]
                            best_result = result
                            best_result[0] = best_result[0].format(y=ykey, x=xkey)
                    except:
                        continue
                
                if best_result and best_error < self.relationship_error_threshold:
                    relationships[f"{ykey}_vs_{xkey}"] = best_result
        
        return relationships
    
    def _fit_linear(self, x, y, template):
        """Fit linear: y = a*x + b"""
        a, b = np.polyfit(x, y, 1)
        y_pred = a * x + b
        error = np.mean(np.abs(y_pred - y))
        func = lambda x_val: a * x_val + b
        return (template.format(a=f"{a:.6f}", b=f"{b:.6f}"), func, error)
    
    def _fit_polynomial(self, x, y, degree, template):
        """Fit polynomial"""
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)
        error = np.mean(np.abs(y_pred - y))
        func = lambda x_val: np.polyval(coeffs, x_val)
        # Format template with coefficients
        formatted = template
        for i, c in enumerate(reversed(coeffs)):
            formatted = formatted.replace(f"{{{chr(97+i)}}}", f"{c:.6f}")
        return (formatted, func, error)
    
    def _fit_inverse(self, x, y, template):
        """Fit inverse: y = a/x + b"""
        x_pos = x[x > 0]
        y_inv = y[x > 0]
        if len(x_pos) < 3:
            return None
        # Transform: y = a/x + b -> y = a*(1/x) + b
        inv_x = 1.0 / x_pos
        a, b = np.polyfit(inv_x, y_inv, 1)
        y_pred = a / x_pos + b
        error = np.mean(np.abs(y_pred - y_inv))
        func = lambda x_val: (a / x_val + b) if x_val > 0 else 0
        return (template.format(a=f"{a:.6f}", b=f"{b:.6f}"), func, error)
    
    def _fit_sqrt(self, x, y, template):
        """Fit square root: y = a*sqrt(x) + b"""
        x_pos = x[x > 0]
        y_sqrt = y[x > 0]
        if len(x_pos) < 3:
            return None
        sqrt_x = np.sqrt(x_pos)
        a, b = np.polyfit(sqrt_x, y_sqrt, 1)
        y_pred = a * sqrt_x + b
        error = np.mean(np.abs(y_pred - y_sqrt))
        func = lambda x_val: (a * np.sqrt(x_val) + b) if x_val > 0 else 0
        return (template.format(a=f"{a:.6f}", b=f"{b:.6f}"), func, error)
    
    # v4: Law Stability + Confidence Scoring
    def compute_law_confidence(self, formula: str, func: Callable, error: float) -> float:
        """
        Compute confidence score for a discovered law.
        
        Args:
            formula: Law formula string
            func: Law function
            error: Fitting error
            
        Returns:
            Confidence score [0, 1]
        """
        # Base confidence from error
        error_confidence = 1.0 / (1.0 + error * 100)
        
        # Stability confidence (if law was seen before)
        if formula in self.discovered_laws:
            law = self.discovered_laws[formula]
            stability = law.stability
            age = len(self.history) - law.first_seen
            age_confidence = min(1.0, age / self.stability_window)
            return (error_confidence * 0.5 + stability * 0.3 + age_confidence * 0.2)
        
        return error_confidence
    
    def compute_law_stability(self, formula: str) -> float:
        """
        Compute stability score for a law (how consistent it is over time).
        
        Args:
            formula: Law formula string
            
        Returns:
            Stability score [0, 1]
        """
        if formula not in self.discovered_laws:
            return 0.5  # Default for new laws
        
        law = self.discovered_laws[formula]
        if len(law.history) < 2:
            return 0.5
        
        # Stability = 1 - coefficient of variation
        history_array = np.array(law.history)
        if np.mean(history_array) == 0:
            return 0.0
        
        cv = np.std(history_array) / np.mean(history_array)
        stability = max(0.0, 1.0 - cv)
        
        return float(stability)
    
    def update_law_tracking(self, relationships: Dict[str, Tuple[str, Callable, float]]):
        """
        Update law tracking for stability and confidence.
        
        Args:
            relationships: Dictionary of discovered relationships
        """
        timestep = len(self.history)
        
        for rel_name, (formula, func, error) in relationships.items():
            confidence = self.compute_law_confidence(formula, func, error)
            
            if formula in self.discovered_laws:
                # Update existing law
                law = self.discovered_laws[formula]
                law.history.append(confidence)
                law.last_seen = timestep
                law.confidence = confidence
                law.stability = self.compute_law_stability(formula)
            else:
                # Create new law
                law = DiscoveredLaw(
                    name=rel_name,
                    formula=formula,
                    function=func,
                    confidence=confidence,
                    stability=0.5,
                    first_seen=timestep,
                    last_seen=timestep
                )
                law.history.append(confidence)
                self.discovered_laws[formula] = law
    
    # v5: Multi-Layer Law Fusion
    def fuse_layer_laws(self) -> Dict[str, DiscoveredLaw]:
        """
        Fuse laws from different recursion layers.
        
        Returns:
            Dictionary of fused laws
        """
        fused_laws = {}
        
        # Collect laws from all layers
        all_layer_laws = {}
        for layer, laws in self.layer_laws.items():
            for name, law in laws.items():
                if name not in all_layer_laws:
                    all_layer_laws[name] = []
                all_layer_laws[name].append((layer, law))
        
        # Fuse laws that appear in multiple layers
        for name, layer_laws_list in all_layer_laws.items():
            if len(layer_laws_list) > 1:
                # Law appears in multiple layers - high confidence
                layers = [layer for layer, _ in layer_laws_list]
                laws = [law for _, law in layer_laws_list]
                
                # Average confidence and stability
                avg_confidence = np.mean([l.confidence for l in laws])
                avg_stability = np.mean([l.stability for l in laws])
                
                # Create fused law
                fused_law = DiscoveredLaw(
                    name=f"{name}_fused_L{min(layers)}-L{max(layers)}",
                    formula=laws[0].formula,  # Use formula from first law
                    function=laws[0].function,
                    confidence=avg_confidence,
                    stability=avg_stability,
                    first_seen=min([l.first_seen for l in laws]),
                    last_seen=max([l.last_seen for l in laws])
                )
                
                fused_laws[fused_law.name] = fused_law
        
        return fused_laws
    
    # v6: Basin-Based Law Extraction
    def extract_basin_laws(
        self,
        basin_states: Dict[str, List[Dict[str, float]]]
    ) -> Dict[str, Dict[str, DiscoveredLaw]]:
        """
        Extract laws specific to individual basins.
        
        Args:
            basin_states: Dictionary mapping basin_id to list of states
            
        Returns:
            Dictionary mapping basin_id to discovered laws
        """
        basin_laws = {}
        
        for basin_id, states in basin_states.items():
            if len(states) < 5:
                continue
            
            # Extract laws for this basin
            keys = list(states[0].keys())
            laws = {}
            
            for i, xkey in enumerate(keys):
                for j, ykey in enumerate(keys):
                    if i == j:
                        continue
                    
                    x = np.array([s[xkey] for s in states if xkey in s and ykey in s])
                    y = np.array([s[ykey] for s in states if xkey in s and ykey in s])
                    
                    if len(x) < 3:
                        continue
                    
                    # Fit linear
                    try:
                        a, b = np.polyfit(x, y, 1)
                        y_pred = a * x + b
                        error = np.mean(np.abs(y_pred - y))
                        
                        if error < self.relationship_error_threshold:
                            formula = f"{ykey} = {a:.6f}*{xkey} + {b:.6f}"
                            func = lambda x_val: a * x_val + b
                            confidence = self.compute_law_confidence(formula, func, error)
                            
                            law = DiscoveredLaw(
                                name=f"{basin_id}_{ykey}_vs_{xkey}",
                                formula=formula,
                                function=func,
                                confidence=confidence,
                                stability=0.5
                            )
                            laws[law.name] = law
                    except:
                        continue
            
            if laws:
                basin_laws[basin_id] = laws
        
        return basin_laws
    
    # Combined extraction
    def extract_all(self) -> Dict[str, Any]:
        """
        Extract all types of laws (v1-v6).
        
        Returns:
            Comprehensive dictionary of all discovered laws
        """
        results = {
            "invariants": self.detect_invariants(),
            "linear_relationships": self.detect_functional_relationships(),
            "nonlinear_relationships": self.detect_nonlinear_relationships(),
            "symbolic_expressions": self.detect_symbolic_expressions(),
            "discovered_laws": {},
            "fused_laws": {},
            "basin_laws": {}
        }
        
        # v4: Update law tracking
        all_relationships = {}
        all_relationships.update(results["linear_relationships"])
        all_relationships.update(results["nonlinear_relationships"])
        all_relationships.update(results["symbolic_expressions"])
        
        # Convert to format expected by update_law_tracking
        formatted_rels = {}
        for name, value in all_relationships.items():
            if isinstance(value, tuple) and len(value) >= 2:
                if len(value) == 2:
                    # (a, b) format from linear
                    a, b = value
                    y_name, x_name = name.split("_vs_")
                    formula = f"{y_name} = {a:.6f}*{x_name} + {b:.6f}"
                    func = lambda x_val, a=a, b=b: a * x_val + b
                    error = 0.001  # Default
                else:
                    # (formula, func, error) format
                    formula, func, error = value
                formatted_rels[name] = (formula, func, error)
        
        self.update_law_tracking(formatted_rels)
        
        # Add discovered laws with confidence/stability
        for formula, law in self.discovered_laws.items():
            if law.confidence >= self.min_confidence:
                results["discovered_laws"][law.name] = {
                    "formula": law.formula,
                    "confidence": law.confidence,
                    "stability": law.stability,
                    "first_seen": law.first_seen,
                    "last_seen": law.last_seen
                }
        
        # v5: Fused laws
        results["fused_laws"] = {
            name: {
                "formula": law.formula,
                "confidence": law.confidence,
                "stability": law.stability
            }
            for name, law in self.fuse_layer_laws().items()
        }
        
        return results
    
    # Helper methods from base class
    def detect_invariants(self) -> Dict[str, bool]:
        """Detect invariants (from base class)."""
        if len(self.history) < 2:
            return {}
        
        invariants = {}
        for key in self.history[0].keys():
            values = np.array([h[key] for h in self.history if key in h])
            if len(values) < 2:
                invariants[key] = False
                continue
            std_dev = np.std(values)
            invariants[key] = std_dev < self.invariant_threshold
        
        return invariants
    
    def detect_functional_relationships(self) -> Dict[str, Tuple[float, float]]:
        """Detect linear relationships (from base class)."""
        if len(self.history) < 3:
            return {}
        
        relationships = {}
        keys = list(self.history[0].keys())
        
        for i, xkey in enumerate(keys):
            for j, ykey in enumerate(keys):
                if i == j:
                    continue
                
                x = np.array([h[xkey] for h in self.history if xkey in h and ykey in h])
                y = np.array([h[ykey] for h in self.history if xkey in h and ykey in h])
                
                if len(x) < 3:
                    continue
                
                try:
                    a, b = np.polyfit(x, y, 1)
                    y_pred = a * x + b
                    error = np.mean(np.abs(y_pred - y))
                    
                    if error < self.relationship_error_threshold:
                        relationships[f"{ykey}_vs_{xkey}"] = (a, b)
                except:
                    continue
        
        return relationships
    
    def clear_history(self):
        """Clear recorded history."""
        self.history = []
        self.discovered_laws = {}
        self.law_history = defaultdict(list)
        self.layer_laws = defaultdict(dict)
        self.basin_laws = defaultdict(dict)

