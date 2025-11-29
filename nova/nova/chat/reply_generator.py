"""
Reply Generator: Pure Generation from Geometry

NO search. NO hardcoded templates.
Only real generation from learned patterns.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add project root to path (go up two levels: chat -> nova -> project_root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nova.core.text_to_geometry import TextToGeometry
from nova.core.geometric_transformer import GeometricTransformer
from nova.core.sentence_decoder import SentenceDecoder
from nova.core.geometric_token_learner import GeometricTokenLearner
from nova.core.cluster_decoder import ClusterDecoder


class ReplyGenerator:
    """
    Pure generation engine - creates new text from geometry.
    
    Pipeline:
    1. Query → collapse → signature
    2. Context → collapse → signature (optional)
    3. Transform signatures → response signature
    4. Generate text from response signature (learned patterns)
    """
    
    def __init__(self,
                 interface: TextToGeometry,
                 signature_database_path: Optional[Path] = None,
                 cluster_path: Optional[Path] = None,
                 collapse_steps: int = 15,
                 use_cluster_decoder: bool = True,
                 temperature: float = 0.7,
                 repetition_penalty: float = 0.1):
        """
        Initialize reply generator.
        
        Args:
            interface: TextToGeometry interface (lattice_size will be auto-adjusted to match model)
            signature_database_path: Path to learned patterns file (for grammar)
            cluster_path: Path to geometric clusters file (for cluster decoder)
            collapse_steps: Number of collapse steps
            use_cluster_decoder: Use cluster-based decoder (new) instead of old decoder
        """
        self.collapse_steps = collapse_steps
        self.use_cluster_decoder = use_cluster_decoder
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        
        # Initialize components
        self.transformer = GeometricTransformer(interface)
        
        if use_cluster_decoder and cluster_path:
            # NEW: Use cluster-based decoder
            # cluster_path should be without extension (loads both .json and .pkl)
            token_learner = GeometricTokenLearner()
            token_learner.load(cluster_path)
            
            # CRITICAL: Ensure interface matches the loaded model's lattice size (dynamic)
            if token_learner.lattice_size != interface.geometry.lattice_size:
                print(f"⚠ Interface lattice_size ({interface.geometry.lattice_size}) doesn't match model ({token_learner.lattice_size})")
                print(f"  Auto-adjusting to lattice_size={token_learner.lattice_size}")
                # Recreate interface with correct lattice size
                from nova.core.text_to_geometry import TextToGeometry
                interface = TextToGeometry(
                    lattice_size=token_learner.lattice_size,
                    impulse_scale=interface.impulse_scale,
                    num_clusters=token_learner.num_clusters
                )
                # Recreate transformer with new interface
                self.transformer = GeometricTransformer(interface)
            
            # Update interface's learner to use the loaded one
            interface.learner = token_learner
            
            # PASS THE GRAMMAR PATH HERE
            self.decoder = ClusterDecoder(
                token_learner, 
                interface, 
                patterns_path=signature_database_path,  # Pass grammar patterns
                collapse_steps=collapse_steps
            )
        else:
            # OLD: Use sentence decoder (backward compatibility)
            self.decoder = SentenceDecoder(signature_database_path)
        
        # Store interface reference (may have been recreated)
        self.interface = interface
        
        # Conversation context - single running signature
        self.context_sig: Optional[np.ndarray] = None
        self.alpha = 0.7  # Weight for current signature
        self.beta = 0.3   # Weight for context signature
        self.use_decay = False  # Enable smooth context decay
        self.decay_alpha = 0.85  # Decay weight for old context
        self.decay_beta = 0.15   # Decay weight for new input
    
    def generate_reply(self, 
                      query: str,
                      use_context: bool = True,
                      transform_type: str = "merge") -> Dict:
        """
        Generate reply to query.
        
        Pipeline:
        1. input text → collapse → current_sig
        2. context → context_sig (if exists)
        3. merged_sig = α * current_sig + β * context_sig
        4. Transform merged_sig → response_sig
        5. Generate text from response_sig
        6. Update context_sig = merged_sig
        
        Args:
            query: Input query sentence
            use_context: Whether to use conversation context
            transform_type: Transformation type ("merge", "rotate", "project")
        
        Returns:
            Dict with reply and metadata
        """
        # Step 1: Collapse query to signature
        current_sig = self.interface.get_meaning_signature(
            query, 
            collapse_steps=self.collapse_steps
        )
        self.interface.reset_geometry()
        
        # Step 2: Merge with context (if available)
        had_context = use_context and self.context_sig is not None
        if had_context:
            # Mix new message with old context
            # Higher weight on current query (0.7) vs context (0.3) to stay relevant
            merged_sig = self.alpha * current_sig + self.beta * self.context_sig
            if verbose := False:  # Set to True for debugging
                print(f"  Context merged: {self.alpha:.2f}×current + {self.beta:.2f}×context")
        else:
            # First message - no context yet
            merged_sig = current_sig
        
        # Step 3: Transform to response signature
        # (Optional: can still apply geometric transformations)
        response_sig = self.transformer.transform_signature(
            merged_sig,
            None,  # Already merged, no additional context needed
            transform_type=transform_type
        )
        
        # Step 4: Generate text from response signature
        # Use temperature and repetition penalty to prevent stuttering/looping
        reply = self.decoder.generate_from_signature(
            response_sig, 
            max_tokens=15,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty
        )
        
        # Step 5: Update context AFTER generating reply
        if self.use_decay and self.context_sig is not None:
            # Smooth decay: keeps memory alive but prevents it from dominating forever
            # Decay uses current_sig directly, not merged_sig
            self.context_sig = self.decay_alpha * self.context_sig + self.decay_beta * current_sig
        else:
            # Standard update: replace context with merged signature
            self.context_sig = merged_sig
        
        return {
            'query': query,
            'reply': reply,
            'current_signature': current_sig.tolist(),
            'merged_signature': merged_sig.tolist(),
            'response_signature': response_sig.tolist(),
            'used_context': had_context,
            'transform_type': transform_type
        }
    
    def chat(self, query: str, **kwargs) -> str:
        """
        Simple chat interface - just returns the reply string.
        
        Args:
            query: Input query
            **kwargs: Additional arguments for generate_reply
        
        Returns:
            Reply string
        """
        result = self.generate_reply(query, **kwargs)
        return result['reply']
    
    def reset_context(self):
        """Reset conversation context."""
        self.context_sig = None
    
    def enable_decay(self, decay_alpha: float = 0.85, decay_beta: float = 0.15):
        """
        Enable smooth context decay to prevent long hallucinations.
        
        Args:
            decay_alpha: Weight for old context (default 0.85)
            decay_beta: Weight for new input (default 0.15)
        """
        self.use_decay = True
        self.decay_alpha = decay_alpha
        self.decay_beta = decay_beta
    
    def disable_decay(self):
        """Disable context decay (use standard merge update)."""
        self.use_decay = False
    
    def set_merge_weights(self, alpha: float = 0.7, beta: float = 0.3):
        """
        Set weights for merging current input with context.
        
        Args:
            alpha: Weight for current signature (default 0.7)
            beta: Weight for context signature (default 0.3)
        """
        self.alpha = alpha
        self.beta = beta
    
    def set_context(self, context_sentences: List[str]):
        """Set context from list of sentences."""
        # Collapse all context sentences and merge them
        context_sigs = []
        for sentence in context_sentences:
            sig = self.interface.get_meaning_signature(
                sentence,
                collapse_steps=self.collapse_steps
            )
            self.interface.reset_geometry()
            context_sigs.append(sig)
        
        # Merge all context signatures into one
        if context_sigs:
            self.context_sig = np.mean(context_sigs, axis=0)
        else:
            self.context_sig = None


def main():
    """Demo: Chat with the reply generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Livnium Reply Generator")
    parser.add_argument("--query", type=str, required=True,
                       help="Query sentence")
    parser.add_argument("--patterns", type=str, 
                       default="nova/model/learned_patterns.json",
                       help="Path to learned patterns file (for old decoder)")
    parser.add_argument("--clusters", type=str,
                       default="nova/model/geometric_clusters.json",
                       help="Path to geometric clusters file (for cluster decoder)")
    parser.add_argument("--use-old-decoder", action="store_true",
                       help="Use old sentence decoder instead of cluster decoder")
    parser.add_argument("--collapse-steps", type=int, default=15,
                       help="Number of collapse steps")
    parser.add_argument("--lattice-size", type=int, default=3,
                       help="Geometry lattice size")
    parser.add_argument("--impulse-scale", type=float, default=0.1,
                       help="Character impulse scale")
    parser.add_argument("--transform", type=str, default="merge",
                       choices=["merge", "rotate", "project"],
                       help="Transformation type")
    parser.add_argument("--no-context", action="store_true",
                       help="Don't use conversation context")
    
    args = parser.parse_args()
    
    # Initialize
    interface = TextToGeometry(
        lattice_size=args.lattice_size,
        impulse_scale=args.impulse_scale
    )
    
    patterns_path = Path(args.patterns) if args.patterns and args.use_old_decoder else None
    cluster_path = Path(args.clusters) if args.clusters and not args.use_old_decoder else None
    
    generator = ReplyGenerator(
        interface,
        signature_database_path=patterns_path,
        cluster_path=cluster_path,
        collapse_steps=args.collapse_steps,
        use_cluster_decoder=not args.use_old_decoder
    )
    
    # Generate reply
    print("=" * 70)
    print("Livnium Reply Generator - Pure Generation")
    print("=" * 70)
    print()
    print(f"Query: {args.query}")
    print()
    
    result = generator.generate_reply(
        args.query,
        use_context=not args.no_context,
        transform_type=args.transform
    )
    
    print(f"Reply: {result['reply']}")
    print()
    print(f"Transform: {result['transform_type']}")
    print(f"Context used: {result['used_context']}")
    print()


if __name__ == "__main__":
    main()
