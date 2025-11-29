"""
Interactive Chat - Pure Generation

NO search. Only real generation from learned patterns.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path (go up two levels: chat -> nova -> project_root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nova.core.text_to_geometry import TextToGeometry
from nova.chat.reply_generator import ReplyGenerator


def main():
    """Interactive chat demo."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice-size", type=int, default=None, help="Lattice size (auto-detected from model if not specified)")
    parser.add_argument("--collapse-steps", type=int, default=12, help="Collapse steps (default 12, lower preserves emotional details)")
    parser.add_argument("--impulse-scale", type=float, default=0.1, help="Impulse scale (default 0.1, lower = more stable emotional patterns)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0=deterministic, 1.0=very random, default 0.7)")
    parser.add_argument("--repetition-penalty", type=float, default=0.1, help="Repetition penalty (0.0-1.0, lower=heavier penalty, default 0.1)")
    parser.add_argument("--context-alpha", type=float, default=0.4, help="Current query weight (default 0.4, lower = more context)")
    parser.add_argument("--context-beta", type=float, default=0.6, help="Context history weight (default 0.6, higher = more stable empathy)")
    args = parser.parse_args()

    # Paths
    cluster_path = Path("nova/model/geometric_clusters")
    cluster_json_path = cluster_path.with_suffix('.json')
    cluster_pkl_path = cluster_path.with_suffix('.pkl')
    patterns_path = Path("nova/model/learned_patterns.json")
    
    use_cluster_decoder = cluster_json_path.exists() and cluster_pkl_path.exists()
    
    # Auto-detect lattice size from saved model (dynamic)
    actual_lattice_size = args.lattice_size
    if use_cluster_decoder:
        try:
            import json
            with open(cluster_json_path, 'r') as f:
                cluster_data = json.load(f)
                saved_lattice_size = cluster_data.get('lattice_size', None)
                if saved_lattice_size:
                    actual_lattice_size = saved_lattice_size
                    if args.lattice_size and args.lattice_size != saved_lattice_size:
                        print(f"⚠ Model was trained with lattice_size={saved_lattice_size}, using that instead of {args.lattice_size}")
                    else:
                        print(f"✓ Auto-detected lattice_size={saved_lattice_size} from saved model")
                elif args.lattice_size:
                    actual_lattice_size = args.lattice_size
                    print(f"⚠ Could not read lattice_size from model, using provided: {args.lattice_size}")
                else:
                    actual_lattice_size = 3  # Default fallback
                    print(f"⚠ Could not read lattice_size from model, using default: 3")
        except Exception as e:
            print(f"⚠ Could not read lattice_size from model: {e}")
            actual_lattice_size = args.lattice_size or 3
            print(f"  Using lattice_size={actual_lattice_size}")
    else:
        # No model found, use provided or default
        actual_lattice_size = args.lattice_size or 3
    
    print("=" * 70)
    print("Livnium Chat - Pure Generation")
    print(f"Geometry: {actual_lattice_size}x{actual_lattice_size}x{actual_lattice_size} ({actual_lattice_size**3} dimensions)")
    print(f"Steps: {args.collapse_steps}, Impulse: {args.impulse_scale}")
    print(f"Context: {args.context_alpha:.1f}×current + {args.context_beta:.1f}×history")
    print("=" * 70)
    print()
    print("Type 'quit' or 'exit' to end the conversation.")
    print("Type 'reset' to clear conversation context.")
    print()
    
    # Initialize interface with detected/configured lattice size
    interface = TextToGeometry(lattice_size=actual_lattice_size, impulse_scale=args.impulse_scale)
    
    if not use_cluster_decoder:
        print("⚠ Warning: Geometric clusters not found.")
        print("  Run: python3 nova/training/train_text_to_geometry.py")
        if patterns_path.exists():
            print("  Falling back to old decoder (learned_patterns.json)")
            use_cluster_decoder = False
        else:
            print("  Will use empty generation (no patterns learned).")
            cluster_path = None
            patterns_path = None
    
    # Initialize Generator
    generator = ReplyGenerator(
        interface,
        # IMPORTANT: Pass patterns_path for grammar even if using cluster decoder
        signature_database_path=patterns_path,
        cluster_path=cluster_path if use_cluster_decoder else None,
        collapse_steps=args.collapse_steps, 
        use_cluster_decoder=use_cluster_decoder,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty
    )
    
    # Set context blending for stable emotional responses
    # Higher beta (0.6) = more weight on conversation history = more stable empathy
    generator.set_merge_weights(alpha=args.context_alpha, beta=args.context_beta)
    
    if use_cluster_decoder:
        print("✓ Using cluster-based decoder (Geometric Topics + Markov Grammar)")
    else:
        print("✓ Using old decoder (learned patterns)")
    
    print("Ready! Start chatting...")
    print()
    
    # Chat loop
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if query.lower() == 'reset':
                generator.reset_context()
                print("Context reset.")
                continue
            
            # Generate reply with context awareness
            result = generator.generate_reply(query, use_context=True)
            reply = result['reply']
            
            # Show context status (for debugging)
            if result.get('used_context', False):
                print(f"Bot: {reply} [context: ON]")
            else:
                print(f"Bot: {reply} [context: OFF]")
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    main()
