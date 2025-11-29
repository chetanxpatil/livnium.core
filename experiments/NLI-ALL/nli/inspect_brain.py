"""
Inspect Brain: View Learned Word Weights & Polarities

Run this to see what the GlobalLexicon has actually learned.
"""
import os
import sys
import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.nli.native_chain import GlobalLexicon

def main():
    print("="*60)
    print("LIVNIUM BRAIN INSPECTION")
    print("="*60)
    
    lexicon = GlobalLexicon()
    
    # Try to load brain from disk
    brain_path = os.path.join(os.path.dirname(__file__), 'brain_state.pkl')
    if os.path.exists(brain_path):
        print(f"Loading brain from: {brain_path}")
        if lexicon.load_from_file(brain_path):
            print("✓ Brain loaded successfully!")
        else:
            print("⚠️  Failed to load brain file")
    else:
        print(f"⚠️  Brain file not found: {brain_path}")
        print("   (Brain is only in memory during training)")
    
    print()
    
    # Check if brain is empty
    if not lexicon.polarity_store:
        print("Brain is empty! Run training first.")
        print("(No learned word polarities found)")
        print()
        print("To save the brain after training, make sure train_moksha_nli.py")
        print("saves the brain state at the end of training.")
        return

    print(f"Total Words Learned: {len(lexicon.polarity_store)}")
    print(f"Total Letters Learned: {len(lexicon.letter_store)}")
    print("-" * 60)

    # 1. CHECK POLARITY (Semantics)
    # Find words that strongly pull towards each class
    
    # Entailment words (Class 0)
    entailment_words = []
    for word in lexicon.polarity_store:
        polarity = lexicon.polarity_store[word]  # [E, C, N]
        entailment_score = polarity[0]
        contradiction_score = polarity[1]
        neutral_score = polarity[2]
        # Only show words with strong class preference
        max_score = max(entailment_score, contradiction_score, neutral_score)
        if max_score > 0.5:  # Has some class preference
            entailment_words.append((word, entailment_score, contradiction_score, neutral_score))
    
    entailment_words.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTOP 20 ENTALLMENT WORDS (Learned from Data):")
    print(f"{'WORD':<20} {'E-SCORE':<10} {'C-SCORE':<10} {'N-SCORE':<10}")
    for w, e, c, n in entailment_words[:20]:
        print(f"{w:<20} {e:.4f}     {c:.4f}     {n:.4f}")

    # Contradiction words (Class 1)
    contradiction_words = []
    for word in lexicon.polarity_store:
        polarity = lexicon.polarity_store[word]  # [E, C, N]
        contradiction_score = polarity[1]
        entailment_score = polarity[0]
        neutral_score = polarity[2]
        max_score = max(entailment_score, contradiction_score, neutral_score)
        if max_score > 0.5:
            contradiction_words.append((word, entailment_score, contradiction_score, neutral_score))
    
    contradiction_words.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTOP 20 CONTRADICTION WORDS (Learned Negation):")
    print(f"{'WORD':<20} {'E-SCORE':<10} {'C-SCORE':<10} {'N-SCORE':<10}")
    for w, e, c, n in contradiction_words[:20]:
        print(f"{w:<20} {e:.4f}     {c:.4f}     {n:.4f}")

    # Neutral words (Class 2) - words that appear in all classes equally
    neutral_words = []
    for word in lexicon.polarity_store:
        polarity = lexicon.polarity_store[word]  # [E, C, N]
        # Neutral words have balanced polarity (all ~0.33)
        variance = np.var(polarity)
        if variance < 0.01:  # Low variance = balanced = neutral
            neutral_words.append((word, polarity[0], polarity[1], polarity[2]))
    
    neutral_words.sort(key=lambda x: x[1])  # Sort by first score (all similar anyway)
    
    print("\nTOP 20 NEUTRAL WORDS (Balanced Across Classes):")
    print(f"{'WORD':<20} {'E-SCORE':<10} {'C-SCORE':<10} {'N-SCORE':<10}")
    for w, e, c, n in neutral_words[:20]:
        print(f"{w:<20} {e:.4f}     {c:.4f}     {n:.4f}")

if __name__ == "__main__":
    main()