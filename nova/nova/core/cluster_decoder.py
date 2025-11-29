"""
Cluster-Based Decoder (Markov Enhanced)

1. Query -> Collapse -> Cluster ID (The "Topic")
2. Cluster -> Set of valid tokens (The "Vocabulary")
3. Markov Chain -> Orders the tokens (The "Grammar")
"""

import numpy as np
import random
import json
from typing import List, Optional, Dict
from pathlib import Path
from nova.core.geometric_token_learner import GeometricTokenLearner
from nova.core.text_to_geometry import TextToGeometry


class ClusterDecoder:
    def __init__(self, 
                 token_learner: GeometricTokenLearner, 
                 interface: TextToGeometry,
                 patterns_path: Optional[Path] = None,
                 collapse_steps: int = 15,
                 phase1_mode: bool = False):
        """
        Initialize ClusterDecoder.
        
        Args:
            token_learner: GeometricTokenLearner instance
            interface: TextToGeometry interface
            patterns_path: Path to learned patterns JSON file
            collapse_steps: Number of collapse steps
            phase1_mode: If True, output only "entailment", "contradiction", or "neutral"
        """
        self.token_learner = token_learner
        self.interface = interface
        self.collapse_steps = collapse_steps
        self.phase1_mode = phase1_mode
        
        # Phase 1: Only these three words allowed
        self.phase1_vocab = {'entailment', 'contradiction', 'neutral'}
        
        # Load Markov Transitions (Grammar)
        self.word_sequences = {}
        if patterns_path and patterns_path.exists():
            self._load_grammar(patterns_path)
        else:
            if not phase1_mode:
                print("⚠ Warning: No grammar file found. Decoder will output word salad.")
            
    def _load_grammar(self, path: Path):
        print(f"Loading grammar from {path}...")
        try:
            with open(path) as f:
                data = json.load(f)
                # Handle new format (pattern-only) or old format
                raw_seqs = data.get('word_sequences', {})
                
                count = 0
                for word, seqs in raw_seqs.items():
                    next_words = []
                    for seq in seqs:
                        # seq is a list like ['word', 'next_word']
                        if len(seq) > 1:
                            next_words.append(seq[1])  # Get the word that follows
                    if next_words:
                        self.word_sequences[word] = next_words
                        count += 1
            print(f"✓ Grammar loaded: {count} words have transitions")
        except Exception as e:
            print(f"⚠ Error loading grammar: {e}")

    def generate_reply(self, query: str, max_tokens: int = 15) -> str:
        # 1. Get Signature
        try:
            query_sig = self.interface.get_meaning_signature(query, collapse_steps=self.collapse_steps)
            self.interface.reset_geometry()
        except Exception as e:
            print(f"Error getting signature: {e}")
            return ""

        return self.generate_from_signature(query_sig, max_tokens)

    def generate_from_signature(self, signature: np.ndarray, max_tokens: int = 15, 
                                temperature: float = 0.7, repetition_penalty: float = 0.1) -> str:
        """
        Generate text from signature with repetition penalty and temperature sampling.
        
        Args:
            signature: Geometric signature
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = very random)
            repetition_penalty: Penalty multiplier for repeated words (0.0-1.0)
        """
        # Phase 1 Mode: Output only E/C/N
        if self.phase1_mode:
            return self._generate_phase1(signature, temperature)
        
        # 2. Identify the Semantic Cluster (The "Topic")
        cluster_id = self.token_learner.get_cluster_id(signature)
        
        if cluster_id not in self.token_learner.cluster_tokens:
            return "..."
            
        cluster_counts = self.token_learner.cluster_tokens[cluster_id]
        cluster_vocab = set(cluster_counts.keys())
        
        if not cluster_vocab:
            return "..."

        # 3. Generate Sequence (The "Grammar")
        words = list(cluster_counts.keys())
        # Safety check
        if not words: 
            return "..."
        
        # Weighted random start
        base_probs = np.array(list(cluster_counts.values()), dtype=float)
        if base_probs.sum() > 0:
            base_probs /= base_probs.sum()
        else:
            base_probs = np.ones(len(words)) / len(words)
        
        current_word = np.random.choice(words, p=base_probs)
        reply_tokens = [current_word]
        
        # Track recent words to punish repetition (sliding window)
        recent_history = [current_word]
        max_history = 3
        
        for _ in range(max_tokens - 1):
            # Find valid next words from Grammar
            possible_next_words = self.word_sequences.get(current_word, [])
            
            # INTERSECTION: Grammar AND Cluster
            valid_candidates = [w for w in possible_next_words if w in cluster_vocab]
            
            if not valid_candidates:
                # --- PATCH: Intelligent Fallback ---
                # OLD BROKEN LOGIC: valid_candidates = words (entire cluster vocabulary)
                # NEW LOGIC: If we fall off the grammar manifold, stick to high-probability core
                # Sort by frequency in this cluster
                top_words = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Take top 50 (or 20% of cluster, whichever is smaller) to ensure coherence
                limit = min(50, len(top_words))
                valid_candidates = [w for w, count in top_words[:limit]]
                
                # Calculate probabilities for top words
                candidate_probs = np.array([
                    cluster_counts.get(w, 1) for w in valid_candidates
                ], dtype=float)
                if candidate_probs.sum() > 0:
                    candidate_probs /= candidate_probs.sum()
                else:
                    candidate_probs = np.ones(len(valid_candidates)) / len(valid_candidates)
            else:
                # Calculate probabilities based on frequency in cluster
                candidate_probs = np.array([
                    cluster_counts.get(w, 1) for w in valid_candidates
                ], dtype=float)
                if candidate_probs.sum() > 0:
                    candidate_probs /= candidate_probs.sum()
                else:
                    candidate_probs = np.ones(len(valid_candidates)) / len(valid_candidates)
            
            # --- REPETITION PENALTY ---
            weights = []
            for i, candidate in enumerate(valid_candidates):
                score = candidate_probs[i]
                
                # Heavy penalty for immediate repetition
                if candidate == current_word:
                    score *= 0.05  # Almost ban immediate self-loops
                
                # Penalty for recent words (sliding window)
                if candidate in recent_history:
                    # More recent = heavier penalty
                    position = recent_history.index(candidate)
                    penalty = repetition_penalty * (1.0 - position / len(recent_history))
                    score *= penalty
                
                weights.append(max(score, 1e-10))  # Prevent zero weights
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)
            
            # --- TEMPERATURE SAMPLING ---
            # Apply temperature: higher = more random, lower = more deterministic
            if temperature > 0:
                # Convert to log space, apply temperature, convert back
                log_weights = np.log(np.array(weights) + 1e-10)
                tempered_weights = np.exp(log_weights / temperature)
                tempered_weights /= tempered_weights.sum()
                weights = tempered_weights.tolist()
            
            # Sample based on weights
            try:
                current_word = random.choices(valid_candidates, weights=weights, k=1)[0]
            except (ValueError, IndexError):
                # Fallback to random choice
                current_word = random.choice(valid_candidates) if valid_candidates else words[0]
            
            reply_tokens.append(current_word)
            
            # Update recent history (sliding window)
            recent_history.append(current_word)
            if len(recent_history) > max_history:
                recent_history.pop(0)
            
            # --- DYNAMIC STOP CONDITION ---
            # Stop on punctuation or if we have a reasonable length
            if current_word in ['.', '!', '?', '...']:
                break
            
            # Stop if we have decent length and hit a natural break
            if len(reply_tokens) >= 5 and current_word in [',', ';', ':', 'and', 'but', 'or']:
                if random.random() < 0.3:  # 30% chance to stop at natural breaks
                    break

        # Formatting
        text = " ".join(reply_tokens)
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
        text = text.replace("  ", " ")  # Remove double spaces
        if text: 
            text = text.strip()
            if text:
                text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            
        return text
    
    def _generate_phase1(self, signature: np.ndarray, temperature: float = 0.7) -> str:
        """
        Phase 1 generation: Output ONLY "entailment", "contradiction", or "neutral".
        
        Discipline:
        - Must output exactly one word
        - Word must be in phase1_vocab
        - Reward correct, punish wrong
        """
        # Get cluster
        cluster_id = self.token_learner.get_cluster_id(signature)
        
        if cluster_id not in self.token_learner.cluster_tokens:
            # Fallback: return most common label
            return "neutral"
        
        cluster_counts = self.token_learner.cluster_tokens[cluster_id]
        
        # Filter to only Phase 1 vocabulary
        phase1_counts = {
            word: count 
            for word, count in cluster_counts.items() 
            if word.lower() in self.phase1_vocab
        }
        
        if not phase1_counts:
            # No E/C/N in this cluster - find closest cluster or return default
            # Try to find E/C/N in nearby clusters
            for cid in range(self.token_learner.num_clusters):
                if cid == cluster_id:
                    continue
                if cid in self.token_learner.cluster_tokens:
                    nearby_counts = self.token_learner.cluster_tokens[cid]
                    phase1_counts = {
                        word: count 
                        for word, count in nearby_counts.items() 
                        if word.lower() in self.phase1_vocab
                    }
                    if phase1_counts:
                        break
            
            # Still no E/C/N found - return default
            if not phase1_counts:
                return "neutral"
        
        # Select word based on frequency (with temperature)
        words = list(phase1_counts.keys())
        counts = np.array(list(phase1_counts.values()), dtype=float)
        
        # Normalize to probabilities
        if counts.sum() > 0:
            probs = counts / counts.sum()
        else:
            probs = np.ones(len(words)) / len(words)
        
        # Apply temperature
        if temperature > 0:
            log_probs = np.log(probs + 1e-10)
            tempered_probs = np.exp(log_probs / temperature)
            tempered_probs /= tempered_probs.sum()
            probs = tempered_probs
        
        # Sample
        selected_word = np.random.choice(words, p=probs)
        
        # Phase 1 Discipline: Return exactly one word, no extra words
        return selected_word.lower()
