"""
Sentence Decoder: Generate Text from Geometric Signatures

Pure generation - learns patterns from training data, generates new text.
NO search. NO hardcoded templates.
"""

import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path
from collections import defaultdict, Counter


class SentenceDecoder:
    """
    Generates sentences from geometric signatures using learned patterns.
    
    Learns from training data:
    - Word patterns (which words appear with which signature patterns)
    - Sentence structures (learned from real sentences)
    - Word sequences (learned n-grams)
    """
    
    def __init__(self, signature_database_path: Optional[Path] = None):
        """
        Initialize decoder with learned patterns.
        
        Args:
            signature_database_path: Path to training data (for learning patterns)
        """
        self.word_patterns = {}  # signature region → common words
        self.sentence_structures = []  # learned sentence patterns
        self.word_sequences = defaultdict(list)  # learned word sequences
        self.vocabulary = set()
        
        if signature_database_path and signature_database_path.exists():
            self.learn_from_data(signature_database_path)
    
    def learn_from_data(self, patterns_path: Path):
        """
        Load learned patterns from file (no sentences stored).
        
        Like NLI's GlobalLexicon - only loads learned patterns:
        - word_patterns: signature region -> words
        - word_sequences: learned n-grams
        - vocabulary: all learned words
        """
        print(f"Loading learned patterns from {patterns_path}...")
        
        with open(patterns_path) as f:
            data = json.load(f)
        
        # Check if it's new format (pattern-only) or old format (with sentences)
        if 'word_patterns' in data:
            # New format: pattern-only (like NLI)
            word_patterns_json = data['word_patterns']
            word_sequences_json = data.get('word_sequences', {})
            vocabulary_list = data.get('vocabulary', [])
            
            # Convert back from JSON format
            # Convert string keys back to tuples
            import ast
            import re
            for key_str, words in word_patterns_json.items():
                # Parse "[1.2, 3.4, ...]" back to tuple
                try:
                    # Handle old format with np.float64() calls
                    # Replace "np.float64(47.9)" with "47.9"
                    cleaned = re.sub(r'np\.float64\(([^)]+)\)', r'\1', key_str)
                    key_list = ast.literal_eval(cleaned)
                    key_tuple = tuple(float(x) for x in key_list)
                    self.word_patterns[key_tuple] = words
                except (ValueError, SyntaxError) as e:
                    # Skip malformed keys
                    continue
            
            # Convert word_sequences back to tuples
            for word, seqs in word_sequences_json.items():
                self.word_sequences[word] = [tuple(seq) for seq in seqs]
            
            # Set vocabulary
            self.vocabulary = set(vocabulary_list)
            
            print(f"✓ Loaded {len(self.word_patterns)} word patterns")
            print(f"✓ Vocabulary: {len(self.vocabulary)} words")
        else:
            # Old format: full database with sentences (backward compatibility)
            print("⚠️  Old format detected - extracting patterns from database...")
            signatures = data.get('signatures', [])
            
            # Learn word patterns from signature regions
            for sig_data in signatures:
                sentence = sig_data['sentence']
                signature = np.array(sig_data['signature'])
                
                # Extract words
                words = sentence.lower().split()
                self.vocabulary.update(words)
                
                # Learn: which signature regions correspond to which words
                num_regions = min(10, len(signature))
                region_size = len(signature) // num_regions
                
                for i, word in enumerate(words):
                    # NO FILTERING - include all words
                    # Map word to signature region
                    region_idx = min(i % num_regions, num_regions - 1)
                    region_start = region_idx * region_size
                    region_end = region_start + region_size
                    region_sig = signature[region_start:region_end]
                    
                    # Store word with its signature region pattern
                    region_key = tuple(np.round(region_sig, 2))
                    if region_key not in self.word_patterns:
                        self.word_patterns[region_key] = []
                    self.word_patterns[region_key].append(word)
                
                # Learn sentence structures (word patterns) - NO MINIMUM LENGTH
                if len(words) >= 1:  # Changed from 3 to 1
                    # Store 2-word and 3-word sequences
                    for i in range(len(words) - 1):
                        if i + 1 < len(words):
                            seq = tuple(words[i:i+2])
                            self.word_sequences[words[i]].append(seq)
                        if i + 2 < len(words):
                            seq = tuple(words[i:i+3])
                            self.word_sequences[words[i]].append(seq)
            
            # NO FILTERING - keep ALL words per pattern
            # Just deduplicate while preserving order
            for region_key in self.word_patterns:
                words = self.word_patterns[region_key]
                # Remove duplicates but keep all words
                seen = set()
                unique_words = []
                for word in words:
                    if word not in seen:
                        seen.add(word)
                        unique_words.append(word)
                self.word_patterns[region_key] = unique_words
            
            print(f"✓ Learned {len(self.word_patterns)} word patterns")
            print(f"✓ Vocabulary: {len(self.vocabulary)} words")
    
    def generate_from_signature(self, signature: np.ndarray) -> str:
        """
        Generate NEW sentence from signature using learned patterns.
        
        NO hardcoded templates. NO search. NO fallback.
        Pure generation from learned word patterns only.
        
        Args:
            signature: Geometric signature to generate from
        
        Returns:
            Generated sentence (empty if no patterns learned)
        """
        if not self.word_patterns or not self.vocabulary:
            # No training data - return empty
            return ""
        
        # Divide signature into regions
        num_regions = min(10, len(signature))
        region_size = len(signature) // num_regions
        
        # Generate words based on signature regions
        generated_words = []
        used_words = set()
        
        for i in range(num_regions):
            region_start = i * region_size
            region_end = region_start + region_size
            region_sig = signature[region_start:region_end]
            
            # Find closest matching region pattern
            region_key = self._find_closest_region(region_sig)
            
            if region_key and region_key in self.word_patterns:
                # Get words for this region
                candidates = self.word_patterns[region_key]
                # Pick word not yet used
                for word in candidates:
                    if word not in used_words:
                        generated_words.append(word)
                        used_words.add(word)
                        break
        
        # If we have words, build sentence
        if generated_words:
            # Use learned sequences to connect words
            sentence_words = self._build_sentence_from_words(generated_words)
            sentence = " ".join(sentence_words)
        else:
            # No words generated - return empty
            return ""
        
        # Clean up
        if sentence:
            sentence = sentence.capitalize()
            if sentence[-1] not in '.!?':
                sentence += '.'
        
        return sentence
    
    def _find_closest_region(self, region_sig: np.ndarray) -> Optional[tuple]:
        """Find closest matching region pattern."""
        if not self.word_patterns:
            return None
        
        region_sig_rounded = tuple(np.round(region_sig, 2))
        
        # Exact match
        if region_sig_rounded in self.word_patterns:
            return region_sig_rounded
        
        # Find closest by Euclidean distance
        min_dist = float('inf')
        closest_key = None
        
        for key in self.word_patterns.keys():
            key_array = np.array(key)
            if len(key_array) == len(region_sig):
                dist = np.linalg.norm(key_array - region_sig)
                if dist < min_dist:
                    min_dist = dist
                    closest_key = key
        
        return closest_key
    
    def _build_sentence_from_words(self, words: List[str]) -> List[str]:
        """Build sentence using learned word sequences."""
        if not words:
            return []
        
        if len(words) == 1:
            return words
        
        # Use learned sequences to connect words
        sentence = [words[0]]
        
        for i in range(1, len(words)):
            prev_word = sentence[-1]
            current_word = words[i]
            
            # Check if we have learned sequence
            if prev_word in self.word_sequences:
                sequences = self.word_sequences[prev_word]
                # Find sequence that includes current_word
                for seq in sequences:
                    if len(seq) >= 2 and seq[1] == current_word:
                        # Use the sequence
                        sentence.extend(list(seq[1:]))
                        break
                else:
                    # No learned sequence, just add word
                    sentence.append(current_word)
            else:
                sentence.append(current_word)
        
        # Limit length
        return sentence[:15]
    
    def decode(self, signature: np.ndarray, **kwargs) -> str:
        """
        Main decode method - always generates.
        
        Args:
            signature: Geometric signature to decode
            **kwargs: Ignored (for compatibility)
        
        Returns:
            Generated sentence
        """
        return self.generate_from_signature(signature)
