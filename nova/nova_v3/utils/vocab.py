"""
Vocabulary Builder

Simple vocabulary from text.
"""

from typing import List, Dict, Set
from collections import Counter
import re


class Vocabulary:
    """
    Simple vocabulary class.
    """
    
    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        
        # Special tokens
        self.pad_idx = 0
        self.unk_idx = 1
        
        # Initialize with special tokens
        self.word2idx['<PAD>'] = self.pad_idx
        self.word2idx['<UNK>'] = self.unk_idx
        self.idx2word[self.pad_idx] = '<PAD>'
        self.idx2word[self.unk_idx] = '<UNK>'
        
        self.next_idx = 2
    
    def add_word(self, word: str):
        """Add word to vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.next_idx
            self.idx2word[self.next_idx] = word
            self.next_idx += 1
        self.word_counts[word] += 1
    
    def build_from_texts(self, texts: List[str], min_count: int = 1):
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of text strings
            min_count: Minimum word count to include
        """
        # Count words
        for text in texts:
            words = self.tokenize(text)
            for word in words:
                self.word_counts[word] += 1
        
        # Add words that meet min_count
        for word, count in self.word_counts.items():
            if count >= min_count and word not in self.word2idx:
                self.word2idx[word] = self.next_idx
                self.idx2word[self.next_idx] = word
                self.next_idx += 1
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on whitespace
        words = re.findall(r'\w+', text.lower())
        return words
    
    def encode(self, text: str, max_len: int = None) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            max_len: Maximum sequence length (None = no limit)
            
        Returns:
            List of token IDs
        """
        words = self.tokenize(text)
        ids = [self.word2idx.get(word, self.unk_idx) for word in words]
        
        if max_len:
            ids = ids[:max_len]
            # Pad if needed
            while len(ids) < max_len:
                ids.append(self.pad_idx)
        
        return ids
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)

    def id_to_token_list(self) -> List[str]:
        """Return list mapping id -> token (index-aligned)."""
        return [self.idx2word.get(i, "") for i in range(len(self))]


def build_vocab_from_snli(samples: List[Dict], min_count: int = 2) -> Vocabulary:
    """
    Build vocabulary from SNLI samples.
    
    Args:
        samples: List of SNLI samples (dict with 'premise' and 'hypothesis')
        min_count: Minimum word count
        
    Returns:
        Vocabulary object
    """
    vocab = Vocabulary()
    
    # Collect all texts
    texts = []
    for sample in samples:
        texts.append(sample['premise'])
        texts.append(sample['hypothesis'])
    
    # Build vocabulary
    vocab.build_from_texts(texts, min_count=min_count)
    
    return vocab
