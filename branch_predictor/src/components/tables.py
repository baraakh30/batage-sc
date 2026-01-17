"""
Weight Tables and Indexing Schemes

Various table implementations and indexing functions for predictors.
"""

import numpy as np
from typing import Callable, Optional, Tuple


class WeightTable:
    """
    Weight table for perceptron-style predictors.
    
    Supports various indexing schemes and weight configurations.
    """
    
    def __init__(self, num_entries: int, weights_per_entry: int,
                 weight_bits: int = 8):
        """
        Initialize weight table.
        
        Args:
            num_entries: Number of table entries
            weights_per_entry: Weights per entry (e.g., history_length + 1)
            weight_bits: Bits per weight
        """
        self.num_entries = num_entries
        self.weights_per_entry = weights_per_entry
        self.weight_bits = weight_bits
        
        # Compute weight bounds
        self.weight_max = (1 << (weight_bits - 1)) - 1
        self.weight_min = -(1 << (weight_bits - 1))
        
        # Initialize table
        self.table = np.zeros((num_entries, weights_per_entry), dtype=np.int16)
        
        # Access statistics
        self.reads = 0
        self.writes = 0
        
    def read(self, index: int) -> np.ndarray:
        """Read weights at index."""
        self.reads += 1
        return self.table[index % self.num_entries]
    
    def write(self, index: int, weights: np.ndarray) -> None:
        """Write weights at index."""
        self.writes += 1
        clipped = np.clip(weights, self.weight_min, self.weight_max)
        self.table[index % self.num_entries] = clipped
    
    def update_weight(self, index: int, weight_idx: int, delta: int) -> None:
        """Update a single weight."""
        self.writes += 1
        idx = index % self.num_entries
        new_val = self.table[idx, weight_idx] + delta
        self.table[idx, weight_idx] = np.clip(new_val, 
                                               self.weight_min, 
                                               self.weight_max)
    
    def reset(self) -> None:
        """Reset table to zeros."""
        self.table.fill(0)
        self.reads = 0
        self.writes = 0
    
    def get_storage_bits(self) -> int:
        """Get total storage in bits."""
        return self.num_entries * self.weights_per_entry * self.weight_bits
    
    def get_statistics(self) -> dict:
        """Get table statistics."""
        flat = self.table.flatten()
        return {
            'entries': self.num_entries,
            'weights_per_entry': self.weights_per_entry,
            'weight_bits': self.weight_bits,
            'total_bits': self.get_storage_bits(),
            'total_kb': self.get_storage_bits() / 8 / 1024,
            'reads': self.reads,
            'writes': self.writes,
            'weight_mean': float(np.mean(flat)),
            'weight_std': float(np.std(flat)),
            'saturated_high': int(np.sum(flat == self.weight_max)),
            'saturated_low': int(np.sum(flat == self.weight_min))
        }


class IndexingScheme:
    """
    Various indexing schemes for predictor tables.
    """
    
    @staticmethod
    def simple_mod(pc: int, table_size: int) -> int:
        """Simple modulo indexing."""
        return pc % table_size
    
    @staticmethod
    def xor_fold(pc: int, history_int: int, table_size: int) -> int:
        """XOR PC with history (gshare-style)."""
        return (pc ^ history_int) % table_size
    
    @staticmethod
    def hash_pc(pc: int, table_size: int) -> int:
        """Hash PC for better distribution."""
        # Simple hash function
        h = pc
        h ^= (h >> 16)
        h *= 0x85ebca6b
        h ^= (h >> 13)
        h *= 0xc2b2ae35
        h ^= (h >> 16)
        return h % table_size
    
    @staticmethod
    def multi_hash(pc: int, history_int: int, 
                   table_size: int, table_idx: int) -> int:
        """
        Multi-table indexing with different hash per table.
        
        Args:
            pc: Program counter
            history_int: History as integer
            table_size: Size of each table
            table_idx: Which table (0, 1, 2, ...)
        """
        # Different folding for each table
        shift = table_idx * 3
        folded_pc = (pc >> shift) ^ pc
        folded_hist = (history_int >> shift) ^ history_int
        
        return (folded_pc ^ folded_hist) % table_size
    
    @staticmethod
    def history_to_int(history: np.ndarray, max_bits: int = 32) -> int:
        """Convert history array to integer."""
        result = 0
        for i, bit in enumerate(history[:max_bits]):
            if bit > 0:
                result |= (1 << i)
        return result


class TaggedTable:
    """
    Tagged table entry for TAGE-style predictors.
    
    Each entry has: tag, counter, useful bits
    """
    
    def __init__(self, num_entries: int, tag_bits: int = 8,
                 counter_bits: int = 3, useful_bits: int = 2):
        self.num_entries = num_entries
        self.tag_bits = tag_bits
        self.counter_bits = counter_bits
        self.useful_bits = useful_bits
        
        # Storage
        self.tags = np.zeros(num_entries, dtype=np.uint32)
        self.counters = np.zeros(num_entries, dtype=np.int8)
        self.useful = np.zeros(num_entries, dtype=np.uint8)
        self.valid = np.zeros(num_entries, dtype=bool)
        
        # Counter bounds
        self.counter_max = (1 << (counter_bits - 1)) - 1
        self.counter_min = -(1 << (counter_bits - 1))
        
        # Useful bounds
        self.useful_max = (1 << useful_bits) - 1
        
    def lookup(self, index: int, tag: int) -> Tuple[bool, bool, int]:
        """
        Lookup entry.
        
        Returns:
            (hit, prediction, confidence)
        """
        idx = index % self.num_entries
        
        if not self.valid[idx]:
            return False, False, 0
        
        tag_mask = (1 << self.tag_bits) - 1
        if (self.tags[idx] & tag_mask) != (tag & tag_mask):
            return False, False, 0
        
        prediction = self.counters[idx] >= 0
        confidence = abs(self.counters[idx])
        
        return True, prediction, confidence
    
    def allocate(self, index: int, tag: int, taken: bool) -> bool:
        """
        Allocate new entry.
        
        Returns:
            True if allocation successful
        """
        idx = index % self.num_entries
        
        # Check if we can allocate (existing entry not useful)
        if self.valid[idx] and self.useful[idx] > 0:
            self.useful[idx] -= 1
            return False
        
        # Allocate
        self.tags[idx] = tag
        self.counters[idx] = 1 if taken else -1  # Weak
        self.useful[idx] = 0
        self.valid[idx] = True
        
        return True
    
    def update(self, index: int, tag: int, taken: bool, 
               correct: bool) -> None:
        """Update existing entry."""
        idx = index % self.num_entries
        
        if not self.valid[idx]:
            return
        
        tag_mask = (1 << self.tag_bits) - 1
        if (self.tags[idx] & tag_mask) != (tag & tag_mask):
            return
        
        # Update counter
        if taken:
            self.counters[idx] = min(self.counter_max, 
                                     self.counters[idx] + 1)
        else:
            self.counters[idx] = max(self.counter_min,
                                     self.counters[idx] - 1)
        
        # Update useful
        if correct:
            self.useful[idx] = min(self.useful_max, self.useful[idx] + 1)
    
    def get_storage_bits(self) -> int:
        """Total storage in bits."""
        bits_per_entry = (self.tag_bits + self.counter_bits + 
                         self.useful_bits + 1)  # +1 for valid
        return self.num_entries * bits_per_entry
