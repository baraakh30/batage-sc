"""
Global History Register

Manages the global branch history used by predictors.
Supports both speculative and committed history.
"""

import numpy as np
from typing import Optional
from collections import deque


class GlobalHistoryRegister:
    """
    Global Branch History Register.
    
    Maintains a shift register of recent branch outcomes.
    Supports both bipolar (-1/+1) and binary (0/1) encodings.
    """
    
    def __init__(self, length: int = 64, 
                 use_path_history: bool = False,
                 path_bits: int = 16):
        """
        Initialize the history register.
        
        Args:
            length: Number of branch outcomes to track
            use_path_history: Include path (address) information
            path_bits: Bits to use from branch addresses for path history
        """
        self.length = length
        self.use_path_history = use_path_history
        self.path_bits = path_bits
        
        # Main history register (binary: 0 = not taken, 1 = taken)
        self._history = np.zeros(length, dtype=np.int8)
        
        # Path history (folded PC bits)
        self._path_history = np.zeros(path_bits, dtype=np.int8)
        
        # Speculative history for recovery
        self._speculative_history: Optional[np.ndarray] = None
        self._checkpoint_stack = deque(maxlen=64)
        
    def update(self, taken: bool, pc: int = 0) -> None:
        """
        Update history with new branch outcome.
        
        Args:
            taken: Branch outcome (True = taken)
            pc: Program counter (for path history)
        """
        # Shift history left and insert new outcome
        self._history = np.roll(self._history, 1)
        self._history[0] = 1 if taken else 0
        
        # Update path history if enabled
        if self.use_path_history and pc != 0:
            self._update_path_history(pc)
    
    def _update_path_history(self, pc: int) -> None:
        """Update path history with branch address."""
        # Fold PC into path history using XOR
        for i in range(self.path_bits):
            bit = (pc >> i) & 1
            self._path_history[i] ^= bit
    
    def get_history(self, as_bipolar: bool = True) -> np.ndarray:
        """
        Get current history.
        
        Args:
            as_bipolar: Return as bipolar (-1/+1) instead of binary (0/1)
            
        Returns:
            History array
        """
        if as_bipolar:
            return np.where(self._history > 0, 1, -1)
        return self._history.copy()
    
    def get_combined_history(self, as_bipolar: bool = True) -> np.ndarray:
        """
        Get history combined with path history.
        
        Returns:
            Combined history array [branch_history | path_history]
        """
        history = self.get_history(as_bipolar)
        
        if not self.use_path_history:
            return history
        
        path = self._path_history if not as_bipolar else \
               np.where(self._path_history > 0, 1, -1)
        
        return np.concatenate([history, path])
    
    def checkpoint(self) -> int:
        """
        Create a checkpoint for speculative execution.
        
        Returns:
            Checkpoint ID
        """
        checkpoint = {
            'history': self._history.copy(),
            'path_history': self._path_history.copy()
        }
        self._checkpoint_stack.append(checkpoint)
        return len(self._checkpoint_stack) - 1
    
    def restore(self, checkpoint_id: Optional[int] = None) -> None:
        """
        Restore history to a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore (default: most recent)
        """
        if not self._checkpoint_stack:
            return
        
        if checkpoint_id is None:
            checkpoint = self._checkpoint_stack.pop()
        else:
            # Restore to specific checkpoint and discard newer ones
            while len(self._checkpoint_stack) > checkpoint_id + 1:
                self._checkpoint_stack.pop()
            checkpoint = self._checkpoint_stack.pop()
        
        self._history = checkpoint['history']
        self._path_history = checkpoint['path_history']
    
    def speculative_update(self, taken: bool, pc: int = 0) -> None:
        """
        Speculatively update history (can be rolled back).
        """
        if self._speculative_history is None:
            self._speculative_history = self._history.copy()
        
        self.update(taken, pc)
    
    def commit_speculative(self) -> None:
        """Commit speculative updates."""
        self._speculative_history = None
    
    def rollback_speculative(self) -> None:
        """Roll back speculative updates."""
        if self._speculative_history is not None:
            self._history = self._speculative_history
            self._speculative_history = None
    
    def reset(self) -> None:
        """Reset history to initial state."""
        self._history.fill(0)
        self._path_history.fill(0)
        self._speculative_history = None
        self._checkpoint_stack.clear()
    
    def get_hash(self, bits: int = 16) -> int:
        """
        Get a hash of current history for table indexing.
        
        Args:
            bits: Number of bits in the hash
            
        Returns:
            Hash value
        """
        # Fold history into hash
        result = 0
        for i, bit in enumerate(self._history):
            if bit > 0:
                result ^= (1 << (i % bits))
        
        return result
    
    @property
    def total_length(self) -> int:
        """Total history length including path history."""
        if self.use_path_history:
            return self.length + self.path_bits
        return self.length
    
    def __len__(self) -> int:
        return self.length
    
    def __repr__(self) -> str:
        hist_str = ''.join(str(b) for b in self._history[:16])
        return f"GHR({self.length}): {hist_str}..."
