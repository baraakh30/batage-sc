"""
Hard-to-Predict (H2P) Branch Detector

Standalone module for H2P detection.
(Also available as part of HybridPredictor)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import OrderedDict


class H2PDetector:
    """
    Hard-to-Predict Branch Detector.
    
    Uses per-branch misprediction rate tracking to identify
    branches that are difficult for the primary predictor.
    """
    
    def __init__(self, config: dict):
        """
        Initialize H2P detector.
        
        Args:
            config: Configuration dictionary with:
                - tracking_window: Window for misprediction tracking
                - h2p_threshold: Misprediction rate threshold for H2P
                - confidence_threshold: Low confidence threshold
                - max_tracked_branches: Maximum branches to track
        """
        self.window_size = config.get('tracking_window', 1000)
        self.h2p_threshold = config.get('h2p_threshold', 0.3)
        self.confidence_threshold = config.get('confidence_threshold', 0.1)
        self.max_tracked = config.get('max_tracked_branches', 8192)
        
        # Adaptive threshold settings
        self.adaptive = config.get('adaptive_threshold', True)
        self.min_threshold = config.get('min_threshold', 0.1)
        self.max_threshold = config.get('max_threshold', 0.5)
        
        # Per-branch tracking using LRU cache
        # Format: {pc: BranchRecord}
        self._branch_records: OrderedDict[int, 'BranchRecord'] = OrderedDict()
        
        # H2P classification cache
        self._h2p_cache: Dict[int, bool] = {}
        
        # Global statistics
        self._global_correct = 0
        self._global_total = 0
        
    def record_outcome(self, pc: int, correct: bool, 
                      confidence: Optional[float] = None) -> None:
        """
        Record prediction outcome for a branch.
        
        Args:
            pc: Program counter of the branch
            correct: Whether prediction was correct
            confidence: Prediction confidence (optional)
        """
        # Update global stats
        self._global_total += 1
        if correct:
            self._global_correct += 1
        
        # Get or create branch record
        if pc in self._branch_records:
            # Move to end (most recently used)
            self._branch_records.move_to_end(pc)
            record = self._branch_records[pc]
        else:
            # Evict oldest if at capacity
            if len(self._branch_records) >= self.max_tracked:
                oldest_pc, _ = self._branch_records.popitem(last=False)
                if oldest_pc in self._h2p_cache:
                    del self._h2p_cache[oldest_pc]
            
            record = BranchRecord(self.window_size)
            self._branch_records[pc] = record
        
        # Update record
        record.update(correct, confidence)
        
        # Invalidate H2P cache for this branch
        if pc in self._h2p_cache:
            del self._h2p_cache[pc]
    
    def is_h2p(self, pc: int, min_samples: int = 10) -> bool:
        """
        Check if a branch is classified as hard-to-predict.
        
        Args:
            pc: Program counter
            min_samples: Minimum samples required for classification
            
        Returns:
            True if branch is hard-to-predict
        """
        # Check cache
        if pc in self._h2p_cache:
            return self._h2p_cache[pc]
        
        # Check if we have enough data
        if pc not in self._branch_records:
            return False
        
        record = self._branch_records[pc]
        
        if record.total < min_samples:
            return False
        
        # Get adaptive threshold
        threshold = self._get_adaptive_threshold()
        
        # Check misprediction rate
        is_hard = record.misprediction_rate >= threshold
        
        # Cache result
        self._h2p_cache[pc] = is_hard
        
        return is_hard
    
    def get_misprediction_rate(self, pc: int) -> float:
        """Get misprediction rate for a specific branch."""
        if pc not in self._branch_records:
            return 0.0
        return self._branch_records[pc].misprediction_rate
    
    def get_branch_info(self, pc: int) -> Optional[dict]:
        """Get detailed information about a branch."""
        if pc not in self._branch_records:
            return None
        
        record = self._branch_records[pc]
        return {
            'pc': hex(pc),
            'total': record.total,
            'correct': record.correct,
            'misprediction_rate': record.misprediction_rate,
            'avg_confidence': record.avg_confidence,
            'is_h2p': self.is_h2p(pc)
        }
    
    def _get_adaptive_threshold(self) -> float:
        """Get adaptive H2P threshold based on global performance."""
        if not self.adaptive or self._global_total < 100:
            return self.h2p_threshold
        
        # Calculate global misprediction rate
        global_mispred = 1.0 - (self._global_correct / self._global_total)
        
        # Adjust threshold:
        # - High global misprediction -> higher threshold (fewer H2P)
        # - Low global misprediction -> lower threshold (more H2P)
        adjustment = (global_mispred - 0.05) * 0.5
        adjusted = self.h2p_threshold + adjustment
        
        return np.clip(adjusted, self.min_threshold, self.max_threshold)
    
    def get_h2p_branches(self) -> list:
        """Get list of all H2P branch PCs."""
        return [pc for pc in self._branch_records if self.is_h2p(pc)]
    
    def get_statistics(self) -> dict:
        """Get detector statistics."""
        h2p_count = len(self.get_h2p_branches())
        total_tracked = len(self._branch_records)
        
        return {
            'tracked_branches': total_tracked,
            'h2p_branches': h2p_count,
            'h2p_ratio': h2p_count / max(1, total_tracked),
            'current_threshold': self._get_adaptive_threshold(),
            'global_accuracy': self._global_correct / max(1, self._global_total),
            'global_misprediction_rate': 1.0 - (self._global_correct / max(1, self._global_total))
        }
    
    def reset(self) -> None:
        """Reset detector state."""
        self._branch_records.clear()
        self._h2p_cache.clear()
        self._global_correct = 0
        self._global_total = 0


class BranchRecord:
    """Record for tracking a single branch's prediction history."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.correct = 0.0  # Exponential moving average
        self.total = 0
        self._confidence_sum = 0.0
        self._confidence_count = 0
        
    def update(self, correct: bool, confidence: Optional[float] = None) -> None:
        """Update record with new outcome."""
        self.total += 1
        
        # Use exponential moving average for smoothing
        alpha = min(1.0 / self.total, 1.0 / 10)  # At least 10-sample smoothing
        
        if correct:
            self.correct = self.correct * (1 - alpha) + alpha
        else:
            self.correct = self.correct * (1 - alpha)
        
        # Track confidence
        if confidence is not None:
            self._confidence_sum += confidence
            self._confidence_count += 1
    
    @property
    def misprediction_rate(self) -> float:
        """Current misprediction rate."""
        if self.total == 0:
            return 0.0
        return 1.0 - self.correct
    
    @property
    def avg_confidence(self) -> float:
        """Average prediction confidence."""
        if self._confidence_count == 0:
            return 0.0
        return self._confidence_sum / self._confidence_count
