"""
Base Predictor Interface

Abstract base class for all branch predictors in the LOABP framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PredictionResult:
    """Result of a branch prediction."""
    prediction: bool          # True = Taken, False = Not Taken
    confidence: float         # Confidence score (higher = more confident)
    predictor_used: str       # Which predictor made the decision
    raw_sum: Optional[float] = None  # Raw computation result (e.g., perceptron sum)
    
    @property
    def taken(self) -> bool:
        return self.prediction
    
    @property
    def not_taken(self) -> bool:
        return not self.prediction


@dataclass 
class BranchInfo:
    """Information about a branch instruction."""
    pc: int                   # Program counter (branch address)
    target: int               # Branch target address
    is_conditional: bool      # True if conditional branch
    is_call: bool            # True if call instruction
    is_return: bool          # True if return instruction
    is_indirect: bool        # True if indirect branch
    
    @classmethod
    def from_trace(cls, pc: int, target: int = 0, branch_type: int = 0):
        """Create BranchInfo from trace data."""
        return cls(
            pc=pc,
            target=target,
            is_conditional=(branch_type & 0x1) != 0,
            is_call=(branch_type & 0x2) != 0,
            is_return=(branch_type & 0x4) != 0,
            is_indirect=(branch_type & 0x8) != 0
        )


class BasePredictor(ABC):
    """Abstract base class for branch predictors."""
    
    def __init__(self, name: str, config: dict):
        """
        Initialize the predictor.
        
        Args:
            name: Name identifier for this predictor
            config: Configuration dictionary
        """
        self.name = name
        self.config = config
        self.stats = PredictorStats()
    
    @abstractmethod
    def predict(self, pc: int, history: np.ndarray) -> PredictionResult:
        """
        Make a branch prediction.
        
        Args:
            pc: Program counter of the branch
            history: Global branch history register (as numpy array of 0/1 or -1/+1)
            
        Returns:
            PredictionResult with prediction and confidence
        """
        pass
    
    @abstractmethod
    def update(self, pc: int, history: np.ndarray, 
               taken: bool, prediction: PredictionResult) -> None:
        """
        Update the predictor based on actual outcome.
        
        Args:
            pc: Program counter of the branch
            history: Global branch history register used for prediction
            taken: Actual branch outcome (True = taken)
            prediction: The prediction that was made
        """
        pass
    
    @abstractmethod
    def get_hardware_cost(self) -> dict:
        """
        Estimate hardware implementation cost.
        
        Returns:
            Dictionary with storage (bits/bytes) and other metrics
        """
        pass
    
    def reset(self) -> None:
        """Reset predictor state (optional override)."""
        self.stats = PredictorStats()
    
    def get_stats(self) -> 'PredictorStats':
        """Get current statistics."""
        return self.stats


class PredictorStats:
    """Statistics tracking for a predictor."""
    
    def __init__(self):
        self.predictions = 0
        self.correct = 0
        self.mispredictions = 0
        self.taken_predicted = 0
        self.taken_actual = 0
        self.updates = 0
        
        # Per-confidence tracking
        self.high_confidence_correct = 0
        self.high_confidence_total = 0
        self.low_confidence_correct = 0
        self.low_confidence_total = 0
    
    def record_prediction(self, prediction: PredictionResult, 
                         actual: bool, confidence_threshold: float = 0.5):
        """Record a prediction result."""
        self.predictions += 1
        
        if prediction.taken:
            self.taken_predicted += 1
        if actual:
            self.taken_actual += 1
            
        correct = (prediction.prediction == actual)
        if correct:
            self.correct += 1
        else:
            self.mispredictions += 1
            
        # Confidence tracking
        if prediction.confidence >= confidence_threshold:
            self.high_confidence_total += 1
            if correct:
                self.high_confidence_correct += 1
        else:
            self.low_confidence_total += 1
            if correct:
                self.low_confidence_correct += 1
    
    def record_update(self):
        """Record that an update was performed."""
        self.updates += 1
    
    @property
    def accuracy(self) -> float:
        """Overall prediction accuracy."""
        if self.predictions == 0:
            return 0.0
        return self.correct / self.predictions
    
    @property
    def mpki(self) -> float:
        """Mispredictions per 1000 instructions (assuming 1 branch per instruction)."""
        if self.predictions == 0:
            return 0.0
        return (self.mispredictions / self.predictions) * 1000
    
    @property
    def misprediction_rate(self) -> float:
        """Misprediction rate."""
        if self.predictions == 0:
            return 0.0
        return self.mispredictions / self.predictions
    
    def __str__(self) -> str:
        return (f"Predictions: {self.predictions}, "
                f"Accuracy: {self.accuracy*100:.2f}%, "
                f"MPKI: {self.mpki:.2f}")


class BimodalPredictor(BasePredictor):
    """
    Simple 2-bit saturating counter predictor (baseline).
    """
    
    def __init__(self, config: dict):
        super().__init__("Bimodal", config)
        self.table_size = config.get('table_size', 4096)
        # 2-bit counters: 0,1 = Not Taken; 2,3 = Taken
        self.table = np.ones(self.table_size, dtype=np.int8) * 2  # Weakly taken
        
    def _index(self, pc: int) -> int:
        """Compute table index from PC."""
        return pc % self.table_size
    
    def predict(self, pc: int, history: np.ndarray) -> PredictionResult:
        idx = self._index(pc)
        counter = self.table[idx]
        taken = counter >= 2
        confidence = abs(counter - 1.5) / 1.5  # 0 to 1
        
        return PredictionResult(
            prediction=taken,
            confidence=confidence,
            predictor_used=self.name,
            raw_sum=float(counter)
        )
    
    def update(self, pc: int, history: np.ndarray,
               taken: bool, prediction: PredictionResult) -> None:
        idx = self._index(pc)
        if taken:
            self.table[idx] = min(3, self.table[idx] + 1)
        else:
            self.table[idx] = max(0, self.table[idx] - 1)
        self.stats.record_update()
    
    def get_hardware_cost(self) -> dict:
        return {
            'table_entries': self.table_size,
            'bits_per_entry': 2,
            'total_bits': self.table_size * 2,
            'total_bytes': (self.table_size * 2) // 8,
            'total_kb': (self.table_size * 2) / 8 / 1024
        }


class GSharePredictor(BasePredictor):
    """
    GShare predictor (baseline) - XOR of PC and global history.
    """
    
    def __init__(self, config: dict):
        super().__init__("GShare", config)
        self.table_size = config.get('table_size', 16384)
        self.history_length = config.get('history_length', 14)
        self.history_mask = (1 << self.history_length) - 1
        # 2-bit counters
        self.table = np.ones(self.table_size, dtype=np.int8) * 2
        
    def _index(self, pc: int, history: np.ndarray) -> int:
        """Compute index using XOR of PC and history."""
        # Convert history array to integer
        hist_int = 0
        for i, bit in enumerate(history[:self.history_length]):
            if bit > 0:  # Works for both 0/1 and -1/+1 encoding
                hist_int |= (1 << i)
        
        return (pc ^ hist_int) % self.table_size
    
    def predict(self, pc: int, history: np.ndarray) -> PredictionResult:
        idx = self._index(pc, history)
        counter = self.table[idx]
        taken = counter >= 2
        confidence = abs(counter - 1.5) / 1.5
        
        return PredictionResult(
            prediction=taken,
            confidence=confidence,
            predictor_used=self.name,
            raw_sum=float(counter)
        )
    
    def update(self, pc: int, history: np.ndarray,
               taken: bool, prediction: PredictionResult) -> None:
        idx = self._index(pc, history)
        if taken:
            self.table[idx] = min(3, self.table[idx] + 1)
        else:
            self.table[idx] = max(0, self.table[idx] - 1)
        self.stats.record_update()
    
    def get_hardware_cost(self) -> dict:
        return {
            'table_entries': self.table_size,
            'bits_per_entry': 2,
            'history_bits': self.history_length,
            'total_bits': self.table_size * 2 + self.history_length,
            'total_bytes': (self.table_size * 2 + self.history_length) // 8,
            'total_kb': (self.table_size * 2) / 8 / 1024
        }
