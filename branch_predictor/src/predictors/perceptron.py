"""
Perceptron Branch Predictor

Implementation based on Jiménez & Lin (2001, 2002).
Uses a table of perceptrons indexed by branch address.
Each perceptron computes a weighted sum of the global history.
"""

import numpy as np
from typing import Optional
from .base import BasePredictor, PredictionResult


class PerceptronPredictor(BasePredictor):
    """
    Perceptron-based branch predictor.
    
    Key features:
    - Linear correlation learning with global history
    - Hardware-efficient signed integer weights
    - Threshold-based training (only update when needed)
    
    Based on: "Dynamic Branch Prediction with Perceptrons" (Jiménez & Lin, 2001)
    """
    
    def __init__(self, config: dict):
        super().__init__("Perceptron", config)
        
        # Configuration
        self.history_length = config.get('history_length', 64)
        self.table_size = config.get('table_size', 4096)
        self.weight_bits = config.get('weight_bits', 8)
        self.theta_multiplier = config.get('theta', 1.93)
        
        # Compute weight bounds and training threshold
        self.weight_max = (1 << (self.weight_bits - 1)) - 1   # e.g., 127 for 8-bit
        self.weight_min = -(1 << (self.weight_bits - 1))      # e.g., -128 for 8-bit
        
        # Training threshold (from Jiménez & Lin)
        # theta = floor(1.93 * history_length + 14)
        self.theta = int(self.theta_multiplier * self.history_length + 14)
        
        # Weight table: [table_size, history_length + 1]
        # +1 for bias weight (w0)
        self.weights = np.zeros(
            (self.table_size, self.history_length + 1), 
            dtype=np.int16
        )
        
        # Learning rate (weight increment/decrement)
        self.learning_rate = config.get('learning_rate', 1)
        
        # Statistics tracking
        self.sum_magnitudes = []  # Track prediction confidence
        
    def _get_index(self, pc: int) -> int:
        """Hash PC to get table index."""
        # Simple modulo hashing (can be improved with more sophisticated hashing)
        return pc % self.table_size
    
    def _compute_sum(self, weights: np.ndarray, history: np.ndarray) -> int:
        """
        Compute perceptron output.
        
        Args:
            weights: Weight vector [history_length + 1]
            history: History bits as bipolar values (+1/-1)
            
        Returns:
            Weighted sum (dot product)
        """
        # Ensure history is in bipolar form (-1, +1)
        bipolar_history = np.where(history > 0, 1, -1)
        
        # Bias weight (w0) is always multiplied by 1
        # Sum = w0 + sum(wi * xi) for i = 1 to history_length
        total = weights[0]  # Bias
        total += np.dot(weights[1:self.history_length + 1], 
                       bipolar_history[:self.history_length])
        
        return int(total)
    
    def predict(self, pc: int, history: np.ndarray) -> PredictionResult:
        """
        Make a prediction using the perceptron.
        
        Prediction rule: Taken if sum >= 0, Not Taken otherwise
        """
        idx = self._get_index(pc)
        weights = self.weights[idx]
        
        # Compute weighted sum
        perceptron_sum = self._compute_sum(weights, history)
        
        # Prediction: sign of the sum
        taken = perceptron_sum >= 0
        
        # Confidence: magnitude of sum relative to theta
        # Higher |sum| means more confident
        confidence = min(abs(perceptron_sum) / self.theta, 1.0)
        
        return PredictionResult(
            prediction=taken,
            confidence=confidence,
            predictor_used=self.name,
            raw_sum=float(perceptron_sum)
        )
    
    def update(self, pc: int, history: np.ndarray,
               taken: bool, prediction: PredictionResult) -> None:
        """
        Update perceptron weights using the perceptron learning rule.
        
        Update conditions (from Jiménez & Lin):
        1. Always update if prediction was wrong
        2. Update if |sum| <= theta (low confidence)
        """
        perceptron_sum = prediction.raw_sum
        predicted_taken = prediction.prediction
        
        # Check if update is needed
        wrong = (predicted_taken != taken)
        low_confidence = abs(perceptron_sum) <= self.theta
        
        if not (wrong or low_confidence):
            return
        
        idx = self._get_index(pc)
        
        # Training direction: +1 if taken, -1 if not taken
        t = 1 if taken else -1
        
        # Convert history to bipolar
        bipolar_history = np.where(history > 0, 1, -1)
        
        # Update bias weight
        new_bias = self.weights[idx, 0] + self.learning_rate * t
        self.weights[idx, 0] = np.clip(new_bias, self.weight_min, self.weight_max)
        
        # Update history weights
        for i in range(self.history_length):
            xi = bipolar_history[i] if i < len(bipolar_history) else -1
            delta = self.learning_rate * t * xi
            new_weight = self.weights[idx, i + 1] + delta
            self.weights[idx, i + 1] = np.clip(new_weight, 
                                                self.weight_min, 
                                                self.weight_max)
        
        self.stats.record_update()
    
    def get_confidence(self, pc: int, history: np.ndarray) -> float:
        """Get prediction confidence without making formal prediction."""
        idx = self._get_index(pc)
        weights = self.weights[idx]
        perceptron_sum = self._compute_sum(weights, history)
        return min(abs(perceptron_sum) / self.theta, 1.0)
    
    def get_hardware_cost(self) -> dict:
        """Estimate hardware implementation cost."""
        # Each entry has (history_length + 1) weights
        weights_per_entry = self.history_length + 1
        bits_per_weight = self.weight_bits
        total_weight_bits = self.table_size * weights_per_entry * bits_per_weight
        
        # Additional storage: theta register, index logic
        overhead_bits = 32  # Approximately
        
        total_bits = total_weight_bits + overhead_bits
        
        return {
            'table_entries': self.table_size,
            'weights_per_entry': weights_per_entry,
            'bits_per_weight': bits_per_weight,
            'total_weight_bits': total_weight_bits,
            'overhead_bits': overhead_bits,
            'total_bits': total_bits,
            'total_bytes': total_bits // 8,
            'total_kb': total_bits / 8 / 1024,
            'theta': self.theta
        }
    
    def reset(self) -> None:
        """Reset predictor state."""
        super().reset()
        self.weights.fill(0)
        self.sum_magnitudes.clear()
    
    def get_weight_statistics(self) -> dict:
        """Get statistics about weight distribution."""
        flat_weights = self.weights.flatten()
        return {
            'mean': float(np.mean(flat_weights)),
            'std': float(np.std(flat_weights)),
            'min': int(np.min(flat_weights)),
            'max': int(np.max(flat_weights)),
            'zeros': int(np.sum(flat_weights == 0)),
            'saturated_pos': int(np.sum(flat_weights == self.weight_max)),
            'saturated_neg': int(np.sum(flat_weights == self.weight_min))
        }


class FastPathPerceptron(PerceptronPredictor):
    """
    Fast-path perceptron for low-latency prediction.
    
    Optimizations:
    - Shorter history for faster computation
    - Precomputed partial sums
    - Simplified indexing
    
    Based on: "Neural Methods for Dynamic Branch Prediction" (Jiménez & Lin, 2002)
    """
    
    def __init__(self, config: dict):
        # Use shorter history for fast path
        fast_config = config.copy()
        fast_config['history_length'] = config.get('fast_history_length', 16)
        super().__init__(fast_config)
        self.name = "FastPathPerceptron"
        
        # Partial sum cache for speculative updates
        self._partial_sums = {}
        self._max_cache_size = 256
    
    def predict_fast(self, pc: int, history: np.ndarray) -> PredictionResult:
        """
        Fast prediction using cached partial sums when possible.
        """
        idx = self._get_index(pc)
        
        # Check cache
        cache_key = (idx, tuple(history[:8].tolist()))  # Use first 8 bits as key
        
        if cache_key in self._partial_sums:
            partial = self._partial_sums[cache_key]
            # Complete the sum with remaining history
            weights = self.weights[idx]
            bipolar_history = np.where(history > 0, 1, -1)
            remaining_sum = np.dot(weights[9:self.history_length + 1],
                                  bipolar_history[8:self.history_length])
            perceptron_sum = partial + remaining_sum
        else:
            # Full computation
            perceptron_sum = self._compute_sum(self.weights[idx], history)
            
            # Cache partial sum
            if len(self._partial_sums) < self._max_cache_size:
                weights = self.weights[idx]
                bipolar_history = np.where(history > 0, 1, -1)
                partial = weights[0] + np.dot(weights[1:9], bipolar_history[:8])
                self._partial_sums[cache_key] = partial
        
        taken = perceptron_sum >= 0
        confidence = min(abs(perceptron_sum) / self.theta, 1.0)
        
        return PredictionResult(
            prediction=taken,
            confidence=confidence,
            predictor_used=self.name,
            raw_sum=float(perceptron_sum)
        )
    
    def invalidate_cache(self, pc: int) -> None:
        """Invalidate cache entries for a specific PC."""
        idx = self._get_index(pc)
        keys_to_remove = [k for k in self._partial_sums if k[0] == idx]
        for k in keys_to_remove:
            del self._partial_sums[k]
    
    def update(self, pc: int, history: np.ndarray,
               taken: bool, prediction: PredictionResult) -> None:
        """Update with cache invalidation."""
        super().update(pc, history, taken, prediction)
        self.invalidate_cache(pc)
