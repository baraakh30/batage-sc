"""
Hybrid Branch Predictor (LOABP)

Lightweight Online-Adaptive Branch Predictor that combines:
1. Fast perceptron predictor for most branches
2. Lightweight BNN for hard-to-predict branches
3. H2P detection and adaptive selection
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .base import BasePredictor, PredictionResult, PredictorStats
from .perceptron import PerceptronPredictor, FastPathPerceptron
from .bnn import BinaryNeuralNetwork


class H2PDetector:
    """
    Hard-to-Predict (H2P) Branch Detector.
    
    Tracks per-branch misprediction rates and classifies
    branches as H2P when they exceed a threshold.
    """
    
    def __init__(self, config: dict):
        self.window_size = config.get('tracking_window', 1000)
        self.h2p_threshold = config.get('h2p_threshold', 0.3)
        self.confidence_threshold = config.get('confidence_threshold', 0.1)
        self.max_tracked = config.get('max_tracked_branches', 8192)
        
        # Adaptive threshold
        self.adaptive = config.get('adaptive_threshold', True)
        self.min_threshold = config.get('min_threshold', 0.1)
        self.max_threshold = config.get('max_threshold', 0.5)
        
        # Per-branch tracking: {pc: (correct, total)}
        self.branch_stats: Dict[int, Tuple[int, int]] = {}
        
        # Recently classified H2P branches (cache)
        self.h2p_cache: Dict[int, bool] = {}
        
        # Global misprediction rate for adaptive threshold
        self.global_correct = 0
        self.global_total = 0
        
    def record_outcome(self, pc: int, correct: bool):
        """Record prediction outcome for a branch."""
        # Update global stats
        self.global_total += 1
        if correct:
            self.global_correct += 1
        
        # Update per-branch stats
        if pc in self.branch_stats:
            old_correct, old_total = self.branch_stats[pc]
            new_total = min(old_total + 1, self.window_size)
            # Exponential moving average style
            alpha = 1.0 / new_total
            new_correct = old_correct * (1 - alpha) + (1 if correct else 0)
            self.branch_stats[pc] = (new_correct, new_total)
        else:
            if len(self.branch_stats) >= self.max_tracked:
                # Evict least-seen branch
                min_pc = min(self.branch_stats, 
                           key=lambda p: self.branch_stats[p][1])
                del self.branch_stats[min_pc]
                if min_pc in self.h2p_cache:
                    del self.h2p_cache[min_pc]
            
            self.branch_stats[pc] = (1 if correct else 0, 1)
        
        # Invalidate cache for this branch
        if pc in self.h2p_cache:
            del self.h2p_cache[pc]
    
    def is_h2p(self, pc: int) -> bool:
        """Check if a branch is classified as hard-to-predict."""
        # Check cache
        if pc in self.h2p_cache:
            return self.h2p_cache[pc]
        
        if pc not in self.branch_stats:
            return False
        
        correct, total = self.branch_stats[pc]
        
        # Need minimum samples
        if total < 10:
            return False
        
        # Get threshold (possibly adaptive)
        threshold = self._get_threshold()
        
        # Misprediction rate
        accuracy = correct / total
        mispred_rate = 1.0 - accuracy
        
        is_hard = mispred_rate >= threshold
        
        # Cache result
        self.h2p_cache[pc] = is_hard
        
        return is_hard
    
    def get_misprediction_rate(self, pc: int) -> float:
        """Get misprediction rate for a specific branch."""
        if pc not in self.branch_stats:
            return 0.0
        correct, total = self.branch_stats[pc]
        if total == 0:
            return 0.0
        return 1.0 - (correct / total)
    
    def _get_threshold(self) -> float:
        """Get current H2P threshold (possibly adaptive)."""
        if not self.adaptive or self.global_total < 100:
            return self.h2p_threshold
        
        # Adjust threshold based on global misprediction rate
        global_accuracy = self.global_correct / self.global_total
        global_mispred = 1.0 - global_accuracy
        
        # If global misprediction is high, raise threshold
        # If global misprediction is low, lower threshold
        adjusted = self.h2p_threshold + (global_mispred - 0.1) * 0.5
        
        return np.clip(adjusted, self.min_threshold, self.max_threshold)
    
    def get_stats(self) -> dict:
        """Get detector statistics."""
        num_h2p = sum(1 for pc in self.branch_stats if self.is_h2p(pc))
        return {
            'tracked_branches': len(self.branch_stats),
            'h2p_branches': num_h2p,
            'h2p_ratio': num_h2p / max(1, len(self.branch_stats)),
            'current_threshold': self._get_threshold(),
            'global_accuracy': self.global_correct / max(1, self.global_total)
        }


class HybridPredictor(BasePredictor):
    """
    LOABP: Lightweight Online-Adaptive Branch Predictor
    
    Combines fast perceptron with lightweight BNN for H2P branches.
    
    Selection strategies:
    - 'confidence': Use BNN when perceptron confidence is low
    - 'h2p_only': Use BNN only for detected H2P branches
    - 'always_bnn': Always use BNN (for comparison)
    - 'always_perceptron': Always use perceptron (for comparison)
    """
    
    def __init__(self, config: dict):
        super().__init__("LOABP", config)
        
        # Sub-predictors
        perceptron_config = config.get('perceptron', {})
        bnn_config = config.get('bnn', {})
        h2p_config = config.get('h2p_detector', {})
        hybrid_config = config.get('hybrid', {})
        
        # Initialize components
        self.perceptron = PerceptronPredictor(perceptron_config)
        self.bnn = BinaryNeuralNetwork(bnn_config)
        self.h2p_detector = H2PDetector(h2p_config)
        
        # Selection strategy - default to 'confidence' for balanced approach
        self.strategy = hybrid_config.get('selection_strategy', 'confidence')
        self.confidence_threshold = hybrid_config.get(
            'perceptron_confidence_threshold', 3  # Lower threshold = use perceptron more
        )
        self.use_bnn_for_h2p = hybrid_config.get('use_bnn_for_h2p', True)
        self.h2p_warmup = 0  # Don't use BNN until we have enough H2P data
        self.h2p_warmup_limit = hybrid_config.get('h2p_warmup_limit', 5000)  # Longer warmup
        
        # Component enable flags
        self.perceptron_enabled = perceptron_config.get('enabled', True)
        self.bnn_enabled = bnn_config.get('enabled', True)
        
        # Training config
        training_config = config.get('training', {})
        self.update_on_correct = training_config.get('update_on_correct', False)
        self.update_on_mispredict = training_config.get('update_on_mispredict', True)
        self.update_on_low_confidence = training_config.get(
            'update_on_low_confidence', True
        )
        
        # Per-predictor statistics
        self.perceptron_stats = PredictorStats()
        self.bnn_stats = PredictorStats()
        
        # Selection statistics
        self.perceptron_selections = 0
        self.bnn_selections = 0
        
    def predict(self, pc: int, history: np.ndarray) -> PredictionResult:
        """
        Make a prediction using the appropriate sub-predictor.
        """
        # Get perceptron prediction (always computed for confidence)
        perceptron_pred = self.perceptron.predict(pc, history)
        
        # Determine which predictor to use
        use_bnn = self._should_use_bnn(pc, perceptron_pred)
        
        if use_bnn and self.bnn_enabled:
            self.bnn_selections += 1
            bnn_pred = self.bnn.predict(pc, history)
            
            # Return BNN prediction but include perceptron info
            return PredictionResult(
                prediction=bnn_pred.prediction,
                confidence=bnn_pred.confidence,
                predictor_used="BNN",
                raw_sum=bnn_pred.raw_sum
            )
        else:
            self.perceptron_selections += 1
            return PredictionResult(
                prediction=perceptron_pred.prediction,
                confidence=perceptron_pred.confidence,
                predictor_used="Perceptron",
                raw_sum=perceptron_pred.raw_sum
            )
    
    def _should_use_bnn(self, pc: int, perceptron_pred: PredictionResult) -> bool:
        """Determine if BNN should be used for this prediction."""
        # Always need warmup before using BNN
        if self.h2p_warmup < self.h2p_warmup_limit:
            self.h2p_warmup += 1
            return False
            
        if self.strategy == 'always_bnn':
            return True
        elif self.strategy == 'always_perceptron':
            return False
        elif self.strategy == 'perceptron_primary':
            # Use perceptron primarily, BNN only for very low confidence H2P branches
            is_h2p = self.use_bnn_for_h2p and self.h2p_detector.is_h2p(pc)
            very_low_confidence = abs(perceptron_pred.raw_sum) < 2  # Very low threshold
            return is_h2p and very_low_confidence
        elif self.strategy == 'h2p_only':
            return self.h2p_detector.is_h2p(pc)
        elif self.strategy == 'confidence':
            # Use BNN if perceptron confidence is low AND branch is H2P
            # This is more conservative - requires both conditions
            low_confidence = abs(perceptron_pred.raw_sum) < self.confidence_threshold
            is_h2p = self.use_bnn_for_h2p and self.h2p_detector.is_h2p(pc)
            return low_confidence and is_h2p  # Changed from OR to AND
        else:
            return False
    
    def update(self, pc: int, history: np.ndarray,
               taken: bool, prediction: PredictionResult) -> None:
        """
        Update predictors based on actual outcome.
        """
        correct = (prediction.prediction == taken)
        
        # Update H2P detector
        self.h2p_detector.record_outcome(pc, correct)
        
        # Determine which predictors to update
        should_update = (
            (not correct and self.update_on_mispredict) or
            (correct and self.update_on_correct) or
            (prediction.confidence < 0.5 and self.update_on_low_confidence)
        )
        
        if should_update:
            # Always update perceptron
            perceptron_pred = self.perceptron.predict(pc, history)
            self.perceptron.update(pc, history, taken, perceptron_pred)
            
            # Update BNN for H2P branches or when BNN was used
            if (prediction.predictor_used == "BNN" or 
                self.h2p_detector.is_h2p(pc)):
                bnn_pred = self.bnn.predict(pc, history)
                self.bnn.update(pc, history, taken, bnn_pred)
        
        # Track per-predictor stats
        if prediction.predictor_used == "BNN":
            self.bnn_stats.record_prediction(prediction, taken)
        else:
            self.perceptron_stats.record_prediction(prediction, taken)
        
        self.stats.record_update()
    
    def get_hardware_cost(self) -> dict:
        """Total hardware cost of hybrid predictor."""
        perceptron_cost = self.perceptron.get_hardware_cost()
        bnn_cost = self.bnn.get_hardware_cost()
        
        # H2P detector cost (simplified)
        h2p_entries = len(self.h2p_detector.branch_stats)
        h2p_bits_per_entry = 64 + 32  # PC hash + stats
        h2p_total_bits = h2p_entries * h2p_bits_per_entry
        
        total_bits = (
            perceptron_cost['total_bits'] + 
            bnn_cost['total_bits'] + 
            h2p_total_bits
        )
        
        return {
            'perceptron': perceptron_cost,
            'bnn': bnn_cost,
            'h2p_detector_bits': h2p_total_bits,
            'total_bits': total_bits,
            'total_bytes': total_bits // 8,
            'total_kb': total_bits / 8 / 1024
        }
    
    def get_detailed_stats(self) -> dict:
        """Get detailed statistics about predictor behavior."""
        total_selections = self.perceptron_selections + self.bnn_selections
        
        return {
            'overall': {
                'predictions': self.stats.predictions,
                'accuracy': self.stats.accuracy,
                'mpki': self.stats.mpki,
                'misprediction_rate': self.stats.misprediction_rate
            },
            'perceptron': {
                'selections': self.perceptron_selections,
                'selection_ratio': self.perceptron_selections / max(1, total_selections),
                'accuracy': self.perceptron_stats.accuracy,
                'mpki': self.perceptron_stats.mpki
            },
            'bnn': {
                'selections': self.bnn_selections,
                'selection_ratio': self.bnn_selections / max(1, total_selections),
                'accuracy': self.bnn_stats.accuracy,
                'mpki': self.bnn_stats.mpki
            },
            'h2p_detector': self.h2p_detector.get_stats(),
            'hardware_cost': self.get_hardware_cost()
        }
    
    def reset(self) -> None:
        """Reset all components."""
        super().reset()
        self.perceptron.reset()
        self.bnn.reset()
        self.h2p_detector = H2PDetector(self.config.get('h2p_detector', {}))
        self.perceptron_stats = PredictorStats()
        self.bnn_stats = PredictorStats()
        self.perceptron_selections = 0
        self.bnn_selections = 0


class AdaptiveHybridPredictor(HybridPredictor):
    """
    Enhanced hybrid predictor with adaptive strategy selection.
    
    Automatically adjusts selection strategy based on observed performance.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "AdaptiveLOABP"
        
        # Adaptation parameters
        self.adaptation_window = config.get('adaptation_window', 10000)
        self.window_counter = 0
        
        # Track performance per strategy
        self.strategy_performance = {
            'confidence': {'correct': 0, 'total': 0},
            'h2p_only': {'correct': 0, 'total': 0}
        }
        
        # Current adaptation state
        self.exploring = True
        self.explore_counter = 0
        self.explore_limit = 5000
        
    def predict(self, pc: int, history: np.ndarray) -> PredictionResult:
        """Make prediction with potential strategy exploration."""
        self.window_counter += 1
        
        # Periodically explore different strategies
        if self.exploring and self.explore_counter < self.explore_limit:
            self.explore_counter += 1
            
            # Try both strategies and track which would be better
            old_strategy = self.strategy
            
            # This is tracked but doesn't affect the actual prediction
            for strat in ['confidence', 'h2p_only']:
                self.strategy = strat
                pred = super().predict(pc, history)
                # Strategy performance is tracked in update()
            
            self.strategy = old_strategy
        
        # Adapt strategy based on observed performance
        if self.window_counter >= self.adaptation_window:
            self._adapt_strategy()
            self.window_counter = 0
        
        return super().predict(pc, history)
    
    def _adapt_strategy(self):
        """Adapt strategy based on observed performance."""
        best_strategy = None
        best_accuracy = 0
        
        for strat, perf in self.strategy_performance.items():
            if perf['total'] > 0:
                accuracy = perf['correct'] / perf['total']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_strategy = strat
        
        if best_strategy and best_strategy != self.strategy:
            self.strategy = best_strategy
            
        # Reset performance tracking
        for strat in self.strategy_performance:
            self.strategy_performance[strat] = {'correct': 0, 'total': 0}
        
        # Reduce exploration over time
        self.explore_limit = max(1000, self.explore_limit - 500)
