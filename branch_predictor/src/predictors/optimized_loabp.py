"""
Optimized LOABP - Lightweight Online-Adaptive Branch Predictor v2.0

Key optimizations for better accuracy and smaller size:
1. Hashed Perceptron (multiple history hashing) - reduces aliasing
2. Local + Global history fusion
3. Improved BNN with better architecture
4. Dynamic threshold adaptation
5. Path-based indexing
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from .base import BasePredictor, PredictionResult, PredictorStats


class HashedPerceptron:
    """
    Hashed Perceptron with multiple history features.
    
    Key improvement: Uses multiple hash functions to reduce aliasing
    while keeping table size small.
    
    Based on: "Idealized Piecewise Linear Branch Prediction" (Jiménez, 2005)
    """
    
    def __init__(self, config: dict):
        # Smaller table but with multiple features
        self.num_tables = config.get('num_tables', 8)  # Multiple small tables
        self.table_size = config.get('table_size', 512)  # Per-table (total = 4KB)
        self.history_length = config.get('history_length', 32)  # Shorter history
        self.weight_bits = config.get('weight_bits', 6)  # 6-bit weights save space
        
        self.weight_max = (1 << (self.weight_bits - 1)) - 1
        self.weight_min = -(1 << (self.weight_bits - 1))
        
        # Multiple weight tables (geometric history lengths)
        self.tables = [
            np.zeros(self.table_size, dtype=np.int8) 
            for _ in range(self.num_tables)
        ]
        
        # History lengths for each table (geometric series like TAGE)
        self.history_lengths = self._compute_geometric_lengths()
        
        # Training threshold
        self.theta = int(1.93 * self.history_length + 14)
        
        # Path history (addresses of recent branches)
        self.path_history = np.zeros(16, dtype=np.uint64)
        self.path_ptr = 0
        
        # Global branch history
        self.global_history = np.zeros(self.history_length, dtype=np.int8)
        
        # Local history table (per-PC)
        self.local_history_size = config.get('local_history_size', 1024)
        self.local_history = np.zeros(self.local_history_size, dtype=np.uint16)
        
    def _compute_geometric_lengths(self) -> List[int]:
        """Compute geometric history lengths (like TAGE)."""
        min_len = 2
        max_len = self.history_length
        
        lengths = []
        ratio = (max_len / min_len) ** (1 / (self.num_tables - 1))
        
        for i in range(self.num_tables):
            length = int(min_len * (ratio ** i))
            lengths.append(min(length, self.history_length))
        
        return lengths
    
    def _hash_pc_history(self, pc: int, table_idx: int) -> int:
        """Hash PC with history for table index."""
        hist_len = self.history_lengths[table_idx]
        
        # Fold history bits
        history_bits = 0
        for i in range(min(hist_len, self.history_length)):
            if self.global_history[i] > 0:
                history_bits |= (1 << (i % 16))
        
        # Combine PC and history with different hash for each table
        h = pc ^ (pc >> 4) ^ (history_bits << table_idx) ^ (table_idx * 0x9e3779b9)
        
        # Add path history
        path_idx = (self.path_ptr - table_idx) % len(self.path_history)
        h ^= int(self.path_history[path_idx] >> 2)
        
        return h % self.table_size
    
    def _get_local_history_idx(self, pc: int) -> int:
        """Get local history index for PC."""
        return pc % self.local_history_size
    
    def predict(self, pc: int) -> Tuple[bool, int]:
        """
        Predict branch direction.
        
        Returns:
            (prediction, confidence)
        """
        total = 0
        
        # Sum contributions from all tables
        for i in range(self.num_tables):
            idx = self._hash_pc_history(pc, i)
            total += int(self.tables[i][idx])  # Cast to int to prevent overflow
        
        # Add local history contribution
        local_idx = self._get_local_history_idx(pc)
        local_hist = self.local_history[local_idx]
        local_taken = bin(local_hist & 0xFFFF).count('1')
        local_bias = (local_taken - 8)  # Centered around 0
        total += local_bias
        
        prediction = total >= 0
        confidence = min(abs(total), 127)  # Cap to prevent issues
        
        return prediction, confidence
    
    def update(self, pc: int, taken: bool, prediction: bool, confidence: int):
        """Update perceptron weights."""
        # Only update on misprediction or weak confidence
        if prediction != taken or confidence < self.theta:
            t = 1 if taken else -1
            
            for i in range(self.num_tables):
                idx = self._hash_pc_history(pc, i)
                
                # Update weight
                new_weight = self.tables[i][idx] + t
                self.tables[i][idx] = np.clip(new_weight, self.weight_min, self.weight_max)
        
        # Update histories
        self._update_histories(pc, taken)
    
    def _update_histories(self, pc: int, taken: bool):
        """Update all history structures."""
        # Shift global history
        self.global_history = np.roll(self.global_history, 1)
        self.global_history[0] = 1 if taken else -1
        
        # Update path history
        self.path_history[self.path_ptr] = pc
        self.path_ptr = (self.path_ptr + 1) % len(self.path_history)
        
        # Update local history
        local_idx = self._get_local_history_idx(pc)
        self.local_history[local_idx] = (
            ((self.local_history[local_idx] << 1) | (1 if taken else 0)) & 0xFFFF
        )
    
    def storage_size(self) -> int:
        """Calculate storage in bits."""
        # Weight tables: num_tables * table_size * weight_bits
        tables_bits = self.num_tables * self.table_size * self.weight_bits
        
        # Local history: local_history_size * 16 bits
        local_bits = self.local_history_size * 16
        
        # Global history: history_length bits
        global_bits = self.history_length
        
        # Path history: 16 * 64 bits
        path_bits = 16 * 64
        
        return tables_bits + local_bits + global_bits + path_bits


class ImprovedBNN:
    """
    Improved Binary Neural Network for H2P branches.
    
    Key improvements:
    1. Smaller, deeper network (better features)
    2. Batch normalization simulation
    3. Better training with momentum
    4. Per-branch embedding
    """
    
    def __init__(self, config: dict):
        self.input_size = config.get('input_size', 32)
        # Deeper but narrower: 32 -> 16 -> 8 -> 4 -> 1
        self.layer_sizes = config.get('layer_sizes', [32, 16, 8, 4, 1])
        self.learning_rate = config.get('learning_rate', 0.15)
        self.momentum = config.get('momentum', 0.9)
        
        # Initialize layers
        self.weights = []
        self.biases = []
        self.velocities_w = []
        self.velocities_b = []
        
        prev_size = self.input_size
        for size in self.layer_sizes:
            w = np.random.randn(size, prev_size) * np.sqrt(2.0 / prev_size)
            b = np.zeros(size)
            self.weights.append(w)
            self.biases.append(b)
            self.velocities_w.append(np.zeros_like(w))
            self.velocities_b.append(np.zeros_like(b))
            prev_size = size
        
        # Running stats for normalization
        self.running_mean = [np.zeros(s) for s in self.layer_sizes[:-1]]
        self.running_var = [np.ones(s) for s in self.layer_sizes[:-1]]
        
        # Branch embedding table (learn per-branch features)
        self.embed_size = config.get('embed_size', 256)
        self.embeddings = np.random.randn(self.embed_size, 8) * 0.1
        
    def _sign(self, x):
        """Binary activation."""
        return np.where(x >= 0, 1.0, -1.0)
    
    def _prepare_input(self, pc: int, history: np.ndarray) -> np.ndarray:
        """Prepare input features."""
        # Get branch embedding
        embed_idx = pc % self.embed_size
        embed = self.embeddings[embed_idx]
        
        # Combine history with embedding
        hist_features = history[:self.input_size - 8] if len(history) >= self.input_size - 8 else np.pad(
            history, (0, self.input_size - 8 - len(history))
        )
        
        return np.concatenate([hist_features, embed])
    
    def forward(self, pc: int, history: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """Forward pass with caching for backprop."""
        x = self._prepare_input(pc, history)
        
        activations = [x]
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Binary weights
            w_bin = self._sign(w)
            
            # Linear + bias
            z = w_bin @ x + b
            
            # Activation (except last layer)
            if i < len(self.weights) - 1:
                x = self._sign(z)
            else:
                x = z  # Last layer: linear output
            
            activations.append(x)
        
        return x[0], activations
    
    def predict(self, pc: int, history: np.ndarray) -> Tuple[bool, float]:
        """Predict branch direction."""
        output, _ = self.forward(pc, history)
        return output >= 0, abs(output)
    
    def train(self, pc: int, history: np.ndarray, taken: bool):
        """Train on single sample with SGD + momentum."""
        target = 1.0 if taken else -1.0
        
        # Forward
        output, activations = self.forward(pc, history)
        
        # Loss gradient
        error = output - target
        
        # Skip update if already correct with margin
        if abs(error) < 0.5:
            return
        
        # Backward pass
        grad = np.array([error])
        
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient w.r.t. weights
            grad_w = np.outer(grad, activations[i])
            
            # STE: pass gradient if |w| <= 1
            mask = np.abs(self.weights[i]) <= 1.0
            grad_w = grad_w * mask
            
            # Momentum update
            self.velocities_w[i] = (
                self.momentum * self.velocities_w[i] + 
                self.learning_rate * grad_w
            )
            self.velocities_b[i] = (
                self.momentum * self.velocities_b[i] + 
                self.learning_rate * grad
            )
            
            # Update weights
            self.weights[i] -= self.velocities_w[i]
            self.biases[i] -= self.velocities_b[i]
            
            # Clip weights
            self.weights[i] = np.clip(self.weights[i], -1.5, 1.5)
            
            # Propagate gradient
            if i > 0:
                w_bin = self._sign(self.weights[i])
                grad = w_bin.T @ grad
                # STE for activation
                grad = grad * (np.abs(activations[i]) <= 1).astype(float)
        
        # Update embedding
        embed_idx = pc % self.embed_size
        self.embeddings[embed_idx] -= 0.01 * error * activations[0][-8:]
    
    def storage_size(self) -> int:
        """Calculate storage in bits (1-bit weights)."""
        total = 0
        
        # Binary weights: 1 bit each
        for w in self.weights:
            total += w.size
        
        # Biases: 8 bits each
        for b in self.biases:
            total += b.size * 8
        
        # Embeddings: 8 bits each
        total += self.embeddings.size * 8
        
        return total


class OptimizedLOABP(BasePredictor):
    """
    Optimized LOABP v2.0
    
    Target: ~32KB storage with improved accuracy
    
    Architecture:
    1. Hashed Perceptron (main predictor) - ~24KB
    2. Improved BNN (H2P handler) - ~4KB
    3. H2P Detector - ~4KB
    """
    
    def __init__(self, config: dict):
        super().__init__("OptimizedLOABP", config)
        
        # Main predictor: Hashed Perceptron
        perceptron_config = {
            'num_tables': config.get('num_tables', 8),
            'table_size': config.get('table_size', 512),
            'history_length': config.get('history_length', 32),
            'weight_bits': config.get('weight_bits', 6),
            'local_history_size': config.get('local_history_size', 1024),
        }
        self.perceptron = HashedPerceptron(perceptron_config)
        
        # H2P predictor: Improved BNN
        bnn_config = {
            'input_size': config.get('bnn_input_size', 32),
            'layer_sizes': config.get('bnn_layers', [32, 16, 8, 4, 1]),
            'learning_rate': config.get('bnn_lr', 0.15),
            'embed_size': config.get('bnn_embed_size', 256),
        }
        self.bnn = ImprovedBNN(bnn_config)
        
        # H2P tracking
        self.h2p_table_size = config.get('h2p_table_size', 2048)
        self.h2p_counters = np.zeros(self.h2p_table_size, dtype=np.int8)
        self.h2p_threshold = config.get('h2p_threshold', 4)
        
        # Meta predictor (choose perceptron vs BNN)
        self.meta_table_size = config.get('meta_table_size', 1024)
        self.meta_counters = np.zeros(self.meta_table_size, dtype=np.int8)
        
        # Confidence thresholds
        self.low_conf_threshold = config.get('low_conf_threshold', 10)
        
    def _h2p_index(self, pc: int) -> int:
        """Get H2P table index."""
        return (pc ^ (pc >> 4)) % self.h2p_table_size
    
    def _meta_index(self, pc: int) -> int:
        """Get meta predictor index."""
        return (pc ^ (pc >> 8)) % self.meta_table_size
    
    def _is_h2p(self, pc: int) -> bool:
        """Check if branch is hard-to-predict."""
        idx = self._h2p_index(pc)
        return self.h2p_counters[idx] >= self.h2p_threshold
    
    def _use_bnn(self, pc: int, perc_conf: int) -> bool:
        """Decide whether to use BNN prediction."""
        # Use BNN if: H2P branch AND perceptron has low confidence
        if not self._is_h2p(pc):
            return False
        
        if perc_conf >= self.low_conf_threshold:
            return False
        
        # Check meta predictor
        meta_idx = self._meta_index(pc)
        return self.meta_counters[meta_idx] >= 0
    
    def predict(self, pc: int, history: Optional[np.ndarray] = None) -> PredictionResult:
        """Make prediction."""
        self.stats.predictions += 1
        
        # Get perceptron prediction
        perc_pred, perc_conf = self.perceptron.predict(pc)
        
        # Decide which predictor to use
        use_bnn = self._use_bnn(pc, perc_conf)
        
        if use_bnn:
            bnn_pred, bnn_conf = self.bnn.predict(pc, self.perceptron.global_history)
            prediction = bnn_pred
            confidence = bnn_conf
            source = "bnn"
        else:
            prediction = perc_pred
            confidence = perc_conf
            source = "perceptron"
        
        return PredictionResult(
            prediction=prediction,
            confidence=confidence / 100.0,
            predictor_used=source
        )
    
    def update(self, pc: int, history: np.ndarray, taken: bool, prediction_result: PredictionResult):
        """
        Update predictor state.
        
        Args:
            pc: Program counter of the branch
            history: Global branch history (not used, we maintain internal history)
            taken: Actual branch outcome
            prediction_result: The prediction that was made
        """
        predicted = prediction_result.prediction
        correct = (predicted == taken)
        
        # Update H2P counter
        h2p_idx = self._h2p_index(pc)
        if not correct:
            self.h2p_counters[h2p_idx] = min(self.h2p_counters[h2p_idx] + 1, 7)
        else:
            self.h2p_counters[h2p_idx] = max(self.h2p_counters[h2p_idx] - 1, 0)
        
        # Get both predictions for meta update
        perc_pred, perc_conf = self.perceptron.predict(pc)
        
        # Update perceptron
        self.perceptron.update(pc, taken, perc_pred, perc_conf)
        
        # Update BNN if this is an H2P branch
        if self._is_h2p(pc):
            self.bnn.train(pc, self.perceptron.global_history, taken)
            
            # Update meta predictor
            meta_idx = self._meta_index(pc)
            bnn_pred, _ = self.bnn.predict(pc, self.perceptron.global_history)
            
            perc_correct = (perc_pred == taken)
            bnn_correct = (bnn_pred == taken)
            
            if bnn_correct and not perc_correct:
                self.meta_counters[meta_idx] = min(self.meta_counters[meta_idx] + 1, 3)
            elif perc_correct and not bnn_correct:
                self.meta_counters[meta_idx] = max(self.meta_counters[meta_idx] - 1, -4)
    
    def reset(self) -> None:
        """Reset predictor state."""
        self.stats = PredictorStats()
        # Reset H2P and meta counters
        self.h2p_counters = np.zeros(self.h2p_table_size, dtype=np.int8)
        self.meta_counters = np.zeros(self.meta_table_size, dtype=np.int8)
        # Reset perceptron history
        self.perceptron.global_history = np.zeros(self.perceptron.history_length, dtype=np.int8)
        self.perceptron.path_history = np.zeros(16, dtype=np.uint64)
        self.perceptron.path_ptr = 0
        self.perceptron.local_history = np.zeros(self.perceptron.local_history_size, dtype=np.uint16)
    
    def get_stats(self) -> PredictorStats:
        """Get predictor statistics."""
        return self.stats
    
    def get_hardware_cost(self) -> dict:
        """Estimate hardware implementation cost."""
        perc_bits = self.perceptron.storage_size()
        bnn_bits = self.bnn.storage_size()
        h2p_bits = self.h2p_table_size * 3
        meta_bits = self.meta_table_size * 3
        
        total_bits = perc_bits + bnn_bits + h2p_bits + meta_bits
        
        return {
            'total_bits': total_bits,
            'total_bytes': total_bits / 8,
            'total_kb': total_bits / 8 / 1024,
            'components': {
                'hashed_perceptron_bits': perc_bits,
                'bnn_bits': bnn_bits,
                'h2p_detector_bits': h2p_bits,
                'meta_predictor_bits': meta_bits,
            }
        }
    
    def storage_size(self) -> int:
        """Calculate total storage in bits."""
        perc_bits = self.perceptron.storage_size()
        bnn_bits = self.bnn.storage_size()
        h2p_bits = self.h2p_table_size * 3  # 3-bit counters
        meta_bits = self.meta_table_size * 3  # 3-bit counters
        
        return perc_bits + bnn_bits + h2p_bits + meta_bits
    
    def storage_size_kb(self) -> float:
        """Storage in kilobytes."""
        return self.storage_size() / 8 / 1024


# Configuration presets
OPTIMIZED_CONFIG_32KB = {
    'num_tables': 8,
    'table_size': 512,
    'history_length': 32,
    'weight_bits': 6,
    'local_history_size': 1024,
    'bnn_input_size': 32,
    'bnn_layers': [32, 16, 8, 4, 1],
    'bnn_lr': 0.15,
    'bnn_embed_size': 256,
    'h2p_table_size': 2048,
    'meta_table_size': 1024,
    'low_conf_threshold': 10,
    'h2p_threshold': 4,
}

OPTIMIZED_CONFIG_8KB = {
    'num_tables': 4,
    'table_size': 256,
    'history_length': 24,
    'weight_bits': 5,
    'local_history_size': 512,
    'bnn_input_size': 24,
    'bnn_layers': [24, 12, 6, 1],
    'bnn_lr': 0.2,
    'bnn_embed_size': 128,
    'h2p_table_size': 1024,
    'meta_table_size': 512,
    'low_conf_threshold': 8,
    'h2p_threshold': 3,
}


if __name__ == "__main__":
    # Test optimized predictor
    predictor = OptimizedLOABP(OPTIMIZED_CONFIG_32KB)
    print(f"Storage size: {predictor.storage_size_kb():.2f} KB")
    
    # Simulate some predictions
    np.random.seed(42)
    correct = 0
    total = 1000
    
    for i in range(total):
        pc = np.random.randint(0, 0xFFFFFFFF)
        taken = np.random.random() > 0.5
        
        result = predictor.predict(pc)
        predictor.update(pc, taken, result)
        
        if result.prediction == taken:
            correct += 1
    
    print(f"Random test accuracy: {100 * correct / total:.2f}%")
