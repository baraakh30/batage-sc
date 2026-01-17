"""
Binary Neural Network (BNN) for Hard-to-Predict Branches

Lightweight neural network with quantized weights for hardware efficiency.
Uses 1-bit weights (+1/-1) and XNOR-popcount operations.
Supports online training with straight-through estimator.
"""

import numpy as np
from typing import List, Tuple, Optional
from .base import BasePredictor, PredictionResult


def sign_function(x: np.ndarray) -> np.ndarray:
    """Binarize to +1/-1."""
    return np.where(x >= 0, 1, -1)


def straight_through_estimator(gradient: np.ndarray, 
                                weights: np.ndarray,
                                clip_value: float = 1.0) -> np.ndarray:
    """
    Straight-through estimator for backprop through sign function.
    
    During forward: use sign(w)
    During backward: pass gradient through if |w| <= clip_value
    """
    mask = np.abs(weights) <= clip_value
    return gradient * mask


class BinaryLinearLayer:
    """
    Binary linear layer with 1-bit weights.
    
    Forward: y = sign(W) @ x
    Uses XNOR-popcount for efficient hardware implementation.
    """
    
    def __init__(self, input_size: int, output_size: int,
                 learning_rate: float = 0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.base_lr = learning_rate
        
        # Full-precision weights for training (quantized during inference)
        # Xavier initialization for better convergence
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / (input_size + output_size))
        self.bias = np.zeros(output_size)
        
        # Momentum for faster convergence
        self.momentum = 0.9
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.bias)
        
        # Cache for backward pass
        self._input_cache = None
        self._binary_weights_cache = None
        
        # Update counter for LR scheduling
        self.update_count = 0
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with binary weights.
        
        For hardware: XNOR(sign(W), sign(x)) then popcount
        Here we simulate: sign(W) @ x
        """
        self._input_cache = x.copy()
        
        # Binarize weights
        binary_weights = sign_function(self.weights)
        self._binary_weights_cache = binary_weights
        
        # Compute output
        output = binary_weights @ x + self.bias
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass using straight-through estimator with momentum.
        """
        if self._input_cache is None:
            raise RuntimeError("Must call forward before backward")
        
        # Gradient w.r.t. weights (using STE)
        grad_weights = np.outer(grad_output, self._input_cache)
        grad_weights = straight_through_estimator(grad_weights, self.weights)
        
        # Gradient w.r.t. bias
        grad_bias = grad_output
        
        # Gradient w.r.t. input
        grad_input = self._binary_weights_cache.T @ grad_output
        
        # Update with momentum
        self.velocity_w = self.momentum * self.velocity_w + self.learning_rate * grad_weights
        self.velocity_b = self.momentum * self.velocity_b + self.learning_rate * grad_bias
        
        self.weights -= self.velocity_w
        self.bias -= self.velocity_b
        
        # Clip weights to prevent explosion
        self.weights = np.clip(self.weights, -1.5, 1.5)
        
        # Adaptive LR decay
        self.update_count += 1
        if self.update_count % 10000 == 0:
            self.learning_rate = max(0.005, self.learning_rate * 0.95)
        
        return grad_input
    
    def get_binary_weights(self) -> np.ndarray:
        """Get quantized binary weights."""
        return sign_function(self.weights)
    
    def get_storage_bits(self) -> int:
        """Estimate storage for binary weights."""
        # 1 bit per weight + full precision bias
        return self.output_size * self.input_size + self.output_size * 32


class BinaryNeuralNetwork(BasePredictor):
    """
    Binary Neural Network for hard-to-predict branch prediction.
    
    Architecture:
    - Input: Global history (binary vector)
    - Hidden layers: Binary linear + sign activation
    - Output: Single neuron (Taken/Not Taken)
    
    Key features:
    - 1-bit weights for hardware efficiency
    - Online training with straight-through estimator
    - Selective activation for H2P branches only
    """
    
    def __init__(self, config: dict):
        super().__init__("BNN", config)
        
        # Architecture configuration
        self.input_size = config.get('input_size', 64)
        self.hidden_sizes = config.get('hidden_sizes', [48, 24])  # Larger hidden layers
        self.learning_rate = config.get('learning_rate', 0.03)  # Moderate LR
        
        # Build network
        self.layers: List[BinaryLinearLayer] = []
        self._build_network()
        
        # Training settings
        self.use_straight_through = config.get('use_straight_through', True)
        self.train_on_mispredict = config.get('train_on_mispredict', True)
        self.always_train = config.get('always_train', True)
        
        # Per-branch bias table for specialization
        self.bias_table_size = config.get('bias_table_size', 4096)
        self.branch_bias = np.zeros(self.bias_table_size, dtype=np.int8)  # -4 to +3
        self.branch_count = np.zeros(self.bias_table_size, dtype=np.uint8)  # 0 to 255
        
        # Per-branch model tracking (optional)
        self.per_branch_enabled = config.get('per_branch_models', False)
        self.branch_models = {}  # pc -> BinaryNeuralNetwork
        self.max_branch_models = config.get('max_branch_models', 64)
        
    def _build_network(self):
        """Build the network layers."""
        sizes = [self.input_size] + self.hidden_sizes + [1]
        
        for i in range(len(sizes) - 1):
            layer = BinaryLinearLayer(
                sizes[i], sizes[i + 1], 
                self.learning_rate
            )
            self.layers.append(layer)
    
    def _forward(self, x: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """
        Forward pass through the network.
        
        Returns:
            output: Final output value
            activations: List of activations for each layer
        """
        activations = [x]
        current = x
        
        for i, layer in enumerate(self.layers):
            current = layer.forward(current)
            
            # Apply sign activation for hidden layers
            if i < len(self.layers) - 1:
                current = sign_function(current)
            
            activations.append(current)
        
        return float(current[0]), activations
    
    def _backward(self, loss_grad: float, activations: List[np.ndarray]):
        """
        Backward pass through the network.
        """
        grad = np.array([loss_grad])
        
        # Backward through layers in reverse
        for i in range(len(self.layers) - 1, -1, -1):
            # Apply STE for sign activation (hidden layers)
            if i < len(self.layers) - 1:
                grad = straight_through_estimator(
                    grad, 
                    activations[i + 1]
                )
            
            grad = self.layers[i].backward(grad)
    
    def predict(self, pc: int, history: np.ndarray) -> PredictionResult:
        """
        Make a prediction using the BNN.
        """
        # Prepare input: convert history to bipolar (-1, +1)
        if history is None or len(history) == 0:
            x = np.ones(self.input_size) * -1.0
        else:
            x = np.where(history[:self.input_size] > 0, 1.0, -1.0)
            if len(x) < self.input_size:
                x = np.pad(x, (0, self.input_size - len(x)), 
                          constant_values=-1.0)
        
        # Add PC-based features - use folded XOR for better distribution
        pc_hash = pc ^ (pc >> 7) ^ (pc >> 14) ^ (pc >> 21)
        for i in range(min(24, self.input_size)):
            bit = (pc_hash >> i) & 1
            # XOR history with PC bits for per-branch differentiation
            if bit:
                x[i] = -x[i]  # Flip if PC bit is 1
        
        # Forward pass
        output, _ = self._forward(x)
        
        # Add per-branch bias
        bias_idx = pc % self.bias_table_size
        branch_bias = float(self.branch_bias[bias_idx])
        output += branch_bias * 0.5  # Scaled bias contribution
        
        # Prediction: sign of output
        taken = output >= 0
        
        # Confidence: magnitude of output (normalized)
        confidence = min(abs(output) / 10.0, 1.0)
        
        return PredictionResult(
            prediction=taken,
            confidence=confidence,
            predictor_used=self.name,
            raw_sum=output
        )
    
    def update(self, pc: int, history: np.ndarray,
               taken: bool, prediction: PredictionResult) -> None:
        """
        Update the BNN using online learning.
        """
        # Prepare input - same as predict
        if history is None or len(history) == 0:
            x = np.ones(self.input_size) * -1.0
        else:
            x = np.where(history[:self.input_size] > 0, 1.0, -1.0)
            if len(x) < self.input_size:
                x = np.pad(x, (0, self.input_size - len(x)), constant_values=-1.0)
        
        # Add PC-based features (same as predict)
        pc_hash = pc ^ (pc >> 7) ^ (pc >> 14) ^ (pc >> 21)
        for i in range(min(24, self.input_size)):
            bit = (pc_hash >> i) & 1
            if bit:
                x[i] = -x[i]
        
        # Target: +1 for taken, -1 for not taken
        target = 1.0 if taken else -1.0
        
        # Forward pass
        output, activations = self._forward(x)
        
        # Add per-branch bias for prediction check
        bias_idx = pc % self.bias_table_size
        output_with_bias = output + float(self.branch_bias[bias_idx]) * 0.5
        
        # Check if prediction was correct
        predicted_correct = (output_with_bias >= 0) == taken
        
        # Update per-branch bias (always)
        if taken:
            self.branch_bias[bias_idx] = min(3, int(self.branch_bias[bias_idx]) + 1)
        else:
            self.branch_bias[bias_idx] = max(-4, int(self.branch_bias[bias_idx]) - 1)
        self.branch_count[bias_idx] = min(255, int(self.branch_count[bias_idx]) + 1)
        
        # Update neural network only when wrong or low confidence
        if not predicted_correct or abs(output) < 3.0:
            error = output - target
            loss_grad = np.clip(error, -2.0, 2.0)
            self._backward(loss_grad, activations)
        
        self.stats.record_update()
    
    def train_batch(self, samples: List[Tuple[int, np.ndarray, bool]]):
        """
        Train on a batch of samples (for offline or mini-batch training).
        
        Args:
            samples: List of (pc, history, taken) tuples
        """
        for pc, history, taken in samples:
            pred = self.predict(pc, history)
            self.update(pc, history, taken, pred)
    
    def get_hardware_cost(self) -> dict:
        """Estimate hardware implementation cost."""
        total_weight_bits = 0
        total_bias_bits = 0
        layer_info = []
        
        for i, layer in enumerate(self.layers):
            weight_bits = layer.input_size * layer.output_size  # 1 bit per weight
            bias_bits = layer.output_size * 8  # 8-bit bias (reduced precision)
            
            total_weight_bits += weight_bits
            total_bias_bits += bias_bits
            
            layer_info.append({
                'layer': i,
                'input_size': layer.input_size,
                'output_size': layer.output_size,
                'weight_bits': weight_bits,
                'bias_bits': bias_bits
            })
        
        total_bits = total_weight_bits + total_bias_bits
        
        return {
            'layers': len(self.layers),
            'layer_info': layer_info,
            'total_weight_bits': total_weight_bits,
            'total_bias_bits': total_bias_bits,
            'total_bits': total_bits,
            'total_bytes': total_bits // 8,
            'total_kb': total_bits / 8 / 1024,
            'architecture': f"{self.input_size}-{'-'.join(map(str, self.hidden_sizes))}-1"
        }
    
    def reset(self) -> None:
        """Reset the network."""
        super().reset()
        self.layers.clear()
        self._build_network()
        self.branch_models.clear()
    
    def get_weight_statistics(self) -> dict:
        """Get statistics about weight distribution."""
        all_weights = []
        for layer in self.layers:
            all_weights.extend(layer.weights.flatten())
        
        all_weights = np.array(all_weights)
        binary_weights = sign_function(all_weights)
        
        return {
            'mean': float(np.mean(all_weights)),
            'std': float(np.std(all_weights)),
            'positive_ratio': float(np.mean(binary_weights > 0)),
            'total_params': len(all_weights)
        }


class EnsembleBNN(BasePredictor):
    """
    Ensemble of small BNNs for improved accuracy.
    
    Uses multiple small networks and majority voting.
    """
    
    def __init__(self, config: dict):
        super().__init__("EnsembleBNN", config)
        
        self.num_networks = config.get('num_networks', 3)
        self.networks: List[BinaryNeuralNetwork] = []
        
        # Create ensemble with different random initializations
        for i in range(self.num_networks):
            net_config = config.copy()
            net_config['hidden_sizes'] = config.get('hidden_sizes', [16, 8])
            self.networks.append(BinaryNeuralNetwork(net_config))
    
    def predict(self, pc: int, history: np.ndarray) -> PredictionResult:
        """Majority voting prediction."""
        votes_taken = 0
        total_confidence = 0
        
        for net in self.networks:
            pred = net.predict(pc, history)
            if pred.prediction:
                votes_taken += 1
            total_confidence += pred.confidence
        
        taken = votes_taken > self.num_networks // 2
        confidence = total_confidence / self.num_networks
        
        return PredictionResult(
            prediction=taken,
            confidence=confidence,
            predictor_used=self.name,
            raw_sum=float(votes_taken - self.num_networks / 2)
        )
    
    def update(self, pc: int, history: np.ndarray,
               taken: bool, prediction: PredictionResult) -> None:
        """Update all networks in the ensemble."""
        for net in self.networks:
            net_pred = net.predict(pc, history)
            net.update(pc, history, taken, net_pred)
        self.stats.record_update()
    
    def get_hardware_cost(self) -> dict:
        """Sum of all network costs."""
        total_bits = 0
        for net in self.networks:
            cost = net.get_hardware_cost()
            total_bits += cost['total_bits']
        
        return {
            'num_networks': self.num_networks,
            'total_bits': total_bits,
            'total_bytes': total_bits // 8,
            'total_kb': total_bits / 8 / 1024
        }
