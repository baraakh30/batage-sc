"""
Multi-Perspective Perceptron (MPP) Branch Predictor

A completely different approach from TAGE. Instead of finding one matching
history length, MPP uses MULTIPLE hash functions simultaneously and combines
their predictions using learned weights.

Key insights from CBP winners (Jimenez's hashed perceptron):
1. Use multiple tables indexed by different hash functions
2. Each hash captures a different correlation (PC, global, local, path)
3. Sum all contributions - no "winner takes all" like TAGE
4. Adaptive threshold for training

This is NOT a TAGE variant - it's a fundamentally different approach based on
neural-inspired linear combination of features.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                  Multi-Perspective Perceptron                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PC ──┬──► [Table 0: PC-indexed weights]           ──┐        │
│        │                                              │        │
│        ├──► [Table 1: PC⊕GHist[0:15] weights]       ──┤        │
│        │                                              │        │
│        ├──► [Table 2: PC⊕GHist[0:31] weights]       ──┤        │
│        │                                              │ SUM    │
│        ├──► [Table 3: PC⊕GHist[0:63] weights]       ──┼──►pred │
│        │                                              │        │
│        ├──► [Table 4: PC⊕PathHist weights]          ──┤        │
│        │                                              │        │
│        ├──► [Table 5: PC⊕LocalHist weights]         ──┤        │
│        │                                              │        │
│        └──► [Table 6: PC⊕GHist[0:127] weights]      ──┘        │
│                                                                 │
│   Training: Update all weights when mispredicted or low conf    │
│   Threshold: Adaptive based on recent accuracy                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

"""

import numpy as np
from typing import Optional, List, Tuple
from .base import BasePredictor, PredictionResult


# ============================================================================
# Configuration Presets
# ============================================================================

MPP_64KB = {
    'name': 'MPP-64KB',
    # Table configurations: (log_size, history_length, weight_bits)
    # Total ~64KB = 512Kbits
    'tables': [
        # PC-only tables (capture branch bias)
        {'log_size': 12, 'hist_len': 0, 'weight_bits': 8},      # 4K × 8 = 32Kb
        
        # Short global history (recent correlation)
        {'log_size': 12, 'hist_len': 8, 'weight_bits': 8},      # 4K × 8 = 32Kb
        {'log_size': 12, 'hist_len': 16, 'weight_bits': 8},     # 4K × 8 = 32Kb
        
        # Medium global history  
        {'log_size': 11, 'hist_len': 32, 'weight_bits': 8},     # 2K × 8 = 16Kb
        {'log_size': 11, 'hist_len': 48, 'weight_bits': 8},     # 2K × 8 = 16Kb
        {'log_size': 11, 'hist_len': 64, 'weight_bits': 8},     # 2K × 8 = 16Kb
        
        # Long global history
        {'log_size': 10, 'hist_len': 96, 'weight_bits': 8},     # 1K × 8 = 8Kb
        {'log_size': 10, 'hist_len': 128, 'weight_bits': 8},    # 1K × 8 = 8Kb
        {'log_size': 10, 'hist_len': 192, 'weight_bits': 8},    # 1K × 8 = 8Kb
        {'log_size': 10, 'hist_len': 256, 'weight_bits': 8},    # 1K × 8 = 8Kb
        
        # Very long history
        {'log_size': 9, 'hist_len': 384, 'weight_bits': 8},     # 512 × 8 = 4Kb
        {'log_size': 9, 'hist_len': 512, 'weight_bits': 8},     # 512 × 8 = 4Kb
        
        # Path history tables
        {'log_size': 11, 'hist_len': 16, 'weight_bits': 8, 'use_path': True},  # 2K × 8 = 16Kb
        {'log_size': 10, 'hist_len': 32, 'weight_bits': 8, 'use_path': True},  # 1K × 8 = 8Kb
        
        # Local history tables  
        {'log_size': 10, 'hist_len': 11, 'weight_bits': 8, 'use_local': True}, # 1K × 8 = 8Kb
    ],
    'local_hist_table_size': 10,  # 1K entries for local history
    'local_hist_length': 11,
    'max_global_hist': 600,
    'path_hist_bits': 27,
    'base_theta': 64,  # Base training threshold
    'theta_max': 127,
    'tc_bits': 8,      # Threshold counter bits
}

MPP_32KB = {
    'name': 'MPP-32KB',
    'tables': [
        {'log_size': 11, 'hist_len': 0, 'weight_bits': 8},
        {'log_size': 11, 'hist_len': 8, 'weight_bits': 8},
        {'log_size': 11, 'hist_len': 16, 'weight_bits': 8},
        {'log_size': 10, 'hist_len': 32, 'weight_bits': 8},
        {'log_size': 10, 'hist_len': 48, 'weight_bits': 8},
        {'log_size': 10, 'hist_len': 64, 'weight_bits': 8},
        {'log_size': 9, 'hist_len': 96, 'weight_bits': 8},
        {'log_size': 9, 'hist_len': 128, 'weight_bits': 8},
        {'log_size': 9, 'hist_len': 192, 'weight_bits': 8},
        {'log_size': 10, 'hist_len': 16, 'weight_bits': 8, 'use_path': True},
        {'log_size': 9, 'hist_len': 11, 'weight_bits': 8, 'use_local': True},
    ],
    'local_hist_table_size': 9,
    'local_hist_length': 11,
    'max_global_hist': 300,
    'path_hist_bits': 24,
    'base_theta': 48,
    'theta_max': 95,
    'tc_bits': 7,
}


class MPP(BasePredictor):
    """
    Multi-Perspective Perceptron Branch Predictor.
    
    Key differences from TAGE:
    1. ALL tables contribute to every prediction (no tag matching)
    2. Uses learned weights instead of counters
    3. Multiple perspectives: global, local, path histories
    4. Adaptive training threshold
    
    Based on concepts from:
    - Jimenez's Hashed Perceptron (CBP-1 winner)
    - TAGE-SC-L's multi-component approach
    - Multiperspective Perceptron (CBP-4, CBP-5 winners)
    """
    
    def __init__(self, config: dict = None):
        config = config or MPP_64KB
        name = config.get('name', 'MPP')
        super().__init__(name, config)
        
        self.table_configs = config.get('tables', MPP_64KB['tables'])
        self.n_tables = len(self.table_configs)
        
        self.local_hist_table_size = config.get('local_hist_table_size', 10)
        self.local_hist_length = config.get('local_hist_length', 11)
        self.max_global_hist = config.get('max_global_hist', 600)
        self.path_hist_bits = config.get('path_hist_bits', 27)
        
        self.base_theta = config.get('base_theta', 64)
        self.theta_max = config.get('theta_max', 127)
        self.tc_bits = config.get('tc_bits', 8)
        
        # === Initialize weight tables ===
        self.tables = []
        self.weight_max = []
        self.weight_min = []
        
        for tc in self.table_configs:
            size = 1 << tc['log_size']
            wbits = tc['weight_bits']
            wmax = (1 << (wbits - 1)) - 1
            wmin = -(1 << (wbits - 1))
            
            # Initialize weights to small random values for symmetry breaking
            table = np.zeros(size, dtype=np.int16)
            self.tables.append(table)
            self.weight_max.append(wmax)
            self.weight_min.append(wmin)
        
        # === Local history table ===
        lht_size = 1 << self.local_hist_table_size
        self.local_history_table = np.zeros(lht_size, dtype=np.uint32)
        self.local_hist_mask = (1 << self.local_hist_length) - 1
        
        # === Global history ===
        self.global_hist = np.zeros(self.max_global_hist + 64, dtype=np.uint8)
        self.ghist_ptr = 0
        
        # === Path history ===
        self.path_hist = 0
        self.path_hist_mask = (1 << self.path_hist_bits) - 1
        
        # === Folded global history for different lengths ===
        # Pre-compute masks and folded values for efficiency
        self.hist_lengths = sorted(set(tc['hist_len'] for tc in self.table_configs if tc['hist_len'] > 0))
        self.folded_hist = {length: 0 for length in self.hist_lengths}
        
        # === Adaptive threshold ===
        self.theta = self.base_theta
        self.tc = 0  # Threshold counter
        self.tc_max = (1 << self.tc_bits) - 1
        
        # === Prediction state ===
        self._indices = []
        self._sum = 0
    
    def _hash_fold(self, value: int, bits: int) -> int:
        """Fold a value down to specified number of bits using XOR."""
        result = 0
        while value > 0:
            result ^= value & ((1 << bits) - 1)
            value >>= bits
        return result
    
    def _compute_folded_history(self, length: int) -> int:
        """Compute folded global history of given length."""
        if length == 0:
            return 0
        
        # Collect history bits
        hist_val = 0
        for i in range(min(length, self.max_global_hist)):
            idx = (self.ghist_ptr + i) % len(self.global_hist)
            if self.global_hist[idx]:
                hist_val |= (1 << i)
        
        return hist_val
    
    def _get_table_index(self, pc: int, table_idx: int) -> int:
        """Compute index for a specific table."""
        tc = self.table_configs[table_idx]
        log_size = tc['log_size']
        hist_len = tc['hist_len']
        use_path = tc.get('use_path', False)
        use_local = tc.get('use_local', False)
        
        # Start with PC
        index = (pc >> 2) & 0xFFFFFFFF
        
        if use_local:
            # Use local history
            lht_idx = (pc >> 2) & ((1 << self.local_hist_table_size) - 1)
            local_hist = int(self.local_history_table[lht_idx]) & self.local_hist_mask
            index ^= local_hist
            # Also mix some global history
            if hist_len > 0:
                ghist = self._compute_folded_history(hist_len)
                index ^= self._hash_fold(ghist, log_size)
        elif use_path:
            # Use path history mixed with global
            index ^= self.path_hist
            if hist_len > 0:
                ghist = self._compute_folded_history(hist_len)
                index ^= self._hash_fold(ghist, log_size)
        elif hist_len > 0:
            # Standard global history
            ghist = self._compute_folded_history(hist_len)
            index ^= self._hash_fold(ghist, log_size)
            # Extra mixing for longer histories
            if hist_len > 32:
                index ^= self._hash_fold(ghist >> 16, log_size)
        
        return index & ((1 << log_size) - 1)
    
    def predict(self, pc: int, history: np.ndarray = None) -> PredictionResult:
        """
        Make prediction by summing contributions from all tables.
        """
        # Compute indices for all tables
        self._indices = []
        self._sum = 0
        
        for i in range(self.n_tables):
            idx = self._get_table_index(pc, i)
            self._indices.append(idx)
            self._sum += int(self.tables[i][idx])
        
        # Prediction based on sum
        prediction = self._sum >= 0
        
        # Confidence based on magnitude
        confidence = min(1.0, abs(self._sum) / (self.theta * 2))
        
        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            predictor_used=self.name,
            raw_sum=float(self._sum)
        )
    
    def update(self, pc: int, history: np.ndarray, taken: bool,
               prediction: PredictionResult) -> None:
        """
        Update weights using perceptron learning rule.
        
        Key insight: Only update when:
        1. Misprediction occurred, OR
        2. Prediction was correct but confidence was low (|sum| < theta)
        
        This "training on low confidence" is crucial for perceptron accuracy.
        """
        pred = prediction.prediction
        outcome = 1 if taken else -1
        
        # Determine if we should train
        mispredicted = (pred != taken)
        low_confidence = abs(self._sum) < self.theta
        
        if mispredicted or low_confidence:
            # Update all weights
            for i in range(self.n_tables):
                idx = self._indices[i]
                weight = int(self.tables[i][idx])
                
                # Perceptron update: w += outcome
                if taken:
                    if weight < self.weight_max[i]:
                        self.tables[i][idx] = weight + 1
                else:
                    if weight > self.weight_min[i]:
                        self.tables[i][idx] = weight - 1
        
        # === Adaptive threshold (from Jimenez) ===
        if mispredicted:
            # Increase threshold on misprediction
            self.tc = min(self.tc_max, self.tc + 1)
            if self.tc == self.tc_max:
                self.tc = 0
                self.theta = min(self.theta_max, self.theta + 1)
        elif low_confidence:
            # Decrease threshold when correct but low confidence
            self.tc = max(0, self.tc - 1)
            if self.tc == 0:
                self.tc = self.tc_max
                self.theta = max(self.base_theta // 2, self.theta - 1)
        
        # === Update histories ===
        # Update local history
        lht_idx = (pc >> 2) & ((1 << self.local_hist_table_size) - 1)
        local_val = int(self.local_history_table[lht_idx])
        self.local_history_table[lht_idx] = ((local_val << 1) | int(taken)) & self.local_hist_mask
        
        # Update path history
        self.path_hist = ((self.path_hist << 1) | ((pc >> 2) & 1)) & self.path_hist_mask
        
        # Update global history (circular buffer)
        self.ghist_ptr = (self.ghist_ptr - 1) % len(self.global_hist)
        self.global_hist[self.ghist_ptr] = 1 if taken else 0
    
    def get_hardware_cost(self) -> dict:
        """Estimate hardware implementation cost."""
        bits = 0
        
        # Weight tables
        for i, tc in enumerate(self.table_configs):
            size = 1 << tc['log_size']
            wbits = tc['weight_bits']
            bits += size * wbits
        
        # Local history table
        bits += (1 << self.local_hist_table_size) * self.local_hist_length
        
        # Global history
        bits += self.max_global_hist
        
        # Path history
        bits += self.path_hist_bits
        
        # Threshold counter
        bits += self.tc_bits + 8  # tc + theta
        
        return {
            'n_tables': self.n_tables,
            'total_bits': bits,
            'total_bytes': bits // 8,
            'total_kb': bits / 8 / 1024,
        }
    
    def reset(self) -> None:
        """Reset predictor state."""
        super().reset()
        
        for table in self.tables:
            table.fill(0)
        
        self.local_history_table.fill(0)
        self.global_hist.fill(0)
        self.ghist_ptr = 0
        self.path_hist = 0
        self.theta = self.base_theta
        self.tc = 0
        self._indices = []
        self._sum = 0
