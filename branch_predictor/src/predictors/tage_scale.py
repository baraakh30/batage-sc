"""
TAGE-SCALE: Statistical Confidence-Aware Local Enhancement

A surgical improvement to TAGE that learns PER-BRANCH when corrections help.

Design Philosophy:
==================
"Don't correct TAGE globally - learn which specific branches benefit from correction."

Key Insight:
============
The problem with all previous correctors (SC, perceptron, neural) is they apply
corrections too broadly. TAGE is already ~96.7% accurate - we only need to fix
~3.3% of predictions. But generic correctors also flip correct predictions.

TAGE-SCALE solves this by:
1. Learning which specific branches consistently benefit from correction
2. Only applying corrections for those branches
3. Using a minimal corrector (less is more)

Key Innovations:
================

1. Per-Branch Correction Counter (PBC) - 1.5KB
   - 4K entries × 3 bits per entry
   - Tracks correction success/failure per PC hash
   - Only consider corrections when counter is high
   
2. Pattern Reliability Table (PRT) - 1KB  
   - 2K entries × 4 bits per entry
   - Indexed by (PC XOR recent_outcomes)
   - Low reliability = TAGE is uncertain here, consider correction
   
3. Micro-Corrector - ~4KB
   - Tiny perceptron: 512 entries × 32 history bits
   - Only consulted when PBC and PRT both say "correction may help"
   
4. Confidence-Gated Decision
   - TAGE high confidence → always trust TAGE
   - TAGE low confidence + PBC high + PRT low → consider corrector
   

"""

import numpy as np
from typing import Optional, List, Tuple
from .base import BasePredictor, PredictionResult


# ============================================================================
# Configuration Presets
# ============================================================================

TAGE_SCALE_64KB = {
    'name': 'TAGE-SCALE-64KB',
    
    # === TAGE Core (same as Original TAGE 64KB) ===
    'num_tables': 12,
    'min_hist': 4,
    'max_hist': 640,
    'log_table_sizes': [14, 10, 10, 11, 11, 11, 11, 10, 10, 10, 10, 9, 9],
    'tag_widths': [0, 7, 7, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15],
    'counter_bits': 3,
    'useful_bits': 2,
    
    # === Per-Branch Correction Counter ===
    'pbc_size': 4096,           # 4K entries
    'pbc_bits': 3,              # 3-bit saturating counter
    'pbc_threshold': 4,         # Correction considered when >= 4
    
    # === Pattern Reliability Table ===
    'prt_size': 2048,           # 2K entries
    'prt_bits': 4,              # 4-bit counter
    'prt_reliability_threshold': 8,  # Pattern unreliable when < 8
    'prt_history_bits': 8,      # Use last 8 outcomes for pattern
    
    # === Micro-Corrector ===
    'mc_size': 512,             # 512 entries
    'mc_history_bits': 32,      # 32-bit local history
    'mc_weight_bits': 6,        # 6-bit weights
    'mc_threshold_factor': 1.93, # Theta factor
    
    # === TAGE Standard Config ===
    'bimodal_size': 16384,
    'bimodal_hyst_ratio': 2,
    'use_alt_size': 128,
    'use_alt_bits': 4,
    'u_reset_period': 19,
    'path_hist_bits': 27,
    'max_alloc': 1,
}

TAGE_SCALE_32KB = {
    'name': 'TAGE-SCALE-32KB',
    'num_tables': 10,
    'min_hist': 4,
    'max_hist': 300,
    'log_table_sizes': [13, 9, 9, 10, 10, 10, 10, 9, 9, 9, 9],
    'tag_widths': [0, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12],
    'counter_bits': 3,
    'useful_bits': 2,
    'pbc_size': 2048,
    'pbc_bits': 3,
    'pbc_threshold': 4,
    'prt_size': 1024,
    'prt_bits': 4,
    'prt_reliability_threshold': 8,
    'prt_history_bits': 8,
    'mc_size': 256,
    'mc_history_bits': 24,
    'mc_weight_bits': 6,
    'mc_threshold_factor': 1.93,
    'bimodal_size': 8192,
    'bimodal_hyst_ratio': 2,
    'use_alt_size': 64,
    'use_alt_bits': 4,
    'u_reset_period': 18,
    'path_hist_bits': 24,
    'max_alloc': 1,
}


# ============================================================================
# Folded History Helper
# ============================================================================

class FoldedHistory:
    """Efficient folded history for indexing and tagging."""
    __slots__ = ['comp', 'comp_length', 'orig_length', 'outpoint']
    
    def __init__(self):
        self.comp = 0
        self.comp_length = 0
        self.orig_length = 0
        self.outpoint = 0
        
    def init(self, original_length: int, compressed_length: int):
        self.orig_length = original_length
        self.comp_length = compressed_length
        if compressed_length > 0:
            self.outpoint = original_length % compressed_length
        else:
            self.outpoint = 0
        self.comp = 0
    
    def update(self, new_bit: int, old_bit: int):
        if self.comp_length <= 0:
            return
        self.comp = (self.comp << 1) | new_bit
        self.comp ^= old_bit << self.outpoint
        self.comp ^= (self.comp >> self.comp_length)
        self.comp &= (1 << self.comp_length) - 1


# ============================================================================
# TAGE Entry
# ============================================================================

class TAGEEntry:
    """Entry in a TAGE tagged table."""
    __slots__ = ['ctr', 'tag', 'u']
    
    def __init__(self):
        self.ctr = 0
        self.tag = 0
        self.u = 0


# ============================================================================
# TAGE-SCALE Predictor
# ============================================================================

class TAGE_SCALE(BasePredictor):
    """
    TAGE-SCALE: Statistical Confidence-Aware Local Enhancement.
    
    Learns per-branch when corrections are beneficial, instead of
    applying generic corrections globally.
    """
    
    def __init__(self, config: dict = None):
        config = config or TAGE_SCALE_64KB
        name = config.get('name', 'TAGE-SCALE')
        super().__init__(name, config)
        
        # === Core TAGE Parameters ===
        self.num_tables = config.get('num_tables', 12)
        self.min_hist = config.get('min_hist', 4)
        self.max_hist = config.get('max_hist', 640)
        self.log_table_sizes = config.get('log_table_sizes')
        self.tag_widths = config.get('tag_widths')
        self.counter_bits = config.get('counter_bits', 3)
        self.useful_bits = config.get('useful_bits', 2)
        
        # Counter bounds
        self.ctr_max = (1 << (self.counter_bits - 1)) - 1
        self.ctr_min = -(1 << (self.counter_bits - 1))
        self.u_max = (1 << self.useful_bits) - 1
        
        # Calculate history lengths (geometric series)
        self.history_lengths = self._calculate_history_lengths()
        
        # === Bimodal Base Predictor ===
        self.bimodal_size = config.get('bimodal_size', 16384)
        self.bimodal_hyst_ratio = config.get('bimodal_hyst_ratio', 2)
        self.bimodal_pred = np.zeros(self.bimodal_size, dtype=np.bool_)
        hyst_size = self.bimodal_size >> self.bimodal_hyst_ratio
        self.bimodal_hyst = np.ones(hyst_size, dtype=np.bool_)
        
        # === Tagged Tables ===
        self.tables: List[List[TAGEEntry]] = []
        self.table_sizes = []
        for i in range(1, self.num_tables + 1):
            table_size = 1 << self.log_table_sizes[i]
            self.table_sizes.append(table_size)
            table = [TAGEEntry() for _ in range(table_size)]
            self.tables.append(table)
        
        # === Path History ===
        self.path_hist_bits = config.get('path_hist_bits', 27)
        self.path_hist = 0
        
        # === Global History ===
        self.global_hist = np.zeros(self.max_hist + 256, dtype=np.uint8)
        self.pt_ghist = 0
        
        # === Folded Histories ===
        self.folded_indices: List[FoldedHistory] = []
        self.folded_tags: List[List[FoldedHistory]] = [[], []]
        
        for i in range(self.num_tables):
            hist_len = self.history_lengths[i + 1]
            table_bits = self.log_table_sizes[i + 1]
            
            ci = FoldedHistory()
            ci.init(hist_len, table_bits)
            self.folded_indices.append(ci)
            
            tag_width = self.tag_widths[i + 1]
            ct0 = FoldedHistory()
            ct0.init(hist_len, tag_width)
            ct1 = FoldedHistory()
            ct1.init(hist_len, max(1, tag_width - 1))
            self.folded_tags[0].append(ct0)
            self.folded_tags[1].append(ct1)
        
        # === useAltPredForNewlyAllocated ===
        self.use_alt_size = config.get('use_alt_size', 128)
        self.use_alt_bits = config.get('use_alt_bits', 4)
        self.use_alt_max = (1 << self.use_alt_bits) - 1
        self.use_alt_on_na = np.zeros(self.use_alt_size, dtype=np.int8)
        
        # === Update Policy ===
        self.u_reset_period = config.get('u_reset_period', 19)
        self.max_alloc = config.get('max_alloc', 1)
        self.tick = 1 << (self.u_reset_period - 1)
        
        # =====================================================================
        # SCALE Components (NEW)
        # =====================================================================
        
        # === Per-Branch Correction Counter (PBC) ===
        # Tracks whether corrections help for each PC hash
        self.pbc_size = config.get('pbc_size', 4096)
        self.pbc_bits = config.get('pbc_bits', 3)
        self.pbc_threshold = config.get('pbc_threshold', 4)
        self.pbc_max = (1 << self.pbc_bits) - 1
        self.pbc = np.zeros(self.pbc_size, dtype=np.int8)  # Signed for +/-
        
        # === Pattern Reliability Table (PRT) ===
        # Tracks if (PC, recent_pattern) is reliably predicted
        self.prt_size = config.get('prt_size', 2048)
        self.prt_bits = config.get('prt_bits', 4)
        self.prt_reliability_threshold = config.get('prt_reliability_threshold', 8)
        self.prt_history_bits = config.get('prt_history_bits', 8)
        self.prt_max = (1 << self.prt_bits) - 1
        self.prt = np.ones(self.prt_size, dtype=np.uint8) * (self.prt_max // 2)
        self.recent_outcomes = 0  # Rolling window of recent branch outcomes
        
        # === Micro-Corrector (MC) ===
        # Tiny perceptron consulted only when PBC and PRT both indicate correction may help
        self.mc_size = config.get('mc_size', 512)
        self.mc_history_bits = config.get('mc_history_bits', 32)
        self.mc_weight_bits = config.get('mc_weight_bits', 6)
        self.mc_threshold_factor = config.get('mc_threshold_factor', 1.93)
        self.mc_max = (1 << (self.mc_weight_bits - 1)) - 1
        self.mc_min = -(1 << (self.mc_weight_bits - 1))
        self.mc_theta = int(self.mc_threshold_factor * self.mc_history_bits + 14)
        
        # Perceptron weights: [mc_size, mc_history_bits + 1]
        self.mc_weights = np.zeros((self.mc_size, self.mc_history_bits + 1), dtype=np.int8)
        
        # Local history for MC
        self.local_hist_size = 1024
        self.local_hist = np.zeros(self.local_hist_size, dtype=np.uint64)
        
        # === Prediction State ===
        self._pred_info = {}
        
        # === Statistics ===
        self._tage_preds = 0
        self._tage_correct = 0
        self._corrections_considered = 0
        self._corrections_applied = 0
        self._corrections_helped = 0
        self._corrections_hurt = 0
    
    def _calculate_history_lengths(self) -> List[int]:
        """Calculate geometric history lengths."""
        hist_lengths = [0]  # Index 0 for bimodal
        
        for i in range(1, self.num_tables + 1):
            if i == 1:
                length = self.min_hist
            elif i == self.num_tables:
                length = self.max_hist
            else:
                ratio = (self.max_hist / self.min_hist) ** ((i - 1) / (self.num_tables - 1))
                length = int(self.min_hist * ratio + 0.5)
            hist_lengths.append(length)
        
        return hist_lengths
    
    # ========================================================================
    # TAGE Core Functions
    # ========================================================================
    
    def _bimodal_index(self, pc: int) -> int:
        return (pc >> 2) & (self.bimodal_size - 1)
    
    def _get_bimodal_pred(self, pc: int) -> bool:
        return self.bimodal_pred[self._bimodal_index(pc)]
    
    def _update_bimodal(self, pc: int, taken: bool):
        idx = self._bimodal_index(pc)
        hyst_idx = idx >> self.bimodal_hyst_ratio
        
        pred = self.bimodal_pred[idx]
        hyst = self.bimodal_hyst[hyst_idx]
        
        inter = (int(pred) << 1) + int(hyst)
        if taken:
            if inter < 3:
                inter += 1
        elif inter > 0:
            inter -= 1
        
        self.bimodal_pred[idx] = (inter >> 1) != 0
        self.bimodal_hyst[hyst_idx] = (inter & 1) != 0
    
    def _F(self, A: int, size: int, bank: int) -> int:
        """Path history shuffle function."""
        table_bits = self.log_table_sizes[bank + 1]
        if table_bits <= 0:
            return 0
        
        A = A & ((1 << size) - 1)
        A1 = A & ((1 << table_bits) - 1)
        A2 = A >> table_bits
        
        effective_bank = (bank + 1) % table_bits if table_bits > 0 else 0
        shift_right = max(0, table_bits - effective_bank)
        
        if effective_bank > 0:
            A2 = ((A2 << effective_bank) & ((1 << table_bits) - 1)) + \
                 (A2 >> shift_right if shift_right > 0 else 0)
        A = A1 ^ A2
        if effective_bank > 0:
            A = ((A << effective_bank) & ((1 << table_bits) - 1)) + \
                (A >> shift_right if shift_right > 0 else 0)
        
        return A
    
    def _tage_index(self, pc: int, table: int) -> int:
        """Compute TAGE table index."""
        hist_len = self.history_lengths[table + 1]
        table_bits = self.log_table_sizes[table + 1]
        path_len = min(hist_len, self.path_hist_bits)
        
        shifted_pc = (pc >> 2) & 0xFFFFFFFF
        
        index = (shifted_pc ^
                 (shifted_pc >> (abs(table_bits - table) + 1)) ^
                 self.folded_indices[table].comp ^
                 self._F(self.path_hist, path_len, table))
        
        return int(index & (self.table_sizes[table] - 1))
    
    def _tage_tag(self, pc: int, table: int) -> int:
        """Compute TAGE tag."""
        tag_width = self.tag_widths[table + 1]
        
        tag = (((pc >> 2) & 0xFFFFFFFF) ^
               self.folded_tags[0][table].comp ^
               (self.folded_tags[1][table].comp << 1))
        
        return int(tag & ((1 << tag_width) - 1))
    
    # ========================================================================
    # SCALE Component Functions
    # ========================================================================
    
    def _pbc_index(self, pc: int) -> int:
        """Per-Branch Correction Counter index."""
        return (pc >> 2) & (self.pbc_size - 1)
    
    def _should_consider_correction(self, pc: int) -> bool:
        """Check if this branch benefits from corrections historically."""
        idx = self._pbc_index(pc)
        return self.pbc[idx] >= self.pbc_threshold
    
    def _prt_index(self, pc: int) -> int:
        """Pattern Reliability Table index - combines PC with recent outcomes."""
        pattern = self.recent_outcomes & ((1 << self.prt_history_bits) - 1)
        return ((pc >> 2) ^ pattern) & (self.prt_size - 1)
    
    def _is_pattern_unreliable(self, pc: int) -> bool:
        """Check if current pattern has low reliability (TAGE struggles here)."""
        idx = self._prt_index(pc)
        return self.prt[idx] < self.prt_reliability_threshold
    
    def _mc_index(self, pc: int) -> int:
        """Micro-Corrector index."""
        return (pc >> 2) & (self.mc_size - 1)
    
    def _mc_predict(self, pc: int) -> Tuple[bool, int]:
        """Get Micro-Corrector prediction."""
        mc_idx = self._mc_index(pc)
        local_idx = (pc >> 2) & (self.local_hist_size - 1)
        local_hist = self.local_hist[local_idx]
        
        weights = self.mc_weights[mc_idx]
        
        # Compute perceptron sum
        total = int(weights[self.mc_history_bits])  # Bias
        
        for i in range(self.mc_history_bits):
            bit = (local_hist >> i) & 1
            w = int(weights[i])
            if bit:
                total += w
            else:
                total -= w
        
        return total >= 0, total
    
    def _mc_update(self, pc: int, taken: bool, mc_sum: int):
        """Update Micro-Corrector."""
        mc_idx = self._mc_index(pc)
        local_idx = (pc >> 2) & (self.local_hist_size - 1)
        local_hist = self.local_hist[local_idx]
        
        weights = self.mc_weights[mc_idx]
        t_val = 1 if taken else -1
        
        # Update if wrong or below threshold
        mc_pred = mc_sum >= 0
        if mc_pred != taken or abs(mc_sum) < self.mc_theta:
            # Update bias
            new_bias = int(weights[self.mc_history_bits]) + t_val
            weights[self.mc_history_bits] = np.clip(new_bias, self.mc_min, self.mc_max)
            
            # Update history weights
            for i in range(self.mc_history_bits):
                bit = (local_hist >> i) & 1
                if bit:
                    delta = t_val
                else:
                    delta = -t_val
                new_w = int(weights[i]) + delta
                weights[i] = np.clip(new_w, self.mc_min, self.mc_max)
    
    # ========================================================================
    # Prediction
    # ========================================================================
    
    def predict(self, pc: int, history: np.ndarray = None) -> PredictionResult:
        """
        TAGE-SCALE prediction.
        
        1. Standard TAGE lookup
        2. Check if correction should be considered (PBC + PRT)
        3. If yes, consult MC and potentially flip prediction
        """
        # === Compute indices and tags for all tables ===
        indices = []
        tags = []
        for i in range(self.num_tables):
            indices.append(self._tage_index(pc, i))
            tags.append(self._tage_tag(pc, i))
        
        # === Find provider (longest matching history) ===
        provider = -1
        for i in range(self.num_tables - 1, -1, -1):
            idx = indices[i]
            if self.tables[i][idx].tag == tags[i]:
                provider = i
                break
        
        # === Find alternate provider ===
        alt_provider = -1
        if provider >= 0:
            for i in range(provider - 1, -1, -1):
                idx = indices[i]
                if self.tables[i][idx].tag == tags[i]:
                    alt_provider = i
                    break
        
        # === Compute TAGE prediction ===
        if provider >= 0:
            provider_idx = indices[provider]
            provider_entry = self.tables[provider][provider_idx]
            provider_pred = provider_entry.ctr >= 0
            provider_ctr = provider_entry.ctr
            
            if alt_provider >= 0:
                alt_idx = indices[alt_provider]
                alt_pred = self.tables[alt_provider][alt_idx].ctr >= 0
            else:
                alt_pred = self._get_bimodal_pred(pc)
            
            # Weak entry check
            weak_entry = abs(2 * provider_ctr + 1) <= 1
            
            # UseAlt logic
            use_alt_idx = (pc >> 2) & (self.use_alt_size - 1)
            use_alt = self.use_alt_on_na[use_alt_idx] < 0
            
            if weak_entry and use_alt:
                tage_pred = alt_pred
            else:
                tage_pred = provider_pred
            
            # TAGE confidence
            tage_conf = abs(provider_ctr) / self.ctr_max
        else:
            alt_pred = self._get_bimodal_pred(pc)
            provider_pred = alt_pred
            tage_pred = alt_pred
            tage_conf = 0.5
            weak_entry = False
            provider_ctr = 0
        
        # === SCALE: Decide if correction should be considered ===
        final_pred = tage_pred
        mc_pred = tage_pred
        mc_sum = 0
        correction_considered = False
        correction_applied = False
        
        # Only consider correction if:
        # 1. TAGE has low confidence (weak entry or from short history)
        # 2. This branch historically benefits from corrections (PBC)
        # 3. Current pattern is unreliable (PRT)
        
        low_tage_conf = weak_entry or (provider >= 0 and provider < self.num_tables // 3)
        should_consider = self._should_consider_correction(pc)
        pattern_unreliable = self._is_pattern_unreliable(pc)
        
        if low_tage_conf and should_consider and pattern_unreliable:
            correction_considered = True
            mc_pred, mc_sum = self._mc_predict(pc)
            
            # Only apply correction if MC is confident and disagrees
            mc_confident = abs(mc_sum) >= self.mc_theta // 2
            
            if mc_confident and mc_pred != tage_pred:
                final_pred = mc_pred
                correction_applied = True
        
        # Compute final confidence
        if correction_applied:
            confidence = min(abs(mc_sum) / self.mc_theta, 1.0)
        else:
            confidence = tage_conf
        
        # Save prediction info
        self._pred_info = {
            'provider': provider,
            'alt_provider': alt_provider,
            'indices': indices,
            'tags': tags,
            'tage_pred': tage_pred,
            'provider_pred': provider_pred,
            'alt_pred': alt_pred,
            'weak_entry': weak_entry,
            'provider_ctr': provider_ctr,
            'final_pred': final_pred,
            'mc_pred': mc_pred,
            'mc_sum': mc_sum,
            'correction_considered': correction_considered,
            'correction_applied': correction_applied,
        }
        
        return PredictionResult(
            prediction=final_pred,
            confidence=confidence,
            predictor_used=self.name,
            raw_sum=float(provider_ctr if provider >= 0 else mc_sum)
        )
    
    # ========================================================================
    # Update
    # ========================================================================
    
    def update(self, pc: int, history: np.ndarray, taken: bool,
               prediction: PredictionResult) -> None:
        """
        Update TAGE-SCALE components.
        """
        info = self._pred_info
        
        provider = info['provider']
        alt_provider = info['alt_provider']
        indices = info['indices']
        tags = info['tags']
        tage_pred = info['tage_pred']
        provider_pred = info['provider_pred']
        alt_pred = info['alt_pred']
        weak_entry = info['weak_entry']
        final_pred = info['final_pred']
        mc_pred = info['mc_pred']
        mc_sum = info['mc_sum']
        correction_considered = info['correction_considered']
        correction_applied = info['correction_applied']
        
        # === Track statistics ===
        self._tage_preds += 1
        if tage_pred == taken:
            self._tage_correct += 1
        
        if correction_considered:
            self._corrections_considered += 1
        
        if correction_applied:
            self._corrections_applied += 1
            if final_pred == taken:
                self._corrections_helped += 1
            else:
                self._corrections_hurt += 1
        
        # === Update Per-Branch Correction Counter (PBC) ===
        # Key insight: Only update when correction was considered
        if correction_considered:
            pbc_idx = self._pbc_index(pc)
            tage_correct = (tage_pred == taken)
            mc_correct = (mc_pred == taken)
            
            if mc_correct and not tage_correct:
                # Correction would have helped
                self.pbc[pbc_idx] = min(self.pbc_max, int(self.pbc[pbc_idx]) + 1)
            elif tage_correct and not mc_correct:
                # Correction would have hurt
                self.pbc[pbc_idx] = max(-self.pbc_max, int(self.pbc[pbc_idx]) - 1)
            # If both correct or both wrong, no strong signal
        
        # === Update Pattern Reliability Table (PRT) ===
        prt_idx = self._prt_index(pc)
        tage_correct = (tage_pred == taken)
        
        if tage_correct:
            # TAGE was right for this pattern - increase reliability
            self.prt[prt_idx] = min(self.prt_max, int(self.prt[prt_idx]) + 1)
        else:
            # TAGE was wrong - decrease reliability
            self.prt[prt_idx] = max(0, int(self.prt[prt_idx]) - 1)
        
        # === Update Micro-Corrector ===
        # Always update MC to keep it trained
        self._mc_update(pc, taken, mc_sum)
        
        # === Update Local History ===
        local_idx = (pc >> 2) & (self.local_hist_size - 1)
        t_int = 1 if taken else 0
        self.local_hist[local_idx] = ((int(self.local_hist[local_idx]) << 1) | t_int) & \
                                     ((1 << 64) - 1)
        
        # === Update Recent Outcomes (for PRT indexing) ===
        self.recent_outcomes = ((self.recent_outcomes << 1) | t_int) & \
                               ((1 << self.prt_history_bits) - 1)
        
        # === Standard TAGE Update ===
        tage_wrong = (tage_pred != taken)
        alloc = tage_wrong and (provider < self.num_tables - 1)
        
        # Update useAltPredForNewlyAllocated
        if provider >= 0 and weak_entry:
            if provider_pred == taken:
                alloc = False
            
            if provider_pred != alt_pred:
                use_alt_idx = (pc >> 2) & (self.use_alt_size - 1)
                if alt_pred == taken:
                    self.use_alt_on_na[use_alt_idx] = min(
                        self.use_alt_max, int(self.use_alt_on_na[use_alt_idx]) + 1)
                else:
                    self.use_alt_on_na[use_alt_idx] = max(
                        -self.use_alt_max - 1, int(self.use_alt_on_na[use_alt_idx]) - 1)
        
        # Handle allocation
        if alloc:
            self._handle_allocation(provider, taken, indices, tags)
        
        # Update TAGE tables
        if provider >= 0:
            provider_idx = indices[provider]
            entry = self.tables[provider][provider_idx]
            
            # Update counter
            if taken:
                entry.ctr = min(self.ctr_max, entry.ctr + 1)
            else:
                entry.ctr = max(self.ctr_min, entry.ctr - 1)
            
            # Update alternate if provider has u=0
            if entry.u == 0:
                if alt_provider >= 0:
                    alt_idx = indices[alt_provider]
                    alt_entry = self.tables[alt_provider][alt_idx]
                    if taken:
                        alt_entry.ctr = min(self.ctr_max, alt_entry.ctr + 1)
                    else:
                        alt_entry.ctr = max(self.ctr_min, alt_entry.ctr - 1)
                else:
                    self._update_bimodal(pc, taken)
            
            # Update useful bits
            if tage_pred != alt_pred:
                if tage_pred == taken:
                    entry.u = min(self.u_max, entry.u + 1)
                else:
                    entry.u = max(0, entry.u - 1)
        else:
            self._update_bimodal(pc, taken)
        
        # === Update Global History ===
        self._update_histories(pc, taken)
        
        # === Periodic useful bit reset ===
        self.tick += 1
        if (self.tick & ((1 << self.u_reset_period) - 1)) == 0:
            for table in self.tables:
                for entry in table:
                    entry.u >>= 1
    
    def _handle_allocation(self, provider: int, taken: bool,
                          indices: List[int], tags: List[int]):
        """Handle entry allocation."""
        start = provider + 1 if provider >= 0 else 0
        
        # Find minimum useful value
        min_u = self.u_max + 1
        for i in range(start, self.num_tables):
            idx = indices[i]
            if self.tables[i][idx].u < min_u:
                min_u = self.tables[i][idx].u
        
        if min_u > 0:
            # No free entry - decay useful bits
            for i in range(start, self.num_tables):
                idx = indices[i]
                if self.tables[i][idx].u > 0:
                    self.tables[i][idx].u -= 1
        else:
            # Allocate in entries with u=0
            num_allocated = 0
            for i in range(start, self.num_tables):
                idx = indices[i]
                if self.tables[i][idx].u == 0:
                    self.tables[i][idx].tag = tags[i]
                    self.tables[i][idx].ctr = 0 if taken else -1
                    num_allocated += 1
                    if num_allocated >= self.max_alloc:
                        break
    
    def _update_histories(self, pc: int, taken: bool):
        """Update global and path histories."""
        # Path history
        path_bit = (pc >> 2) & 1
        self.path_hist = ((self.path_hist << 1) + path_bit) & \
                         ((1 << self.path_hist_bits) - 1)
        
        # Global history buffer
        if self.pt_ghist <= 0:
            buffer_size = len(self.global_hist)
            for i in range(min(self.max_hist, buffer_size)):
                if i < buffer_size and buffer_size - self.max_hist + i >= 0:
                    self.global_hist[buffer_size - self.max_hist + i] = self.global_hist[i]
            self.pt_ghist = max(1, buffer_size - self.max_hist)
        
        self.pt_ghist -= 1
        t_int = 1 if taken else 0
        self.global_hist[self.pt_ghist] = t_int
        
        # Folded histories
        new_bit = t_int
        for i in range(self.num_tables):
            hist_len = self.history_lengths[i + 1]
            idx = self.pt_ghist + hist_len
            old_bit = int(self.global_hist[idx]) if 0 <= idx < len(self.global_hist) else 0
            
            self.folded_indices[i].update(new_bit, old_bit)
            self.folded_tags[0][i].update(new_bit, old_bit)
            self.folded_tags[1][i].update(new_bit, old_bit)
    
    # ========================================================================
    # Hardware Cost Estimation
    # ========================================================================
    
    def get_hardware_cost(self) -> dict:
        """Estimate hardware implementation cost."""
        bits = 0
        components = {}
        
        # Tagged tables
        tage_bits = 0
        for i in range(self.num_tables):
            bits_per_entry = self.counter_bits + self.useful_bits + self.tag_widths[i + 1]
            tage_bits += self.table_sizes[i] * bits_per_entry
        bits += tage_bits
        components['tage_tables'] = tage_bits
        
        # Bimodal
        bimodal_bits = self.bimodal_size + (self.bimodal_size >> self.bimodal_hyst_ratio)
        bits += bimodal_bits
        components['bimodal'] = bimodal_bits
        
        # Per-Branch Correction Counter
        pbc_bits = self.pbc_size * self.pbc_bits
        bits += pbc_bits
        components['pbc'] = pbc_bits
        
        # Pattern Reliability Table
        prt_bits = self.prt_size * self.prt_bits
        bits += prt_bits
        components['prt'] = prt_bits
        
        # Micro-Corrector
        mc_bits = self.mc_size * (self.mc_history_bits + 1) * self.mc_weight_bits
        bits += mc_bits
        components['micro_corrector'] = mc_bits
        
        # Local history
        local_bits = self.local_hist_size * 64
        bits += local_bits
        components['local_history'] = local_bits
        
        # UseAlt
        use_alt_bits = self.use_alt_size * self.use_alt_bits
        bits += use_alt_bits
        components['use_alt'] = use_alt_bits
        
        # Histories
        hist_bits = self.max_hist + self.path_hist_bits + self.prt_history_bits
        bits += hist_bits
        components['histories'] = hist_bits
        
        return {
            'name': self.name,
            'total_bits': bits,
            'total_bytes': bits // 8,
            'total_kb': bits / 8 / 1024,
            'components': components,
            'breakdown_kb': {k: v / 8 / 1024 for k, v in components.items()},
        }
    
    def get_statistics(self) -> dict:
        """Get prediction statistics."""
        tage_acc = self._tage_correct / self._tage_preds * 100 if self._tage_preds > 0 else 0
        
        correction_rate = self._corrections_applied / self._tage_preds * 100 if self._tage_preds > 0 else 0
        correction_accuracy = (self._corrections_helped / self._corrections_applied * 100 
                              if self._corrections_applied > 0 else 0)
        
        return {
            'tage_predictions': self._tage_preds,
            'tage_accuracy': tage_acc,
            'corrections_considered': self._corrections_considered,
            'corrections_applied': self._corrections_applied,
            'corrections_helped': self._corrections_helped,
            'corrections_hurt': self._corrections_hurt,
            'correction_rate': correction_rate,
            'correction_accuracy': correction_accuracy,
        }
    
    def reset(self) -> None:
        """Reset predictor state."""
        super().reset()
        
        # Bimodal
        self.bimodal_pred.fill(False)
        self.bimodal_hyst.fill(True)
        
        # Tagged tables
        for table in self.tables:
            for entry in table:
                entry.ctr = 0
                entry.tag = 0
                entry.u = 0
        
        # PBC
        self.pbc.fill(0)
        
        # PRT
        self.prt.fill(self.prt_max // 2)
        self.recent_outcomes = 0
        
        # MC
        self.mc_weights.fill(0)
        
        # Local history
        self.local_hist.fill(0)
        
        # UseAlt
        self.use_alt_on_na.fill(0)
        
        # Histories
        self.global_hist.fill(0)
        self.pt_ghist = 0
        self.path_hist = 0
        
        # Folded histories
        for i in range(self.num_tables):
            self.folded_indices[i].comp = 0
            self.folded_tags[0][i].comp = 0
            self.folded_tags[1][i].comp = 0
        
        # Tick
        self.tick = 1 << (self.u_reset_period - 1)
        
        # Statistics
        self._tage_preds = 0
        self._tage_correct = 0
        self._corrections_considered = 0
        self._corrections_applied = 0
        self._corrections_helped = 0
        self._corrections_hurt = 0
