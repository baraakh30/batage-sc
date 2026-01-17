"""
TAGE-SC-L: TAGE with Statistical Corrector and Loop Predictor

This is a faithful implementation of the CBP-winning TAGE-SC-L architecture.
Based on André Seznec's championship branch predictors.

Key components:
1. TAGE base predictor with 12 tables (geometric history)
2. Statistical Corrector (SC) - captures local/global correlations TAGE misses
3. Loop Predictor - specialized for loop branches with fixed iteration counts

The key insight from CBP winners:
- SC doesn't "correct" TAGE - it provides ADDITIONAL perspectives
- Each component captures patterns the others miss
- Only combine when there's high confidence disagreement

"""

import numpy as np
from typing import List, Tuple, Optional
from .base import BasePredictor, PredictionResult


# ============================================================================
# Configuration
# ============================================================================

TAGE_SC_L_64KB = {
    'name': 'TAGE-SC-L-64KB',
    
    # TAGE configuration (similar to gem5 8-component)
    'num_tage_tables': 12,
    'tage_table_bits': [10, 10, 11, 11, 11, 11, 10, 10, 10, 10, 9, 9],
    'tage_tag_bits': [7, 7, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15],
    'tage_history_lengths': [4, 6, 10, 16, 25, 40, 64, 101, 160, 254, 403, 640],
    'tage_ctr_bits': 3,
    'tage_u_bits': 2,
    
    # Bimodal
    'bimodal_bits': 14,  # 16K entries
    
    # Loop predictor
    'loop_table_bits': 6,   # 64 entries
    'loop_tag_bits': 14,
    'loop_ctr_bits': 14,    # Up to 16K iterations
    'loop_conf_bits': 2,    # 4-state confidence
    'loop_age_bits': 8,
    
    # Statistical Corrector
    'sc_num_tables': 6,
    'sc_table_bits': 10,    # 1K entries each
    'sc_ctr_bits': 6,       # Larger counters for better resolution
    
    # Global parameters
    'path_bits': 32,
    'use_alt_bits': 4,
    'use_alt_size': 128,
    'u_reset_period': 19,
}

TAGE_SC_L_32KB = {
    'name': 'TAGE-SC-L-32KB',
    'num_tage_tables': 10,
    'tage_table_bits': [9, 9, 10, 10, 10, 10, 9, 9, 9, 8],
    'tage_tag_bits': [7, 7, 8, 8, 9, 10, 10, 11, 12, 13],
    'tage_history_lengths': [4, 6, 10, 16, 25, 40, 64, 101, 160, 254],
    'tage_ctr_bits': 3,
    'tage_u_bits': 2,
    'bimodal_bits': 13,
    'loop_table_bits': 5,
    'loop_tag_bits': 12,
    'loop_ctr_bits': 12,
    'loop_conf_bits': 2,
    'loop_age_bits': 8,
    'sc_num_tables': 5,
    'sc_table_bits': 9,
    'sc_ctr_bits': 5,
    'path_bits': 27,
    'use_alt_bits': 4,
    'use_alt_size': 64,
    'u_reset_period': 18,
}


# ============================================================================
# Helper Classes
# ============================================================================

class FoldedHistory:
    """Folded history for efficient indexing."""
    __slots__ = ['comp', 'comp_length', 'orig_length', 'outpoint']
    
    def __init__(self):
        self.comp = 0
        self.comp_length = 0
        self.orig_length = 0
        self.outpoint = 0
    
    def init(self, orig_len: int, comp_len: int):
        self.orig_length = orig_len
        self.comp_length = comp_len
        self.outpoint = orig_len % comp_len if comp_len > 0 else 0
        self.comp = 0
    
    def update(self, new_bit: int, old_bit: int):
        if self.comp_length <= 0:
            return
        self.comp = (self.comp << 1) | new_bit
        self.comp ^= old_bit << self.outpoint
        self.comp ^= self.comp >> self.comp_length
        self.comp &= (1 << self.comp_length) - 1


class TAGEEntry:
    __slots__ = ['ctr', 'tag', 'u']
    def __init__(self):
        self.ctr = 0
        self.tag = 0
        self.u = 0


class LoopEntry:
    """Loop predictor entry."""
    __slots__ = ['tag', 'current_iter', 'past_iter', 'confidence', 'age', 'dir']
    def __init__(self):
        self.tag = 0
        self.current_iter = 0  # Current iteration count
        self.past_iter = 0     # Last recorded loop length
        self.confidence = 0    # Confidence in loop length
        self.age = 0           # Age for replacement
        self.dir = True        # Loop exit direction


# ============================================================================
# TAGE-SC-L Predictor
# ============================================================================

class TAGE_SC_L(BasePredictor):
    """
    TAGE-SC-L: Championship Branch Predictor
    
    Three coordinated components:
    1. TAGE - Handles most branches with geometric history
    2. Loop - Specializes in loop branches (fixed iteration patterns)
    3. SC - Captures correlations TAGE misses using multiple feature tables
    """
    
    def __init__(self, config: dict = None):
        config = config or TAGE_SC_L_64KB
        super().__init__(config.get('name', 'TAGE-SC-L'), config)
        
        # ==================== TAGE Component ====================
        self.num_tage_tables = config['num_tage_tables']
        self.tage_table_bits = config['tage_table_bits']
        self.tage_tag_bits = config['tage_tag_bits']
        self.tage_history_lengths = config['tage_history_lengths']
        self.tage_ctr_bits = config['tage_ctr_bits']
        self.tage_u_bits = config['tage_u_bits']
        
        self.ctr_max = (1 << (self.tage_ctr_bits - 1)) - 1
        self.ctr_min = -(1 << (self.tage_ctr_bits - 1))
        self.u_max = (1 << self.tage_u_bits) - 1
        
        # TAGE tables
        self.tage_tables: List[List[TAGEEntry]] = []
        for i in range(self.num_tage_tables):
            size = 1 << self.tage_table_bits[i]
            self.tage_tables.append([TAGEEntry() for _ in range(size)])
        
        # Folded histories
        self.tage_idx_fold: List[FoldedHistory] = []
        self.tage_tag_fold: List[List[FoldedHistory]] = [[], []]
        
        for i in range(self.num_tage_tables):
            hist_len = self.tage_history_lengths[i]
            
            idx_fold = FoldedHistory()
            idx_fold.init(hist_len, self.tage_table_bits[i])
            self.tage_idx_fold.append(idx_fold)
            
            tag_fold0 = FoldedHistory()
            tag_fold0.init(hist_len, self.tage_tag_bits[i])
            tag_fold1 = FoldedHistory()
            tag_fold1.init(hist_len, max(1, self.tage_tag_bits[i] - 1))
            self.tage_tag_fold[0].append(tag_fold0)
            self.tage_tag_fold[1].append(tag_fold1)
        
        # Bimodal
        self.bimodal_bits = config['bimodal_bits']
        bimodal_size = 1 << self.bimodal_bits
        self.bimodal_pred = np.zeros(bimodal_size, dtype=np.bool_)
        self.bimodal_hyst = np.ones(bimodal_size >> 2, dtype=np.bool_)
        
        # UseAlt
        self.use_alt_size = config['use_alt_size']
        self.use_alt_bits = config['use_alt_bits']
        self.use_alt_max = (1 << self.use_alt_bits) - 1
        self.use_alt_on_na = np.zeros(self.use_alt_size, dtype=np.int8)
        
        # ==================== Loop Predictor ====================
        self.loop_table_bits = config['loop_table_bits']
        self.loop_tag_bits = config['loop_tag_bits']
        self.loop_ctr_bits = config['loop_ctr_bits']
        self.loop_conf_bits = config['loop_conf_bits']
        
        loop_size = 1 << self.loop_table_bits
        self.loop_table = [LoopEntry() for _ in range(loop_size)]
        self.loop_tag_mask = (1 << self.loop_tag_bits) - 1
        self.loop_ctr_max = (1 << self.loop_ctr_bits) - 1
        self.loop_conf_max = (1 << self.loop_conf_bits) - 1
        
        # Loop state
        self.loop_useful = False  # Track if loop override helped
        self.loop_use_counter = 0  # Counter for loop override decisions
        
        # ==================== Statistical Corrector ====================
        self.sc_num_tables = config['sc_num_tables']
        self.sc_table_bits = config['sc_table_bits']
        self.sc_ctr_bits = config['sc_ctr_bits']
        
        sc_size = 1 << self.sc_table_bits
        self.sc_max = (1 << (self.sc_ctr_bits - 1)) - 1
        self.sc_min = -(1 << (self.sc_ctr_bits - 1))
        
        # SC tables with different indexing
        self.sc_tables = [np.zeros(sc_size, dtype=np.int8) 
                         for _ in range(self.sc_num_tables)]
        
        # SC threshold - learned during execution
        self.sc_threshold = 6
        
        # Per-branch SC confidence (small table)
        self.sc_conf_table = np.zeros(256, dtype=np.int8)  # 256 entries, 4-bit counters
        
        # ==================== Global State ====================
        self.path_bits = config['path_bits']
        self.path_hist = 0
        
        self.max_hist = max(self.tage_history_lengths)
        self.ghist = np.zeros(self.max_hist + 256, dtype=np.uint8)
        self.pt_ghist = 0
        self.ghist_int = 0  # For fast integer ops
        
        # Local history
        self.local_hist_table = np.zeros(1024, dtype=np.uint32)
        
        # Useful bit reset
        self.u_reset_period = config['u_reset_period']
        self.tick = 1 << (self.u_reset_period - 1)
        
        # Prediction state
        self._info = {}
        
        # Stats
        self.tage_pred_count = 0
        self.loop_override_count = 0
        self.sc_override_count = 0
    
    # ==================== TAGE Methods ====================
    
    def _bimodal_idx(self, pc: int) -> int:
        return (pc >> 2) & ((1 << self.bimodal_bits) - 1)
    
    def _bimodal_pred(self, pc: int) -> bool:
        return bool(self.bimodal_pred[self._bimodal_idx(pc)])
    
    def _bimodal_update(self, pc: int, taken: bool):
        idx = self._bimodal_idx(pc)
        hyst_idx = idx >> 2
        
        pred = self.bimodal_pred[idx]
        hyst = self.bimodal_hyst[hyst_idx]
        
        inter = (int(pred) << 1) + int(hyst)
        if taken:
            inter = min(3, inter + 1)
        else:
            inter = max(0, inter - 1)
        
        self.bimodal_pred[idx] = inter >> 1
        self.bimodal_hyst[hyst_idx] = inter & 1
    
    def _tage_idx(self, pc: int, table: int) -> int:
        """Compute TAGE table index."""
        table_bits = self.tage_table_bits[table]
        hist_len = self.tage_history_lengths[table]
        
        shifted_pc = (pc >> 2) & 0xFFFFFFFF
        
        # Mix PC, folded history, and path history
        idx = shifted_pc ^ self.tage_idx_fold[table].comp
        
        # Add path history contribution
        path_len = min(hist_len, self.path_bits)
        path_contrib = self.path_hist & ((1 << path_len) - 1)
        idx ^= path_contrib ^ (path_contrib >> table_bits)
        
        return idx & ((1 << table_bits) - 1)
    
    def _tage_tag(self, pc: int, table: int) -> int:
        """Compute TAGE tag."""
        tag_bits = self.tage_tag_bits[table]
        shifted_pc = (pc >> 2) & 0xFFFFFFFF
        
        tag = shifted_pc ^ self.tage_tag_fold[0][table].comp
        tag ^= self.tage_tag_fold[1][table].comp << 1
        
        return tag & ((1 << tag_bits) - 1)
    
    def _tage_predict(self, pc: int) -> Tuple[bool, int, int, float, bool, bool]:
        """
        TAGE prediction.
        Returns: (pred, provider, alt_provider, confidence, weak_entry, alt_pred)
        """
        # Compute all indices and tags
        indices = []
        tags = []
        for i in range(self.num_tage_tables):
            indices.append(self._tage_idx(pc, i))
            tags.append(self._tage_tag(pc, i))
        
        # Find provider (longest matching)
        provider = -1
        for i in range(self.num_tage_tables - 1, -1, -1):
            idx = indices[i]
            if self.tage_tables[i][idx].tag == tags[i]:
                provider = i
                break
        
        # Find alternate
        alt_provider = -1
        if provider > 0:
            for i in range(provider - 1, -1, -1):
                idx = indices[i]
                if self.tage_tables[i][idx].tag == tags[i]:
                    alt_provider = i
                    break
        
        # Get predictions
        if provider >= 0:
            idx = indices[provider]
            entry = self.tage_tables[provider][idx]
            provider_pred = entry.ctr >= 0
            provider_ctr = entry.ctr
            confidence = abs(provider_ctr) / self.ctr_max
            weak_entry = abs(2 * provider_ctr + 1) <= 1
            
            if alt_provider >= 0:
                alt_idx = indices[alt_provider]
                alt_pred = self.tage_tables[alt_provider][alt_idx].ctr >= 0
            else:
                alt_pred = self._bimodal_pred(pc)
            
            # UseAlt decision
            use_alt_idx = (pc >> 2) & (self.use_alt_size - 1)
            use_alt = self.use_alt_on_na[use_alt_idx] < 0
            
            if weak_entry and use_alt:
                tage_pred = alt_pred
            else:
                tage_pred = provider_pred
        else:
            tage_pred = self._bimodal_pred(pc)
            alt_pred = tage_pred
            confidence = 0.5
            weak_entry = False
        
        # Save indices/tags for update
        self._info['indices'] = indices
        self._info['tags'] = tags
        self._info['provider'] = provider
        self._info['alt_provider'] = alt_provider
        self._info['alt_pred'] = alt_pred
        self._info['weak_entry'] = weak_entry
        self._info['tage_pred'] = tage_pred
        
        return tage_pred, provider, alt_provider, confidence, weak_entry, alt_pred
    
    # ==================== Loop Predictor ====================
    
    def _loop_idx(self, pc: int) -> int:
        return (pc >> 2) & ((1 << self.loop_table_bits) - 1)
    
    def _loop_tag(self, pc: int) -> int:
        return (pc >> (2 + self.loop_table_bits)) & self.loop_tag_mask
    
    def _loop_predict(self, pc: int) -> Tuple[bool, bool, bool]:
        """
        Loop predictor.
        Returns: (prediction, valid, confident)
        """
        idx = self._loop_idx(pc)
        tag = self._loop_tag(pc)
        entry = self.loop_table[idx]
        
        if entry.tag != tag:
            return False, False, False
        
        # Check if we have a confident loop
        if entry.confidence < self.loop_conf_max:
            return False, True, False
        
        # Predict based on iteration count
        if entry.current_iter == entry.past_iter:
            # At loop exit point
            pred = not entry.dir  # Predict exit
            return pred, True, True
        else:
            # Not at exit, predict continue
            pred = entry.dir
            return pred, True, True
    
    def _loop_update(self, pc: int, taken: bool, tage_pred: bool):
        """Update loop predictor."""
        idx = self._loop_idx(pc)
        tag = self._loop_tag(pc)
        entry = self.loop_table[idx]
        
        if entry.tag != tag:
            # Different branch - check for allocation
            if tage_pred != taken:  # Misprediction
                # Allocate new entry
                entry.tag = tag
                entry.current_iter = 0
                entry.past_iter = 0
                entry.confidence = 0
                entry.age = 0
                entry.dir = taken
            return
        
        # Same branch - update
        if taken == entry.dir:
            # Continuing loop
            if entry.current_iter < self.loop_ctr_max:
                entry.current_iter += 1
        else:
            # Loop exit
            if entry.current_iter == entry.past_iter:
                # Consistent loop length - increase confidence
                entry.confidence = min(self.loop_conf_max, entry.confidence + 1)
            elif entry.past_iter == 0:
                # First loop completion
                entry.past_iter = entry.current_iter
            else:
                # Inconsistent - decrease confidence
                entry.confidence = max(0, entry.confidence - 1)
                if entry.confidence == 0:
                    entry.past_iter = 0  # Reset
            
            entry.current_iter = 0
    
    # ==================== Statistical Corrector ====================
    
    def _sc_idx(self, pc: int, table: int) -> int:
        """Compute SC index with different hash per table."""
        pc_hash = (pc >> 2) & 0xFFFFFFFF
        local_idx = pc_hash & 1023
        local_hist = int(self.local_hist_table[local_idx]) & 0xFFFFFFFF
        
        if table == 0:
            # PC only
            return pc_hash & ((1 << self.sc_table_bits) - 1)
        elif table == 1:
            # PC ^ global history
            return (pc_hash ^ self.ghist_int) & ((1 << self.sc_table_bits) - 1)
        elif table == 2:
            # PC ^ path history
            return (pc_hash ^ self.path_hist) & ((1 << self.sc_table_bits) - 1)
        elif table == 3:
            # PC ^ local history
            return (pc_hash ^ local_hist) & ((1 << self.sc_table_bits) - 1)
        elif table == 4:
            # PC ^ (global history >> 4) - different history window
            return (pc_hash ^ (self.ghist_int >> 4)) & ((1 << self.sc_table_bits) - 1)
        else:
            # PC ^ (local ^ global) - combined
            return (pc_hash ^ local_hist ^ self.ghist_int) & ((1 << self.sc_table_bits) - 1)
    
    def _sc_predict(self, pc: int) -> Tuple[bool, int]:
        """SC prediction: returns (prediction, sum)."""
        total = 0
        for t in range(self.sc_num_tables):
            idx = self._sc_idx(pc, t)
            total += int(self.sc_tables[t][idx])
        
        return total >= 0, total
    
    def _sc_update(self, pc: int, taken: bool, tage_pred: bool, sc_used: bool):
        """Update SC tables."""
        # Only update if SC was considered (TAGE was weak)
        for t in range(self.sc_num_tables):
            idx = self._sc_idx(pc, t)
            if taken:
                self.sc_tables[t][idx] = min(self.sc_max, int(self.sc_tables[t][idx]) + 1)
            else:
                self.sc_tables[t][idx] = max(self.sc_min, int(self.sc_tables[t][idx]) - 1)
        
        # Update per-branch SC confidence
        conf_idx = (pc >> 2) & 255
        if sc_used:
            if (self._sc_predict(pc)[0] == taken) != (tage_pred == taken):
                # SC was right, TAGE was wrong
                self.sc_conf_table[conf_idx] = min(7, int(self.sc_conf_table[conf_idx]) + 1)
            else:
                self.sc_conf_table[conf_idx] = max(-8, int(self.sc_conf_table[conf_idx]) - 1)
    
    # ==================== Main Predict/Update ====================
    
    def predict(self, pc: int, history: np.ndarray = None) -> PredictionResult:
        """
        Combined prediction from TAGE + Loop + SC.
        """
        # === TAGE Prediction ===
        tage_pred, provider, alt_provider, confidence, weak_entry, alt_pred = self._tage_predict(pc)
        self.tage_pred_count += 1
        
        # === Loop Prediction ===
        loop_pred, loop_valid, loop_confident = self._loop_predict(pc)
        
        # === SC Prediction ===
        sc_pred, sc_sum = self._sc_predict(pc)
        
        # === Combine Predictions ===
        final_pred = tage_pred
        pred_source = "tage"
        
        # 1. Loop predictor has highest priority if confident
        if loop_valid and loop_confident and loop_pred != tage_pred:
            # Use loop override counter to decide
            if self.loop_use_counter >= 0:
                final_pred = loop_pred
                pred_source = "loop"
                self.loop_override_count += 1
        
        # 2. SC can override if TAGE is weak and SC is strong
        elif provider >= 0 and weak_entry:
            sc_conf_idx = (pc >> 2) & 255
            sc_confidence = self.sc_conf_table[sc_conf_idx]
            
            # Only override if SC has good track record for this branch
            # AND SC sum exceeds threshold
            if abs(sc_sum) >= self.sc_threshold and sc_confidence >= 2:
                if sc_pred != tage_pred:
                    final_pred = sc_pred
                    pred_source = "sc"
                    self.sc_override_count += 1
        
        # Save state
        self._info['tage_pred'] = tage_pred
        self._info['final_pred'] = final_pred
        self._info['pred_source'] = pred_source
        self._info['loop_pred'] = loop_pred
        self._info['loop_valid'] = loop_valid
        self._info['loop_confident'] = loop_confident
        self._info['sc_pred'] = sc_pred
        self._info['sc_sum'] = sc_sum
        self._info['confidence'] = confidence
        self._info['weak_entry'] = weak_entry
        
        return PredictionResult(
            prediction=final_pred,
            confidence=confidence,
            predictor_used=self.name,
            raw_sum=float(sc_sum)
        )
    
    def update(self, pc: int, history: np.ndarray, taken: bool,
               prediction: PredictionResult) -> None:
        """Update all components."""
        info = self._info
        tage_pred = info['tage_pred']
        provider = info['provider']
        alt_provider = info['alt_provider']
        indices = info['indices']
        tags = info['tags']
        alt_pred = info['alt_pred']
        weak_entry = info['weak_entry']
        pred_source = info['pred_source']
        
        # === Update Loop ===
        self._loop_update(pc, taken, tage_pred)
        
        # Update loop use counter
        if pred_source == "loop":
            if taken == info['loop_pred']:
                self.loop_use_counter = min(7, self.loop_use_counter + 1)
            else:
                self.loop_use_counter = max(-8, self.loop_use_counter - 1)
        
        # === Update SC ===
        sc_used = pred_source == "sc" or (provider >= 0 and weak_entry)
        self._sc_update(pc, taken, tage_pred, sc_used)
        
        # === Update TAGE ===
        # Allocation on misprediction
        alloc = (tage_pred != taken) and (provider < self.num_tage_tables - 1)
        
        if provider >= 0:
            # Update UseAlt
            if weak_entry and tage_pred != alt_pred:
                use_alt_idx = (pc >> 2) & (self.use_alt_size - 1)
                if alt_pred == taken:
                    self.use_alt_on_na[use_alt_idx] = min(
                        self.use_alt_max, 
                        int(self.use_alt_on_na[use_alt_idx]) + 1
                    )
                else:
                    self.use_alt_on_na[use_alt_idx] = max(
                        -self.use_alt_max - 1,
                        int(self.use_alt_on_na[use_alt_idx]) - 1
                    )
            
            # Don't allocate if provider was correct
            provider_entry = self.tage_tables[provider][indices[provider]]
            if (provider_entry.ctr >= 0) == taken and weak_entry:
                alloc = False
        
        # Handle allocation
        if alloc:
            self._tage_allocate(provider, indices, tags, taken)
        
        # Update TAGE tables
        self._tage_update_tables(pc, taken, provider, alt_provider, indices, tage_pred, alt_pred)
        
        # Update histories
        self._update_histories(pc, taken)
        
        # Periodic useful bit reset
        self.tick += 1
        if (self.tick & ((1 << self.u_reset_period) - 1)) == 0:
            self._reset_useful()
    
    def _tage_allocate(self, provider: int, indices: List[int], 
                       tags: List[int], taken: bool):
        """Allocate new TAGE entries."""
        import random
        
        start = provider + 1 if provider >= 0 else 0
        
        # Find min useful
        min_u = self.u_max + 1
        for i in range(start, self.num_tage_tables):
            idx = indices[i]
            if self.tage_tables[i][idx].u < min_u:
                min_u = self.tage_tables[i][idx].u
        
        # Decay useful bits if no free entry
        if min_u > 0:
            for i in range(start, self.num_tage_tables):
                idx = indices[i]
                if self.tage_tables[i][idx].u > 0:
                    self.tage_tables[i][idx].u -= 1
        
        # Allocate
        allocated = 0
        for i in range(start, self.num_tage_tables):
            idx = indices[i]
            if self.tage_tables[i][idx].u == 0:
                # Allocate
                entry = self.tage_tables[i][idx]
                entry.tag = tags[i]
                entry.ctr = 0 if taken else -1
                entry.u = 0
                allocated += 1
                if allocated >= 2:  # Max 2 allocations
                    break
    
    def _tage_update_tables(self, pc: int, taken: bool, provider: int,
                            alt_provider: int, indices: List[int],
                            tage_pred: bool, alt_pred: bool):
        """Update TAGE counter and useful bits."""
        if provider >= 0:
            idx = indices[provider]
            entry = self.tage_tables[provider][idx]
            
            # Update counter
            if taken:
                entry.ctr = min(self.ctr_max, entry.ctr + 1)
            else:
                entry.ctr = max(self.ctr_min, entry.ctr - 1)
            
            # Update useful
            if tage_pred != alt_pred:
                if tage_pred == taken:
                    entry.u = min(self.u_max, entry.u + 1)
                else:
                    entry.u = max(0, entry.u - 1)
            
            # Update alt if provider u == 0
            if entry.u == 0:
                if alt_provider >= 0:
                    alt_idx = indices[alt_provider]
                    alt_entry = self.tage_tables[alt_provider][alt_idx]
                    if taken:
                        alt_entry.ctr = min(self.ctr_max, alt_entry.ctr + 1)
                    else:
                        alt_entry.ctr = max(self.ctr_min, alt_entry.ctr - 1)
                else:
                    self._bimodal_update(pc, taken)
        else:
            self._bimodal_update(pc, taken)
    
    def _reset_useful(self):
        """Periodic reset of useful bits."""
        for i in range(self.num_tage_tables):
            for entry in self.tage_tables[i]:
                entry.u >>= 1
    
    def _update_histories(self, pc: int, taken: bool):
        """Update all histories."""
        # Path history
        path_bit = (pc >> 2) & 1
        self.path_hist = ((self.path_hist << 1) | path_bit) & ((1 << self.path_bits) - 1)
        
        # Global history
        if self.pt_ghist <= 0:
            # Roll over
            for i in range(self.max_hist):
                if len(self.ghist) > self.max_hist:
                    self.ghist[len(self.ghist) - self.max_hist + i] = self.ghist[i]
            self.pt_ghist = len(self.ghist) - self.max_hist
        
        self.pt_ghist -= 1
        self.ghist[self.pt_ghist] = 1 if taken else 0
        
        # Update ghist_int (keep last 64 bits)
        self.ghist_int = ((self.ghist_int << 1) | (1 if taken else 0)) & 0xFFFFFFFFFFFFFFFF
        
        # Update folded histories
        new_bit = 1 if taken else 0
        for i in range(self.num_tage_tables):
            hist_len = self.tage_history_lengths[i]
            old_idx = self.pt_ghist + hist_len
            old_bit = int(self.ghist[old_idx]) if old_idx < len(self.ghist) else 0
            
            self.tage_idx_fold[i].update(new_bit, old_bit)
            self.tage_tag_fold[0][i].update(new_bit, old_bit)
            self.tage_tag_fold[1][i].update(new_bit, old_bit)
        
        # Local history
        local_idx = (pc >> 2) & 1023
        self.local_hist_table[local_idx] = (
            ((int(self.local_hist_table[local_idx]) << 1) | (1 if taken else 0)) & 0xFFFF
        )
    
    def get_hardware_cost(self) -> dict:
        """Estimate hardware cost."""
        bits = 0
        
        # TAGE tables
        for i in range(self.num_tage_tables):
            size = 1 << self.tage_table_bits[i]
            bits_per_entry = self.tage_ctr_bits + self.tage_tag_bits[i] + self.tage_u_bits
            bits += size * bits_per_entry
        
        # Bimodal
        bits += (1 << self.bimodal_bits)
        bits += (1 << (self.bimodal_bits - 2))
        
        # Loop
        loop_size = 1 << self.loop_table_bits
        loop_entry_bits = self.loop_tag_bits + 2 * self.loop_ctr_bits + self.loop_conf_bits + 8 + 1
        bits += loop_size * loop_entry_bits
        
        # SC
        sc_size = 1 << self.sc_table_bits
        bits += self.sc_num_tables * sc_size * self.sc_ctr_bits
        bits += 256 * 4  # SC confidence table
        
        # UseAlt
        bits += self.use_alt_size * self.use_alt_bits
        
        # Histories
        bits += self.max_hist + self.path_bits + 1024 * 16
        
        return {
            'total_bits': bits,
            'total_kb': bits / 8 / 1024,
        }
    
    def reset(self) -> None:
        """Reset all state."""
        super().reset()
        
        # TAGE
        for table in self.tage_tables:
            for entry in table:
                entry.ctr = 0
                entry.tag = 0
                entry.u = 0
        
        self.bimodal_pred.fill(False)
        self.bimodal_hyst.fill(True)
        self.use_alt_on_na.fill(0)
        
        for i in range(self.num_tage_tables):
            self.tage_idx_fold[i].comp = 0
            self.tage_tag_fold[0][i].comp = 0
            self.tage_tag_fold[1][i].comp = 0
        
        # Loop
        for entry in self.loop_table:
            entry.tag = 0
            entry.current_iter = 0
            entry.past_iter = 0
            entry.confidence = 0
            entry.age = 0
            entry.dir = True
        
        self.loop_use_counter = 0
        
        # SC
        for table in self.sc_tables:
            table.fill(0)
        self.sc_conf_table.fill(0)
        
        # Histories
        self.path_hist = 0
        self.ghist.fill(0)
        self.pt_ghist = 0
        self.ghist_int = 0
        self.local_hist_table.fill(0)
        
        self.tick = 1 << (self.u_reset_period - 1)
        
        # Stats
        self.tage_pred_count = 0
        self.loop_override_count = 0
        self.sc_override_count = 0
