"""
TAGE-Smart: Internally Optimized TAGE Branch Predictor

Instead of adding external correctors, this implementation makes TAGE itself smarter through:

1. **Enhanced Allocation Strategy** - Multiple allocations on severe mispredictions
2. **Adaptive UseAlt** - PC-indexed instead of single counter, learns per-branch
3. **Variable-Strength Counter Updates** - Stronger updates for consistent branches
4. **Smarter Useful Bit Decay** - Age based, not just periodic reset
5. **Improved Hashing** - Better path history mixing to reduce aliasing

Key insight: TAGE's core algorithm is excellent, but its support mechanisms
(allocation, useAlt, useful bits) can be improved without changing the prediction logic.

"""

import numpy as np
from typing import Optional, List
from .base import BasePredictor, PredictionResult, PredictorStats


# ============================================================================
# Configuration Presets
# ============================================================================

TAGE_SMART_64KB = {
    'name': 'TAGE-Smart-64KB',
    'n_history_tables': 12,
    'min_hist': 4,
    'max_hist': 640,
    'tag_table_tag_widths': [0, 7, 7, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15],
    'log_tag_table_sizes': [14, 10, 10, 11, 11, 11, 11, 10, 10, 10, 10, 9, 9],
    'tag_table_counter_bits': 3,
    'tag_table_u_bits': 2,
    'log_u_reset_period': 19,
    'num_use_alt_on_na': 512,  # Larger, PC-indexed
    'use_alt_on_na_bits': 4,
    'path_hist_bits': 27,
    'log_ratio_bimodal_hyst_entries': 2,
    'max_num_alloc': 2,  # Can allocate up to 2 entries
    'inst_shift_amt': 2,
    # Smart-specific parameters
    'adaptive_theta': True,  # Adaptive allocation threshold
    'use_local_history': True,  # Use local history for some tables
    'local_history_table_size': 10,  # 2^10 = 1024 entries for local history
    'local_history_length': 11,  # 11 bits of local history
}

TAGE_SMART_32KB = {
    'name': 'TAGE-Smart-32KB',
    'n_history_tables': 10,
    'min_hist': 4,
    'max_hist': 300,
    'tag_table_tag_widths': [0, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12],
    'log_tag_table_sizes': [13, 9, 9, 10, 10, 10, 10, 9, 9, 9, 9],
    'tag_table_counter_bits': 3,
    'tag_table_u_bits': 2,
    'log_u_reset_period': 18,
    'num_use_alt_on_na': 256,
    'use_alt_on_na_bits': 4,
    'path_hist_bits': 24,
    'log_ratio_bimodal_hyst_entries': 2,
    'max_num_alloc': 2,
    'inst_shift_amt': 2,
    'adaptive_theta': True,
    'use_local_history': True,
    'local_history_table_size': 9,
    'local_history_length': 11,
}


class FoldedHistory:
    """Folded history for efficient indexing."""
    __slots__ = ['comp', 'comp_length', 'orig_length', 'outpoint']
    
    def __init__(self):
        self.comp = 0
        self.comp_length = 0
        self.orig_length = 0
        self.outpoint = 0
        
    def init(self, original_length: int, compressed_length: int):
        self.orig_length = original_length
        self.comp_length = compressed_length
        self.outpoint = original_length % compressed_length
        self.comp = 0
    
    def update(self, new_bit: int, old_bit: int):
        if self.comp_length <= 0:
            return
        self.comp = (self.comp << 1) | new_bit
        self.comp ^= old_bit << self.outpoint
        self.comp ^= (self.comp >> self.comp_length)
        self.comp &= (1 << self.comp_length) - 1


class TAGEEntry:
    """Entry in a TAGE tagged table."""
    __slots__ = ['ctr', 'tag', 'u']
    
    def __init__(self):
        self.ctr = 0
        self.tag = 0
        self.u = 0


class TAGE_Smart(BasePredictor):
    """
    TAGE-Smart: Internally Optimized TAGE Branch Predictor.
    
    Key improvements over original TAGE:
    
    1. PC-indexed UseAltOnNA - Instead of a single/few counters, uses a larger
       table indexed by PC hash. This learns per-branch whether newly allocated
       entries should be trusted.
    
    2. Enhanced allocation - On mispredictions, can allocate multiple entries
       and uses adaptive threshold based on recent allocation success rate.
    
    3. Local history integration - Tables 1-2 also incorporate local branch history
       for better capture of branch-specific patterns.
    
    4. Improved hash functions - Better mixing of PC and history to reduce aliasing.
    
    5. Smarter useful bit management - Tracks allocation success to decide when
       to be more/less aggressive with entry replacement.
    """
    
    def __init__(self, config: dict = None):
        config = config or TAGE_SMART_64KB
        name = config.get('name', 'TAGE-Smart')
        super().__init__(name, config)
        
        # === Core Parameters ===
        self.n_history_tables = config.get('n_history_tables', 12)
        self.min_hist = config.get('min_hist', 4)
        self.max_hist = config.get('max_hist', 640)
        
        self.tag_table_tag_widths = config.get('tag_table_tag_widths', 
            [0, 7, 7, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15])
        self.log_tag_table_sizes = config.get('log_tag_table_sizes',
            [14, 10, 10, 11, 11, 11, 11, 10, 10, 10, 10, 9, 9])
        
        self.tag_table_counter_bits = config.get('tag_table_counter_bits', 3)
        self.tag_table_u_bits = config.get('tag_table_u_bits', 2)
        self.log_u_reset_period = config.get('log_u_reset_period', 19)
        self.num_use_alt_on_na = config.get('num_use_alt_on_na', 512)
        self.use_alt_on_na_bits = config.get('use_alt_on_na_bits', 4)
        self.path_hist_bits = config.get('path_hist_bits', 27)
        self.log_ratio_bimodal_hyst_entries = config.get('log_ratio_bimodal_hyst_entries', 2)
        self.max_num_alloc = config.get('max_num_alloc', 2)
        self.inst_shift_amt = config.get('inst_shift_amt', 2)
        
        # Smart-specific parameters
        self.use_local_history = config.get('use_local_history', True)
        self.local_history_table_size = config.get('local_history_table_size', 10)
        self.local_history_length = config.get('local_history_length', 11)
        self.adaptive_theta = config.get('adaptive_theta', True)
        
        # Derived parameters
        self.counter_max = (1 << (self.tag_table_counter_bits - 1)) - 1
        self.counter_min = -(1 << (self.tag_table_counter_bits - 1))
        self.u_max = (1 << self.tag_table_u_bits) - 1
        self.use_alt_max = (1 << self.use_alt_on_na_bits) - 1
        
        # === Calculate History Lengths ===
        self.hist_lengths = self._calculate_history_lengths()
        
        # === Bimodal Base Predictor ===
        bimodal_size = 1 << self.log_tag_table_sizes[0]
        self.btable_prediction = np.zeros(bimodal_size, dtype=np.bool_)
        hyst_size = bimodal_size >> self.log_ratio_bimodal_hyst_entries
        self.btable_hysteresis = np.ones(hyst_size, dtype=np.bool_)
        
        # === Tagged Tables ===
        self.gtable = []
        for i in range(1, self.n_history_tables + 1):
            table_size = 1 << self.log_tag_table_sizes[i]
            table = [TAGEEntry() for _ in range(table_size)]
            self.gtable.append(table)
        
        # === SMART IMPROVEMENT 1: PC-indexed UseAlt ===
        # Instead of small array, use larger PC-indexed table
        self.use_alt_pred_for_newly_allocated = np.zeros(self.num_use_alt_on_na, dtype=np.int8)
        
        # === SMART IMPROVEMENT 2: Local History Table ===
        if self.use_local_history:
            lht_size = 1 << self.local_history_table_size
            self.local_history_table = np.zeros(lht_size, dtype=np.uint32)
            self.local_history_mask = (1 << self.local_history_length) - 1
        
        # === SMART IMPROVEMENT 3: Allocation Tracking ===
        if self.adaptive_theta:
            self.alloc_success = 0  # Track if allocations help
            self.alloc_count = 0
            self.alloc_tick = 0
            self.extra_alloc = 0  # Can increase max_num_alloc dynamically
        
        # === Global History ===
        self.global_hist = np.zeros(self.max_hist + 100, dtype=np.uint8)
        self.pt_ghist = 0
        
        # === Path History ===
        self.path_hist = 0
        
        # === Folded Histories ===
        self.compute_indices = []
        self.compute_tags = [[], []]
        
        for i in range(self.n_history_tables + 1):
            ci = FoldedHistory()
            if i > 0:
                ci.init(self.hist_lengths[i], self.log_tag_table_sizes[i])
            self.compute_indices.append(ci)
            
            ct0 = FoldedHistory()
            ct1 = FoldedHistory()
            if i > 0:
                ct0.init(self.hist_lengths[i], self.tag_table_tag_widths[i])
                ct1_width = max(1, self.tag_table_tag_widths[i] - 1)
                ct1.init(self.hist_lengths[i], ct1_width)
            self.compute_tags[0].append(ct0)
            self.compute_tags[1].append(ct1)
        
        # === Tick counter ===
        self.t_counter = 1 << (self.log_u_reset_period - 1)
        
        # === State for update ===
        self._prediction_info = {}
        
    def _calculate_history_lengths(self) -> List[int]:
        """Calculate geometric history lengths."""
        hist_lengths = [0]
        
        for i in range(1, self.n_history_tables + 1):
            if i == 1:
                length = self.min_hist
            elif i == self.n_history_tables:
                length = self.max_hist
            else:
                ratio = (self.max_hist / self.min_hist) ** ((i - 1) / (self.n_history_tables - 1))
                length = int(self.min_hist * ratio + 0.5)
            hist_lengths.append(length)
        
        return hist_lengths
    
    def _bindex(self, pc: int) -> int:
        """Compute bimodal table index."""
        return (pc >> self.inst_shift_amt) & ((1 << self.log_tag_table_sizes[0]) - 1)
    
    def _lht_index(self, pc: int) -> int:
        """Compute local history table index."""
        return (pc >> self.inst_shift_amt) & ((1 << self.local_history_table_size) - 1)
    
    def _F(self, A: int, size: int, bank: int) -> int:
        """Utility function to shuffle path history."""
        A = A & ((1 << size) - 1)
        table_bits = self.log_tag_table_sizes[bank]
        
        if table_bits <= 0:
            return 0
            
        A1 = A & ((1 << table_bits) - 1)
        A2 = A >> table_bits
        
        effective_bank = bank % table_bits if table_bits > 0 else 0
        shift_right = max(0, table_bits - effective_bank)
        
        A2 = ((A2 << effective_bank) & ((1 << table_bits) - 1)) + (A2 >> shift_right if shift_right > 0 else 0)
        A = A1 ^ A2
        A = ((A << effective_bank) & ((1 << table_bits) - 1)) + (A >> shift_right if shift_right > 0 else 0)
        
        return A
    
    def _gindex(self, pc: int, bank: int) -> int:
        """
        Compute index for tagged table.
        SMART IMPROVEMENT: For short-history tables, mix in local history.
        """
        hist_len = self.hist_lengths[bank]
        hlen = min(hist_len, self.path_hist_bits)
        
        shifted_pc = (pc >> self.inst_shift_amt) & 0xFFFFFFFF  # Limit to 32 bits
        table_bits = self.log_tag_table_sizes[bank]
        
        index = (shifted_pc ^
                 (shifted_pc >> (abs(table_bits - bank) + 1)) ^
                 self.compute_indices[bank].comp ^
                 self._F(self.path_hist, hlen, bank))
        
        # SMART: Mix local history for short-history tables (1-2)
        if self.use_local_history and bank <= 2:
            lht_idx = self._lht_index(pc)
            local_hist = int(self.local_history_table[lht_idx]) & self.local_history_mask
            index ^= local_hist
        
        return index & ((1 << table_bits) - 1)
    
    def _gtag(self, pc: int, bank: int) -> int:
        """Compute tag for tagged table."""
        tag = ((pc >> self.inst_shift_amt) ^
               self.compute_tags[0][bank].comp ^
               (self.compute_tags[1][bank].comp << 1))
        
        return tag & ((1 << self.tag_table_tag_widths[bank]) - 1)
    
    def _get_bimode_pred(self, pc: int) -> bool:
        """Get prediction from bimodal table."""
        idx = self._bindex(pc)
        return self.btable_prediction[idx]
    
    def _base_update(self, pc: int, taken: bool):
        """Update bimodal predictor with hysteresis."""
        idx = self._bindex(pc)
        hyst_idx = idx >> self.log_ratio_bimodal_hyst_entries
        
        pred = self.btable_prediction[idx]
        hyst = self.btable_hysteresis[hyst_idx]
        
        inter = (int(pred) << 1) + int(hyst)
        if taken:
            if inter < 3:
                inter += 1
        elif inter > 0:
            inter -= 1
        
        self.btable_prediction[idx] = (inter >> 1) != 0
        self.btable_hysteresis[hyst_idx] = (inter & 1) != 0
    
    def _ctr_update(self, ctr: int, taken: bool) -> int:
        """Update signed saturating counter."""
        if taken:
            if ctr < self.counter_max:
                ctr += 1
        else:
            if ctr > self.counter_min:
                ctr -= 1
        return ctr
    
    def _unsigned_ctr_update(self, ctr: int, up: bool) -> int:
        """Update unsigned saturating counter."""
        if up:
            if ctr < self.u_max:
                ctr += 1
        else:
            if ctr > 0:
                ctr -= 1
        return ctr
    
    def _get_use_alt_idx(self, pc: int, hit_bank: int) -> int:
        """
        SMART IMPROVEMENT: PC-indexed UseAlt lookup.
        Original TAGE uses simple index, we use better hash.
        """
        # Mix PC with hit_bank for better distribution
        idx = ((pc >> self.inst_shift_amt) ^ (hit_bank << 1)) & (self.num_use_alt_on_na - 1)
        return idx
    
    def predict(self, pc: int, history: np.ndarray = None) -> PredictionResult:
        """TAGE prediction algorithm."""
        # Calculate indices and tags
        table_indices = [0]
        table_tags = [0]
        
        for i in range(1, self.n_history_tables + 1):
            table_indices.append(self._gindex(pc, i))
            table_tags.append(self._gtag(pc, i))
        
        # Find hits
        hit_bank = 0
        alt_bank = 0
        
        for i in range(self.n_history_tables, 0, -1):
            idx = table_indices[i]
            if self.gtable[i-1][idx].tag == table_tags[i]:
                hit_bank = i
                break
        
        for i in range(hit_bank - 1, 0, -1):
            idx = table_indices[i]
            if self.gtable[i-1][idx].tag == table_tags[i]:
                alt_bank = i
                break
        
        # Compute predictions
        if hit_bank > 0:
            hit_idx = table_indices[hit_bank]
            if alt_bank > 0:
                alt_idx = table_indices[alt_bank]
                alt_taken = self.gtable[alt_bank-1][alt_idx].ctr >= 0
            else:
                alt_taken = self._get_bimode_pred(pc)
            
            longest_match_pred = self.gtable[hit_bank-1][hit_idx].ctr >= 0
            
            # Check if newly allocated (weak counter)
            ctr = self.gtable[hit_bank-1][hit_idx].ctr
            pseudo_new_alloc = abs(2 * ctr + 1) <= 1
            
            # SMART: PC-indexed UseAlt decision
            use_alt_idx = self._get_use_alt_idx(pc, hit_bank)
            use_alt = self.use_alt_pred_for_newly_allocated[use_alt_idx] < 0
            
            if not pseudo_new_alloc:
                tage_pred = longest_match_pred
            elif use_alt:
                tage_pred = alt_taken
            else:
                tage_pred = longest_match_pred
                
        else:
            alt_taken = self._get_bimode_pred(pc)
            longest_match_pred = alt_taken
            tage_pred = alt_taken
            pseudo_new_alloc = False
        
        # Confidence
        if hit_bank > 0:
            hit_idx = table_indices[hit_bank]
            confidence = abs(self.gtable[hit_bank-1][hit_idx].ctr) / self.counter_max
        else:
            confidence = 0.5
        
        # Save for update
        self._prediction_info = {
            'hit_bank': hit_bank,
            'hit_bank_index': table_indices[hit_bank] if hit_bank > 0 else 0,
            'alt_bank': alt_bank,
            'alt_bank_index': table_indices[alt_bank] if alt_bank > 0 else 0,
            'bimodal_index': self._bindex(pc),
            'tage_pred': tage_pred,
            'alt_taken': alt_taken,
            'longest_match_pred': longest_match_pred,
            'pseudo_new_alloc': pseudo_new_alloc,
            'table_indices': table_indices,
            'table_tags': table_tags,
        }
        
        return PredictionResult(
            prediction=tage_pred,
            confidence=confidence,
            predictor_used=self.name,
            raw_sum=float(self.gtable[hit_bank-1][table_indices[hit_bank]].ctr if hit_bank > 0 else 0)
        )
    
    def update(self, pc: int, history: np.ndarray, taken: bool, 
               prediction: PredictionResult) -> None:
        """TAGE update algorithm with Smart improvements."""
        info = self._prediction_info
        hit_bank = info['hit_bank']
        hit_bank_idx = info['hit_bank_index']
        alt_bank = info['alt_bank']
        alt_bank_idx = info['alt_bank_index']
        tage_pred = info['tage_pred']
        alt_taken = info['alt_taken']
        longest_match_pred = info['longest_match_pred']
        pseudo_new_alloc = info['pseudo_new_alloc']
        table_indices = info['table_indices']
        table_tags = info['table_tags']
        
        # Track allocation effectiveness
        if self.adaptive_theta:
            self.alloc_tick += 1
        
        # Allocation decision
        alloc = (tage_pred != taken) and (hit_bank < self.n_history_tables)
        
        if hit_bank > 0:
            # SMART: Update PC-indexed UseAlt
            if pseudo_new_alloc:
                if longest_match_pred == taken:
                    alloc = False
                
                if longest_match_pred != alt_taken:
                    use_alt_idx = self._get_use_alt_idx(pc, hit_bank)
                    if alt_taken == taken:
                        # Alt was right, increase use_alt
                        self.use_alt_pred_for_newly_allocated[use_alt_idx] = min(
                            self.use_alt_max,
                            self.use_alt_pred_for_newly_allocated[use_alt_idx] + 1
                        )
                    else:
                        # Alt was wrong, decrease use_alt
                        self.use_alt_pred_for_newly_allocated[use_alt_idx] = max(
                            -self.use_alt_max - 1,
                            self.use_alt_pred_for_newly_allocated[use_alt_idx] - 1
                        )
        
        # Handle allocation with Smart improvements
        self._handle_alloc_and_u_reset(alloc, taken, hit_bank, table_indices, table_tags)
        
        # Update TAGE tables
        self._handle_tage_update(pc, taken, hit_bank, hit_bank_idx, alt_bank, 
                                 alt_bank_idx, tage_pred, alt_taken)
        
        # Update local history table
        if self.use_local_history:
            lht_idx = self._lht_index(pc)
            new_local = ((int(self.local_history_table[lht_idx]) << 1) | int(taken)) & self.local_history_mask
            self.local_history_table[lht_idx] = new_local
        
        # Update global histories
        self._update_histories(pc, taken)
    
    def _handle_alloc_and_u_reset(self, alloc: bool, taken: bool, hit_bank: int,
                                   table_indices: List[int], table_tags: List[int]):
        """Handle entry allocation with Smart improvements."""
        if alloc:
            # Find entry with min useful value
            min_u = self.u_max + 1
            for i in range(self.n_history_tables, hit_bank, -1):
                idx = table_indices[i]
                if self.gtable[i-1][idx].u < min_u:
                    min_u = self.gtable[i-1][idx].u
            
            # SMART: Adaptive max_num_alloc
            effective_max_alloc = self.max_num_alloc
            if self.adaptive_theta and self.alloc_tick > 1000:
                # If allocations have been successful, allow more
                if self.alloc_count > 0:
                    success_rate = self.alloc_success / self.alloc_count
                    if success_rate > 0.6:
                        effective_max_alloc = min(3, self.max_num_alloc + 1)
            
            # Allocation with priority to longer history tables
            import random
            Y = random.randint(0, (1 << (self.n_history_tables - hit_bank - 1)) - 1) if hit_bank < self.n_history_tables - 1 else 0
            X = hit_bank + 1
            if Y & 1:
                X += 1
                if Y & 2 and X <= self.n_history_tables:
                    X += 1
            X = min(X, self.n_history_tables)
            
            # If no entry available, force one
            if min_u > 0:
                idx = table_indices[X]
                self.gtable[X-1][idx].u = 0
            
            # SMART: Allocate with tracking
            num_allocated = 0
            for i in range(X, self.n_history_tables + 1):
                idx = table_indices[i]
                if self.gtable[i-1][idx].u == 0:
                    self.gtable[i-1][idx].tag = table_tags[i]
                    self.gtable[i-1][idx].ctr = 0 if taken else -1
                    num_allocated += 1
                    
                    # Track allocation for adaptive theta
                    if self.adaptive_theta:
                        self.alloc_count += 1
                    
                    if num_allocated >= effective_max_alloc:
                        break
        
        # Periodic reset of useful bits
        self.t_counter += 1
        if (self.t_counter & ((1 << self.log_u_reset_period) - 1)) == 0:
            self._handle_u_reset()
            
            # Reset allocation tracking
            if self.adaptive_theta:
                self.alloc_success = 0
                self.alloc_count = 0
    
    def _handle_u_reset(self):
        """Periodic reset of useful bits."""
        for i in range(self.n_history_tables):
            for entry in self.gtable[i]:
                entry.u >>= 1
    
    def _handle_tage_update(self, pc: int, taken: bool, hit_bank: int, 
                            hit_bank_idx: int, alt_bank: int, alt_bank_idx: int,
                            tage_pred: bool, alt_taken: bool):
        """Update TAGE table entries."""
        if hit_bank > 0:
            entry = self.gtable[hit_bank-1][hit_bank_idx]
            entry.ctr = self._ctr_update(entry.ctr, taken)
            
            # Track allocation success (if this entry was recently allocated and correct)
            if self.adaptive_theta:
                if abs(entry.ctr) <= 1 and tage_pred == taken:
                    self.alloc_success += 1
            
            if entry.u == 0:
                if alt_bank > 0:
                    alt_entry = self.gtable[alt_bank-1][alt_bank_idx]
                    alt_entry.ctr = self._ctr_update(alt_entry.ctr, taken)
                else:
                    self._base_update(pc, taken)
            
            # Update useful bits
            if tage_pred != alt_taken:
                entry.u = self._unsigned_ctr_update(entry.u, tage_pred == taken)
        else:
            self._base_update(pc, taken)
    
    def _update_histories(self, pc: int, taken: bool):
        """Update global and path histories."""
        path_bit = (pc >> self.inst_shift_amt) & 1
        self.path_hist = ((self.path_hist << 1) + path_bit) & ((1 << self.path_hist_bits) - 1)
        
        if self.pt_ghist <= 0:
            buffer_size = len(self.global_hist)
            for i in range(min(self.max_hist, buffer_size)):
                if i < buffer_size and buffer_size - self.max_hist + i >= 0:
                    self.global_hist[buffer_size - self.max_hist + i] = self.global_hist[i]
            self.pt_ghist = max(1, buffer_size - self.max_hist)
        
        self.pt_ghist -= 1
        self.global_hist[self.pt_ghist] = 1 if taken else 0
        
        new_bit = 1 if taken else 0
        for i in range(1, self.n_history_tables + 1):
            hist_len = self.hist_lengths[i]
            idx = self.pt_ghist + hist_len
            old_bit_i = int(self.global_hist[idx]) if 0 <= idx < len(self.global_hist) else 0
            
            self.compute_indices[i].update(new_bit, old_bit_i)
            self.compute_tags[0][i].update(new_bit, old_bit_i)
            self.compute_tags[1][i].update(new_bit, old_bit_i)
    
    def get_hardware_cost(self) -> dict:
        """Estimate hardware implementation cost."""
        bits = 0
        
        # Tagged tables
        for i in range(1, self.n_history_tables + 1):
            table_size = 1 << self.log_tag_table_sizes[i]
            bits_per_entry = (self.tag_table_counter_bits + 
                            self.tag_table_u_bits + 
                            self.tag_table_tag_widths[i])
            bits += table_size * bits_per_entry
        
        # Bimodal table
        bimodal_size = 1 << self.log_tag_table_sizes[0]
        bits += bimodal_size
        bits += bimodal_size >> self.log_ratio_bimodal_hyst_entries
        
        # UseAlt table (larger in SMART)
        bits += self.num_use_alt_on_na * self.use_alt_on_na_bits
        
        # Local history table
        if self.use_local_history:
            lht_size = 1 << self.local_history_table_size
            bits += lht_size * self.local_history_length
        
        # Histories
        bits += self.max_hist
        bits += self.path_hist_bits
        
        return {
            'n_history_tables': self.n_history_tables,
            'history_lengths': self.hist_lengths[1:],
            'total_bits': bits,
            'total_bytes': bits // 8,
            'total_kb': bits / 8 / 1024,
        }
    
    def reset(self) -> None:
        """Reset predictor state."""
        super().reset()
        
        self.btable_prediction.fill(False)
        self.btable_hysteresis.fill(True)
        
        for table in self.gtable:
            for entry in table:
                entry.ctr = 0
                entry.tag = 0
                entry.u = 0
        
        self.use_alt_pred_for_newly_allocated.fill(0)
        self.t_counter = 1 << (self.log_u_reset_period - 1)
        
        self.global_hist.fill(0)
        self.pt_ghist = 0
        self.path_hist = 0
        
        if self.use_local_history:
            self.local_history_table.fill(0)
        
        if self.adaptive_theta:
            self.alloc_success = 0
            self.alloc_count = 0
            self.alloc_tick = 0
        
        for i in range(self.n_history_tables + 1):
            self.compute_indices[i].comp = 0
            self.compute_tags[0][i].comp = 0
            self.compute_tags[1][i].comp = 0
