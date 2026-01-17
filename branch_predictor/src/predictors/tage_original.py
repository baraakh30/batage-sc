"""
Original TAGE Branch Predictor Implementation

Faithful implementation of the TAGE (TAgged GEometric history length) predictor
as described by André Seznec in:
- "A case for (partially) tagged geometric history length branch prediction" (JILP 2006)
- "A New Case for the TAGE Branch Predictor" (MICRO 2011)

Based on the gem5 reference implementation:
https://github.com/gem5/gem5/blob/stable/src/cpu/pred/tage_base.cc

This is the de-facto standard TAGE used in CBP competitions and academic research.
Key features:
- Bimodal base predictor
- N partially-tagged tables with geometrically increasing history lengths
- Folded history for efficient indexing
- Useful bit management for entry replacement
- UseAltOnNewlyAllocated for weak entry handling

"""

import numpy as np
from typing import Optional, List
from .base import BasePredictor, PredictionResult, PredictorStats


# ============================================================================
# Configuration Presets - Based on gem5 8C-TAGE (~64KB)
# ============================================================================

# Original TAGE 8C configuration from CBP (approximately 64KB)
TAGE_8C_64KB = {
    'name': 'TAGE-8C-64KB',
    'n_history_tables': 12,
    'min_hist': 4,
    'max_hist': 640,
    # Tag widths per table (table 0 is bimodal - no tag)
    'tag_table_tag_widths': [0, 7, 7, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15],
    # Log2 of table sizes
    'log_tag_table_sizes': [14, 10, 10, 11, 11, 11, 11, 10, 10, 10, 10, 9, 9],
    'tag_table_counter_bits': 3,
    'tag_table_u_bits': 2,
    'log_u_reset_period': 19,
    'num_use_alt_on_na': 128,
    'use_alt_on_na_bits': 4,
    'path_hist_bits': 27,
    'log_ratio_bimodal_hyst_entries': 2,
    'max_num_alloc': 1,
    'inst_shift_amt': 2,  # For instruction alignment
}

# Smaller TAGE configuration (~32KB)
TAGE_32KB = {
    'name': 'TAGE-32KB',
    'n_history_tables': 10,
    'min_hist': 4,
    'max_hist': 300,
    'tag_table_tag_widths': [0, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12],
    'log_tag_table_sizes': [13, 9, 9, 10, 10, 10, 10, 9, 9, 9, 9],
    'tag_table_counter_bits': 3,
    'tag_table_u_bits': 2,
    'log_u_reset_period': 18,
    'num_use_alt_on_na': 64,
    'use_alt_on_na_bits': 4,
    'path_hist_bits': 24,
    'log_ratio_bimodal_hyst_entries': 2,
    'max_num_alloc': 1,
    'inst_shift_amt': 2,
}

# Medium TAGE configuration (~8KB) 
TAGE_8KB = {
    'name': 'TAGE-8KB',
    'n_history_tables': 7,
    'min_hist': 4,
    'max_hist': 130,
    'tag_table_tag_widths': [0, 7, 8, 8, 9, 9, 10, 11],
    'log_tag_table_sizes': [12, 8, 8, 8, 8, 8, 8, 8],
    'tag_table_counter_bits': 3,
    'tag_table_u_bits': 2,
    'log_u_reset_period': 17,
    'num_use_alt_on_na': 32,
    'use_alt_on_na_bits': 4,
    'path_hist_bits': 16,
    'log_ratio_bimodal_hyst_entries': 2,
    'max_num_alloc': 1,
    'inst_shift_amt': 2,
}

# Small TAGE configuration (~4KB)
TAGE_4KB = {
    'name': 'TAGE-4KB',
    'n_history_tables': 5,
    'min_hist': 4,
    'max_hist': 64,
    'tag_table_tag_widths': [0, 7, 8, 8, 9, 10],
    'log_tag_table_sizes': [11, 7, 7, 7, 7, 7],
    'tag_table_counter_bits': 3,
    'tag_table_u_bits': 2,
    'log_u_reset_period': 16,
    'num_use_alt_on_na': 16,
    'use_alt_on_na_bits': 4,
    'path_hist_bits': 12,
    'log_ratio_bimodal_hyst_entries': 2,
    'max_num_alloc': 1,
    'inst_shift_amt': 2,
}


class FoldedHistory:
    """
    Folded history for efficient indexing.
    Compresses long history into shorter value using XOR folding.
    Based on gem5 FoldedHistory implementation.
    """
    
    def __init__(self):
        self.comp = 0
        self.comp_length = 0
        self.orig_length = 0
        self.outpoint = 0
        
    def init(self, original_length: int, compressed_length: int):
        """Initialize folded history parameters."""
        self.orig_length = original_length
        self.comp_length = compressed_length
        self.outpoint = original_length % compressed_length
        self.comp = 0
    
    def update(self, new_bit: int, old_bit: int):
        """Update folded history with new bit shifting in and old bit shifting out."""
        if self.comp_length <= 0:
            return
        # Shift in new bit
        self.comp = (self.comp << 1) | new_bit
        # XOR out the bit that's leaving the window
        self.comp ^= old_bit << self.outpoint
        # XOR to fold
        self.comp ^= (self.comp >> self.comp_length)
        # Mask to compressed length
        self.comp &= (1 << self.comp_length) - 1


class TAGEEntry:
    """Entry in a TAGE tagged table."""
    __slots__ = ['ctr', 'tag', 'u']
    
    def __init__(self):
        self.ctr = 0   # Signed counter (prediction)
        self.tag = 0   # Partial tag
        self.u = 0     # Usefulness counter


class OriginalTAGE(BasePredictor):
    """
    Original TAGE Branch Predictor.
    
    Faithful implementation based on gem5 reference code.
    Uses geometric history lengths with partial tagging.
    
    Architecture:
    - Table 0: Bimodal base predictor (no tags)
    - Tables 1-N: Tagged tables with increasing history lengths
    
    The history lengths follow a geometric series:
        L[i] = minHist * (maxHist/minHist)^((i-1)/(N-1))
    """
    
    def __init__(self, config: dict = None):
        config = config or TAGE_8KB
        name = config.get('name', 'OriginalTAGE')
        super().__init__(name, config)
        
        # === Core Parameters ===
        self.n_history_tables = config.get('n_history_tables', 7)
        self.min_hist = config.get('min_hist', 4)
        self.max_hist = config.get('max_hist', 130)
        
        self.tag_table_tag_widths = config.get('tag_table_tag_widths', 
            [0, 7, 8, 8, 9, 9, 10, 11])
        self.log_tag_table_sizes = config.get('log_tag_table_sizes',
            [12, 8, 8, 8, 8, 8, 8, 8])
        
        self.tag_table_counter_bits = config.get('tag_table_counter_bits', 3)
        self.tag_table_u_bits = config.get('tag_table_u_bits', 2)
        self.log_u_reset_period = config.get('log_u_reset_period', 17)
        self.num_use_alt_on_na = config.get('num_use_alt_on_na', 32)
        self.use_alt_on_na_bits = config.get('use_alt_on_na_bits', 4)
        self.path_hist_bits = config.get('path_hist_bits', 16)
        self.log_ratio_bimodal_hyst_entries = config.get('log_ratio_bimodal_hyst_entries', 2)
        self.max_num_alloc = config.get('max_num_alloc', 1)
        self.inst_shift_amt = config.get('inst_shift_amt', 2)
        
        # Derived parameters
        self.counter_max = (1 << (self.tag_table_counter_bits - 1)) - 1
        self.counter_min = -(1 << (self.tag_table_counter_bits - 1))
        self.u_max = (1 << self.tag_table_u_bits) - 1
        self.use_alt_max = (1 << self.use_alt_on_na_bits) - 1
        
        # === Calculate History Lengths (Geometric Series) ===
        self.hist_lengths = self._calculate_history_lengths()
        
        # === Bimodal Base Predictor (Table 0) ===
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
        
        # === useAltPredForNewlyAllocated ===
        # Counter to decide whether to use alt prediction for newly allocated entries
        self.use_alt_pred_for_newly_allocated = np.zeros(self.num_use_alt_on_na, dtype=np.int8)
        
        # === Global History ===
        self.global_hist = np.zeros(self.max_hist + 100, dtype=np.uint8)
        self.pt_ghist = 0  # Pointer to current position in global history
        
        # === Path History ===
        self.path_hist = 0
        
        # === Folded Histories ===
        # For efficient index and tag computation
        self.compute_indices = []
        self.compute_tags = [[], []]  # Two sets of tag folded histories
        
        for i in range(self.n_history_tables + 1):
            ci = FoldedHistory()
            if i > 0:
                ci.init(self.hist_lengths[i], self.log_tag_table_sizes[i])
            self.compute_indices.append(ci)
            
            ct0 = FoldedHistory()
            ct1 = FoldedHistory()
            if i > 0:
                ct0.init(self.hist_lengths[i], self.tag_table_tag_widths[i])
                # Ensure we don't get negative width
                ct1_width = max(1, self.tag_table_tag_widths[i] - 1)
                ct1.init(self.hist_lengths[i], ct1_width)
            self.compute_tags[0].append(ct0)
            self.compute_tags[1].append(ct1)
        
        # === Tick counter for periodic reset ===
        self.t_counter = 1 << (self.log_u_reset_period - 1)
        
        # === State for update ===
        self._prediction_info = {}
        
    def _calculate_history_lengths(self) -> List[int]:
        """
        Calculate geometric history lengths.
        L[1] = minHist, L[N] = maxHist
        L[i] = minHist * (maxHist/minHist)^((i-1)/(N-1))
        """
        hist_lengths = [0]  # Index 0 for bimodal (no history)
        
        for i in range(1, self.n_history_tables + 1):
            if i == 1:
                length = self.min_hist
            elif i == self.n_history_tables:
                length = self.max_hist
            else:
                # Geometric interpolation
                ratio = (self.max_hist / self.min_hist) ** ((i - 1) / (self.n_history_tables - 1))
                length = int(self.min_hist * ratio + 0.5)
            hist_lengths.append(length)
        
        return hist_lengths
    
    def _bindex(self, pc: int) -> int:
        """Compute bimodal table index."""
        return (pc >> self.inst_shift_amt) & ((1 << self.log_tag_table_sizes[0]) - 1)
    
    def _F(self, A: int, size: int, bank: int) -> int:
        """
        Utility function to shuffle path history for index computation.
        From gem5 TAGEBase::F()
        """
        A = A & ((1 << size) - 1)
        table_bits = self.log_tag_table_sizes[bank]
        
        # Guard against negative shifts
        if table_bits <= 0:
            return 0
            
        A1 = A & ((1 << table_bits) - 1)
        A2 = A >> table_bits
        
        # Ensure bank doesn't exceed table_bits to avoid negative shifts
        effective_bank = bank % table_bits if table_bits > 0 else 0
        shift_right = max(0, table_bits - effective_bank)
        
        A2 = ((A2 << effective_bank) & ((1 << table_bits) - 1)) + (A2 >> shift_right if shift_right > 0 else 0)
        A = A1 ^ A2
        A = ((A << effective_bank) & ((1 << table_bits) - 1)) + (A >> shift_right if shift_right > 0 else 0)
        
        return A
    
    def _gindex(self, pc: int, bank: int) -> int:
        """
        Compute index for tagged table.
        Uses PC, folded global history, and path history.
        """
        hist_len = self.hist_lengths[bank]
        hlen = min(hist_len, self.path_hist_bits)
        
        shifted_pc = pc >> self.inst_shift_amt
        table_bits = self.log_tag_table_sizes[bank]
        
        index = (shifted_pc ^
                 (shifted_pc >> (abs(table_bits - bank) + 1)) ^
                 self.compute_indices[bank].comp ^
                 self._F(self.path_hist, hlen, bank))
        
        return index & ((1 << table_bits) - 1)
    
    def _gtag(self, pc: int, bank: int) -> int:
        """
        Compute tag for tagged table.
        Uses PC and folded global history.
        """
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
        
        # 2-bit counter logic with separate hysteresis
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
        """Get index for useAltPredForNewlyAllocated."""
        return (pc >> self.inst_shift_amt) & (self.num_use_alt_on_na - 1)
    
    def predict(self, pc: int, history: np.ndarray = None) -> PredictionResult:
        """
        TAGE prediction algorithm.
        
        1. Calculate indices and tags for all tables
        2. Find longest matching history (provider)
        3. Find alternate (second longest match or bimodal)
        4. Decide between provider and alternate
        """
        # === Calculate indices and tags ===
        table_indices = [0]  # Index 0 unused for bimodal
        table_tags = [0]
        
        for i in range(1, self.n_history_tables + 1):
            table_indices.append(self._gindex(pc, i))
            table_tags.append(self._gtag(pc, i))
        
        # === Find hits ===
        hit_bank = 0
        alt_bank = 0
        
        # Look for longest matching history
        for i in range(self.n_history_tables, 0, -1):
            idx = table_indices[i]
            if self.gtable[i-1][idx].tag == table_tags[i]:
                hit_bank = i
                break
        
        # Look for alternate
        for i in range(hit_bank - 1, 0, -1):
            idx = table_indices[i]
            if self.gtable[i-1][idx].tag == table_tags[i]:
                alt_bank = i
                break
        
        # === Compute predictions ===
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
            
            # Decide whether to use alt
            use_alt_idx = self._get_use_alt_idx(pc, hit_bank)
            use_alt = self.use_alt_pred_for_newly_allocated[use_alt_idx] < 0
            
            if use_alt or not pseudo_new_alloc:
                tage_pred = longest_match_pred
            else:
                tage_pred = alt_taken
            
        else:
            # No hit - use bimodal
            alt_taken = self._get_bimode_pred(pc)
            longest_match_pred = alt_taken
            tage_pred = alt_taken
            pseudo_new_alloc = False
        
        # Compute confidence
        if hit_bank > 0:
            hit_idx = table_indices[hit_bank]
            confidence = abs(self.gtable[hit_bank-1][hit_idx].ctr) / self.counter_max
        else:
            confidence = 0.5
        
        # Save prediction info for update
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
        """
        TAGE update algorithm.
        
        1. Update counters
        2. Update useful bits
        3. Allocate new entries on misprediction
        4. Periodic useful bit reset
        5. Update histories
        """
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
        
        # === Try to allocate on misprediction ===
        alloc = (tage_pred != taken) and (hit_bank < self.n_history_tables)
        
        if hit_bank > 0:
            # Manage useAltPredForNewlyAllocated
            if pseudo_new_alloc:
                if longest_match_pred == taken:
                    alloc = False
                
                if longest_match_pred != alt_taken:
                    use_alt_idx = self._get_use_alt_idx(pc, hit_bank)
                    if alt_taken == taken:
                        self.use_alt_pred_for_newly_allocated[use_alt_idx] = min(
                            self.use_alt_max,
                            self.use_alt_pred_for_newly_allocated[use_alt_idx] + 1
                        )
                    else:
                        self.use_alt_pred_for_newly_allocated[use_alt_idx] = max(
                            -self.use_alt_max - 1,
                            self.use_alt_pred_for_newly_allocated[use_alt_idx] - 1
                        )
        
        # === Handle allocation ===
        self._handle_alloc_and_u_reset(alloc, taken, hit_bank, table_indices, table_tags)
        
        # === Update TAGE tables ===
        self._handle_tage_update(pc, taken, hit_bank, hit_bank_idx, alt_bank, 
                                 alt_bank_idx, tage_pred, alt_taken)
        
        # === Update histories ===
        self._update_histories(pc, taken)
    
    def _handle_alloc_and_u_reset(self, alloc: bool, taken: bool, hit_bank: int,
                                   table_indices: List[int], table_tags: List[int]):
        """Handle entry allocation and useful bit reset."""
        if alloc:
            # Find entry with min useful value
            min_u = self.u_max + 1
            for i in range(self.n_history_tables, hit_bank, -1):
                idx = table_indices[i]
                if self.gtable[i-1][idx].u < min_u:
                    min_u = self.gtable[i-1][idx].u
            
            # Allocation with some randomness (simplified)
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
            
            # Allocate entries
            num_allocated = 0
            for i in range(X, self.n_history_tables + 1):
                idx = table_indices[i]
                if self.gtable[i-1][idx].u == 0:
                    self.gtable[i-1][idx].tag = table_tags[i]
                    self.gtable[i-1][idx].ctr = 0 if taken else -1
                    num_allocated += 1
                    if num_allocated >= self.max_num_alloc:
                        break
        
        # Periodic reset of useful bits
        self.t_counter += 1
        if (self.t_counter & ((1 << self.log_u_reset_period) - 1)) == 0:
            self._handle_u_reset()
    
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
            # Update hit entry counter
            entry = self.gtable[hit_bank-1][hit_bank_idx]
            entry.ctr = self._ctr_update(entry.ctr, taken)
            
            # Update alternate if hit entry has u=0
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
            # Only bimodal
            self._base_update(pc, taken)
    
    def _update_histories(self, pc: int, taken: bool):
        """Update global and path histories."""
        # Update path history
        path_bit = (pc >> self.inst_shift_amt) & 1
        self.path_hist = ((self.path_hist << 1) + path_bit) & ((1 << self.path_hist_bits) - 1)
        
        # Shift in new bit - handle rollover first
        if self.pt_ghist <= 0:
            # Rollover: copy recent history to end of buffer
            buffer_size = len(self.global_hist)
            for i in range(min(self.max_hist, buffer_size)):
                if i < buffer_size and buffer_size - self.max_hist + i >= 0:
                    self.global_hist[buffer_size - self.max_hist + i] = self.global_hist[i]
            self.pt_ghist = max(1, buffer_size - self.max_hist)
        
        self.pt_ghist -= 1
        self.global_hist[self.pt_ghist] = 1 if taken else 0
        
        # Update folded histories
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
        bits += bimodal_size  # Prediction bits
        bits += bimodal_size >> self.log_ratio_bimodal_hyst_entries  # Hysteresis bits
        
        # useAltPredForNewlyAllocated
        bits += self.num_use_alt_on_na * self.use_alt_on_na_bits
        
        # Histories
        bits += self.max_hist  # Global history
        bits += self.path_hist_bits  # Path history
        
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
        
        # Reset bimodal
        self.btable_prediction.fill(False)
        self.btable_hysteresis.fill(True)
        
        # Reset tagged tables
        for table in self.gtable:
            for entry in table:
                entry.ctr = 0
                entry.tag = 0
                entry.u = 0
        
        # Reset counters
        self.use_alt_pred_for_newly_allocated.fill(0)
        self.t_counter = 1 << (self.log_u_reset_period - 1)
        
        # Reset histories
        self.global_hist.fill(0)
        self.pt_ghist = 0
        self.path_hist = 0
        
        # Reset folded histories
        for i in range(self.n_history_tables + 1):
            self.compute_indices[i].comp = 0
            self.compute_tags[0][i].comp = 0
            self.compute_tags[1][i].comp = 0
