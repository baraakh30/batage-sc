"""
BATAGE-SC: Branch-Aware TAGE with Statistical Corrector

A surgical improvement to TAGE that makes the predictor SMARTER rather than
adding complex components on top.

Key Innovations:
================

1. Branch Type Classification (Zero-latency)
   - 2-bit counter per PC tracks branch behavior during normal operation
   - Classifies branches as: LOOP-LIKE, DATA-DEPENDENT, or UNKNOWN
   - Classification guides table access priority (no extra latency)

2. Selective Table Access
   - Loop-like branches: prioritize short history tables (4-64 bits)
   - Data-dependent: prioritize long history tables (128+ bits)
   - Reduces wasted lookups, improves effective capacity

3. In-Place Statistical Corrector (IPSC)
   - 4 small bimodal tables with different hash functions
   - Computed IN PARALLEL with TAGE (zero latency overhead)
   - Only overrides when TAGE is weak AND SC has strong agreement
   - Much smaller than traditional SC (~2KB vs 8-16KB)

4. Enhanced Allocation with H2P Awareness
   - Tracks misprediction frequency per branch
   - Aggressive allocation for hard-to-predict branches
   - Conservative for easy branches (saves space)

5. Improved Tag Design
   - Uses PC + path history + global history for better tag coverage
   - Reduces aliasing in tagged tables

"""

import numpy as np
from typing import Optional, List, Tuple
from .base import BasePredictor, PredictionResult


# ============================================================================
# Configuration Presets
# ============================================================================

BATAGE_SC_64KB = {
    'name': 'BATAGE-SC-64KB',
    
    # === TAGE Core Configuration ===
    'num_tables': 12,
    'history_lengths': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096],
    'table_sizes': [4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128],
    'tag_widths': [8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14],
    'counter_bits': 3,
    'useful_bits': 2,
    
    # === Bimodal Base ===
    'bimodal_size': 16384,
    'bimodal_hyst_ratio': 2,
    
    # === Branch Type Classifier ===
    'classifier_size': 4096,  # 4K entries, 2 bits each = 1KB
    'loop_threshold': 2,       # Counter >= 2 means loop-like
    'data_threshold': -2,      # Counter <= -2 means data-dependent
    
    # === In-Place Statistical Corrector ===
    'sc_num_tables': 4,
    'sc_table_size': 1024,     # 4 tables x 1K entries x 3 bits = 1.5KB
    'sc_counter_bits': 3,
    'sc_threshold': 6,         # SC sum threshold for override (balanced)
    'sc_confidence_threshold': 0.35,  # Only consider SC when TAGE confidence < this
    'sc_loop_only': True,      # Only use SC for loop-like branches (safer)
    
    # === H2P Tracking ===
    'h2p_table_size': 2048,
    'h2p_threshold': 6,        # Mispredicts to be considered H2P
    
    # === Update Policy ===
    'use_alt_size': 128,
    'use_alt_bits': 4,
    'u_reset_period': 19,
    'max_alloc': 2,
    'max_alloc_h2p': 3,        # More aggressive allocation for H2P
    
    # === History Configuration ===
    'path_hist_bits': 32,
    'local_hist_size': 1024,
    'local_hist_bits': 16,
}

BATAGE_SC_32KB = {
    'name': 'BATAGE-SC-32KB',
    'num_tables': 10,
    'history_lengths': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    'table_sizes': [2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128],
    'tag_widths': [7, 8, 8, 9, 9, 10, 10, 11, 11, 12],
    'counter_bits': 3,
    'useful_bits': 2,
    'bimodal_size': 8192,
    'bimodal_hyst_ratio': 2,
    'classifier_size': 2048,
    'loop_threshold': 2,
    'data_threshold': -2,
    'sc_num_tables': 4,
    'sc_table_size': 512,
    'sc_counter_bits': 3,
    'sc_threshold': 7,
    'sc_confidence_threshold': 0.25,
    'h2p_table_size': 1024,
    'h2p_threshold': 5,
    'use_alt_size': 64,
    'use_alt_bits': 4,
    'u_reset_period': 18,
    'max_alloc': 2,
    'max_alloc_h2p': 3,
    'path_hist_bits': 27,
    'local_hist_size': 512,
    'local_hist_bits': 14,
}

BATAGE_SC_8KB = {
    'name': 'BATAGE-SC-8KB',
    'num_tables': 7,
    'history_lengths': [4, 8, 16, 32, 64, 128, 256],
    'table_sizes': [1024, 1024, 512, 512, 256, 256, 128],
    'tag_widths': [7, 7, 8, 8, 9, 9, 10],
    'counter_bits': 3,
    'useful_bits': 1,
    'bimodal_size': 4096,
    'bimodal_hyst_ratio': 2,
    'classifier_size': 1024,
    'loop_threshold': 2,
    'data_threshold': -2,
    'sc_num_tables': 3,
    'sc_table_size': 256,
    'sc_counter_bits': 3,
    'sc_threshold': 6,
    'sc_confidence_threshold': 0.25,
    'h2p_table_size': 512,
    'h2p_threshold': 4,
    'use_alt_size': 32,
    'use_alt_bits': 4,
    'u_reset_period': 17,
    'max_alloc': 1,
    'max_alloc_h2p': 2,
    'path_hist_bits': 16,
    'local_hist_size': 256,
    'local_hist_bits': 10,
}


# ============================================================================
# Branch Type Enumeration
# ============================================================================

class BranchType:
    UNKNOWN = 0
    LOOP_LIKE = 1
    DATA_DEPENDENT = 2


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
        self.ctr = 0   # Signed counter
        self.tag = 0   # Partial tag
        self.u = 0     # Usefulness


# ============================================================================
# BATAGE-SC Predictor
# ============================================================================

class BATAGE_SC(BasePredictor):
    """
    Branch-Aware TAGE with Statistical Corrector.
    
    Makes TAGE smarter by:
    1. Classifying branch types and prioritizing relevant tables
    2. Using a lightweight parallel statistical corrector
    3. H2P-aware allocation policy
    """
    
    def __init__(self, config: dict = None):
        config = config or BATAGE_SC_64KB
        name = config.get('name', 'BATAGE-SC')
        super().__init__(name, config)
        
        # === Core TAGE Parameters ===
        self.num_tables = config.get('num_tables', 12)
        self.history_lengths = config.get('history_lengths')
        self.table_sizes = config.get('table_sizes')
        self.tag_widths = config.get('tag_widths')
        self.counter_bits = config.get('counter_bits', 3)
        self.useful_bits = config.get('useful_bits', 2)
        
        # Counter bounds
        self.ctr_max = (1 << (self.counter_bits - 1)) - 1
        self.ctr_min = -(1 << (self.counter_bits - 1))
        self.u_max = (1 << self.useful_bits) - 1
        
        # === Bimodal Base Predictor ===
        self.bimodal_size = config.get('bimodal_size', 16384)
        self.bimodal_hyst_ratio = config.get('bimodal_hyst_ratio', 2)
        self.bimodal_pred = np.zeros(self.bimodal_size, dtype=np.bool_)
        hyst_size = self.bimodal_size >> self.bimodal_hyst_ratio
        self.bimodal_hyst = np.ones(hyst_size, dtype=np.bool_)
        
        # === Tagged Tables ===
        self.tables: List[List[TAGEEntry]] = []
        for i in range(self.num_tables):
            table_size = self.table_sizes[i]
            table = [TAGEEntry() for _ in range(table_size)]
            self.tables.append(table)
        
        # === Branch Type Classifier ===
        # 2-bit saturating counter per PC hash
        # Positive = loop-like, Negative = data-dependent
        self.classifier_size = config.get('classifier_size', 4096)
        self.classifier = np.zeros(self.classifier_size, dtype=np.int8)
        self.loop_threshold = config.get('loop_threshold', 2)
        self.data_threshold = config.get('data_threshold', -2)
        
        # === In-Place Statistical Corrector ===
        self.sc_num_tables = config.get('sc_num_tables', 4)
        self.sc_table_size = config.get('sc_table_size', 1024)
        self.sc_counter_bits = config.get('sc_counter_bits', 3)
        self.sc_max = (1 << (self.sc_counter_bits - 1)) - 1
        self.sc_min = -(1 << (self.sc_counter_bits - 1))
        self.sc_threshold = config.get('sc_threshold', 5)
        self.sc_confidence_threshold = config.get('sc_confidence_threshold', 0.4)
        
        # SC tables
        self.sc_tables = [np.zeros(self.sc_table_size, dtype=np.int8) 
                         for _ in range(self.sc_num_tables)]
        
        # SC loop-only mode - only apply SC override for loop-like branches
        self.sc_loop_only = config.get('sc_loop_only', True)
        
        # === H2P Tracking ===
        self.h2p_table_size = config.get('h2p_table_size', 2048)
        self.h2p_threshold = config.get('h2p_threshold', 6)
        self.h2p_counts = np.zeros(self.h2p_table_size, dtype=np.uint8)
        
        # === Local History Table ===
        self.local_hist_size = config.get('local_hist_size', 1024)
        self.local_hist_bits = config.get('local_hist_bits', 16)
        self.local_hist = np.zeros(self.local_hist_size, dtype=np.uint32)
        
        # === Path History ===
        self.path_hist_bits = config.get('path_hist_bits', 32)
        self.path_hist = 0
        
        # === Global History ===
        self.max_hist = max(self.history_lengths) if self.history_lengths else 4096
        self.global_hist = np.zeros(self.max_hist + 256, dtype=np.uint8)
        self.pt_ghist = 0
        self.ghist_int = 0  # Integer version for fast ops
        
        # === Folded Histories ===
        self.folded_indices: List[FoldedHistory] = []
        self.folded_tags: List[List[FoldedHistory]] = [[], []]
        
        for i in range(self.num_tables):
            hist_len = self.history_lengths[i]
            table_bits = int(np.log2(self.table_sizes[i]))
            
            ci = FoldedHistory()
            ci.init(hist_len, table_bits)
            self.folded_indices.append(ci)
            
            tag_width = self.tag_widths[i]
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
        self.max_alloc = config.get('max_alloc', 2)
        self.max_alloc_h2p = config.get('max_alloc_h2p', 3)
        
        # Tick counter
        self.tick = 1 << (self.u_reset_period - 1)
        
        # === Prediction State ===
        self._pred_info = {}
        
        # === Statistics ===
        self._tage_predictions = 0
        self._tage_correct = 0
        self._sc_overrides = 0
        self._sc_override_correct = 0
        self._loop_branches = 0
        self._data_branches = 0
    
    # ========================================================================
    # Branch Type Classification
    # ========================================================================
    
    def _classify_index(self, pc: int) -> int:
        """Get classifier table index."""
        return int((pc >> 2) & (self.classifier_size - 1))
    
    def _get_branch_type(self, pc: int) -> int:
        """Get branch type classification."""
        idx = self._classify_index(pc)
        counter = self.classifier[idx]
        
        if counter >= self.loop_threshold:
            return BranchType.LOOP_LIKE
        elif counter <= self.data_threshold:
            return BranchType.DATA_DEPENDENT
        else:
            return BranchType.UNKNOWN
    
    def _update_classifier(self, pc: int, is_loop_behavior: bool):
        """Update branch classifier based on observed behavior."""
        idx = self._classify_index(pc)
        if is_loop_behavior:
            self.classifier[idx] = min(3, self.classifier[idx] + 1)
        else:
            self.classifier[idx] = max(-4, self.classifier[idx] - 1)
    
    def _get_priority_tables(self, branch_type: int) -> Tuple[List[int], List[int]]:
        """
        Get prioritized table ordering based on branch type.
        
        Returns:
            (priority_tables, secondary_tables)
        """
        if branch_type == BranchType.LOOP_LIKE:
            # Short history tables first (0-5), then long
            mid = min(6, self.num_tables)
            return list(range(mid-1, -1, -1)), list(range(self.num_tables-1, mid-1, -1))
        elif branch_type == BranchType.DATA_DEPENDENT:
            # Long history tables first, then short
            mid = max(self.num_tables - 6, 0)
            return list(range(self.num_tables-1, mid-1, -1)), list(range(mid-1, -1, -1))
        else:
            # Default: longest history first (standard TAGE)
            return list(range(self.num_tables-1, -1, -1)), []
    
    # ========================================================================
    # Indexing Functions
    # ========================================================================
    
    def _bimodal_index(self, pc: int) -> int:
        return int((pc >> 2) & (self.bimodal_size - 1))
    
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
        table_bits = int(np.log2(self.table_sizes[bank]))
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
        hist_len = self.history_lengths[table]
        table_bits = int(np.log2(self.table_sizes[table]))
        path_len = min(hist_len, self.path_hist_bits)
        
        shifted_pc = (pc >> 2) & 0xFFFFFFFF
        
        index = (shifted_pc ^
                 (shifted_pc >> (abs(table_bits - table) + 1)) ^
                 self.folded_indices[table].comp ^
                 self._F(self.path_hist, path_len, table))
        
        return int(index & (self.table_sizes[table] - 1))
    
    def _tage_tag(self, pc: int, table: int) -> int:
        """Compute TAGE tag with enhanced mixing."""
        tag_width = self.tag_widths[table]
        
        # Enhanced tag: PC + folded history + path contribution
        tag = (((pc >> 2) & 0xFFFFFFFF) ^
               self.folded_tags[0][table].comp ^
               (self.folded_tags[1][table].comp << 1))
        
        # Add path history contribution for better coverage
        path_contribution = (self.path_hist >> (table * 2)) & 0xF
        tag ^= path_contribution
        
        return int(tag & ((1 << tag_width) - 1))
    
    # ========================================================================
    # Statistical Corrector
    # ========================================================================
    
    def _sc_index(self, pc: int, table: int) -> int:
        """Compute SC table index with different hash per table."""
        local_idx = (pc >> 2) & (self.local_hist_size - 1)
        local_hist = int(self.local_hist[local_idx]) & 0xFFFFFFFF
        pc_hash = (pc >> 2) & 0xFFFFFFFF
        
        if table == 0:
            # PC-based
            return pc_hash & (self.sc_table_size - 1)
        elif table == 1:
            # PC ^ global history
            return (pc_hash ^ self.ghist_int) & (self.sc_table_size - 1)
        elif table == 2:
            # PC ^ path history
            return (pc_hash ^ self.path_hist) & (self.sc_table_size - 1)
        else:
            # PC ^ local history
            return (pc_hash ^ local_hist) & (self.sc_table_size - 1)
    
    def _sc_predict(self, pc: int) -> Tuple[bool, int, float]:
        """
        Get Statistical Corrector prediction.
        
        Returns:
            (prediction, sum, confidence)
        """
        total = 0
        for t in range(self.sc_num_tables):
            idx = self._sc_index(pc, t)
            total += int(self.sc_tables[t][idx])
        
        pred = total >= 0
        # Confidence based on sum magnitude vs threshold
        conf = min(abs(total) / (self.sc_threshold * 2), 1.0)
        
        return pred, total, conf
    
    def _sc_update(self, pc: int, taken: bool):
        """Update SC tables."""
        for t in range(self.sc_num_tables):
            idx = self._sc_index(pc, t)
            if taken:
                self.sc_tables[t][idx] = min(self.sc_max, 
                                             int(self.sc_tables[t][idx]) + 1)
            else:
                self.sc_tables[t][idx] = max(self.sc_min, 
                                             int(self.sc_tables[t][idx]) - 1)
    
    # ========================================================================
    # H2P Tracking
    # ========================================================================
    
    def _h2p_index(self, pc: int) -> int:
        return int((pc >> 2) & (self.h2p_table_size - 1))
    
    def _is_h2p(self, pc: int) -> bool:
        return self.h2p_counts[self._h2p_index(pc)] >= self.h2p_threshold
    
    def _update_h2p(self, pc: int, mispredicted: bool):
        idx = self._h2p_index(pc)
        if mispredicted:
            self.h2p_counts[idx] = min(255, int(self.h2p_counts[idx]) + 2)
        else:
            if self.h2p_counts[idx] > 0:
                self.h2p_counts[idx] = int(self.h2p_counts[idx]) - 1
    
    # ========================================================================
    # Counter Update Functions
    # ========================================================================
    
    def _ctr_update(self, ctr: int, taken: bool) -> int:
        if taken:
            return min(self.ctr_max, ctr + 1)
        else:
            return max(self.ctr_min, ctr - 1)
    
    def _unsigned_ctr_update(self, ctr: int, up: bool) -> int:
        if up:
            return min(self.u_max, ctr + 1)
        else:
            return max(0, ctr - 1)
    
    # ========================================================================
    # Prediction
    # ========================================================================
    
    def predict(self, pc: int, history: np.ndarray = None) -> PredictionResult:
        """
        BATAGE-SC prediction.
        
        1. Classify branch type
        2. Compute indices/tags for all tables
        3. Find provider using priority ordering
        4. Compute SC prediction in parallel
        5. Decide final prediction
        """
        # === Get branch type and table priority ===
        branch_type = self._get_branch_type(pc)
        priority_tables, secondary_tables = self._get_priority_tables(branch_type)
        
        # === Compute indices and tags for all tables ===
        indices = []
        tags = []
        for i in range(self.num_tables):
            indices.append(self._tage_index(pc, i))
            tags.append(self._tage_tag(pc, i))
        
        # === Find provider using priority ordering ===
        provider = -1
        provider_from_priority = False
        
        # First check priority tables
        for i in priority_tables:
            idx = indices[i]
            if self.tables[i][idx].tag == tags[i]:
                provider = i
                provider_from_priority = True
                break
        
        # If no hit in priority, check secondary tables
        if provider < 0:
            for i in secondary_tables:
                idx = indices[i]
                if self.tables[i][idx].tag == tags[i]:
                    provider = i
                    break
        
        # === Find alternate provider ===
        alt_provider = -1
        if provider >= 0:
            # Look for alternate in all tables with shorter history
            for i in range(self.num_tables - 1, -1, -1):
                if i != provider and i < provider:
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
            provider_conf = abs(provider_ctr) / self.ctr_max
            
            if alt_provider >= 0:
                alt_idx = indices[alt_provider]
                alt_pred = self.tables[alt_provider][alt_idx].ctr >= 0
            else:
                alt_pred = self._get_bimodal_pred(pc)
            
            # Check weak entry
            weak_entry = abs(2 * provider_ctr + 1) <= 1
            
            # UseAlt logic
            use_alt_idx = (pc >> 2) & (self.use_alt_size - 1)
            use_alt = self.use_alt_on_na[use_alt_idx] < 0
            
            if weak_entry and use_alt:
                tage_pred = alt_pred
            else:
                tage_pred = provider_pred
        else:
            alt_pred = self._get_bimodal_pred(pc)
            provider_pred = alt_pred
            tage_pred = alt_pred
            provider_conf = 0.5
            weak_entry = False
            provider_ctr = 0
        
        # === Compute SC prediction (in parallel) ===
        sc_pred, sc_sum, sc_conf = self._sc_predict(pc)
        
        # === Get branch type for SC decision ===
        branch_type = self._get_branch_type(pc)
        
        # === Final Decision Logic ===
        final_pred = tage_pred
        use_sc = False
        
        # Only consider SC when:
        # 1. TAGE confidence is low (provider is weak)
        # 2. We have a provider (don't override bimodal - it's often right)
        # 3. SC has strong agreement (sum magnitude > threshold)
        # 4. SC and TAGE disagree
        # 5. Supermajority of SC tables agree
        # 6. If sc_loop_only mode: only for loop-like branches
        sc_type_ok = (not self.sc_loop_only) or (branch_type == BranchType.LOOP_LIKE)
        
        if (provider >= 0 and  # Don't override bimodal
            sc_type_ok and
            provider_conf < self.sc_confidence_threshold and 
            abs(sc_sum) >= self.sc_threshold and
            sc_pred != tage_pred):
            # Count how many SC tables agree
            agreement_count = 0
            for t in range(self.sc_num_tables):
                idx = self._sc_index(pc, t)
                if (self.sc_tables[t][idx] >= 0) == sc_pred:
                    agreement_count += 1
            
            # Require supermajority (3 out of 4 for 4 tables)
            supermajority = (self.sc_num_tables * 3 + 3) // 4  # ceil(0.75 * n)
            if agreement_count >= supermajority:
                final_pred = sc_pred
                use_sc = True
        
        # Compute final confidence
        if use_sc:
            confidence = sc_conf
        else:
            confidence = provider_conf
        
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
            'provider_conf': provider_conf,
            'sc_pred': sc_pred,
            'sc_sum': sc_sum,
            'sc_conf': sc_conf,
            'use_sc': use_sc,
            'final_pred': final_pred,
            'branch_type': branch_type,
            'provider_from_priority': provider_from_priority,
        }
        
        return PredictionResult(
            prediction=final_pred,
            confidence=confidence,
            predictor_used=self.name,
            raw_sum=float(provider_ctr if provider >= 0 else sc_sum)
        )
    
    # ========================================================================
    # Update
    # ========================================================================
    
    def update(self, pc: int, history: np.ndarray, taken: bool,
               prediction: PredictionResult) -> None:
        """
        Update BATAGE-SC components.
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
        use_sc = info['use_sc']
        branch_type = info['branch_type']
        
        # Track statistics
        self._tage_predictions += 1
        if tage_pred == taken:
            self._tage_correct += 1
        if use_sc:
            self._sc_overrides += 1
            if final_pred == taken:
                self._sc_override_correct += 1
        
        # Track branch types
        if branch_type == BranchType.LOOP_LIKE:
            self._loop_branches += 1
        elif branch_type == BranchType.DATA_DEPENDENT:
            self._data_branches += 1
        
        # === Update Branch Classifier ===
        # Detect loop-like behavior: branch taken consecutively or has regular pattern
        # Simple heuristic: if prediction was from a short-history table, likely loop-like
        if provider >= 0:
            is_loop_behavior = provider < self.num_tables // 2
        else:
            # Bimodal hit often indicates stable/loop-like behavior
            is_loop_behavior = True
        self._update_classifier(pc, is_loop_behavior)
        
        # === Update H2P tracking ===
        mispredicted = (final_pred != taken)
        self._update_h2p(pc, mispredicted)
        is_h2p = self._is_h2p(pc)
        
        # === Update SC tables ===
        # Always update SC (it's lightweight)
        self._sc_update(pc, taken)
        
        # === Update useAltPredForNewlyAllocated ===
        tage_wrong = (tage_pred != taken)
        alloc = tage_wrong and (provider < self.num_tables - 1)
        
        if provider >= 0 and weak_entry:
            if provider_pred == taken:
                alloc = False
            
            if provider_pred != alt_pred:
                use_alt_idx = int((pc >> 2) & (self.use_alt_size - 1))
                if alt_pred == taken:
                    self.use_alt_on_na[use_alt_idx] = min(
                        self.use_alt_max, int(self.use_alt_on_na[use_alt_idx]) + 1)
                else:
                    self.use_alt_on_na[use_alt_idx] = max(
                        -self.use_alt_max - 1, int(self.use_alt_on_na[use_alt_idx]) - 1)
        
        # === Allocation ===
        if alloc:
            # Determine allocation aggressiveness based on H2P status
            max_alloc = self.max_alloc_h2p if is_h2p else self.max_alloc
            self._handle_allocation(provider, taken, indices, tags, max_alloc, branch_type)
        
        # === Update TAGE tables ===
        if provider >= 0:
            provider_idx = indices[provider]
            entry = self.tables[provider][provider_idx]
            entry.ctr = self._ctr_update(entry.ctr, taken)
            
            # Update alternate if provider has u=0
            if entry.u == 0:
                if alt_provider >= 0:
                    alt_idx = indices[alt_provider]
                    alt_entry = self.tables[alt_provider][alt_idx]
                    alt_entry.ctr = self._ctr_update(alt_entry.ctr, taken)
                else:
                    self._update_bimodal(pc, taken)
            
            # Update useful bits
            if tage_pred != alt_pred:
                entry.u = self._unsigned_ctr_update(entry.u, tage_pred == taken)
        else:
            self._update_bimodal(pc, taken)
        
        # === Update Local History ===
        local_idx = int((pc >> 2) & (self.local_hist_size - 1))
        t_int = 1 if taken else 0
        self.local_hist[local_idx] = int(((int(self.local_hist[local_idx]) << 1) | t_int) & \
                                     ((1 << self.local_hist_bits) - 1))
        
        # === Update Global History ===
        self._update_histories(pc, taken)
        
        # === Periodic useful bit reset ===
        self.tick += 1
        if (self.tick & ((1 << self.u_reset_period) - 1)) == 0:
            for table in self.tables:
                for entry in table:
                    entry.u >>= 1
    
    def _handle_allocation(self, provider: int, taken: bool, indices: List[int],
                          tags: List[int], max_alloc: int, branch_type: int):
        """
        Handle entry allocation with branch-type awareness.
        """
        start = provider + 1 if provider >= 0 else 0
        
        # Get allocation order based on branch type
        if branch_type == BranchType.LOOP_LIKE:
            # For loop branches, prefer allocating in short-history tables
            alloc_order = list(range(start, min(start + 4, self.num_tables)))
        elif branch_type == BranchType.DATA_DEPENDENT:
            # For data-dependent, prefer long-history tables
            alloc_order = list(range(self.num_tables - 1, start - 1, -1))
        else:
            # Default: standard TAGE order (try all from start upward)
            alloc_order = list(range(start, self.num_tables))
        
        # Find minimum useful value
        min_u = self.u_max + 1
        for i in alloc_order:
            idx = indices[i]
            if self.tables[i][idx].u < min_u:
                min_u = self.tables[i][idx].u
        
        # Allocation
        num_allocated = 0
        
        if min_u > 0:
            # No free entry - decay useful bits
            for i in alloc_order:
                idx = indices[i]
                if self.tables[i][idx].u > 0:
                    self.tables[i][idx].u -= 1
        else:
            # Allocate in entries with u=0
            for i in alloc_order:
                idx = indices[i]
                if self.tables[i][idx].u == 0:
                    self.tables[i][idx].tag = tags[i]
                    self.tables[i][idx].ctr = 0 if taken else -1
                    num_allocated += 1
                    if num_allocated >= max_alloc:
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
        
        # Integer history (for fast SC indexing)
        self.ghist_int = ((self.ghist_int << 1) | t_int) & 0xFFFFFFFF
        
        # Folded histories
        new_bit = t_int
        for i in range(self.num_tables):
            hist_len = self.history_lengths[i]
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
            bits_per_entry = (self.counter_bits + self.useful_bits + self.tag_widths[i])
            tage_bits += self.table_sizes[i] * bits_per_entry
        bits += tage_bits
        components['tage_tables'] = tage_bits
        
        # Bimodal
        bimodal_bits = self.bimodal_size + (self.bimodal_size >> self.bimodal_hyst_ratio)
        bits += bimodal_bits
        components['bimodal'] = bimodal_bits
        
        # Branch classifier
        classifier_bits = self.classifier_size * 2  # 2 bits per entry
        bits += classifier_bits
        components['classifier'] = classifier_bits
        
        # Statistical Corrector
        sc_bits = self.sc_num_tables * self.sc_table_size * self.sc_counter_bits
        bits += sc_bits
        components['statistical_corrector'] = sc_bits
        
        # H2P tracking
        h2p_bits = self.h2p_table_size * 8  # 8 bits per entry
        bits += h2p_bits
        components['h2p_tracking'] = h2p_bits
        
        # Local history
        local_bits = self.local_hist_size * self.local_hist_bits
        bits += local_bits
        components['local_history'] = local_bits
        
        # UseAlt
        use_alt_bits = self.use_alt_size * self.use_alt_bits
        bits += use_alt_bits
        components['use_alt'] = use_alt_bits
        
        # Histories
        hist_bits = self.max_hist + self.path_hist_bits + 32  # ghist_int
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
        base_stats = self.stats.get_stats()
        
        tage_accuracy = (self._tage_correct / self._tage_predictions * 100 
                        if self._tage_predictions > 0 else 0)
        sc_accuracy = (self._sc_override_correct / self._sc_overrides * 100 
                      if self._sc_overrides > 0 else 0)
        
        return {
            **base_stats,
            'tage_predictions': self._tage_predictions,
            'tage_accuracy': tage_accuracy,
            'sc_overrides': self._sc_overrides,
            'sc_override_accuracy': sc_accuracy,
            'loop_branches': self._loop_branches,
            'data_dependent_branches': self._data_branches,
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
        
        # Classifier
        self.classifier.fill(0)
        
        # SC
        for t in range(self.sc_num_tables):
            self.sc_tables[t].fill(0)
        
        # H2P
        self.h2p_counts.fill(0)
        
        # Local history
        self.local_hist.fill(0)
        
        # UseAlt
        self.use_alt_on_na.fill(0)
        
        # Histories
        self.global_hist.fill(0)
        self.pt_ghist = 0
        self.ghist_int = 0
        self.path_hist = 0
        
        # Folded histories
        for i in range(self.num_tables):
            self.folded_indices[i].comp = 0
            self.folded_tags[0][i].comp = 0
            self.folded_tags[1][i].comp = 0
        
        # Tick
        self.tick = 1 << (self.u_reset_period - 1)
        
        # Statistics
        self._tage_predictions = 0
        self._tage_correct = 0
        self._sc_overrides = 0
        self._sc_override_correct = 0
        self._loop_branches = 0
        self._data_branches = 0
