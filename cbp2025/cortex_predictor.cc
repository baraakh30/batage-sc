/*
 * CORTEX: COrrelation-Rich TEXture Predictor Implementation
 * CBP2025 Championship Submission
 */

#include "cortex_predictor.h"
#include <cstdio>

// Global predictor instance
CORTEX_Predictor cond_predictor_impl;

// ============================================================================
// Constructor
// ============================================================================
CORTEX_Predictor::CORTEX_Predictor() {
    // Allocate TAGE tables
    for (int i = 0; i < CORTEX_NUM_TABLES; i++) {
        tables[i] = new CortexEntry[CORTEX_TABLE_SIZES[i]];
        memset(tables[i], 0, CORTEX_TABLE_SIZES[i] * sizeof(CortexEntry));
    }
    
    // Allocate large bimodal
    bimodal_pred = new int8_t[CORTEX_BIMODAL_SIZE];
    bimodal_hyst = new int8_t[CORTEX_BIMODAL_SIZE >> CORTEX_BIMODAL_HYST_SHIFT];
    memset(bimodal_pred, 0, CORTEX_BIMODAL_SIZE);
    memset(bimodal_hyst, 1, CORTEX_BIMODAL_SIZE >> CORTEX_BIMODAL_HYST_SHIFT);
    
    // Initialize tables
    memset(reg_useful, 0, sizeof(reg_useful));
    memset(reg_pred_table, 0, sizeof(reg_pred_table));
    memset(load_corr_table, 0, sizeof(load_corr_table));
    memset(distant_loads, 0, sizeof(distant_loads));
    memset(value_stability, 0, sizeof(value_stability));
    memset(ls_distance_table, 0, sizeof(ls_distance_table));
    memset(inst_mix_table, 0, sizeof(inst_mix_table));
    memset(clusters, 0, sizeof(clusters));
    memset(calibration, 0, sizeof(calibration));
    memset(sc_tables, 0, sizeof(sc_tables));
    memset(imli_table, 0, sizeof(imli_table));
    memset(imli_oh_table, 0, sizeof(imli_oh_table));
    memset(h2p_counts, 0, sizeof(h2p_counts));
    memset(use_alt_on_na, 0, sizeof(use_alt_on_na));
    
    // Initialize counters
    tick = 1ULL << (CORTEX_U_RESET_PERIOD - 1);
    total_instr_count = 0;
    imli_counter = 0;
    last_br_pc = 0;
    last_target = 0;
    last_load_pc = 0;
    last_store_pc = 0;
    last_load_distance = 0;
    
    // Initialize folded histories
    for (int i = 0; i < CORTEX_NUM_TABLES; i++) {
        int table_bits = (int)log2(CORTEX_TABLE_SIZES[i]);
        active_hist.ch_i[i].init(CORTEX_HISTORY_LENGTHS[i], table_bits);
        active_hist.ch_t[0][i].init(CORTEX_HISTORY_LENGTHS[i], CORTEX_TAG_WIDTHS[i]);
        active_hist.ch_t[1][i].init(CORTEX_HISTORY_LENGTHS[i], std::max(1, CORTEX_TAG_WIDTHS[i] - 1));
    }
}

// ============================================================================
// Destructor
// ============================================================================
CORTEX_Predictor::~CORTEX_Predictor() {
    for (int i = 0; i < CORTEX_NUM_TABLES; i++) {
        delete[] tables[i];
    }
    delete[] bimodal_pred;
    delete[] bimodal_hyst;
}

// ============================================================================
// Setup and Terminate
// ============================================================================
void CORTEX_Predictor::setup() {
    // Initialization complete
}

void CORTEX_Predictor::terminate() {
    // Cleanup if needed
}

// ============================================================================
// Helper: Unique ID
// ============================================================================
uint64_t CORTEX_Predictor::get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
    assert(piece < 16);
    return (seq_no << 4) | (piece & 0x000F);
}

// ============================================================================
// Helper: Register Digest Computation (from RUNLTS approach)
// ============================================================================
uint16_t CORTEX_Predictor::compute_reg_digest(uint64_t value, bool is_int) {
    if (is_int) {
        // For integer: leading zeros, trailing zeros, low bits
        int leading = __builtin_clzll(value | 1);
        int trailing = __builtin_ctzll(value | (1ULL << 63));
        uint8_t low6 = value & 0x3F;
        
        // 4 bits leading, 4 bits trailing, 4 bits low
        return ((leading & 0xF) << 8) | ((trailing & 0xF) << 4) | (low6 >> 2);
    } else {
        // For FP: extract sign + exponent bits
        uint64_t sign = (value >> 63) & 1;
        uint64_t exp = (value >> 52) & 0x7FF;
        uint64_t mantissa_high = (value >> 48) & 0xF;
        
        return ((sign & 1) << 11) | ((exp & 0x3F) << 5) | (mantissa_high & 0x1F);
    }
}

// ============================================================================
// Helper: Instruction Mix Update
// ============================================================================
void CORTEX_Predictor::update_inst_mix(uint8_t inst_class) {
    // Track instruction types in sliding window
    uint8_t old_type = active_hist.recent_inst_types[active_hist.inst_mix_ptr];
    active_hist.recent_inst_types[active_hist.inst_mix_ptr] = inst_class;
    active_hist.inst_mix_ptr = (active_hist.inst_mix_ptr + 1) % CORTEX_INST_MIX_WINDOW;
    
    // Update counts (decremented old, increment new)
    if (old_type <= 1) {  // ALU
        if (active_hist.inst_mix.alu_count > 0) active_hist.inst_mix.alu_count--;
    } else if (old_type <= 2) {  // MEM
        if (active_hist.inst_mix.mem_count > 0) active_hist.inst_mix.mem_count--;
    } else if (old_type <= 5) {  // BR
        if (active_hist.inst_mix.br_count > 0) active_hist.inst_mix.br_count--;
    } else if (old_type == 6) {  // FP
        if (active_hist.inst_mix.fp_count > 0) active_hist.inst_mix.fp_count--;
    }
    
    if (inst_class <= 1) {
        if (active_hist.inst_mix.alu_count < 15) active_hist.inst_mix.alu_count++;
    } else if (inst_class <= 2) {
        if (active_hist.inst_mix.mem_count < 15) active_hist.inst_mix.mem_count++;
    } else if (inst_class <= 5) {
        if (active_hist.inst_mix.br_count < 15) active_hist.inst_mix.br_count++;
    } else if (inst_class == 6) {
        if (active_hist.inst_mix.fp_count < 15) active_hist.inst_mix.fp_count++;
    }
}

// ============================================================================
// Helper: Branch Clustering
// ============================================================================
int CORTEX_Predictor::get_cluster_index(uint64_t pc) {
    uint32_t hash = ((pc >> 2) ^ (pc >> 10) ^ (pc >> 18)) & (CORTEX_CLUSTER_SIZE - 1);
    return hash;
}

int CORTEX_Predictor::find_or_create_cluster(uint64_t pc) {
    int idx = get_cluster_index(pc);
    BranchCluster& cluster = clusters[idx];
    
    // Check if PC already in cluster
    for (int i = 0; i < cluster.num_members && i < CORTEX_CLUSTER_MEMBERS; i++) {
        if (cluster.member_pcs[i] == pc) {
            return idx;
        }
    }
    
    // Add if space
    if (cluster.num_members < CORTEX_CLUSTER_MEMBERS) {
        cluster.member_pcs[cluster.num_members++] = pc;
    }
    
    return idx;
}

// ============================================================================
// Helper: Confidence Calibration
// ============================================================================
void CORTEX_Predictor::update_calibration(uint64_t pc, bool pred, bool outcome, bool high_conf) {
    int idx = (pc >> 2) & (CORTEX_CALIB_SIZE - 1);
    CalibEntry& entry = calibration[idx];
    
    // Update histories
    entry.pred_history = ((entry.pred_history << 1) | (pred ? 1 : 0)) & 0xFFFF;
    entry.outcome_history = ((entry.outcome_history << 1) | (outcome ? 1 : 0)) & 0xFFFF;
    
    // Update confidence accuracy
    if (high_conf) {
        bool correct = (pred == outcome);
        if (correct) {
            entry.confidence_accuracy = std::min(255, (int)entry.confidence_accuracy + 2);
        } else {
            entry.confidence_accuracy = std::max(0, (int)entry.confidence_accuracy - 3);
        }
    }
    
    // Adjust calibration based on pattern
    uint16_t xor_pattern = entry.pred_history ^ entry.outcome_history;
    int mismatches = __builtin_popcount(xor_pattern);
    
    if (mismatches > 8) {
        // Many mismatches - lower threshold (more conservative)
        entry.calibration = std::min((int8_t)15, (int8_t)(entry.calibration + 1));
    } else if (mismatches < 4) {
        // Few mismatches - raise threshold (more aggressive)
        entry.calibration = std::max((int8_t)-15, (int8_t)(entry.calibration - 1));
    }
}

int CORTEX_Predictor::get_calibrated_threshold(uint64_t pc) {
    int idx = (pc >> 2) & (CORTEX_CALIB_SIZE - 1);
    const CalibEntry& entry = calibration[idx];
    
    // Base threshold
    int base_threshold = 8;
    
    // Adjust based on calibration
    int adjusted = base_threshold + entry.calibration;
    
    // Further adjust based on confidence accuracy
    if (entry.confidence_accuracy < 100) {
        adjusted += 4;  // Less confident, higher threshold
    } else if (entry.confidence_accuracy > 180) {
        adjusted -= 2;  // More confident, lower threshold
    }
    
    return std::max(4, std::min(20, adjusted));
}

// ============================================================================
// Indexing Functions
// ============================================================================
int CORTEX_Predictor::bimodal_index(uint64_t pc) const {
    return (pc >> 2) & (CORTEX_BIMODAL_SIZE - 1);
}

bool CORTEX_Predictor::get_bimodal_pred(uint64_t pc) const {
    return bimodal_pred[bimodal_index(pc)] >= 0;
}

void CORTEX_Predictor::update_bimodal(uint64_t pc, bool taken) {
    int idx = bimodal_index(pc);
    int hyst_idx = idx >> CORTEX_BIMODAL_HYST_SHIFT;
    
    int8_t pred = bimodal_pred[idx];
    int8_t hyst = bimodal_hyst[hyst_idx];
    
    int inter = ((pred >= 0) ? 2 : 0) + ((hyst > 0) ? 1 : 0);
    if (taken) {
        if (inter < 3) inter++;
    } else {
        if (inter > 0) inter--;
    }
    
    bimodal_pred[idx] = (inter >= 2) ? 1 : -1;
    bimodal_hyst[hyst_idx] = (inter & 1) ? 1 : 0;
}

uint64_t F_hash(uint64_t A, int size, int bank) {
    if (size <= 0) return 0;
    uint64_t mask = (1ULL << size) - 1;
    A &= mask;
    
    int shift = bank % size;
    if (shift == 0) shift = 1;
    
    uint64_t A1 = A & ((1ULL << (size / 2)) - 1);
    uint64_t A2 = A >> (size / 2);
    
    A2 = ((A2 << shift) & ((1ULL << (size / 2)) - 1)) | (A2 >> ((size / 2) - shift));
    
    return A1 ^ A2;
}

int CORTEX_Predictor::tage_index(uint64_t pc, int table, const CortexHist& hist) const {
    int hist_len = CORTEX_HISTORY_LENGTHS[table];
    int table_bits = (int)log2(CORTEX_TABLE_SIZES[table]);
    int path_len = std::min(hist_len, CORTEX_PATH_BITS);
    
    uint64_t pc_hash = pc >> 2;
    
    uint64_t index = pc_hash ^
                    (pc_hash >> (abs(table_bits - table) + 1)) ^
                    hist.ch_i[table].comp ^
                    F_hash(hist.path_hist, path_len, table);
    
    return index & (CORTEX_TABLE_SIZES[table] - 1);
}

uint16_t CORTEX_Predictor::tage_tag(uint64_t pc, int table, const CortexHist& hist) const {
    int tag_width = CORTEX_TAG_WIDTHS[table];
    
    uint64_t tag = (pc >> 2) ^
                  hist.ch_t[0][table].comp ^
                  (hist.ch_t[1][table].comp << 1);
    
    // Add path contribution
    uint64_t path_contrib = (hist.path_hist >> (table * 2)) & 0xF;
    tag ^= path_contrib;
    
    return tag & ((1U << tag_width) - 1);
}

// ============================================================================
// Statistical Corrector
// ============================================================================
int CORTEX_Predictor::sc_index(uint64_t pc, int bank, const CortexHist& hist) const {
    int local_idx = (pc >> 2) & (CORTEX_LOCAL_SIZE - 1);
    uint32_t local_val = hist.local_hist[local_idx];
    uint64_t pc_hash = pc >> 2;
    
    uint64_t result;
    switch (bank) {
        case 0:
            // PC only
            result = pc_hash;
            break;
        case 1:
            // PC ^ global
            result = pc_hash ^ hist.ghist_int;
            break;
        case 2:
            // PC ^ path
            result = pc_hash ^ hist.path_hist;
            break;
        case 3:
            // PC ^ local
            result = pc_hash ^ local_val;
            break;
        case 4:
            // PC ^ instruction mix
            result = pc_hash ^ hist.inst_mix.get_digest();
            break;
        default:
            // PC ^ ghist ^ path
            result = pc_hash ^ hist.ghist_int ^ (hist.path_hist >> 8);
            break;
    }
    
    return result & (CORTEX_SC_BANK_SIZE - 1);
}

void CORTEX_Predictor::sc_predict(uint64_t pc, const CortexHist& hist, bool& pred, int& sum) {
    int total = 0;
    
    // Standard SC banks
    for (int b = 0; b < CORTEX_SC_BANKS; b++) {
        int idx = sc_index(pc, b, hist);
        total += sc_tables[b][idx];
    }
    
    // IMLI contribution (3x weight - key finding!)
    if (imli_counter > 0) {
        int imli_idx = ((pc >> 2) ^ imli_counter) & (CORTEX_IMLI_SIZE - 1);
        total += imli_table[imli_idx] * CORTEX_IMLI_WEIGHT;
        
        int oh_idx = ((pc >> 2) ^ (1 << (imli_counter & 7))) & (CORTEX_IMLI_OH_SIZE - 1);
        total += imli_oh_table[oh_idx] * CORTEX_IMLI_WEIGHT;
    }
    
    // NEW: Instruction mix contribution
    int mix_idx = ((pc >> 2) ^ hist.inst_mix.get_digest()) & (CORTEX_INST_MIX_SIZE - 1);
    total += inst_mix_table[mix_idx];
    
    // NEW: Load-store distance contribution
    if (last_load_distance < 256) {
        int ls_idx = ((pc >> 2) ^ last_load_distance) & (CORTEX_LS_DISTANCE_SIZE - 1);
        total += ls_distance_table[ls_idx];
    }
    
    // NEW: Branch cluster shared bias
    int cluster_idx = get_cluster_index(pc);
    total += clusters[cluster_idx].shared_bias;
    
    // NEW: Register value contribution (top 2 most useful registers)
    int best_regs[2] = {-1, -1};
    int best_useful[2] = {-128, -128};
    
    for (int bank = 0; bank < CORTEX_REG_BANKS; bank++) {
        int regs_per_bank = (CORTEX_NUM_INT_REGS + CORTEX_REG_BANKS - 1) / CORTEX_REG_BANKS;
        int reg_start = bank * regs_per_bank;
        int reg_end = std::min(CORTEX_NUM_INT_REGS, reg_start + regs_per_bank);
        
        for (int r = reg_start; r < reg_end; r++) {
            if (hist.int_regs[r].age < 8) {  // Recent value
                int useful_idx = r % CORTEX_REG_USEFUL_SIZE;
                int useful_val = reg_useful[bank][useful_idx][r];
                
                if (useful_val > best_useful[0]) {
                    best_useful[1] = best_useful[0];
                    best_regs[1] = best_regs[0];
                    best_useful[0] = useful_val;
                    best_regs[0] = r;
                } else if (useful_val > best_useful[1]) {
                    best_useful[1] = useful_val;
                    best_regs[1] = r;
                }
            }
        }
    }
    
    // Add contribution from top 2 registers
    for (int i = 0; i < 2; i++) {
        if (best_regs[i] >= 0) {
            int bank = best_regs[i] / ((CORTEX_NUM_INT_REGS + CORTEX_REG_BANKS - 1) / CORTEX_REG_BANKS);
            uint16_t digest = hist.int_regs[best_regs[i]].value;
            int reg_idx = ((pc >> 2) ^ digest ^ best_regs[i]) & (CORTEX_REG_TABLE_SIZE - 1);
            total += reg_pred_table[bank][reg_idx];
        }
    }
    
    pred = (total >= 0);
    sum = total;
}

void CORTEX_Predictor::sc_update(uint64_t pc, bool taken, const CortexHist& hist) {
    int8_t update = taken ? 1 : -1;
    
    // Update SC banks
    for (int b = 0; b < CORTEX_SC_BANKS; b++) {
        int idx = sc_index(pc, b, hist);
        sc_tables[b][idx] = std::max(CORTEX_SC_MIN, std::min(CORTEX_SC_MAX, 
                                                             (int)sc_tables[b][idx] + update));
    }
    
    // Update IMLI
    if (imli_counter > 0) {
        int imli_idx = ((pc >> 2) ^ imli_counter) & (CORTEX_IMLI_SIZE - 1);
        imli_table[imli_idx] = std::max(CORTEX_SC_MIN, std::min(CORTEX_SC_MAX, 
                                                                (int)imli_table[imli_idx] + update));
        
        int oh_idx = ((pc >> 2) ^ (1 << (imli_counter & 7))) & (CORTEX_IMLI_OH_SIZE - 1);
        imli_oh_table[oh_idx] = std::max(CORTEX_SC_MIN, std::min(CORTEX_SC_MAX, 
                                                                 (int)imli_oh_table[oh_idx] + update));
    }
    
    // Update instruction mix
    int mix_idx = ((pc >> 2) ^ hist.inst_mix.get_digest()) & (CORTEX_INST_MIX_SIZE - 1);
    inst_mix_table[mix_idx] = std::max(CORTEX_SC_MIN, std::min(CORTEX_SC_MAX, 
                                                               (int)inst_mix_table[mix_idx] + update));
    
    // Update LS distance
    if (last_load_distance < 256) {
        int ls_idx = ((pc >> 2) ^ last_load_distance) & (CORTEX_LS_DISTANCE_SIZE - 1);
        ls_distance_table[ls_idx] = std::max(CORTEX_SC_MIN, std::min(CORTEX_SC_MAX, 
                                                                     (int)ls_distance_table[ls_idx] + update));
    }
}

// ============================================================================
// H2P Tracking
// ============================================================================
bool CORTEX_Predictor::is_h2p(uint64_t pc) const {
    int idx = (pc >> 2) & (CORTEX_H2P_SIZE - 1);
    return h2p_counts[idx] >= CORTEX_H2P_THRESHOLD;
}

void CORTEX_Predictor::update_h2p(uint64_t pc, bool mispred) {
    int idx = (pc >> 2) & (CORTEX_H2P_SIZE - 1);
    if (mispred) {
        h2p_counts[idx] = std::min(65535, (int)h2p_counts[idx] + 4);
    } else {
        if (h2p_counts[idx] > 0) {
            h2p_counts[idx]--;
        }
    }
}

// ============================================================================
// Prediction
// ============================================================================
bool CORTEX_Predictor::predict(uint64_t seq_no, uint8_t piece, uint64_t pc, bool tage_sc_l_pred) {
    CortexHist& hist = active_hist;
    
    // Compute indices and tags
    for (int i = 0; i < CORTEX_NUM_TABLES; i++) {
        hist.indices[i] = tage_index(pc, i, hist);
        hist.tags[i] = tage_tag(pc, i, hist);
    }
    
    // Find provider (longest matching)
    int provider = -1;
    for (int i = CORTEX_NUM_TABLES - 1; i >= 0; i--) {
        int idx = hist.indices[i];
        if (tables[i][idx].tag == hist.tags[i]) {
            provider = i;
            break;
        }
    }
    
    // Find alternate
    int alt_provider = -1;
    if (provider >= 0) {
        for (int i = provider - 1; i >= 0; i--) {
            int idx = hist.indices[i];
            if (tables[i][idx].tag == hist.tags[i]) {
                alt_provider = i;
                break;
            }
        }
    }
    
    // TAGE prediction
    bool provider_pred, alt_pred, tage_pred;
    int8_t provider_ctr = 0;
    bool weak_provider = false;
    
    if (provider >= 0) {
        int prov_idx = hist.indices[provider];
        provider_ctr = tables[provider][prov_idx].ctr;
        provider_pred = (provider_ctr >= 0);
        
        if (alt_provider >= 0) {
            alt_pred = (tables[alt_provider][hist.indices[alt_provider]].ctr >= 0);
        } else {
            alt_pred = get_bimodal_pred(pc);
        }
        
        weak_provider = (abs(2 * provider_ctr + 1) <= 1);
        
        // UseAlt
        int use_alt_idx = (pc >> 2) & (CORTEX_USE_ALT_SIZE - 1);
        bool use_alt = (use_alt_on_na[use_alt_idx] < 0);
        
        if (weak_provider && use_alt) {
            tage_pred = alt_pred;
        } else {
            tage_pred = provider_pred;
        }
    } else {
        alt_pred = get_bimodal_pred(pc);
        provider_pred = alt_pred;
        tage_pred = alt_pred;
    }
    
    // SC prediction
    bool sc_pred;
    int sc_sum;
    sc_predict(pc, hist, sc_pred, sc_sum);
    
    // Final decision with calibrated threshold
    bool final_pred = tage_pred;
    bool use_sc = false;
    
    int threshold = get_calibrated_threshold(pc);
    
    if (provider >= 0 &&
        abs(provider_ctr) <= 1 &&
        abs(sc_sum) >= threshold &&
        sc_pred != tage_pred) {
        
        final_pred = sc_pred;
        use_sc = true;
    }
    
    // Save state
    hist.provider = provider;
    hist.alt_provider = alt_provider;
    hist.tage_pred = tage_pred;
    hist.sc_pred = sc_pred;
    hist.final_pred = final_pred;
    hist.sc_sum = sc_sum;
    hist.provider_ctr = provider_ctr;
    hist.weak_provider = weak_provider;
    hist.use_sc = use_sc;
    
    // Checkpoint
    pred_time_histories.emplace(get_unique_inst_id(seq_no, piece), hist);
    
    return final_pred;
}

bool CORTEX_Predictor::predict_using_given_hist(uint64_t seq_no, uint8_t piece, uint64_t PC, 
                                               const CortexHist& hist_to_use, bool pred_time_predict) {
    return hist_to_use.final_pred;
}

// ============================================================================
// History Update
// ============================================================================
void CORTEX_Predictor::history_update(uint64_t seq_no, uint8_t piece, uint64_t pc, 
                                     bool taken, uint64_t next_pc) {
    CortexHist& hist = active_hist;
    
    total_instr_count++;
    
    // IMLI tracking
    bool is_backward = (next_pc < pc);
    if (taken && is_backward) {
        if (pc == last_br_pc && next_pc == last_target) {
            imli_counter++;
        } else {
            imli_counter = 1;
        }
    } else if (!taken && is_backward) {
        imli_counter = 0;
    }
    last_br_pc = pc;
    last_target = next_pc;
    
    // Path history
    hist.path_hist = ((hist.path_hist << 1) | ((pc >> 2) & 1)) & ((1ULL << CORTEX_PATH_BITS) - 1);
    
    // Local history
    int local_idx = (pc >> 2) & (CORTEX_LOCAL_SIZE - 1);
    hist.local_hist[local_idx] = ((hist.local_hist[local_idx] << 1) | (taken ? 1 : 0)) & 
                                  ((1ULL << CORTEX_LOCAL_BITS) - 1);
    
    // Global history
    if (hist.pt_ghist <= 0) {
        for (int i = 0; i < CORTEX_HIST_SIZE / 2; i++) {
            hist.ghist[CORTEX_HIST_SIZE - 1 - i] = hist.ghist[CORTEX_HIST_SIZE / 2 - 1 - i];
        }
        hist.pt_ghist = CORTEX_HIST_SIZE / 2;
    }
    hist.pt_ghist--;
    hist.ghist[hist.pt_ghist] = taken ? 1 : 0;
    hist.ghist_int = ((hist.ghist_int << 1) | (taken ? 1 : 0));
    
    // Update folded histories
    for (int i = 0; i < CORTEX_NUM_TABLES; i++) {
        hist.ch_i[i].update(hist.ghist, hist.pt_ghist);
        hist.ch_t[0][i].update(hist.ghist, hist.pt_ghist);
        hist.ch_t[1][i].update(hist.ghist, hist.pt_ghist);
    }
    
    // Age register values
    for (int i = 0; i < CORTEX_NUM_INT_REGS; i++) {
        if (hist.int_regs[i].age < 15) {
            hist.int_regs[i].age++;
        }
    }
}

// ============================================================================
// Allocation
// ============================================================================
void CORTEX_Predictor::handle_allocation(int provider, bool taken, const CortexHist& hist, bool h2p) {
    int start = (provider >= 0) ? provider + 1 : 0;
    int max_alloc = h2p ? CORTEX_MAX_ALLOC_H2P : CORTEX_MAX_ALLOC_NORMAL;
    
    // Find minimum u
    int min_u = CORTEX_U_MAX + 1;
    for (int i = start; i < CORTEX_NUM_TABLES; i++) {
        int idx = hist.indices[i];
        if (tables[i][idx].u < min_u) {
            min_u = tables[i][idx].u;
        }
    }
    
    if (min_u > 0) {
        // Decay u bits
        for (int i = start; i < CORTEX_NUM_TABLES; i++) {
            int idx = hist.indices[i];
            if (tables[i][idx].u > 0) {
                tables[i][idx].u--;
            }
        }
    } else {
        // Allocate
        int allocated = 0;
        for (int i = start; i < CORTEX_NUM_TABLES && allocated < max_alloc; i++) {
            int idx = hist.indices[i];
            if (tables[i][idx].u == 0) {
                tables[i][idx].tag = hist.tags[i];
                tables[i][idx].ctr = taken ? 0 : -1;
                tables[i][idx].u = 0;
                allocated++;
            }
        }
    }
}

// ============================================================================
// Update
// ============================================================================
void CORTEX_Predictor::update(uint64_t seq_no, uint8_t piece, uint64_t pc, 
                              bool resolveDir, bool predDir, uint64_t nextPC) {
    auto pred_hist_key = get_unique_inst_id(seq_no, piece);
    
    if (pred_time_histories.find(pred_hist_key) == pred_time_histories.end()) {
        return;
    }
    
    const CortexHist& pred_hist = pred_time_histories.at(pred_hist_key);
    update_internal(pc, resolveDir, predDir, nextPC, pred_hist);
    pred_time_histories.erase(pred_hist_key);
}

void CORTEX_Predictor::update(uint64_t pc, bool resolveDir, bool pred_taken, uint64_t nextPC, 
                              const CortexHist& hist_to_use) {
    update_internal(pc, resolveDir, pred_taken, nextPC, hist_to_use);
}

void CORTEX_Predictor::update_internal(uint64_t pc, bool taken, bool predDir, uint64_t nextPC,
                                      const CortexHist& pred_hist) {
    int provider = pred_hist.provider;
    int alt_provider = pred_hist.alt_provider;
    bool tage_pred = pred_hist.tage_pred;
    bool provider_pred = (pred_hist.provider_ctr >= 0);
    bool alt_pred = (alt_provider >= 0) ? 
                    (tables[alt_provider][pred_hist.indices[alt_provider]].ctr >= 0) :
                    get_bimodal_pred(pc);
    bool final_pred = pred_hist.final_pred;
    bool weak_provider = pred_hist.weak_provider;
    
    bool mispredicted = (final_pred != taken);
    bool tage_mispred = (tage_pred != taken);
    
    // Update H2P
    update_h2p(pc, mispredicted);
    bool h2p = is_h2p(pc);
    
    // Update SC
    sc_update(pc, taken, pred_hist);
    
    // Update calibration
    update_calibration(pc, final_pred, taken, abs(pred_hist.sc_sum) > 10);
    
    // Update UseAlt
    if (provider >= 0 && weak_provider && provider_pred != alt_pred) {
        int use_alt_idx = (pc >> 2) & (CORTEX_USE_ALT_SIZE - 1);
        if (alt_pred == taken) {
            use_alt_on_na[use_alt_idx] = std::min((int)((1 << CORTEX_USE_ALT_BITS) - 1), 
                                                  (int)use_alt_on_na[use_alt_idx] + 1);
        } else {
            use_alt_on_na[use_alt_idx] = std::max(-(int)(1 << CORTEX_USE_ALT_BITS), 
                                                  (int)use_alt_on_na[use_alt_idx] - 1);
        }
    }
    
    // Allocation
    bool alloc = tage_mispred && (provider < CORTEX_NUM_TABLES - 1);
    if (alloc) {
        handle_allocation(provider, taken, pred_hist, h2p);
    }
    
    // Update TAGE
    if (provider >= 0) {
        int prov_idx = pred_hist.indices[provider];
        CortexEntry& entry = tables[provider][prov_idx];
        
        entry.ctr = std::max(CORTEX_CTR_MIN, std::min(CORTEX_CTR_MAX, 
                                                      (int)entry.ctr + (taken ? 1 : -1)));
        
        // Update u
        if (tage_pred != alt_pred) {
            bool inc = (tage_pred == taken);
            entry.u = inc ? std::min(CORTEX_U_MAX, (int)entry.u + 1) : 
                           std::max(0, (int)entry.u - 1);
        }
        
        // Update alt if provider u=0
        if (entry.u == 0) {
            if (alt_provider >= 0) {
                int alt_idx = pred_hist.indices[alt_provider];
                CortexEntry& alt_entry = tables[alt_provider][alt_idx];
                alt_entry.ctr = std::max(CORTEX_CTR_MIN, std::min(CORTEX_CTR_MAX, 
                                                                  (int)alt_entry.ctr + (taken ? 1 : -1)));
            } else {
                update_bimodal(pc, taken);
            }
        }
    } else {
        update_bimodal(pc, taken);
    }
    
    // Update cluster bias
    int cluster_idx = find_or_create_cluster(pc);
    int8_t& bias = clusters[cluster_idx].shared_bias;
    bias = std::max(CORTEX_SC_MIN, std::min(CORTEX_SC_MAX, (int)bias + (taken ? 1 : -1)));
    
    // Periodic u reset
    tick++;
    if ((tick & ((1ULL << CORTEX_U_RESET_PERIOD) - 1)) == 0) {
        for (int i = 0; i < CORTEX_NUM_TABLES; i++) {
            for (int j = 0; j < CORTEX_TABLE_SIZES[i]; j++) {
                tables[i][j].u >>= 1;
            }
        }
    }
}

// ============================================================================
// Decode/Execute Notifications (for value tracking)
// ============================================================================
void CORTEX_Predictor::notify_decode(uint64_t pc, uint8_t inst_class, 
                                    const std::vector<uint64_t>& src_regs, uint64_t dst_reg) {
    // Update instruction mix
    update_inst_mix(inst_class);
    
    // Track load/store PCs for distance calculation
    if (inst_class == 1) {  // Load
        last_load_pc = pc;
        last_load_distance = 0;
    } else if (inst_class == 2) {  // Store
        last_store_pc = pc;
    }
}

void CORTEX_Predictor::notify_execute(uint64_t pc, uint8_t inst_class, uint64_t dst_reg, 
                                     uint64_t dst_value, uint64_t mem_va) {
    // Update register values
    if (dst_reg < CORTEX_NUM_INT_REGS && inst_class != 6) {
        active_hist.int_regs[dst_reg].value = compute_reg_digest(dst_value, true);
        active_hist.int_regs[dst_reg].age = 0;
    } else if (dst_reg < CORTEX_NUM_FP_REGS && inst_class == 6) {
        active_hist.fp_regs[dst_reg].value = compute_reg_digest(dst_value, false);
        active_hist.fp_regs[dst_reg].age = 0;
    }
    
    // Track load values
    if (inst_class == 1 && active_hist.load_queue_tail < CORTEX_LOAD_QUEUE_SIZE) {
        LoadEntry& entry = active_hist.load_queue[active_hist.load_queue_tail++];
        entry.pc = pc;
        entry.addr = mem_va;
        entry.value = dst_value;
        entry.distance = last_load_distance;
        entry.valid = true;
    }
    
    // Increment distance
    if (last_load_distance < 65535) {
        last_load_distance++;
    }
}
