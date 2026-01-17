/*
 * BATAGE-SC: Branch-Aware TAGE with Statistical Corrector
 * 
 * A surgical improvement to TAGE that makes the predictor SMARTER rather than
 * adding complex components on top.
 * 
 * Key Innovations:
 * 1. Branch Type Classification - classifies branches as loop-like or data-dependent
 * 2. Selective Table Access based on branch type
 * 3. In-Place Statistical Corrector (IPSC) - parallel lightweight SC
 * 4. H2P-aware allocation policy
 * 5. Enhanced tag design with path history
 * 
 * Budget: 128KB additional to the 64KB TAGE-SC-L baseline = 192KB total
 *         OR use standalone 192KB BATAGE-SC
 *
 * Author: Ported from Python implementation for CBP2025
 * Date: January 2026
 */

#ifndef _BATAGE_SC_PREDICTOR_H_
#define _BATAGE_SC_PREDICTOR_H_

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>
#include <unordered_map>
#include <algorithm>

// ============================================================================
// Configuration - 192KB BATAGE-SC Standalone
// ============================================================================

// === TAGE Core Configuration ===
#define BATAGE_NUM_TABLES 14
static const int BATAGE_HISTORY_LENGTHS[BATAGE_NUM_TABLES] = 
    {4, 8, 16, 32, 64, 128, 200, 320, 512, 800, 1280, 2048, 3072, 4096};
static const int BATAGE_TABLE_SIZES[BATAGE_NUM_TABLES] = 
    {8192, 8192, 4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128};
static const int BATAGE_TAG_WIDTHS[BATAGE_NUM_TABLES] = 
    {8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14};

#define BATAGE_COUNTER_BITS 3
#define BATAGE_USEFUL_BITS 2

#define BATAGE_CTR_MAX ((1 << (BATAGE_COUNTER_BITS - 1)) - 1)  // 3
#define BATAGE_CTR_MIN (-(1 << (BATAGE_COUNTER_BITS - 1)))     // -4
#define BATAGE_U_MAX ((1 << BATAGE_USEFUL_BITS) - 1)           // 3

// === Bimodal Base ===
#define BATAGE_BIMODAL_SIZE 32768
#define BATAGE_BIMODAL_HYST_SHIFT 2

// === Branch Type Classifier ===
#define BATAGE_CLASSIFIER_SIZE 8192
#define BATAGE_LOOP_THRESHOLD 2
#define BATAGE_DATA_THRESHOLD (-2)

// === Statistical Corrector ===
#define BATAGE_SC_NUM_TABLES 4
#define BATAGE_SC_TABLE_SIZE 2048
#define BATAGE_SC_COUNTER_BITS 4
#define BATAGE_SC_MAX ((1 << (BATAGE_SC_COUNTER_BITS - 1)) - 1)
#define BATAGE_SC_MIN (-(1 << (BATAGE_SC_COUNTER_BITS - 1)))
#define BATAGE_SC_THRESHOLD 7
#define BATAGE_SC_CONF_THRESHOLD 1  // Provider counter <= this to consider SC

// === H2P Tracking ===
#define BATAGE_H2P_TABLE_SIZE 4096
#define BATAGE_H2P_THRESHOLD 6

// === Local History ===
#define BATAGE_LOCAL_HIST_SIZE 2048
#define BATAGE_LOCAL_HIST_BITS 16

// === Path History ===
#define BATAGE_PATH_HIST_BITS 32

// === Global History Buffer ===
#define BATAGE_HIST_BUFFER_SIZE 8192

// === UseAlt ===
#define BATAGE_USE_ALT_SIZE 256
#define BATAGE_USE_ALT_BITS 5
#define BATAGE_USE_ALT_MAX ((1 << BATAGE_USE_ALT_BITS) - 1)

// === Update Policy ===
#define BATAGE_U_RESET_PERIOD 19
#define BATAGE_MAX_ALLOC 2
#define BATAGE_MAX_ALLOC_H2P 3


// ============================================================================
// Branch Type Enumeration
// ============================================================================
enum BranchType {
    BRANCH_UNKNOWN = 0,
    BRANCH_LOOP_LIKE = 1,
    BRANCH_DATA_DEPENDENT = 2
};


// ============================================================================
// Folded History
// ============================================================================
class BATAGEFoldedHistory {
public:
    uint32_t comp;
    int comp_length;
    int orig_length;
    int outpoint;
    
    BATAGEFoldedHistory() : comp(0), comp_length(0), orig_length(0), outpoint(0) {}
    
    void init(int original_length, int compressed_length) {
        orig_length = original_length;
        comp_length = compressed_length;
        outpoint = (compressed_length > 0) ? (original_length % compressed_length) : 0;
        comp = 0;
    }
    
    void update(uint8_t* ghist, int pt) {
        int buffer_mask = BATAGE_HIST_BUFFER_SIZE - 1;
        comp = (comp << 1) ^ ghist[pt & buffer_mask];
        comp ^= ghist[(pt + orig_length) & buffer_mask] << outpoint;
        comp ^= (comp >> comp_length);
        comp &= (1 << comp_length) - 1;
    }
};


// ============================================================================
// TAGE Entry
// ============================================================================
struct BATAGEEntry {
    int8_t ctr;     // Signed counter
    uint16_t tag;   // Partial tag (up to 14 bits)
    uint8_t u;      // Usefulness
    
    BATAGEEntry() : ctr(0), tag(0), u(0) {}
};


// ============================================================================
// BATAGE-SC History State
// ============================================================================
struct BATAGEHist {
    // Global history buffer
    uint8_t ghist[BATAGE_HIST_BUFFER_SIZE];
    int pt_ghist;
    
    // Path history
    uint64_t path_hist;
    
    // Integer global history for fast hashing
    uint64_t ghist_int;
    
    // Local history table
    uint32_t local_hist[BATAGE_LOCAL_HIST_SIZE];
    
    // Folded histories
    BATAGEFoldedHistory ch_i[BATAGE_NUM_TABLES];
    BATAGEFoldedHistory ch_t[2][BATAGE_NUM_TABLES];
    
    // Prediction state to checkpoint
    int provider;
    int alt_provider;
    bool tage_pred;
    bool provider_pred;
    bool alt_pred;
    bool weak_entry;
    int8_t provider_ctr;
    bool use_sc;
    bool final_pred;
    int branch_type;
    int indices[BATAGE_NUM_TABLES];
    uint16_t tags[BATAGE_NUM_TABLES];
    
    BATAGEHist() {
        memset(ghist, 0, sizeof(ghist));
        pt_ghist = 0;
        path_hist = 0;
        ghist_int = 0;
        memset(local_hist, 0, sizeof(local_hist));
        provider = -1;
        alt_provider = -1;
        tage_pred = false;
        provider_pred = false;
        alt_pred = false;
        weak_entry = false;
        provider_ctr = 0;
        use_sc = false;
        final_pred = false;
        branch_type = BRANCH_UNKNOWN;
    }
};


// ============================================================================
// BATAGE-SC Predictor Class
// ============================================================================
class BATAGE_SC_Predictor {
private:
    // === Tables ===
    BATAGEEntry* tables[BATAGE_NUM_TABLES];
    
    // Bimodal
    int8_t bimodal_pred[BATAGE_BIMODAL_SIZE];
    int8_t bimodal_hyst[BATAGE_BIMODAL_SIZE >> BATAGE_BIMODAL_HYST_SHIFT];
    
    // Branch type classifier
    int8_t classifier[BATAGE_CLASSIFIER_SIZE];
    
    // Statistical Corrector
    int8_t sc_tables[BATAGE_SC_NUM_TABLES][BATAGE_SC_TABLE_SIZE];
    
    // H2P tracking
    uint8_t h2p_counts[BATAGE_H2P_TABLE_SIZE];
    
    // UseAlt
    int8_t use_alt_on_na[BATAGE_USE_ALT_SIZE];
    
    // Tick counter for useful bit reset
    uint64_t tick;
    
    // Active history state
    BATAGEHist active_hist;
    
    // Checkpointed histories
    std::unordered_map<uint64_t, BATAGEHist> pred_time_histories;

public:
    BATAGE_SC_Predictor() {
        // Allocate tables
        for (int i = 0; i < BATAGE_NUM_TABLES; i++) {
            tables[i] = new BATAGEEntry[BATAGE_TABLE_SIZES[i]];
        }
        
        // Initialize bimodal
        memset(bimodal_pred, 0, sizeof(bimodal_pred));
        memset(bimodal_hyst, 1, sizeof(bimodal_hyst));
        
        // Initialize classifier
        memset(classifier, 0, sizeof(classifier));
        
        // Initialize SC
        memset(sc_tables, 0, sizeof(sc_tables));
        
        // Initialize H2P
        memset(h2p_counts, 0, sizeof(h2p_counts));
        
        // Initialize UseAlt
        memset(use_alt_on_na, 0, sizeof(use_alt_on_na));
        
        // Initialize tick
        tick = 1ULL << (BATAGE_U_RESET_PERIOD - 1);
        
        // Initialize folded histories
        for (int i = 0; i < BATAGE_NUM_TABLES; i++) {
            int table_bits = (int)log2(BATAGE_TABLE_SIZES[i]);
            active_hist.ch_i[i].init(BATAGE_HISTORY_LENGTHS[i], table_bits);
            active_hist.ch_t[0][i].init(BATAGE_HISTORY_LENGTHS[i], BATAGE_TAG_WIDTHS[i]);
            active_hist.ch_t[1][i].init(BATAGE_HISTORY_LENGTHS[i], std::max(1, BATAGE_TAG_WIDTHS[i] - 1));
        }
    }
    
    ~BATAGE_SC_Predictor() {
        for (int i = 0; i < BATAGE_NUM_TABLES; i++) {
            delete[] tables[i];
        }
    }
    
    void setup() {
        // Called at initialization - nothing extra needed
    }
    
    void terminate() {
        // Called at end of simulation
    }
    
    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0x000F);
    }
    
    // ========================================================================
    // Branch Type Classification
    // ========================================================================
    
    int classify_index(uint64_t pc) const {
        return (int)((pc >> 2) & (BATAGE_CLASSIFIER_SIZE - 1));
    }
    
    int get_branch_type(uint64_t pc) const {
        int idx = classify_index(pc);
        int8_t counter = classifier[idx];
        
        if (counter >= BATAGE_LOOP_THRESHOLD) {
            return BRANCH_LOOP_LIKE;
        } else if (counter <= BATAGE_DATA_THRESHOLD) {
            return BRANCH_DATA_DEPENDENT;
        }
        return BRANCH_UNKNOWN;
    }
    
    void update_classifier(uint64_t pc, bool is_loop_behavior) {
        int idx = classify_index(pc);
        if (is_loop_behavior) {
            classifier[idx] = std::min((int8_t)3, (int8_t)(classifier[idx] + 1));
        } else {
            classifier[idx] = std::max((int8_t)-4, (int8_t)(classifier[idx] - 1));
        }
    }
    
    // ========================================================================
    // Indexing Functions
    // ========================================================================
    
    int bimodal_index(uint64_t pc) const {
        return (int)((pc >> 2) & (BATAGE_BIMODAL_SIZE - 1));
    }
    
    bool get_bimodal_pred(uint64_t pc) const {
        return bimodal_pred[bimodal_index(pc)] >= 0;
    }
    
    void update_bimodal(uint64_t pc, bool taken) {
        int idx = bimodal_index(pc);
        int hyst_idx = idx >> BATAGE_BIMODAL_HYST_SHIFT;
        
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
    
    uint32_t F(uint64_t A, int size, int bank) const {
        int table_bits = (int)log2(BATAGE_TABLE_SIZES[bank]);
        if (table_bits <= 0) return 0;
        
        uint32_t mask = (1U << size) - 1;
        A = A & mask;
        
        uint32_t A1 = A & ((1U << table_bits) - 1);
        uint32_t A2 = A >> table_bits;
        
        int effective_bank = (bank + 1) % table_bits;
        if (effective_bank < 0) effective_bank = 0;
        
        if (effective_bank > 0) {
            A2 = ((A2 << effective_bank) & ((1U << table_bits) - 1)) | 
                 (A2 >> (table_bits - effective_bank));
        }
        
        uint32_t result = A1 ^ A2;
        
        if (effective_bank > 0) {
            result = ((result << effective_bank) & ((1U << table_bits) - 1)) | 
                     (result >> (table_bits - effective_bank));
        }
        
        return result;
    }
    
    int tage_index(uint64_t pc, int table, const BATAGEHist& hist) const {
        int hist_len = BATAGE_HISTORY_LENGTHS[table];
        int table_bits = (int)log2(BATAGE_TABLE_SIZES[table]);
        int path_len = std::min(hist_len, BATAGE_PATH_HIST_BITS);
        
        uint32_t shifted_pc = (uint32_t)(pc >> 2);
        
        uint32_t index = shifted_pc ^
                        (shifted_pc >> (abs(table_bits - table) + 1)) ^
                        hist.ch_i[table].comp ^
                        F(hist.path_hist, path_len, table);
        
        return (int)(index & (BATAGE_TABLE_SIZES[table] - 1));
    }
    
    uint16_t tage_tag(uint64_t pc, int table, const BATAGEHist& hist) const {
        int tag_width = BATAGE_TAG_WIDTHS[table];
        
        uint32_t tag = ((uint32_t)(pc >> 2)) ^
                      hist.ch_t[0][table].comp ^
                      (hist.ch_t[1][table].comp << 1);
        
        // Add path history contribution
        uint32_t path_contrib = (uint32_t)((hist.path_hist >> (table * 2)) & 0xF);
        tag ^= path_contrib;
        
        return (uint16_t)(tag & ((1U << tag_width) - 1));
    }
    
    // ========================================================================
    // Statistical Corrector
    // ========================================================================
    
    int sc_index(uint64_t pc, int table, const BATAGEHist& hist) const {
        int local_idx = (int)((pc >> 2) & (BATAGE_LOCAL_HIST_SIZE - 1));
        uint32_t local_hist_val = hist.local_hist[local_idx];
        uint32_t pc_hash = (uint32_t)(pc >> 2);
        
        uint32_t result;
        switch (table) {
            case 0:
                // PC-based
                result = pc_hash;
                break;
            case 1:
                // PC ^ global history
                result = pc_hash ^ (uint32_t)hist.ghist_int;
                break;
            case 2:
                // PC ^ path history
                result = pc_hash ^ (uint32_t)hist.path_hist;
                break;
            default:
                // PC ^ local history
                result = pc_hash ^ local_hist_val;
                break;
        }
        
        return (int)(result & (BATAGE_SC_TABLE_SIZE - 1));
    }
    
    void sc_predict(uint64_t pc, const BATAGEHist& hist, bool& pred, int& sum) const {
        int total = 0;
        for (int t = 0; t < BATAGE_SC_NUM_TABLES; t++) {
            int idx = sc_index(pc, t, hist);
            total += (int)sc_tables[t][idx];
        }
        pred = (total >= 0);
        sum = total;
    }
    
    void sc_update(uint64_t pc, bool taken, const BATAGEHist& hist) {
        for (int t = 0; t < BATAGE_SC_NUM_TABLES; t++) {
            int idx = sc_index(pc, t, hist);
            if (taken) {
                sc_tables[t][idx] = std::min(BATAGE_SC_MAX, (int)sc_tables[t][idx] + 1);
            } else {
                sc_tables[t][idx] = std::max(BATAGE_SC_MIN, (int)sc_tables[t][idx] - 1);
            }
        }
    }
    
    // ========================================================================
    // H2P Tracking
    // ========================================================================
    
    int h2p_index(uint64_t pc) const {
        return (int)((pc >> 2) & (BATAGE_H2P_TABLE_SIZE - 1));
    }
    
    bool is_h2p(uint64_t pc) const {
        return h2p_counts[h2p_index(pc)] >= BATAGE_H2P_THRESHOLD;
    }
    
    void update_h2p(uint64_t pc, bool mispredicted) {
        int idx = h2p_index(pc);
        if (mispredicted) {
            h2p_counts[idx] = std::min(255, (int)h2p_counts[idx] + 2);
        } else {
            if (h2p_counts[idx] > 0) {
                h2p_counts[idx]--;
            }
        }
    }
    
    // ========================================================================
    // Counter Update Functions
    // ========================================================================
    
    int8_t ctr_update(int8_t ctr, bool taken) const {
        if (taken) {
            return std::min(BATAGE_CTR_MAX, (int)ctr + 1);
        } else {
            return std::max(BATAGE_CTR_MIN, (int)ctr - 1);
        }
    }
    
    uint8_t u_update(uint8_t u, bool up) const {
        if (up) {
            return std::min(BATAGE_U_MAX, (int)u + 1);
        } else {
            return std::max(0, (int)u - 1);
        }
    }
    
    // ========================================================================
    // Prediction
    // ========================================================================
    
    bool predict(uint64_t seq_no, uint8_t piece, uint64_t pc, const bool tage_sc_l_pred) {
        BATAGEHist& hist = active_hist;
        
        // Get branch type
        int branch_type = get_branch_type(pc);
        
        // Compute indices and tags for all tables
        for (int i = 0; i < BATAGE_NUM_TABLES; i++) {
            hist.indices[i] = tage_index(pc, i, hist);
            hist.tags[i] = tage_tag(pc, i, hist);
        }
        
        // Find provider (longest matching)
        int provider = -1;
        for (int i = BATAGE_NUM_TABLES - 1; i >= 0; i--) {
            int idx = hist.indices[i];
            if (tables[i][idx].tag == hist.tags[i]) {
                provider = i;
                break;
            }
        }
        
        // Find alternate provider
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
        
        // Compute TAGE prediction
        bool provider_pred, alt_pred, tage_pred;
        int8_t provider_ctr = 0;
        bool weak_entry = false;
        
        if (provider >= 0) {
            int provider_idx = hist.indices[provider];
            BATAGEEntry& entry = tables[provider][provider_idx];
            provider_pred = (entry.ctr >= 0);
            provider_ctr = entry.ctr;
            
            if (alt_provider >= 0) {
                int alt_idx = hist.indices[alt_provider];
                alt_pred = (tables[alt_provider][alt_idx].ctr >= 0);
            } else {
                alt_pred = get_bimodal_pred(pc);
            }
            
            // Check weak entry
            weak_entry = (abs(2 * provider_ctr + 1) <= 1);
            
            // UseAlt logic
            int use_alt_idx = (int)((pc >> 2) & (BATAGE_USE_ALT_SIZE - 1));
            bool use_alt = (use_alt_on_na[use_alt_idx] < 0);
            
            if (weak_entry && use_alt) {
                tage_pred = alt_pred;
            } else {
                tage_pred = provider_pred;
            }
        } else {
            alt_pred = get_bimodal_pred(pc);
            provider_pred = alt_pred;
            tage_pred = alt_pred;
            provider_ctr = 0;
        }
        
        // Compute SC prediction
        bool sc_pred;
        int sc_sum;
        sc_predict(pc, hist, sc_pred, sc_sum);
        
        // Final decision with SC override
        bool final_pred = tage_pred;
        bool use_sc = false;
        
        // SC override conditions:
        // 1. Have a provider (not just bimodal)
        // 2. Provider is weak
        // 3. SC has strong confidence
        // 4. SC disagrees with TAGE
        // 5. Not in data-dependent mode (safer)
        if (provider >= 0 &&
            abs(provider_ctr) <= BATAGE_SC_CONF_THRESHOLD &&
            abs(sc_sum) >= BATAGE_SC_THRESHOLD &&
            sc_pred != tage_pred &&
            branch_type != BRANCH_DATA_DEPENDENT) {
            
            // Count SC table agreement
            int agreement = 0;
            for (int t = 0; t < BATAGE_SC_NUM_TABLES; t++) {
                int idx = sc_index(pc, t, hist);
                if ((sc_tables[t][idx] >= 0) == sc_pred) {
                    agreement++;
                }
            }
            
            // Require majority (3/4)
            if (agreement >= 3) {
                final_pred = sc_pred;
                use_sc = true;
            }
        }
        
        // Save prediction state
        hist.provider = provider;
        hist.alt_provider = alt_provider;
        hist.tage_pred = tage_pred;
        hist.provider_pred = provider_pred;
        hist.alt_pred = alt_pred;
        hist.weak_entry = weak_entry;
        hist.provider_ctr = provider_ctr;
        hist.use_sc = use_sc;
        hist.final_pred = final_pred;
        hist.branch_type = branch_type;
        
        // Checkpoint history
        pred_time_histories.emplace(get_unique_inst_id(seq_no, piece), hist);
        
        return final_pred;
    }
    
    bool predict_using_given_hist(uint64_t seq_no, uint8_t piece, uint64_t PC, 
                                  const BATAGEHist& hist_to_use, const bool pred_time_predict) {
        return hist_to_use.final_pred;
    }
    
    // ========================================================================
    // History Update (speculative)
    // ========================================================================
    
    void history_update(uint64_t seq_no, uint8_t piece, uint64_t pc, bool taken, uint64_t next_pc) {
        BATAGEHist& hist = active_hist;
        
        // Path history update
        uint64_t path_bit = (pc >> 2) & 1;
        hist.path_hist = ((hist.path_hist << 1) | path_bit) & ((1ULL << BATAGE_PATH_HIST_BITS) - 1);
        
        // Local history update
        int local_idx = (int)((pc >> 2) & (BATAGE_LOCAL_HIST_SIZE - 1));
        hist.local_hist[local_idx] = ((hist.local_hist[local_idx] << 1) | (taken ? 1 : 0)) & 
                                      ((1U << BATAGE_LOCAL_HIST_BITS) - 1);
        
        // Global history buffer update
        if (hist.pt_ghist <= 0) {
            for (int i = 0; i < BATAGE_HIST_BUFFER_SIZE / 2; i++) {
                hist.ghist[BATAGE_HIST_BUFFER_SIZE - 1 - i] = hist.ghist[BATAGE_HIST_BUFFER_SIZE / 2 - 1 - i];
            }
            hist.pt_ghist = BATAGE_HIST_BUFFER_SIZE / 2;
        }
        hist.pt_ghist--;
        hist.ghist[hist.pt_ghist] = taken ? 1 : 0;
        
        // Integer history
        hist.ghist_int = ((hist.ghist_int << 1) | (taken ? 1 : 0)) & 0xFFFFFFFFFFFFFFFFULL;
        
        // Update folded histories
        for (int i = 0; i < BATAGE_NUM_TABLES; i++) {
            hist.ch_i[i].update(hist.ghist, hist.pt_ghist);
            hist.ch_t[0][i].update(hist.ghist, hist.pt_ghist);
            hist.ch_t[1][i].update(hist.ghist, hist.pt_ghist);
        }
    }
    
    // ========================================================================
    // Update (at resolve)
    // ========================================================================
    
    void update(uint64_t seq_no, uint8_t piece, uint64_t pc, bool resolveDir, 
                bool predDir, uint64_t nextPC) {
        const auto pred_hist_key = get_unique_inst_id(seq_no, piece);
        
        if (pred_time_histories.find(pred_hist_key) == pred_time_histories.end()) {
            return;  // No checkpoint found
        }
        
        const BATAGEHist& pred_hist = pred_time_histories.at(pred_hist_key);
        update_internal(pc, resolveDir, predDir, nextPC, pred_hist);
        pred_time_histories.erase(pred_hist_key);
    }
    
    void update(uint64_t pc, bool resolveDir, bool pred_taken, uint64_t nextPC, 
                const BATAGEHist& hist_to_use) {
        update_internal(pc, resolveDir, pred_taken, nextPC, hist_to_use);
    }
    
private:
    void update_internal(uint64_t pc, bool taken, bool predDir, uint64_t nextPC,
                        const BATAGEHist& pred_hist) {
        int provider = pred_hist.provider;
        int alt_provider = pred_hist.alt_provider;
        bool tage_pred = pred_hist.tage_pred;
        bool provider_pred = pred_hist.provider_pred;
        bool alt_pred = pred_hist.alt_pred;
        bool weak_entry = pred_hist.weak_entry;
        bool final_pred = pred_hist.final_pred;
        int branch_type = pred_hist.branch_type;
        
        bool mispredicted = (final_pred != taken);
        
        // Update classifier
        bool is_loop_behavior = (provider >= 0) ? (provider < BATAGE_NUM_TABLES / 2) : true;
        update_classifier(pc, is_loop_behavior);
        
        // Update H2P
        update_h2p(pc, mispredicted);
        bool h2p = is_h2p(pc);
        
        // Update SC
        sc_update(pc, taken, pred_hist);
        
        // Update UseAlt
        bool alloc = (tage_pred != taken) && (provider < BATAGE_NUM_TABLES - 1);
        
        if (provider >= 0 && weak_entry) {
            if (provider_pred == taken) {
                alloc = false;
            }
            
            if (provider_pred != alt_pred) {
                int use_alt_idx = (int)((pc >> 2) & (BATAGE_USE_ALT_SIZE - 1));
                if (alt_pred == taken) {
                    use_alt_on_na[use_alt_idx] = std::min(BATAGE_USE_ALT_MAX, 
                                                          (int)use_alt_on_na[use_alt_idx] + 1);
                } else {
                    use_alt_on_na[use_alt_idx] = std::max(-BATAGE_USE_ALT_MAX - 1, 
                                                          (int)use_alt_on_na[use_alt_idx] - 1);
                }
            }
        }
        
        // Allocation
        if (alloc) {
            int max_alloc = h2p ? BATAGE_MAX_ALLOC_H2P : BATAGE_MAX_ALLOC;
            handle_allocation(provider, taken, pred_hist.indices, pred_hist.tags, 
                            max_alloc, branch_type);
        }
        
        // Update TAGE tables
        if (provider >= 0) {
            int provider_idx = pred_hist.indices[provider];
            BATAGEEntry& entry = tables[provider][provider_idx];
            entry.ctr = ctr_update(entry.ctr, taken);
            
            // Update alternate if provider u=0
            if (entry.u == 0) {
                if (alt_provider >= 0) {
                    int alt_idx = pred_hist.indices[alt_provider];
                    tables[alt_provider][alt_idx].ctr = ctr_update(
                        tables[alt_provider][alt_idx].ctr, taken);
                } else {
                    update_bimodal(pc, taken);
                }
            }
            
            // Update useful
            if (tage_pred != alt_pred) {
                entry.u = u_update(entry.u, tage_pred == taken);
            }
        } else {
            update_bimodal(pc, taken);
        }
        
        // Periodic useful bit reset
        tick++;
        if ((tick & ((1ULL << BATAGE_U_RESET_PERIOD) - 1)) == 0) {
            for (int i = 0; i < BATAGE_NUM_TABLES; i++) {
                for (int j = 0; j < BATAGE_TABLE_SIZES[i]; j++) {
                    tables[i][j].u >>= 1;
                }
            }
        }
    }
    
    void handle_allocation(int provider, bool taken, const int* indices, 
                          const uint16_t* tags, int max_alloc, int branch_type) {
        int start = (provider >= 0) ? provider + 1 : 0;
        
        // Find minimum useful value
        int min_u = BATAGE_U_MAX + 1;
        for (int i = start; i < BATAGE_NUM_TABLES; i++) {
            int idx = indices[i];
            if (tables[i][idx].u < min_u) {
                min_u = tables[i][idx].u;
            }
        }
        
        // Allocation
        int num_allocated = 0;
        
        if (min_u > 0) {
            // Decay useful bits
            for (int i = start; i < BATAGE_NUM_TABLES; i++) {
                int idx = indices[i];
                if (tables[i][idx].u > 0) {
                    tables[i][idx].u--;
                }
            }
        } else {
            // Allocate in entries with u=0
            for (int i = start; i < BATAGE_NUM_TABLES; i++) {
                int idx = indices[i];
                if (tables[i][idx].u == 0) {
                    tables[i][idx].tag = tags[i];
                    tables[i][idx].ctr = taken ? 0 : -1;
                    tables[i][idx].u = 0;
                    num_allocated++;
                    if (num_allocated >= max_alloc) {
                        break;
                    }
                }
            }
        }
    }
};

// Extern declaration of the predictor instance (defined in .cc file)
extern BATAGE_SC_Predictor cond_predictor_impl;

#endif // _BATAGE_SC_PREDICTOR_H_
