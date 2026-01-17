/*
 * CORTEX: COrrelation-Rich TEXture Predictor for CBP2025
 * 
 * NOVEL APPROACH combining insights from RUNLTS and LVC-TAGE-SC-L winners
 * with original micro-architectural execution context tracking.
 * 
 * Key Innovations:
 * 1. Temporal Execution Context Fingerprinting (TECF)
 *    - Value stability tracking (how long values stay unchanged)
 *    - Load-store distance correlation
 *    - Instruction mix fingerprinting
 * 
 * 2. Hybrid Value Correlation
 *    - Register value digests (from RUNLTS)
 *    - Load value correlation (from LVC-TAGE-SC-L)
 *    - NEW: Combined load-register-store patterns
 * 
 * 3. Branch Neighborhood Clustering
 *    - Identify branches that mispredict together
 *    - Share learning across similar branches
 * 
 * 4. Confidence Calibration Network
 *    - Dynamically adjust SC threshold per branch
 *    - Track prediction confidence accuracy
 * 
 * 5. Advanced Allocation with Multi-Modal Learning
 *    - Separate allocators for different branch behaviors
 *    - Fast-track H2P branches with different structures
 * 
 * Budget: 192KB total
 *
 * Author: Advanced Branch Prediction Research
 * Date: January 2026
 */

#ifndef _CORTEX_PREDICTOR_H_
#define _CORTEX_PREDICTOR_H_

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <queue>

// ============================================================================
// Configuration - 192KB CORTEX
// ============================================================================

// === Base TAGE Configuration (optimized) ===
#define CORTEX_NUM_TABLES 16
static const int CORTEX_HISTORY_LENGTHS[CORTEX_NUM_TABLES] = 
    {5, 10, 18, 30, 46, 66, 94, 134, 190, 270, 384, 546, 776, 1104, 1570, 3600};
static const int CORTEX_TABLE_SIZES[CORTEX_NUM_TABLES] = 
    {4096, 4096, 3072, 3072, 2048, 2048, 1536, 1536, 1024, 1024, 768, 768, 512, 512, 384, 256};
static const int CORTEX_TAG_WIDTHS[CORTEX_NUM_TABLES] = 
    {9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 14, 14};

#define CORTEX_COUNTER_BITS 3
#define CORTEX_USEFUL_BITS 2
#define CORTEX_CTR_MAX ((1 << (CORTEX_COUNTER_BITS - 1)) - 1)
#define CORTEX_CTR_MIN (-(1 << (CORTEX_COUNTER_BITS - 1)))
#define CORTEX_U_MAX ((1 << CORTEX_USEFUL_BITS) - 1)

// === Large Bimodal ===
#define CORTEX_BIMODAL_SIZE 131072  // 128K entries like RUNLTS
#define CORTEX_BIMODAL_HYST_SHIFT 2

// === Register Value Correlation (inspired by RUNLTS) ===
#define CORTEX_NUM_INT_REGS 32
#define CORTEX_NUM_FP_REGS 32
#define CORTEX_REG_DIGEST_BITS 12
#define CORTEX_REG_BANKS 8
#define CORTEX_REG_TABLE_SIZE 512
#define CORTEX_REG_USEFUL_SIZE 8

// === Load Value Correlation (inspired by LVC-TAGE-SC-L) ===
#define CORTEX_LOAD_QUEUE_SIZE 20  // Deeper than LVC's 16
#define CORTEX_DISTANT_LOAD_SIZE 128
#define CORTEX_LOAD_MARKING_SIZE 2048
#define CORTEX_LOAD_CORR_SIZE 4096
#define CORTEX_LOAD_CORR_TAG_BITS 12

// === NEW: Temporal Execution Context Fingerprinting ===
#define CORTEX_VALUE_STABILITY_SIZE 1024
#define CORTEX_LS_DISTANCE_SIZE 512
#define CORTEX_INST_MIX_SIZE 256
#define CORTEX_INST_MIX_WINDOW 16  // Track last 16 instructions

// === NEW: Branch Neighborhood Clustering ===
#define CORTEX_CLUSTER_SIZE 512
#define CORTEX_CLUSTER_MEMBERS 8
#define CORTEX_CLUSTER_SIGNATURE_BITS 16

// === NEW: Confidence Calibration ===
#define CORTEX_CALIB_SIZE 2048
#define CORTEX_CALIB_HISTORY 16  // Track last 16 predictions per branch

// === Statistical Corrector (Enhanced) ===
#define CORTEX_SC_BANKS 6
#define CORTEX_SC_BANK_SIZE 2048
#define CORTEX_SC_WEIGHT_BITS 5
#define CORTEX_SC_MAX ((1 << (CORTEX_SC_WEIGHT_BITS - 1)) - 1)
#define CORTEX_SC_MIN (-(1 << (CORTEX_SC_WEIGHT_BITS - 1)))

// === IMLI (with higher weight like RUNLTS) ===
#define CORTEX_IMLI_SIZE 512
#define CORTEX_IMLI_OH_SIZE 512
#define CORTEX_IMLI_WEIGHT 3  // Even higher than RUNLTS's 2

// === H2P Tracking ===
#define CORTEX_H2P_SIZE 8192
#define CORTEX_H2P_THRESHOLD 8

// === Local/Path History ===
#define CORTEX_LOCAL_SIZE 4096
#define CORTEX_LOCAL_BITS 18
#define CORTEX_PATH_BITS 48

// === UseAlt ===
#define CORTEX_USE_ALT_SIZE 512
#define CORTEX_USE_ALT_BITS 6

// === Allocation ===
#define CORTEX_U_RESET_PERIOD 18
#define CORTEX_MAX_ALLOC_NORMAL 2
#define CORTEX_MAX_ALLOC_H2P 4  // More aggressive for H2P

// === History Buffer ===
#define CORTEX_HIST_SIZE 8192

// ============================================================================
// NEW: Execution Context Structures
// ============================================================================

// Register value digest (compact representation)
struct RegDigest {
    uint16_t value : CORTEX_REG_DIGEST_BITS;
    uint8_t age : 4;  // How many instructions since last update
    
    RegDigest() : value(0), age(15) {}
};

// Load tracking entry
struct LoadEntry {
    uint64_t pc;
    uint64_t addr;
    uint64_t value;
    uint16_t distance;  // Instructions since load
    bool valid;
    
    LoadEntry() : pc(0), addr(0), value(0), distance(0), valid(false) {}
};

// Instruction mix fingerprint
struct InstMixFingerprint {
    uint8_t alu_count : 4;
    uint8_t mem_count : 4;
    uint8_t br_count : 4;
    uint8_t fp_count : 4;
    
    InstMixFingerprint() : alu_count(0), mem_count(0), br_count(0), fp_count(0) {}
    
    uint16_t get_digest() const {
        return ((uint16_t)alu_count << 12) | ((uint16_t)mem_count << 8) | 
               ((uint16_t)br_count << 4) | (uint16_t)fp_count;
    }
};

// Branch cluster for neighborhood learning
struct BranchCluster {
    uint16_t signature;  // Compact representation of branch characteristics
    uint64_t member_pcs[CORTEX_CLUSTER_MEMBERS];
    uint8_t num_members;
    int8_t shared_bias;  // Shared learning across cluster
    
    BranchCluster() : signature(0), num_members(0), shared_bias(0) {
        memset(member_pcs, 0, sizeof(member_pcs));
    }
};

// Confidence calibration entry
struct CalibEntry {
    uint16_t pred_history;  // Last 16 predictions
    uint16_t outcome_history;  // Last 16 outcomes
    int8_t calibration;  // Adjustment to SC threshold
    uint8_t confidence_accuracy;  // How often confident predictions were correct
    
    CalibEntry() : pred_history(0), outcome_history(0), calibration(0), confidence_accuracy(128) {}
};

// ============================================================================
// Folded History
// ============================================================================
class CortexFoldedHistory {
public:
    uint64_t comp;
    int comp_length;
    int orig_length;
    int outpoint;
    
    CortexFoldedHistory() : comp(0), comp_length(0), orig_length(0), outpoint(0) {}
    
    void init(int original_length, int compressed_length) {
        orig_length = original_length;
        comp_length = compressed_length;
        outpoint = (compressed_length > 0) ? (original_length % compressed_length) : 0;
        comp = 0;
    }
    
    void update(uint8_t* ghist, int pt) {
        int buffer_mask = CORTEX_HIST_SIZE - 1;
        comp = (comp << 1) ^ ghist[pt & buffer_mask];
        comp ^= ghist[(pt + orig_length) & buffer_mask] << outpoint;
        comp ^= (comp >> comp_length);
        comp &= (1ULL << comp_length) - 1;
    }
};

// ============================================================================
// TAGE Entry
// ============================================================================
struct CortexEntry {
    int8_t ctr;
    uint16_t tag;
    uint8_t u;
    
    CortexEntry() : ctr(0), tag(0), u(0) {}
};

// ============================================================================
// CORTEX History State
// ============================================================================
struct CortexHist {
    // Global history
    uint8_t ghist[CORTEX_HIST_SIZE];
    int pt_ghist;
    uint64_t ghist_int;
    
    // Path history
    uint64_t path_hist;
    
    // Local history
    uint32_t local_hist[CORTEX_LOCAL_SIZE];
    
    // Folded histories
    CortexFoldedHistory ch_i[CORTEX_NUM_TABLES];
    CortexFoldedHistory ch_t[2][CORTEX_NUM_TABLES];
    
    // Register value tracking
    RegDigest int_regs[CORTEX_NUM_INT_REGS];
    RegDigest fp_regs[CORTEX_NUM_FP_REGS];
    
    // Load tracking
    LoadEntry load_queue[CORTEX_LOAD_QUEUE_SIZE];
    int load_queue_head;
    int load_queue_tail;
    
    // Instruction mix
    InstMixFingerprint inst_mix;
    uint8_t recent_inst_types[CORTEX_INST_MIX_WINDOW];
    int inst_mix_ptr;
    
    // Prediction state
    int provider;
    int alt_provider;
    bool tage_pred;
    bool sc_pred;
    bool final_pred;
    int sc_sum;
    int8_t provider_ctr;
    bool weak_provider;
    bool use_sc;
    int indices[CORTEX_NUM_TABLES];
    uint16_t tags[CORTEX_NUM_TABLES];
    
    CortexHist() {
        memset(ghist, 0, sizeof(ghist));
        pt_ghist = 0;
        ghist_int = 0;
        path_hist = 0;
        memset(local_hist, 0, sizeof(local_hist));
        load_queue_head = 0;
        load_queue_tail = 0;
        inst_mix_ptr = 0;
        memset(recent_inst_types, 0, sizeof(recent_inst_types));
        provider = -1;
        alt_provider = -1;
        tage_pred = false;
        sc_pred = false;
        final_pred = false;
        sc_sum = 0;
        provider_ctr = 0;
        weak_provider = false;
        use_sc = false;
    }
};

// ============================================================================
// CORTEX Predictor Class
// ============================================================================
class CORTEX_Predictor {
private:
    // === TAGE Tables ===
    CortexEntry* tables[CORTEX_NUM_TABLES];
    
    // === Bimodal ===
    int8_t* bimodal_pred;
    int8_t* bimodal_hyst;
    
    // === Register Value Correlation Tables ===
    int8_t reg_useful[CORTEX_REG_BANKS][CORTEX_REG_USEFUL_SIZE][CORTEX_NUM_INT_REGS];
    int8_t reg_pred_table[CORTEX_REG_BANKS][CORTEX_REG_TABLE_SIZE];
    
    // === Load Value Correlation ===
    struct LoadCorrEntry {
        uint16_t tag : CORTEX_LOAD_CORR_TAG_BITS;
        uint8_t conf : 4;
        bool dir : 1;
        bool valid : 1;
    };
    LoadCorrEntry load_corr_table[CORTEX_LOAD_CORR_SIZE];
    
    struct DistantLoad {
        uint64_t pc;
        uint64_t value;
        bool valid;
    };
    DistantLoad distant_loads[CORTEX_DISTANT_LOAD_SIZE];
    
    // === Temporal Execution Context ===
    struct ValueStabilityEntry {
        uint16_t stable_cycles;
        uint8_t change_count;
    };
    ValueStabilityEntry value_stability[CORTEX_VALUE_STABILITY_SIZE];
    
    int8_t ls_distance_table[CORTEX_LS_DISTANCE_SIZE];
    int8_t inst_mix_table[CORTEX_INST_MIX_SIZE];
    
    // === Branch Clustering ===
    BranchCluster clusters[CORTEX_CLUSTER_SIZE];
    
    // === Confidence Calibration ===
    CalibEntry calibration[CORTEX_CALIB_SIZE];
    
    // === Statistical Corrector ===
    int8_t sc_tables[CORTEX_SC_BANKS][CORTEX_SC_BANK_SIZE];
    
    // === IMLI ===
    int8_t imli_table[CORTEX_IMLI_SIZE];
    int8_t imli_oh_table[CORTEX_IMLI_OH_SIZE];
    uint32_t imli_counter;
    uint64_t last_br_pc;
    uint64_t last_target;
    
    // === H2P ===
    uint16_t h2p_counts[CORTEX_H2P_SIZE];
    
    // === UseAlt ===
    int8_t use_alt_on_na[CORTEX_USE_ALT_SIZE];
    
    // === Counters ===
    uint64_t tick;
    uint64_t total_instr_count;
    uint64_t last_load_pc;
    uint64_t last_store_pc;
    uint16_t last_load_distance;
    
    // === Active History ===
    CortexHist active_hist;
    
    // === Checkpointed Histories ===
    std::unordered_map<uint64_t, CortexHist> pred_time_histories;
    
public:
    CORTEX_Predictor();
    ~CORTEX_Predictor();
    
    void setup();
    void terminate();
    
    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const;
    
    // Prediction
    bool predict(uint64_t seq_no, uint8_t piece, uint64_t pc, bool tage_sc_l_pred);
    bool predict_using_given_hist(uint64_t seq_no, uint8_t piece, uint64_t PC, 
                                  const CortexHist& hist_to_use, bool pred_time_predict);
    
    // Updates
    void history_update(uint64_t seq_no, uint8_t piece, uint64_t pc, bool taken, uint64_t next_pc);
    void update(uint64_t seq_no, uint8_t piece, uint64_t pc, bool resolveDir, 
                bool predDir, uint64_t nextPC);
    void update(uint64_t pc, bool resolveDir, bool pred_taken, uint64_t nextPC, 
                const CortexHist& hist_to_use);
    
    // Decode notification (for register/load tracking)
    void notify_decode(uint64_t pc, uint8_t inst_class, 
                      const std::vector<uint64_t>& src_regs,
                      uint64_t dst_reg);
    
    // Execute notification (for value updates)
    void notify_execute(uint64_t pc, uint8_t inst_class, uint64_t dst_reg, 
                       uint64_t dst_value, uint64_t mem_va);
    
private:
    // Helper functions
    uint16_t compute_reg_digest(uint64_t value, bool is_int);
    void update_inst_mix(uint8_t inst_class);
    int find_or_create_cluster(uint64_t pc);
    int get_cluster_index(uint64_t pc);
    void update_calibration(uint64_t pc, bool pred, bool outcome, bool high_conf);
    int get_calibrated_threshold(uint64_t pc);
    
    // Indexing
    int bimodal_index(uint64_t pc) const;
    int tage_index(uint64_t pc, int table, const CortexHist& hist) const;
    uint16_t tage_tag(uint64_t pc, int table, const CortexHist& hist) const;
    int sc_index(uint64_t pc, int bank, const CortexHist& hist) const;
    
    // SC prediction
    void sc_predict(uint64_t pc, const CortexHist& hist, bool& pred, int& sum);
    void sc_update(uint64_t pc, bool taken, const CortexHist& hist);
    
    // H2P
    bool is_h2p(uint64_t pc) const;
    void update_h2p(uint64_t pc, bool mispred);
    
    // Bimodal
    bool get_bimodal_pred(uint64_t pc) const;
    void update_bimodal(uint64_t pc, bool taken);
    
    // Allocation
    void handle_allocation(int provider, bool taken, const CortexHist& hist, bool h2p);
    
    // Main update
    void update_internal(uint64_t pc, bool taken, bool predDir, uint64_t nextPC,
                        const CortexHist& pred_hist);
};

extern CORTEX_Predictor cond_predictor_impl;

#endif // _CORTEX_PREDICTOR_H_
