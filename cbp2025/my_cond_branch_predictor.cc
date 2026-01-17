/*
 * CORTEX Predictor Interface for CBP2025
 * Connects CORTEX predictor to CBP2025 simulation framework
 */

#include "lib/sim_common_structs.h"
#include "cortex_predictor.h"
#include <cassert>

// ============================================================================
// CBP2025 Interface Functions
// ============================================================================

void beginCondDirPredictor() {
    cond_predictor_impl.setup();
}

void notify_instr_fetch(uint64_t seq_no, uint8_t piece, uint64_t pc, const uint64_t fetch_cycle) {
    // Not used in CORTEX
}

bool get_cond_dir_prediction(uint64_t seq_no, uint8_t piece, uint64_t pc, const uint64_t pred_cycle) {
    return cond_predictor_impl.predict(seq_no, piece, pc, false);
}

void spec_update(uint64_t seq_no, uint8_t piece, uint64_t pc, InstClass inst_class, 
                const bool resolve_dir, const bool pred_dir, const uint64_t next_pc) {
    assert(is_br(inst_class));
    
    // Update history for all branches (conditional and unconditional)
    if (inst_class == InstClass::condBranchInstClass) {
        cond_predictor_impl.history_update(seq_no, piece, pc, resolve_dir, next_pc);
    } else {
        // Unconditional branches: update with taken=true
        cond_predictor_impl.history_update(seq_no, piece, pc, true, next_pc);
    }
}

void notify_instr_decode(uint64_t seq_no, uint8_t piece, uint64_t pc, 
                        const DecodeInfo& _decode_info, const uint64_t decode_cycle) {
    // Extract register information
    std::vector<uint64_t> src_regs = _decode_info.src_reg_info;
    uint64_t dst_reg = _decode_info.dst_reg_info.value_or(UINT64_MAX);
    uint8_t inst_class = static_cast<uint8_t>(_decode_info.insn_class);
    
    // Notify CORTEX for value tracking
    cond_predictor_impl.notify_decode(pc, inst_class, src_regs, dst_reg);
}

void notify_agen_complete(uint64_t seq_no, uint8_t piece, uint64_t pc, 
                         const DecodeInfo& _decode_info, const uint64_t mem_va, 
                         const uint64_t mem_sz, const uint64_t agen_cycle) {
    // Not used in CORTEX (we use execute notification instead)
}

void notify_instr_execute_resolve(uint64_t seq_no, uint8_t piece, uint64_t pc, 
                                  const bool pred_dir, const ExecuteInfo& _exec_info, 
                                  const uint64_t execute_cycle) {
    const bool is_branch = is_br(_exec_info.dec_info.insn_class);
    
    if (is_branch && is_cond_br(_exec_info.dec_info.insn_class)) {
        // Update predictor for conditional branches
        const bool resolve_dir = _exec_info.taken.value();
        const uint64_t next_pc = _exec_info.next_pc;
        cond_predictor_impl.update(seq_no, piece, pc, resolve_dir, pred_dir, next_pc);
    }
    
    // Track register/memory values for all instructions
    uint8_t inst_class = static_cast<uint8_t>(_exec_info.dec_info.insn_class);
    uint64_t dst_reg = _exec_info.dec_info.dst_reg_info.value_or(UINT64_MAX);
    uint64_t dst_value = _exec_info.dst_reg_value.value_or(0);
    uint64_t mem_va = _exec_info.mem_va.value_or(0);
    
    if (dst_reg != UINT64_MAX || is_mem(static_cast<InstClass>(inst_class))) {
        cond_predictor_impl.notify_execute(pc, inst_class, dst_reg, dst_value, mem_va);
    }
}

void notify_instr_commit(uint64_t seq_no, uint8_t piece, uint64_t pc, 
                        const bool pred_dir, const ExecuteInfo& _exec_info, 
                        const uint64_t commit_cycle) {
    // Not used in CORTEX (we update at execute)
}

void endCondDirPredictor() {
    cond_predictor_impl.terminate();
}
