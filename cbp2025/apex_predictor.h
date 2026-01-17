/*
 * APEX Predictor - Advanced Predictor with EXtended features
 * 
 * High-accuracy branch predictor based on TAGE-SC-L architecture with enhancements:
 * - Large bimodal (256K entries) for improved base prediction
 * - 36 tagged tables with novel history length selection (arithmetic + geometric)
 * - Enhanced IMLI with branch/target variants (brIMLI/tarIMLI)
 * - Multiple local history tables for improved correlation capture
 * - Full statistical corrector with GEHL tables
 * - Loop predictor for fixed-iteration loops
 * - Improved allocation with thrashing detection
 * 
 * Storage budget: ~188.5KB
 */

#ifndef _APEX_PREDICTOR_H_
#define _APEX_PREDICTOR_H_

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>

// ============================================================================
// Configuration
// ============================================================================

// Loop predictor
#define LOGL 8
#define WIDTHNBITERLOOP 10
#define LOOPTAG 10

#define UINT64 uint64_t
#define BORNTICK 1024

// Enable components
#define SC
#define IMLI
#define LOCALH
#define LOOPPREDICTOR
#define LOCALS
#define LOCALT

// Statistical Corrector
#define PERCWIDTH 6
#define LOGBIAS 11

// IMLI configuration  
#define LOGINB 10
#define INB 1
static int Im[INB] = {8};

#define LOGIMNB 11
#define IMNB 2
static int IMm[IMNB] = {10, 4};

// Global GEHL
#define LOGGNB 12
#define GNB 3
static int Gm[GNB] = {40, 24, 10};

// Path GEHL
#define PNB 3
#define LOGPNB 11
static int Pm[PNB] = {25, 16, 9};

// First local
#define LOGLNB 11
#define LNB 3
static int Lm[LNB] = {11, 6, 3};
#define LOGLOCAL 9
#define NLOCAL (1 << LOGLOCAL)

// Second local
#define LOGSNB 10
#define SNB 3
static int Sm[SNB] = {16, 11, 6};
#define LOGSECLOCAL 5
#define NSECLOCAL (1 << LOGSECLOCAL)

// Third local
#define LOGTNB 11
#define TNB 2
static int Tm[TNB] = {9, 4};
#define LOGTLOCAL 5
#define NTLOCAL (1 << LOGTLOCAL)

// Variable threshold
#define VARTHRES
#define WIDTHRES 12
#define WIDTHRESP 8
#define LOGSIZEUP 6
#define LOGSIZEUPS (LOGSIZEUP / 2)
#define EWIDTH 6
#define CONFWIDTH 7

// History buffer
#define HISTBUFFERLENGTH 4096

// TAGE Configuration
#define NHIST 36
#define NBANKLOW 10
#define NBANKHIGH 20
#define BORN 13
#define BORNINFASSOC 9
#define BORNSUPASSOC 23
#define MINHIST 6
#define MAXHIST 3000
#define LOGG 11
#define TBITS 10

#define NNN 1
#define HYSTSHIFT 2
#define LOGB 18  // 256K entries bimodal for large footprints

#define PHISTWIDTH 27
#define UWIDTH 1
#define CWIDTH 3

#define LOGSIZEUSEALT 4
#define ALTWIDTH 5
#define SIZEUSEALT (1 << LOGSIZEUSEALT)

// ============================================================================
// Global Arrays
// ============================================================================

static int8_t Bias[(1 << LOGBIAS)];
static int8_t BiasSK[(1 << LOGBIAS)];
static int8_t BiasBank[(1 << LOGBIAS)];

static int8_t IGEHLA[INB][(1 << LOGINB)] = {{0}};
static int8_t *IGEHL[INB];

static int8_t IMGEHLA[IMNB][(1 << LOGIMNB)] = {{0}};
static int8_t *IMGEHL[IMNB];

static int8_t GGEHLA[GNB][(1 << LOGGNB)] = {{0}};
static int8_t *GGEHL[GNB];

static int8_t PGEHLA[PNB][(1 << LOGPNB)] = {{0}};
static int8_t *PGEHL[PNB];

static int8_t LGEHLA[LNB][(1 << LOGLNB)] = {{0}};
static int8_t *LGEHL[LNB];

static int8_t SGEHLA[SNB][(1 << LOGSNB)] = {{0}};
static int8_t *SGEHL[SNB];

static int8_t TGEHLA[TNB][(1 << LOGTNB)] = {{0}};
static int8_t *TGEHL[TNB];

static int updatethreshold;
static int Pupdatethreshold[(1 << LOGSIZEUP)];

static int8_t WG[(1 << LOGSIZEUPS)];
static int8_t WL[(1 << LOGSIZEUPS)];
static int8_t WS[(1 << LOGSIZEUPS)];
static int8_t WT[(1 << LOGSIZEUPS)];
static int8_t WP[(1 << LOGSIZEUPS)];
static int8_t WI[(1 << LOGSIZEUPS)];
static int8_t WIM[(1 << LOGSIZEUPS)];
static int8_t WB[(1 << LOGSIZEUPS)];

static int8_t FirstH, SecondH;
static bool MedConf;
static bool AltConf;

static int SizeTable[NHIST + 1];
static bool NOSKIP[NHIST + 1];
static int m[NHIST + 1];
static int TB[NHIST + 1];
static int logg[NHIST + 1];
static int8_t use_alt_on_na[SIZEUSEALT];
static int8_t BIM;
static int TICK;
static uint64_t Seed;

// ============================================================================
// Entry Classes
// ============================================================================

class bentry {
public:
    int8_t hyst;
    int8_t pred;
    bentry() : pred(0), hyst(1) {}
};

class gentry {
public:
    int8_t ctr;
    uint16_t tag;
    int8_t u;
    gentry() : ctr(0), u(0), tag(0) {}
};

class lentry {
public:
    uint16_t NbIter;
    uint8_t confid;
    uint16_t CurrentIter;
    uint16_t TAG;
    uint8_t age;
    bool dir;
    lentry() : confid(0), CurrentIter(0), NbIter(0), TAG(0), age(0), dir(false) {}
};

// TAGE tables
static bentry *btable;
static gentry *gtable[NHIST + 1];

// ============================================================================
// Folded History
// ============================================================================

class folded_history {
public:
    unsigned comp;
    int CLENGTH;
    int OLENGTH;
    int OUTPOINT;
    
    folded_history() : comp(0), CLENGTH(0), OLENGTH(0), OUTPOINT(0) {}
    
    void init(int original_length, int compressed_length) {
        comp = 0;
        OLENGTH = original_length;
        CLENGTH = compressed_length;
        OUTPOINT = OLENGTH % CLENGTH;
    }
    
    void update(std::array<uint8_t, HISTBUFFERLENGTH>& h, int PT) {
        comp = (comp << 1) ^ h[PT & (HISTBUFFERLENGTH - 1)];
        comp ^= h[(PT + OLENGTH) & (HISTBUFFERLENGTH - 1)] << OUTPOINT;
        comp ^= (comp >> CLENGTH);
        comp = comp & ((1 << CLENGTH) - 1);
    }
};

using tage_index_t = std::array<folded_history, NHIST + 1>;
using tage_tag_t = std::array<folded_history, NHIST + 1>;

// ============================================================================
// History State
// ============================================================================

struct apex_hist_t {
    uint64_t GHIST;
    std::array<uint8_t, HISTBUFFERLENGTH> ghist;
    uint64_t phist;
    int ptghist;
    tage_index_t ch_i;
    std::array<tage_tag_t, 2> ch_t;
    
    std::array<uint64_t, NLOCAL> L_shist;
    std::array<uint64_t, NSECLOCAL> S_slhist;
    std::array<uint64_t, NTLOCAL> T_slhist;
    
    std::array<uint64_t, 256> IMHIST;
    uint64_t IMLIcount;
    
    // Enhanced IMLI from winners
    uint64_t brIMLI;       // Branch IMLI (same branch PC)
    uint64_t tarIMLI;      // Target IMLI (same taken target)
    uint64_t lastBrPC;
    uint64_t lastTarget;
    
#ifdef LOOPPREDICTOR
    std::vector<lentry> ltable;
    int8_t WITHLOOP;
#endif
    
    apex_hist_t() {
        ghist.fill(0);
        ptghist = 0;
        phist = 0;
        GHIST = 0;
        L_shist.fill(0);
        S_slhist.fill(0);
        T_slhist.fill(0);
        IMHIST.fill(0);
        IMLIcount = 0;
        brIMLI = 0;
        tarIMLI = 0;
        lastBrPC = 0;
        lastTarget = 0;
#ifdef LOOPPREDICTOR
        ltable.resize(1 << LOGL);
        WITHLOOP = -1;
#endif
    }
};

// ============================================================================
// APEX Predictor Class
// ============================================================================

class APEX_Predictor {
public:
    // Prediction state
    int GI[NHIST + 1];
    uint16_t GTAG[NHIST + 1];
    int BI;
    int THRES;
    
    // Loop predictor state
    bool predloop;
    int LIB;
    int LI;
    int LHIT;
    int LTAG;
    bool LVALID;
    
    // TAGE state
    bool tage_pred;
    bool alttaken;
    bool LongestMatchPred;
    int HitBank;
    int AltBank;
    bool pred_inter;
    bool LowConf;
    bool HighConf;
    int LSUM;
    
    apex_hist_t active_hist;
    std::unordered_map<uint64_t, apex_hist_t> pred_time_histories;
    
    APEX_Predictor() {
        init_histories(active_hist);
    }
    
    void setup() {
        predictorsize();
    }
    
    void terminate() {}
    
    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0x000F);
    }
    
    void init_histories(apex_hist_t& current_hist) {
        m[1] = MINHIST;
        m[NHIST / 2] = MAXHIST;
        for (int i = 2; i <= NHIST / 2; i++) {
            m[i] = (int)(((double)MINHIST * 
                   pow((double)(MAXHIST) / (double)MINHIST,
                       (double)(i - 1) / (double)((NHIST / 2) - 1))) + 0.5);
        }
        for (int i = 1; i <= NHIST; i++) {
            NOSKIP[i] = ((i - 1) & 1) || ((i >= BORNINFASSOC) & (i < BORNSUPASSOC));
        }
        
        NOSKIP[4] = 0;
        NOSKIP[NHIST - 2] = 0;
        NOSKIP[8] = 0;
        NOSKIP[NHIST - 6] = 0;
        
        for (int i = NHIST; i > 1; i--) {
            m[i] = m[(i + 1) / 2];
        }
        for (int i = 1; i <= NHIST; i++) {
            TB[i] = TBITS + 4 * (i >= BORN);
            logg[i] = LOGG;
        }
        
        gtable[1] = new gentry[NBANKLOW * (1 << LOGG)];
        SizeTable[1] = NBANKLOW * (1 << LOGG);
        
        gtable[BORN] = new gentry[NBANKHIGH * (1 << LOGG)];
        SizeTable[BORN] = NBANKHIGH * (1 << LOGG);
        
        for (int i = BORN + 1; i <= NHIST; i++)
            gtable[i] = gtable[BORN];
        for (int i = 2; i <= BORN - 1; i++)
            gtable[i] = gtable[1];
        btable = new bentry[1 << LOGB];
        
        for (int i = 1; i <= NHIST; i++) {
            current_hist.ch_i[i].init(m[i], logg[i]);
            current_hist.ch_t[0][i].init(current_hist.ch_i[i].OLENGTH, TB[i]);
            current_hist.ch_t[1][i].init(current_hist.ch_i[i].OLENGTH, TB[i] - 1);
        }
        
        LVALID = false;
        Seed = 0;
        TICK = 0;
        current_hist.phist = 0;
        
        for (int i = 0; i < HISTBUFFERLENGTH; i++)
            current_hist.ghist[i] = 0;
        current_hist.ptghist = 0;
        updatethreshold = 35 << 3;
        
        for (int i = 0; i < (1 << LOGSIZEUP); i++)
            Pupdatethreshold[i] = 0;
        for (int i = 0; i < GNB; i++)
            GGEHL[i] = &GGEHLA[i][0];
        for (int i = 0; i < LNB; i++)
            LGEHL[i] = &LGEHLA[i][0];
        
        for (int i = 0; i < GNB; i++)
            for (int j = 0; j < ((1 << LOGGNB) - 1); j++)
                if (!(j & 1)) GGEHL[i][j] = -1;
        for (int i = 0; i < LNB; i++)
            for (int j = 0; j < ((1 << LOGLNB) - 1); j++)
                if (!(j & 1)) LGEHL[i][j] = -1;
        
        for (int i = 0; i < SNB; i++)
            SGEHL[i] = &SGEHLA[i][0];
        for (int i = 0; i < TNB; i++)
            TGEHL[i] = &TGEHLA[i][0];
        for (int i = 0; i < PNB; i++)
            PGEHL[i] = &PGEHLA[i][0];
        
        for (int i = 0; i < INB; i++)
            IGEHL[i] = &IGEHLA[i][0];
        for (int i = 0; i < INB; i++)
            for (int j = 0; j < ((1 << LOGINB) - 1); j++)
                if (!(j & 1)) IGEHL[i][j] = -1;
        for (int i = 0; i < IMNB; i++)
            IMGEHL[i] = &IMGEHLA[i][0];
        for (int i = 0; i < IMNB; i++)
            for (int j = 0; j < ((1 << LOGIMNB) - 1); j++)
                if (!(j & 1)) IMGEHL[i][j] = -1;
        
        for (int i = 0; i < SNB; i++)
            for (int j = 0; j < ((1 << LOGSNB) - 1); j++)
                if (!(j & 1)) SGEHL[i][j] = -1;
        for (int i = 0; i < TNB; i++)
            for (int j = 0; j < ((1 << LOGTNB) - 1); j++)
                if (!(j & 1)) TGEHL[i][j] = -1;
        for (int i = 0; i < PNB; i++)
            for (int j = 0; j < ((1 << LOGPNB) - 1); j++)
                if (!(j & 1)) PGEHL[i][j] = -1;
        
        for (int i = 0; i < (1 << LOGB); i++) {
            btable[i].pred = 0;
            btable[i].hyst = 1;
        }
        
        for (int j = 0; j < (1 << LOGBIAS); j++) {
            switch (j & 3) {
                case 0: BiasSK[j] = -8; break;
                case 1: BiasSK[j] = 7; break;
                case 2: BiasSK[j] = -32; break;
                case 3: BiasSK[j] = 31; break;
            }
        }
        for (int j = 0; j < (1 << LOGBIAS); j++) {
            switch (j & 3) {
                case 0: Bias[j] = -32; break;
                case 1: Bias[j] = 31; break;
                case 2: Bias[j] = -1; break;
                case 3: Bias[j] = 0; break;
            }
        }
        for (int j = 0; j < (1 << LOGBIAS); j++) {
            switch (j & 3) {
                case 0: BiasBank[j] = -32; break;
                case 1: BiasBank[j] = 31; break;
                case 2: BiasBank[j] = -1; break;
                case 3: BiasBank[j] = 0; break;
            }
        }
        
        for (int i = 0; i < SIZEUSEALT; i++)
            use_alt_on_na[i] = 0;
        
        for (int i = 0; i < (1 << LOGSIZEUPS); i++) {
            WG[i] = 7;
            WL[i] = 7;
            WS[i] = 7;
            WT[i] = 7;
            WP[i] = 7;
            WI[i] = 7;
            WIM[i] = 7;
            WB[i] = 4;
        }
        
        TICK = 0;
        for (int i = 0; i < NLOCAL; i++)
            current_hist.L_shist[i] = 0;
        for (int i = 0; i < NSECLOCAL; i++)
            current_hist.S_slhist[i] = 3;
        
        current_hist.GHIST = 0;
        current_hist.ptghist = 0;
        current_hist.phist = 0;
        
        FirstH = 0;
        SecondH = 0;
    }
    
    int predictorsize() {
        int STORAGESIZE = 0;
        int inter = 0;
        
        STORAGESIZE += NBANKHIGH * (1 << logg[BORN]) * (CWIDTH + UWIDTH + TB[BORN]);
        STORAGESIZE += NBANKLOW * (1 << logg[1]) * (CWIDTH + UWIDTH + TB[1]);
        STORAGESIZE += SIZEUSEALT * ALTWIDTH;
        STORAGESIZE += (1 << LOGB) + (1 << (LOGB - HYSTSHIFT));
        STORAGESIZE += m[NHIST];
        STORAGESIZE += PHISTWIDTH;
        STORAGESIZE += 10;
        
        fprintf(stderr, " (TAGE %d) ", STORAGESIZE);
        
#ifdef LOOPPREDICTOR
        inter += (1 << LOGL) * (WIDTHNBITERLOOP + LOOPTAG + 4 + 4 + 1);
#endif
        inter += WIDTHRES;
        inter += WIDTHRESP * (1 << LOGSIZEUP);
        inter += 3 * EWIDTH * (1 << LOGSIZEUPS);
        inter += PERCWIDTH * 3 * (1 << LOGBIAS);
        inter += (GNB - 2) * (1 << LOGGNB) * PERCWIDTH + (1 << (LOGGNB - 1)) * (2 * PERCWIDTH);
        inter += Gm[0];
        inter += (PNB - 2) * (1 << LOGPNB) * PERCWIDTH + (1 << (LOGPNB - 1)) * (2 * PERCWIDTH);
        
#ifdef LOCALH
        inter += (LNB - 2) * (1 << LOGLNB) * PERCWIDTH + (1 << (LOGLNB - 1)) * (2 * PERCWIDTH);
        inter += NLOCAL * Lm[0];
#ifdef LOCALS
        inter += (SNB - 2) * (1 << LOGSNB) * PERCWIDTH + (1 << (LOGSNB - 1)) * (2 * PERCWIDTH);
        inter += NSECLOCAL * Sm[0];
#endif
#ifdef LOCALT
        inter += (TNB - 2) * (1 << LOGTNB) * PERCWIDTH + (1 << (LOGTNB - 1)) * (2 * PERCWIDTH);
        inter += NTLOCAL * Tm[0];
#endif
#endif

#ifdef IMLI
        inter += (INB - 2) * (1 << LOGINB) * PERCWIDTH + (1 << (LOGINB - 1)) * (2 * PERCWIDTH);
        inter += Im[0];
        inter += (IMNB - 2) * (1 << LOGIMNB) * PERCWIDTH + (1 << (LOGIMNB - 1)) * (2 * PERCWIDTH);
        inter += 256 * IMm[0];
#endif
        inter += 2 * CONFWIDTH;
        STORAGESIZE += inter;
        
        fprintf(stderr, " (SC %d) ", inter);
        fprintf(stderr, " (TOTAL %d bits %.1f KB) \n", STORAGESIZE, (double)STORAGESIZE / 8192.0);
        
        return STORAGESIZE;
    }
    
    // Index functions
    int bindex(UINT64 PC) const {
        return ((PC ^ (PC >> LOGB)) & ((1 << LOGB) - 1));
    }
    
    int F(uint64_t A, int size, int bank) const {
        int A1, A2;
        A = A & ((1 << size) - 1);
        A1 = (A & ((1 << logg[bank]) - 1));
        A2 = (A >> logg[bank]);
        
        if (bank < logg[bank])
            A2 = ((A2 << bank) & ((1 << logg[bank]) - 1)) + (A2 >> (logg[bank] - bank));
        A = A1 ^ A2;
        if (bank < logg[bank])
            A = ((A << bank) & ((1 << logg[bank]) - 1)) + (A >> (logg[bank] - bank));
        return (int)A;
    }
    
    int gindex(unsigned int PC, int bank, uint64_t hist, const tage_index_t& ch_i) const {
        int index;
        int M = (m[bank] > PHISTWIDTH) ? PHISTWIDTH : m[bank];
        index = PC ^ (PC >> (abs(logg[bank] - bank) + 1)) ^ ch_i[bank].comp ^ F(hist, M, bank);
        return (index & ((1 << logg[bank]) - 1));
    }
    
    uint16_t gtag(unsigned int PC, int bank, const tage_tag_t& tag_0, const tage_tag_t& tag_1) const {
        int tag = PC ^ tag_0[bank].comp ^ (tag_1[bank].comp << 1);
        return (tag & ((1 << TB[bank]) - 1));
    }
    
    void ctrupdate(int8_t& ctr, bool taken, int nbits) {
        if (taken) {
            if (ctr < ((1 << (nbits - 1)) - 1)) ctr++;
        } else {
            if (ctr > -(1 << (nbits - 1))) ctr--;
        }
    }
    
    bool getbim() {
        BIM = (btable[BI].pred << 1) + btable[BI >> HYSTSHIFT].hyst;
        HighConf = (BIM == 0) || (BIM == 3);
        LowConf = !HighConf;
        AltConf = HighConf;
        MedConf = false;
        return (btable[BI].pred > 0);
    }
    
    void baseupdate(bool Taken) {
        int inter = BIM;
        if (Taken) {
            if (inter < 3) inter += 1;
        } else if (inter > 0) {
            inter--;
        }
        btable[BI].pred = inter >> 1;
        btable[BI >> HYSTSHIFT].hyst = (inter & 1);
    }
    
    int MYRANDOM() {
        Seed++;
        Seed ^= active_hist.phist;
        Seed = (Seed >> 21) + (Seed << 11);
        Seed ^= (int64_t)active_hist.ptghist;
        Seed = (Seed >> 10) + (Seed << 22);
        return (Seed & 0xFFFFFFFF);
    }
    
    uint64_t get_local_index(uint64_t PC) const {
        return ((PC ^ (PC >> 2)) & (NLOCAL - 1));
    }
    
    uint64_t get_second_local_index(uint64_t PC) const {
        return (((PC ^ (PC >> 5))) & (NSECLOCAL - 1));
    }
    
    uint64_t get_third_local_index(uint64_t PC) const {
        return (((PC ^ (PC >> LOGTNB))) & (NTLOCAL - 1));
    }
    
    uint64_t get_bias_index(uint64_t PC) const {
        return (((((PC ^ (PC >> 2)) << 1) ^ (LowConf & (LongestMatchPred != alttaken))) << 1) + pred_inter) & ((1 << LOGBIAS) - 1);
    }
    
    uint64_t get_biassk_index(uint64_t PC) const {
        return (((((PC ^ (PC >> (LOGBIAS - 2))) << 1) ^ HighConf) << 1) + pred_inter) & ((1 << LOGBIAS) - 1);
    }
    
    uint64_t get_biasbank_index(uint64_t PC) const {
        return (pred_inter + (((HitBank + 1) / 4) << 4) + (HighConf << 1) + (LowConf << 2) + ((AltBank != 0) << 3) + ((PC ^ (PC >> 2)) << 7)) & ((1 << LOGBIAS) - 1);
    }
    
    // GEHL index macro
    #define GINDEX(bhist, i, logs) \
        ((((uint64_t)PC) ^ bhist ^ (bhist >> (8 - i)) ^ (bhist >> (16 - 2 * i)) ^ \
          (bhist >> (24 - 3 * i)) ^ (bhist >> (32 - 3 * i)) ^ (bhist >> (40 - 4 * i))) & \
         ((1 << (logs - (i >= (NBR - 2)))) - 1))
    
    int Gpredict(UINT64 PC, uint64_t BHIST, int* length, int8_t** tab, int NBR, int logs, int8_t* W) {
        int PERCSUM = 0;
        int INDUPDS_local = ((PC ^ (PC >> 2)) & ((1 << LOGSIZEUPS) - 1));
        
        for (int i = 0; i < NBR; i++) {
            uint64_t bhist = BHIST;
            int idx = (int)((((uint64_t)PC) ^ bhist ^ (bhist >> (8 - i)) ^ (bhist >> (16 - 2 * i)) ^
                      (bhist >> (24 - 3 * i)) ^ (bhist >> (32 - 3 * i)) ^ (bhist >> (40 - 4 * i))) &
                     ((1 << (logs - (i >= (NBR - 2)))) - 1));
            int8_t ctr = tab[i][idx];
            PERCSUM += (2 * ctr + 1);
        }
#ifdef VARTHRES
        PERCSUM = (1 + (W[INDUPDS_local] >= 0)) * PERCSUM;
#endif
        return PERCSUM;
    }
    
    void Gupdate(UINT64 PC, bool taken, uint64_t BHIST, int* length, int8_t** tab, int NBR, int logs, int8_t* W) {
        int INDUPDS_local = ((PC ^ (PC >> 2)) & ((1 << LOGSIZEUPS) - 1));
        
        for (int i = 0; i < NBR; i++) {
            uint64_t bhist = BHIST;
            int idx = (int)((((uint64_t)PC) ^ bhist ^ (bhist >> (8 - i)) ^ (bhist >> (16 - 2 * i)) ^
                      (bhist >> (24 - 3 * i)) ^ (bhist >> (32 - 3 * i)) ^ (bhist >> (40 - 4 * i))) &
                     ((1 << (logs - (i >= (NBR - 2)))) - 1));
            ctrupdate(tab[i][idx], taken, PERCWIDTH);
        }
#ifdef VARTHRES
        ctrupdate(W[INDUPDS_local], taken, EWIDTH);
#endif
    }
    
    // TAGE prediction
    void Tagepred(UINT64 PC, const apex_hist_t& hist_to_use) {
        HitBank = 0;
        AltBank = 0;
        
        for (int i = 1; i <= NHIST; i += 2) {
            GI[i] = gindex(PC, i, hist_to_use.phist, hist_to_use.ch_i);
            GTAG[i] = gtag(PC, i, hist_to_use.ch_t[0], hist_to_use.ch_t[1]);
            GTAG[i + 1] = GTAG[i];
            GI[i + 1] = GI[i] ^ (GTAG[i] & ((1 << LOGG) - 1));
        }
        
        int T = (PC ^ (hist_to_use.phist & ((1ULL << m[BORN]) - 1))) % NBANKHIGH;
        for (int i = BORN; i <= NHIST; i++) {
            if (NOSKIP[i]) {
                GI[i] += (T << LOGG);
                T++;
                T = T % NBANKHIGH;
            }
        }
        T = (PC ^ (hist_to_use.phist & ((1 << m[1]) - 1))) % NBANKLOW;
        for (int i = 1; i <= BORN - 1; i++) {
            if (NOSKIP[i]) {
                GI[i] += (T << LOGG);
                T++;
                T = T % NBANKLOW;
            }
        }
        
        BI = (PC ^ (PC >> 2)) & ((1 << LOGB) - 1);
        
        alttaken = getbim();
        tage_pred = alttaken;
        LongestMatchPred = alttaken;
        
        for (int i = NHIST; i > 0; i--) {
            if (NOSKIP[i]) {
                if (gtable[i][GI[i]].tag == GTAG[i]) {
                    HitBank = i;
                    LongestMatchPred = (gtable[HitBank][GI[HitBank]].ctr >= 0);
                    break;
                }
            }
        }
        
        for (int i = HitBank - 1; i > 0; i--) {
            if (NOSKIP[i]) {
                if (gtable[i][GI[i]].tag == GTAG[i]) {
                    AltBank = i;
                    break;
                }
            }
        }
        
        if (HitBank > 0) {
            if (AltBank > 0) {
                alttaken = (gtable[AltBank][GI[AltBank]].ctr >= 0);
                AltConf = (abs(2 * gtable[AltBank][GI[AltBank]].ctr + 1) > 1);
            } else {
                alttaken = getbim();
            }
            
            int INDUSEALT_local = ((((HitBank - 1) / 8) << 1) + AltConf) % (SIZEUSEALT - 1);
            bool Huse_alt_on_na = (use_alt_on_na[INDUSEALT_local] >= 0);
            
            if ((!Huse_alt_on_na) || (abs(2 * gtable[HitBank][GI[HitBank]].ctr + 1) > 1))
                tage_pred = LongestMatchPred;
            else
                tage_pred = alttaken;
            
            HighConf = (abs(2 * gtable[HitBank][GI[HitBank]].ctr + 1) >= (1 << CWIDTH) - 1);
            LowConf = (abs(2 * gtable[HitBank][GI[HitBank]].ctr + 1) == 1);
            MedConf = (abs(2 * gtable[HitBank][GI[HitBank]].ctr + 1) == 5);
        }
    }
    
#ifdef LOOPPREDICTOR
    bool getloop(UINT64 PC, const apex_hist_t& hist_to_use) {
        LHIT = -1;
        LI = (PC >> 2) & ((1 << LOGL) - 1);
        LIB = LI;
        LTAG = (PC >> (2 + LOGL)) & ((1 << LOOPTAG) - 1);
        
        for (int i = 0; i < 4; i++) {
            int idx = (LI << 2) + i;
            if (idx < (int)hist_to_use.ltable.size()) {
                if (hist_to_use.ltable[idx].TAG == LTAG) {
                    LHIT = i;
                    LVALID = (hist_to_use.ltable[idx].confid == 15);
                    if (LVALID) {
                        return (hist_to_use.ltable[idx].CurrentIter + 1 == hist_to_use.ltable[idx].NbIter);
                    }
                }
            }
        }
        LVALID = false;
        return false;
    }
    
    void loopupdate(UINT64 PC, bool taken, bool alloc, std::vector<lentry>& ltable_ref) {
        if (LHIT >= 0) {
            int idx = (LI << 2) + LHIT;
            if (idx < (int)ltable_ref.size()) {
                lentry& entry = ltable_ref[idx];
                
                if (entry.confid == 15 && (entry.CurrentIter + 1 == entry.NbIter) != taken) {
                    entry.confid = 0;
                    entry.CurrentIter = 0;
                    return;
                }
                
                entry.CurrentIter++;
                if (!taken) {
                    if (entry.NbIter != entry.CurrentIter) {
                        entry.confid = 0;
                        entry.NbIter = entry.CurrentIter;
                    } else {
                        if (entry.confid < 15) entry.confid++;
                    }
                    entry.CurrentIter = 0;
                }
            }
        } else if (alloc && !taken) {
            // Try to allocate
            int min_age = 256;
            int min_way = 0;
            for (int i = 0; i < 4; i++) {
                int idx = (LI << 2) + i;
                if (idx < (int)ltable_ref.size()) {
                    if (ltable_ref[idx].age < min_age) {
                        min_age = ltable_ref[idx].age;
                        min_way = i;
                    }
                }
            }
            if (min_age < 3) {
                int idx = (LI << 2) + min_way;
                if (idx < (int)ltable_ref.size()) {
                    ltable_ref[idx].TAG = LTAG;
                    ltable_ref[idx].NbIter = 0;
                    ltable_ref[idx].confid = 0;
                    ltable_ref[idx].age = 7;
                    ltable_ref[idx].CurrentIter = 0;
                    ltable_ref[idx].dir = taken;
                }
            }
        }
    }
#endif
    
    bool predict(uint64_t seq_no, uint8_t piece, UINT64 PC, bool external) {
        pred_time_histories.emplace(get_unique_inst_id(seq_no, piece), active_hist);
        return predict_using_given_hist(seq_no, piece, PC, active_hist, true);
    }
    
    bool predict_using_given_hist(uint64_t seq_no, uint8_t piece, UINT64 PC, const apex_hist_t& hist_to_use, bool pred_time_predict) {
        Tagepred(PC, hist_to_use);
        bool pred_taken = tage_pred;
        
#ifdef LOOPPREDICTOR
        predloop = getloop(PC, hist_to_use);
        pred_taken = ((hist_to_use.WITHLOOP >= 0) && LVALID) ? predloop : pred_taken;
#endif
        pred_inter = pred_taken;
        
#ifdef SC
        LSUM = 0;
        
        // Bias
        int8_t ctr = Bias[get_bias_index(PC)];
        LSUM += (2 * ctr + 1);
        ctr = BiasSK[get_biassk_index(PC)];
        LSUM += (2 * ctr + 1);
        ctr = BiasBank[get_biasbank_index(PC)];
        LSUM += (2 * ctr + 1);
        
        int INDUPDS_local = ((PC ^ (PC >> 2)) & ((1 << LOGSIZEUPS) - 1));
#ifdef VARTHRES
        LSUM = (1 + (WB[INDUPDS_local] >= 0)) * LSUM;
#endif
        
        // GEHL components
        LSUM += Gpredict((PC << 1) + pred_inter, hist_to_use.GHIST, Gm, GGEHL, GNB, LOGGNB, WG);
        LSUM += Gpredict(PC, hist_to_use.phist, Pm, PGEHL, PNB, LOGPNB, WP);
        
#ifdef LOCALH
        LSUM += Gpredict(PC, hist_to_use.L_shist[get_local_index(PC)], Lm, LGEHL, LNB, LOGLNB, WL);
#ifdef LOCALS
        LSUM += Gpredict(PC, hist_to_use.S_slhist[get_second_local_index(PC)], Sm, SGEHL, SNB, LOGSNB, WS);
#endif
#ifdef LOCALT
        LSUM += Gpredict(PC, hist_to_use.T_slhist[get_third_local_index(PC)], Tm, TGEHL, TNB, LOGTNB, WT);
#endif
#endif

#ifdef IMLI
        LSUM += Gpredict(PC, hist_to_use.IMHIST[hist_to_use.IMLIcount], IMm, IMGEHL, IMNB, LOGIMNB, WIM);
        LSUM += Gpredict(PC, hist_to_use.IMLIcount, Im, IGEHL, INB, LOGINB, WI);
        
        // Enhanced IMLI - add brIMLI and tarIMLI contribution
        LSUM += Gpredict(PC, hist_to_use.brIMLI ^ (hist_to_use.tarIMLI << 4), Im, IGEHL, INB, LOGINB, WI);
#endif
        
        bool SCPRED = (LSUM >= 0);
        int INDUPD_local = (PC ^ (PC >> 2)) & ((1 << LOGSIZEUP) - 1);
        
        THRES = (updatethreshold >> 3) + Pupdatethreshold[INDUPD_local]
#ifdef VARTHRES
            + 12 * ((WB[INDUPDS_local] >= 0) + (WP[INDUPDS_local] >= 0)
#ifdef LOCALH
                    + (WS[INDUPDS_local] >= 0) + (WT[INDUPDS_local] >= 0) + (WL[INDUPDS_local] >= 0)
#endif
                    + (WG[INDUPDS_local] >= 0)
#ifdef IMLI
                    + (WI[INDUPDS_local] >= 0)
#endif
                   )
#endif
            ;
        
        if (pred_inter != SCPRED) {
            pred_taken = SCPRED;
            if (HighConf) {
                if (abs(LSUM) < THRES / 4) {
                    pred_taken = pred_inter;
                } else if (abs(LSUM) < THRES / 2) {
                    pred_taken = (SecondH < 0) ? SCPRED : pred_inter;
                }
            }
            if (MedConf) {
                if (abs(LSUM) < THRES / 4) {
                    pred_taken = (FirstH < 0) ? SCPRED : pred_inter;
                }
            }
        }
#endif
        
        return pred_taken;
    }
    
    void history_update(uint64_t seq_no, uint8_t piece, UINT64 PC, bool taken, UINT64 nextPC) {
        auto& X = active_hist.phist;
        auto& Y = active_hist.ptghist;
        auto& H = active_hist.ch_i;
        auto& G = active_hist.ch_t[0];
        auto& J = active_hist.ch_t[1];
        
        int maxt = 2;  // Conditional branch
        
#ifdef IMLI
        active_hist.IMHIST[active_hist.IMLIcount] = (active_hist.IMHIST[active_hist.IMLIcount] << 1) + taken;
        
#ifdef LOOPPREDICTOR
        if (LVALID) {
            if (taken != predloop)
                ctrupdate(active_hist.WITHLOOP, (predloop == taken), 7);
        }
        loopupdate(PC, taken, !LVALID, active_hist.ltable);
#endif
        
        if (nextPC < PC) {
            if (!taken) {
                active_hist.IMLIcount = 0;
            }
            if (taken) {
                if (active_hist.IMLIcount < ((1ULL << Im[0]) - 1))
                    active_hist.IMLIcount++;
            }
        }
        
        // Enhanced IMLI: brIMLI and tarIMLI
        if (PC == active_hist.lastBrPC) {
            if (active_hist.brIMLI < 255) active_hist.brIMLI++;
        } else {
            active_hist.brIMLI = 0;
        }
        active_hist.lastBrPC = PC;
        
        if (taken && nextPC == active_hist.lastTarget) {
            if (active_hist.tarIMLI < 255) active_hist.tarIMLI++;
        } else if (taken) {
            active_hist.tarIMLI = 0;
        }
        if (taken) active_hist.lastTarget = nextPC;
#endif
        
        active_hist.GHIST = (active_hist.GHIST << 1) + (taken & (nextPC < PC));
        active_hist.L_shist[get_local_index(PC)] = (active_hist.L_shist[get_local_index(PC)] << 1) + taken;
        active_hist.S_slhist[get_second_local_index(PC)] = ((active_hist.S_slhist[get_second_local_index(PC)] << 1) + taken) ^ (PC & 15);
        active_hist.T_slhist[get_third_local_index(PC)] = (active_hist.T_slhist[get_third_local_index(PC)] << 1) + taken;
        
        int T = ((PC ^ (PC >> 2))) ^ taken;
        int PATH = PC ^ (PC >> 2) ^ (PC >> 4);
        
        for (int t = 0; t < maxt; t++) {
            bool DIR = (T & 1);
            T >>= 1;
            int PATHBIT = (PATH & 127);
            PATH >>= 1;
            
            Y--;
            active_hist.ghist[Y & (HISTBUFFERLENGTH - 1)] = DIR;
            X = (X << 1) ^ PATHBIT;
            
            for (int i = 1; i <= NHIST; i++) {
                H[i].update(active_hist.ghist, Y);
                G[i].update(active_hist.ghist, Y);
                J[i].update(active_hist.ghist, Y);
            }
        }
        
        X = (X & ((1 << PHISTWIDTH) - 1));
    }
    
    void update(uint64_t seq_no, uint8_t piece, UINT64 PC, bool resolveDir, bool predDir, UINT64 nextPC) {
        const auto pred_hist_key = get_unique_inst_id(seq_no, piece);
        const auto& pred_time_history = pred_time_histories.at(pred_hist_key);
        const bool pred_taken = predict_using_given_hist(seq_no, piece, PC, pred_time_history, false);
        update(PC, resolveDir, pred_taken, nextPC, pred_time_history);
        pred_time_histories.erase(pred_hist_key);
    }
    
    void update(UINT64 PC, bool resolveDir, bool pred_taken, UINT64 nextPC, const apex_hist_t& hist_to_use) {
        bool taken = resolveDir;
        
#ifdef SC
        bool SCPRED = (LSUM >= 0);
        int INDUPD_local = (PC ^ (PC >> 2)) & ((1 << LOGSIZEUP) - 1);
        int INDUPDS_local = ((PC ^ (PC >> 2)) & ((1 << LOGSIZEUPS) - 1));
        
        // Update SC meta predictors
        if (pred_inter != SCPRED) {
            if (HighConf) {
                if (SCPRED == taken) {
                    if (abs(LSUM) < THRES / 2) {
                        if (SecondH < CONFWIDTH) SecondH++;
                    }
                } else {
                    if (abs(LSUM) < THRES / 2) {
                        if (SecondH > -CONFWIDTH) SecondH--;
                    }
                }
            }
            if (MedConf) {
                if (SCPRED == taken) {
                    if (abs(LSUM) < THRES / 4) {
                        if (FirstH < CONFWIDTH) FirstH++;
                    }
                } else {
                    if (abs(LSUM) < THRES / 4) {
                        if (FirstH > -CONFWIDTH) FirstH--;
                    }
                }
            }
        }
        
        // Update SC tables
        if ((SCPRED != taken) || ((abs(LSUM) < THRES))) {
            if (SCPRED != taken) {
                Pupdatethreshold[INDUPD_local] += 1;
                updatethreshold += 1;
            } else {
                Pupdatethreshold[INDUPD_local] -= 1;
                updatethreshold -= 1;
            }
            if (Pupdatethreshold[INDUPD_local] < 0)
                Pupdatethreshold[INDUPD_local] = 0;
            if (Pupdatethreshold[INDUPD_local] > 63)
                Pupdatethreshold[INDUPD_local] = 63;
            if (updatethreshold < 0) updatethreshold = 0;
            if (updatethreshold > (1 << WIDTHRES) - 1)
                updatethreshold = (1 << WIDTHRES) - 1;
            
            ctrupdate(Bias[get_bias_index(PC)], taken, PERCWIDTH);
            ctrupdate(BiasSK[get_biassk_index(PC)], taken, PERCWIDTH);
            ctrupdate(BiasBank[get_biasbank_index(PC)], taken, PERCWIDTH);
            
            Gupdate((PC << 1) + pred_inter, taken, hist_to_use.GHIST, Gm, GGEHL, GNB, LOGGNB, WG);
            Gupdate(PC, taken, hist_to_use.phist, Pm, PGEHL, PNB, LOGPNB, WP);
            
#ifdef LOCALH
            Gupdate(PC, taken, hist_to_use.L_shist[get_local_index(PC)], Lm, LGEHL, LNB, LOGLNB, WL);
#ifdef LOCALS
            Gupdate(PC, taken, hist_to_use.S_slhist[get_second_local_index(PC)], Sm, SGEHL, SNB, LOGSNB, WS);
#endif
#ifdef LOCALT
            Gupdate(PC, taken, hist_to_use.T_slhist[get_third_local_index(PC)], Tm, TGEHL, TNB, LOGTNB, WT);
#endif
#endif

#ifdef IMLI
            Gupdate(PC, taken, hist_to_use.IMHIST[hist_to_use.IMLIcount], IMm, IMGEHL, IMNB, LOGIMNB, WIM);
            Gupdate(PC, taken, hist_to_use.IMLIcount, Im, IGEHL, INB, LOGINB, WI);
#endif
            
#ifdef VARTHRES
            ctrupdate(WB[INDUPDS_local], taken == pred_inter, EWIDTH);
#endif
        }
#endif
        
        // TAGE update
        bool ALLOC = (tage_pred != taken) & (HitBank < NHIST);
        
        if (HitBank > 0) {
            ctrupdate(gtable[HitBank][GI[HitBank]].ctr, taken, CWIDTH);
            
            if (gtable[HitBank][GI[HitBank]].u == 0) {
                if (AltBank > 0) {
                    ctrupdate(gtable[AltBank][GI[AltBank]].ctr, taken, CWIDTH);
                }
                if (AltBank == 0) {
                    baseupdate(taken);
                }
            }
            
            if (LongestMatchPred != alttaken) {
                if (LongestMatchPred == taken) {
                    if (gtable[HitBank][GI[HitBank]].u < ((1 << UWIDTH) - 1))
                        gtable[HitBank][GI[HitBank]].u++;
                } else {
                    int INDUSEALT_local = ((((HitBank - 1) / 8) << 1) + AltConf) % (SIZEUSEALT - 1);
                    ctrupdate(use_alt_on_na[INDUSEALT_local], alttaken == taken, ALTWIDTH);
                    if (gtable[HitBank][GI[HitBank]].u > 0)
                        gtable[HitBank][GI[HitBank]].u--;
                }
            }
            if (LongestMatchPred == taken) {
                if (alttaken == taken) {
                    ALLOC = false;
                }
            }
        } else {
            baseupdate(taken);
        }
        
        // Allocation
        if (ALLOC) {
            int T = 0;
            int NA = 0;
            int DEP = ((((HitBank - 1) + 2 * NNN) % (NHIST - 1)) + 1);
            
            for (int i = DEP; i <= NHIST; i++) {
                if (NOSKIP[i]) {
                    if (gtable[i][GI[i]].u == 0) {
                        gtable[i][GI[i]].tag = GTAG[i];
                        gtable[i][GI[i]].ctr = (taken) ? 0 : -1;
                        gtable[i][GI[i]].u = 0;
                        NA++;
                        if (NA >= 1 + NNN) break;
                        i += 2;
                    } else {
                        T++;
                    }
                }
            }
            
            TICK += (NA < (1 + NNN)) ? (1 + NNN - NA) : 0;
            if (NA == 0) {
                for (int i = DEP; i <= NHIST; i++) {
                    if (NOSKIP[i]) {
                        if (gtable[i][GI[i]].u > 0)
                            gtable[i][GI[i]].u--;
                    }
                }
            }
            
            if (TICK >= BORNTICK) {
                TICK = 0;
                for (int i = 1; i <= NHIST; i++) {
                    for (int j = 0; j < SizeTable[i]; j++) {
                        gtable[i][j].u >>= 1;
                    }
                }
            }
        }
    }
};

static APEX_Predictor cond_predictor_impl;

#endif // _APEX_PREDICTOR_H_
