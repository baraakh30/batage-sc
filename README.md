# Branch Predictor Simulation Framework

## Project Overview

A comprehensive Python-based branch prediction simulation framework for evaluating and comparing various branch prediction algorithms. This framework was developed for the Advanced Computer Architecture course to benchmark our novel **BATAGE-SC** predictor against state-of-the-art designs.

The framework supports:
- **15+ predictor implementations** including TAGE variants, perceptron-based, and neural predictors
- **CBP2025 trace format** compatibility
- **Parallel benchmark execution** for efficient evaluation
- **Comprehensive metrics** collection and visualization

## Implemented Predictors

### TAGE-Family Predictors
| Predictor | Description |
|-----------|-------------|
| **BATAGE-SC** | Our novel Branch-Aware TAGE with in-place Statistical Corrector and H2P tracking |
| **OriginalTAGE** | Reference TAGE from gem5 with 12 tagged tables and geometric history |
| **TAGE-SC-L** | CBP-winning TAGE + Statistical Corrector + Loop predictor |
| **TAGE-Apex** | TAGE with asymmetric tables, H2P Bloom filter, and micro-perceptron |
| **TAGE-Loop** | TAGE with dedicated loop predictor for fixed-iteration loops |
| **TAGE-Phoenix** | TAGE with dual-tag verification and adaptive history selection |
| **TAGE-Smart** | TAGE with enhanced allocation, adaptive UseAlt, and local history |
| **TAGE-Scale** | Scalable TAGE configuration |
| **MTAGE** | Multiplex TAGE separating volatile/stable branches into channels |

### Neural/Perceptron Predictors
| Predictor | Description |
|-----------|-------------|
| **MPP** | Multi-Perspective Perceptron with multiple hash tables summed |
| **Perceptron** | Classic neural predictor with weighted history sum (Jiménez 2001) |
| **BNN** | Binary Neural Network with 1-bit weights |
| **OptimizedLOABP** | Lightweight Online-Adaptive predictor with hashed perceptron |
| **Hybrid** | Configurable hybrid combining perceptron and BNN paths |

### Baseline Predictors
| Predictor | Description |
|-----------|-------------|
| **GShare** | Global history XOR'd with PC |
| **Bimodal** | Simple 2-bit saturating counters indexed by PC |

## Project Structure

```
branch_predictor/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config/
│   └── config.yaml          # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── predictors/          # All predictor implementations
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract base predictor + Bimodal/GShare
│   │   ├── batage_sc.py     # BATAGE-SC (our main contribution)
│   │   ├── tage_original.py # Reference TAGE from gem5
│   │   ├── tage_sc_l.py     # TAGE-SC-L championship predictor
│   │   ├── tage_apex.py     # TAGE-Apex with H2P detection
│   │   ├── tage_loop.py     # TAGE with loop predictor
│   │   ├── tage_phoenix.py  # TAGE-Phoenix with dual-tag
│   │   ├── tage_smart.py    # TAGE-Smart with enhanced allocation
│   │   ├── tage_scale.py    # Scalable TAGE
│   │   ├── mtage.py         # Multiplex TAGE
│   │   ├── mpp.py           # Multi-Perspective Perceptron
│   │   ├── perceptron.py    # Classic perceptron predictor
│   │   ├── bnn.py           # Binary neural network
│   │   ├── hybrid.py        # Hybrid predictor framework
│   │   └── optimized_loabp.py # Optimized LOABP
│   ├── components/          # Shared components
│   │   ├── __init__.py
│   │   ├── history.py       # Global history register
│   │   ├── h2p_detector.py  # Hard-to-predict branch detector
│   │   └── tables.py        # Weight tables and indexing
│   ├── trace/               # Trace handling
│   │   ├── __init__.py
│   │   ├── parser.py        # Multi-format trace parser
│   │   └── formats.py       # Trace format definitions (CBP, ChampSim, etc.)
│   ├── simulation/          # Simulation engine
│   │   ├── __init__.py
│   │   ├── simulator.py     # Main simulation driver
│   │   └── metrics.py       # Performance metrics collection
│   └── utils/
│       ├── __init__.py
│       └── helpers.py       # Utility functions
├── scripts/
│   ├── run_benchmarks.py    # Multi-trace benchmark runner (parallel)
│   ├── analyze_results.py   # Result analysis and visualization
│   ├── extract_traces.py    # Extract compressed trace archives
│   └── debug_batage.py      # BATAGE-SC debugging utilities
└── results/                 # Output directory for results
```

## Key Features

### 1. BATAGE-SC (Our Contribution)
- **Branch Type Classifier**: 2-bit counter classifies branches as loop-like or data-dependent
- **Selective Table Access**: Prioritizes tables based on branch classification
- **In-Place Statistical Corrector**: 4 small tables computed in parallel with TAGE (zero latency)
- **H2P-Aware Allocation**: Aggressive allocation for hard-to-predict branches

### 2. Simulation Framework
- **Trace-driven simulation** with CBP2025 format support
- **Configurable warmup** and simulation lengths
- **Parallel execution** using Python multiprocessing
- **Streaming trace parsing** for memory efficiency

### 3. Metrics Collection
- **MPKI**: Mispredictions per 1000 instructions
- **Accuracy**: Overall prediction accuracy
- **Per-category breakdown**: compress, fp, int, media, web
- **Per-branch statistics** (optional)

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Benchmarks
```bash
# Run all predictors on all traces (parallel)
python scripts/run_benchmarks.py --traces ../data/branch/ --output results/

# Run specific predictors
python scripts/run_benchmarks.py --traces ../data/branch/ --predictors batage_sc tage_original

# Limit trace count per category
python scripts/run_benchmarks.py --traces ../data/branch/ --max-traces 5
```

### Analyze Results
```bash
# Generate summary and plots
python scripts/analyze_results.py --results results/

# Compare specific result files
python scripts/analyze_results.py --results results/results_*.json --compare
```

### Extract Traces
```bash
# Extract single archive
python scripts/extract_traces.py --input traces/int.tar.xz --output data/branch/

# Extract all archives in directory
python scripts/extract_traces.py --input traces/ --output data/branch/ --all
```

## Configuration

Key parameters in `config/config.yaml`:

```yaml
simulation:
  warmup_instructions: 1000000      # Warmup period (no stats)
  simulation_instructions: 10000000 # Instructions to simulate

history:
  length: 64                        # Global history bits
  use_path_history: true            # Include path/address history

hardware:
  target_budget_kb: 32              # Target hardware budget
```

## Evaluation Results (CBP2025 Traces)

| Predictor | Avg MPKI | Avg Accuracy | Storage |
|-----------|----------|--------------|---------|
| **BATAGE-SC-64KB** | **4.64** | **96.16%** | 64KB |
| OriginalTAGE-64KB | 4.65 | 96.15% | 64KB |
| TAGE-SC-L-64KB | 4.67 | 96.13% | 64KB |

## Authors

- Baraa Khanfar (1210640)
- Sadeen Khatib (1212164)
- Toleen Hamed (1211386)

**Supervisor**: Prof. Ayman Hroub  
Birzeit University, January 2026
