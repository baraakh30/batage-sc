#!/usr/bin/env python3
"""
Multi-trace benchmark runner for branch prediction evaluation.

Runs simulations across multiple trace categories and generates comparison reports.
"""

import sys
import os
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictors.tage_phoenix import PHOENIX_32KB, PHOENIX_64KB, TAGE_Phoenix
from src.predictors.hybrid import HybridPredictor
from src.predictors.perceptron import PerceptronPredictor
from src.predictors.bnn import BinaryNeuralNetwork
from src.predictors.base import BimodalPredictor, GSharePredictor
from src.predictors.optimized_loabp import OptimizedLOABP, OPTIMIZED_CONFIG_32KB, OPTIMIZED_CONFIG_8KB






from src.predictors.tage_original import (
    OriginalTAGE,
    TAGE_8C_64KB,
    TAGE_32KB,
    TAGE_8KB,
    TAGE_4KB,
)

from src.predictors.tage_apex import (
    TAGEApex,
    APEX_64KB,
    APEX_32KB,
    APEX_8KB,
)


from src.predictors.batage_sc import (
    BATAGE_SC,
    BATAGE_SC_64KB,
    BATAGE_SC_32KB,
    BATAGE_SC_8KB,
)

from src.predictors.tage_scale import (
    TAGE_SCALE,
    TAGE_SCALE_64KB,
    TAGE_SCALE_32KB,
)

from src.predictors.tage_smart import (
    TAGE_Smart,
    TAGE_SMART_64KB,
    TAGE_SMART_32KB,
)

from src.predictors.mpp import (
    MPP,
    MPP_64KB,
    MPP_32KB,
)

from src.predictors.tage_sc_l import (
    TAGE_SC_L,
    TAGE_SC_L_64KB,
    TAGE_SC_L_32KB,
)

from src.predictors.tage_loop import (
    TAGE_Loop,
    TAGE_LOOP_64KB,
    TAGE_LOOP_32KB,
)



from src.predictors.mtage import (
    MTAGE,
    MTAGE_64KB,
    MTAGE_32KB,
    MTAGE_8KB,
)



from src.simulation.simulator import BranchSimulator, SimulationConfig
from src.trace.parser import TraceParser


def find_traces(base_dir: Path) -> dict:
    """Find all trace files organized by category."""
    traces = {}
    
    # Check for extracted directories
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            category = subdir.name
            trace_files = list(subdir.glob("*_trace.gz"))
            if trace_files:
                traces[category] = sorted(trace_files)
    
    # Also check for .tar.xz archives that need extraction
    for archive in base_dir.glob("*.tar.xz"):
        category = archive.stem  # e.g., 'int' from 'int.tar.xz'
        if category not in traces:
            print(f"Note: {archive.name} needs extraction. Run: tar -xf {archive}")
    
    return traces


def create_predictors(config: dict = None) -> dict:
    """Create all predictor instances for benchmarking.
    
    Selected predictors based on efficiency analysis:
    - NEXUS: NEW State-of-the-art hybrid predictor targeting 99%+ accuracy
    - OriginalTAGE: Reference TAGE from gem5 - the gold standard
    - TAGE-Apex: State-of-the-art predictor targeting >96.72% accuracy
    - LOABP: Best accuracy (~96%), reference implementation
    - Perceptron: Classic neural predictor, good accuracy
    - TAGE-Lite-8KB: Efficient TAGE implementation
    - GShare: Simple, efficient baseline (~4KB)
    - OptimizedLOABP-32KB: Good accuracy with optimizations
    - Bimodal: Simplest baseline
    """
    config = config or {}
    
    return {
        
        # === Original TAGE Baseline ===
        'OriginalTAGE-64KB': OriginalTAGE(TAGE_8C_64KB),
        
        # # === MTAGE (Multiplex TAGE - for comparison) ===
        # 'MTAGE-64KB': MTAGE(MTAGE_64KB),
        
        # # === PHOENIX (for comparison) ===
        # 'PHOENIX-64KB': TAGE_Phoenix(PHOENIX_64KB),
        # === TAGE-Loop (TAGE + Dedicated Loop Predictor) ===
        # 'TAGE-Loop-64KB': TAGE_Loop(TAGE_LOOP_64KB),

        # === TAGE-SC-L (TAGE + Statistical Corrector + Loop Predictor) ===
        'TAGE-SC-L-64KB': TAGE_SC_L(TAGE_SC_L_64KB),

        # # === MPP (Multi-Perspective Perceptron - Novel Architecture) ===
        # 'MPP-64KB': MPP(MPP_64KB),

        # # === TAGE-Smart (Internally Optimized TAGE) ===
        # 'TAGE-Smart-64KB': TAGE_Smart(TAGE_SMART_64KB),

        # # === TAGE-SCALE (NEW - Per-Branch Correction Learning) ===
        # 'TAGE-SCALE-64KB': TAGE_SCALE(TAGE_SCALE_64KB),

        # === BATAGE-SC (Branch-Aware TAGE + Statistical Corrector) ===
        'BATAGE-SC-64KB': BATAGE_SC(BATAGE_SC_64KB),
        # 'BATAGE-SC-32KB': BATAGE_SC(BATAGE_SC_32KB),
        # 'BATAGE-SC-8KB': BATAGE_SC(BATAGE_SC_8KB),

        # === TAGE-Apex (Target: Beat TAGE-64KB) ===
        'TAGE-Apex-64KB': TAGEApex(APEX_64KB),

        # === Reference predictors ===
        # 'LOABP': HybridPredictor(config),
        # 'Perceptron': PerceptronPredictor(config.get('perceptron', {})),
        # 'OptimizedLOABP-32KB': OptimizedLOABP(OPTIMIZED_CONFIG_32KB),
        
        # === Baselines ===
        # 'GShare': GSharePredictor(config.get('gshare', {})),
        # 'Bimodal': BimodalPredictor(config.get('bimodal', {})),
    }


def run_benchmark(trace_path: Path, 
                  warmup: int = 10000, 
                  instructions: int = 100000,
                  verbose: bool = False) -> dict:
    """Run benchmark on a single trace."""
    
    # Create simulation config
    sim_config = SimulationConfig(
        warmup_instructions=warmup,
        simulation_instructions=instructions,
        verbose=verbose,
        log_interval=instructions // 10 if verbose else instructions
    )
    
    # Create simulator and add predictors
    simulator = BranchSimulator(sim_config)
    predictors = create_predictors()
    
    for name, pred in predictors.items():
        simulator.add_predictor(name, pred)
    
    # Run simulation
    results = simulator.run(trace_path, trace_format='cbp2025')
    
    return {
        'trace': trace_path.name,
        'branches': results.branches_simulated,
        'time': results.elapsed_time,
        'predictors': {
            name: {
                'accuracy': metrics.get('accuracy', 0) * 100,
                'mpki': metrics.get('mpki', 0),
                'mispredictions': metrics.get('mispredictions', 0)
            }
            for name, metrics in results.predictor_results.items()
        }
    }


def _run_benchmark_worker(args: tuple) -> dict:
    """Worker function for parallel benchmark execution.
    
    Args:
        args: Tuple of (trace_path, warmup, instructions, verbose)
    
    Returns:
        Dictionary with benchmark results or error info
    """
    trace_path, warmup, instructions, verbose = args
    try:
        result = run_benchmark(trace_path, warmup, instructions, verbose)
        result['category'] = trace_path.parent.name
        result['success'] = True
        return result
    except Exception as e:
        return {
            'trace': trace_path.name,
            'category': trace_path.parent.name,
            'success': False,
            'error': str(e)
        }


def run_all_benchmarks(trace_dir: Path,
                       warmup: int = 10000,
                       instructions: int = 100000,
                       max_traces_per_category: int = None,
                       verbose: bool = False,
                       num_workers: int = None) -> dict:
    """Run benchmarks on all available traces in parallel.
    
    Args:
        trace_dir: Directory containing trace files
        warmup: Number of warmup branches
        instructions: Number of simulation branches
        max_traces_per_category: Maximum traces per category (None for all)
        verbose: Enable verbose output
        num_workers: Number of parallel workers (None for auto)
    
    Returns:
        Dictionary with all benchmark results
    """
    
    traces = find_traces(trace_dir)
    
    if not traces:
        print(f"No traces found in {trace_dir}")
        print("Make sure traces are extracted from .tar.xz archives")
        return {}
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'warmup': warmup,
            'instructions': instructions,
            'num_workers': num_workers,
        },
        'categories': {}
    }
    
    # Collect all trace files with their args
    all_trace_args = []
    for category, trace_files in traces.items():
        # Limit traces if specified
        if max_traces_per_category:
            trace_files = trace_files[:max_traces_per_category]
        
        for trace_path in trace_files:
            all_trace_args.append((trace_path, warmup, instructions, verbose))
        
        # Initialize category in results
        all_results['categories'][category] = []
    
    total_traces = len(all_trace_args)
    print(f"\nRunning {total_traces} traces with {num_workers} parallel workers...")
    print("="*60)
    
    # Run benchmarks in parallel
    completed = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_trace = {
            executor.submit(_run_benchmark_worker, args): args[0] 
            for args in all_trace_args
        }
        
        # Process results as they complete
        for future in as_completed(future_to_trace):
            trace_path = future_to_trace[future]
            completed += 1
            
            try:
                result = future.result()
                category = result['category']
                
                if result['success']:
                    all_results['categories'][category].append(result)
                    best = min(result['predictors'].items(), 
                              key=lambda x: x[1]['mpki'])
                    print(f"[{completed}/{total_traces}] {result['trace']}: "
                          f"{result['branches']:,} branches | "
                          f"Best: {best[0]} (MPKI: {best[1]['mpki']:.2f})")
                else:
                    print(f"[{completed}/{total_traces}] {result['trace']}: Error - {result['error']}")
                    
            except Exception as e:
                print(f"[{completed}/{total_traces}] {trace_path.name}: Exception - {e}")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {completed} traces in {elapsed:.1f}s ({elapsed/max(completed,1):.1f}s avg)")
    
    return all_results


def print_summary(results: dict):
    """Print summary of all benchmark results."""
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Aggregate by predictor
    predictor_totals = {}
    
    for category, traces in results.get('categories', {}).items():
        print(f"\n{category.upper()}:")
        print("-"*40)
        
        for trace_result in traces:
            print(f"  {trace_result['trace']}:")
            for pred_name, pred_stats in trace_result['predictors'].items():
                print(f"    {pred_name}: {pred_stats['accuracy']:.2f}% (MPKI: {pred_stats['mpki']:.2f})")
                
                # Accumulate totals
                if pred_name not in predictor_totals:
                    predictor_totals[pred_name] = {'mpki_sum': 0, 'acc_sum': 0, 'count': 0}
                predictor_totals[pred_name]['mpki_sum'] += pred_stats['mpki']
                predictor_totals[pred_name]['acc_sum'] += pred_stats['accuracy']
                predictor_totals[pred_name]['count'] += 1
    
    # Print overall averages
    print("\n" + "="*80)
    print("OVERALL AVERAGES")
    print("="*80)
    
    for pred_name, totals in sorted(predictor_totals.items(), 
                                     key=lambda x: x[1]['mpki_sum']):
        if totals['count'] > 0:
            avg_mpki = totals['mpki_sum'] / totals['count']
            avg_acc = totals['acc_sum'] / totals['count']
            print(f"{pred_name:12}: Avg MPKI: {avg_mpki:7.2f} | Avg Accuracy: {avg_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Run multi-trace benchmarks')
    parser.add_argument('--trace-dir', '-d', type=str, 
                       default='data/branch',
                       help='Directory containing trace files')
    parser.add_argument('--warmup', '-w', type=int, default=50000,
                       help='Warmup branches')
    parser.add_argument('--instructions', '-n', type=int, default=1000000,
                       help='Simulation branches')
    parser.add_argument('--max-traces', '-m', type=int, default=4,
                       help='Max traces per category')
    parser.add_argument('--workers', '-j', type=int, default=None,
                       help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--output', '-o', type=str, default='results/benchmark.json',
                       help='Output JSON file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    trace_dir = Path(args.trace_dir)
    
    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}")
        sys.exit(1)
    
    print(f"Running benchmarks from: {trace_dir}")
    print(f"Warmup: {args.warmup:,} | Instructions: {args.instructions:,}")
    
    # Run all benchmarks in parallel
    results = run_all_benchmarks(
        trace_dir,
        args.warmup,
        args.instructions,
        args.max_traces,
        args.verbose,
        args.workers
    )
    
    if results:
        # Print summary
        print_summary(results)
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
