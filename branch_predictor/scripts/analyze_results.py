#!/usr/bin/env python3
"""
Analyze simulation results and generate visualizations.

Usage:
    python analyze_results.py --results results/
    python analyze_results.py --results results/results_*.json --compare
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_dir: Path) -> List[Dict]:
    """Load all result files from directory."""
    results = []
    
    for result_file in results_dir.glob("*.json"):
        with open(result_file) as f:
            data = json.load(f)
            data['_filename'] = result_file.name
            results.append(data)
    
    return results


def print_summary(results: List[Dict]) -> None:
    """Print summary of results."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for result in results:
        print(f"\n{result.get('trace_name', 'Unknown')}:")
        print(f"  Branches: {result.get('branches_simulated', 0):,}")
        print(f"  Time: {result.get('elapsed_time', 0):.2f}s")
        
        for pred_name, pred_stats in result.get('predictor_results', {}).items():
            print(f"\n  {pred_name}:")
            print(f"    Accuracy: {pred_stats.get('accuracy', 0)*100:.4f}%")
            print(f"    MPKI: {pred_stats.get('mpki', 0):.4f}")


def generate_comparison_table(results: List[Dict]) -> str:
    """Generate comparison table in markdown format."""
    if not results:
        return "No results to compare"
    
    # Get all predictor names
    all_predictors = set()
    for r in results:
        all_predictors.update(r.get('predictor_results', {}).keys())
    all_predictors = sorted(all_predictors)
    
    # Header
    lines = [
        "| Trace | " + " | ".join(all_predictors) + " |",
        "|" + "---|" * (len(all_predictors) + 1)
    ]
    
    # Data rows
    for result in results:
        trace_name = Path(result.get('trace_name', 'Unknown')).stem
        mpki_values = []
        for pred in all_predictors:
            stats = result.get('predictor_results', {}).get(pred, {})
            mpki = stats.get('mpki', float('nan'))
            mpki_values.append(f"{mpki:.2f}")
        
        lines.append(f"| {trace_name} | " + " | ".join(mpki_values) + " |")
    
    return "\n".join(lines)


def plot_results(results: List[Dict], output_dir: Path) -> None:
    """Generate plots from results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return
    
    if not results:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictor names and data
    all_predictors = set()
    for r in results:
        all_predictors.update(r.get('predictor_results', {}).keys())
    all_predictors = sorted(all_predictors)
    
    # MPKI comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results))
    width = 0.8 / len(all_predictors)
    
    for i, pred in enumerate(all_predictors):
        mpki_values = []
        for result in results:
            stats = result.get('predictor_results', {}).get(pred, {})
            mpki_values.append(stats.get('mpki', 0))
        
        offset = (i - len(all_predictors)/2 + 0.5) * width
        ax.bar(x + offset, mpki_values, width, label=pred)
    
    ax.set_xlabel('Trace')
    ax.set_ylabel('MPKI')
    ax.set_title('Branch Prediction MPKI Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([Path(r.get('trace_name', '')).stem for r in results], 
                       rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mpki_comparison.png', dpi=150)
    plt.close()
    
    print(f"Plot saved to {output_dir / 'mpki_comparison.png'}")
    
    # Accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, pred in enumerate(all_predictors):
        acc_values = []
        for result in results:
            stats = result.get('predictor_results', {}).get(pred, {})
            acc_values.append(stats.get('accuracy', 0) * 100)
        
        offset = (i - len(all_predictors)/2 + 0.5) * width
        ax.bar(x + offset, acc_values, width, label=pred)
    
    ax.set_xlabel('Trace')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Branch Prediction Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([Path(r.get('trace_name', '')).stem for r in results],
                       rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    plt.close()
    
    print(f"Plot saved to {output_dir / 'accuracy_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument('--results', '-r', type=str, required=True,
                       help='Results directory or file pattern')
    parser.add_argument('--output', '-o', type=str, default='results/analysis',
                       help='Output directory for analysis')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Generate comparison table')
    parser.add_argument('--plot', '-p', action='store_true',
                       help='Generate plots')
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    output_dir = Path(args.output)
    
    # Load results
    if results_path.is_dir():
        results = load_results(results_path)
    else:
        # Glob pattern
        results = []
        for f in Path('.').glob(args.results):
            with open(f) as fp:
                results.append(json.load(fp))
    
    if not results:
        print("No results found")
        return
    
    print(f"Loaded {len(results)} result file(s)")
    
    # Print summary
    print_summary(results)
    
    # Generate comparison table
    if args.compare:
        print("\n" + "="*70)
        print("COMPARISON TABLE (Markdown)")
        print("="*70)
        print(generate_comparison_table(results))
    
    # Generate plots
    if args.plot:
        plot_results(results, output_dir)


if __name__ == "__main__":
    main()
