"""
Metrics Collection and Analysis

Collects and analyzes branch prediction performance metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class SimulationResults:
    """Container for simulation results."""
    trace_name: str
    branches_simulated: int
    warmup_branches: int
    elapsed_time: float
    predictor_results: Dict[str, Dict[str, Any]]
    hardware_costs: Dict[str, Dict[str, Any]]
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'trace_name': self.trace_name,
            'branches_simulated': self.branches_simulated,
            'warmup_branches': self.warmup_branches,
            'elapsed_time': self.elapsed_time,
            'predictor_results': self.predictor_results,
            'hardware_costs': self.hardware_costs,
            'config': self.config
        }
    
    def get_summary(self) -> str:
        """Get text summary of results."""
        lines = [
            f"Trace: {self.trace_name}",
            f"Branches: {self.branches_simulated:,}",
            f"Time: {self.elapsed_time:.2f}s",
            ""
        ]
        
        for name, stats in self.predictor_results.items():
            lines.append(f"{name}:")
            lines.append(f"  MPKI: {stats.get('mpki', 0):.4f}")
            lines.append(f"  Accuracy: {stats.get('accuracy', 0)*100:.4f}%")
        
        return "\n".join(lines)


class MetricsCollector:
    """
    Collects and computes branch prediction metrics.
    """
    
    def __init__(self):
        self._predictors: Dict[str, PredictorMetrics] = {}
        self._per_branch_stats: Dict[str, Dict[int, BranchMetrics]] = {}
        
    def register_predictor(self, name: str) -> None:
        """Register a predictor for metrics collection."""
        self._predictors[name] = PredictorMetrics()
        self._per_branch_stats[name] = {}
    
    def record_prediction(self, predictor_name: str, pc: int,
                         prediction: Any, actual: bool,
                         collect_per_branch: bool = False) -> None:
        """
        Record a prediction outcome.
        
        Args:
            predictor_name: Name of predictor
            pc: Program counter
            prediction: PredictionResult object
            actual: Actual branch outcome
            collect_per_branch: Whether to collect per-branch stats
        """
        if predictor_name not in self._predictors:
            self.register_predictor(predictor_name)
        
        metrics = self._predictors[predictor_name]
        correct = (prediction.prediction == actual)
        
        # Update overall metrics
        metrics.total += 1
        if correct:
            metrics.correct += 1
        else:
            metrics.mispredictions += 1
        
        if actual:
            metrics.taken_actual += 1
        if prediction.prediction:
            metrics.taken_predicted += 1
        
        # Confidence tracking
        if hasattr(prediction, 'confidence'):
            if prediction.confidence >= 0.5:
                metrics.high_confidence_total += 1
                if correct:
                    metrics.high_confidence_correct += 1
            else:
                metrics.low_confidence_total += 1
                if correct:
                    metrics.low_confidence_correct += 1
        
        # Per-branch tracking
        if collect_per_branch:
            if pc not in self._per_branch_stats[predictor_name]:
                self._per_branch_stats[predictor_name][pc] = BranchMetrics()
            
            branch_metrics = self._per_branch_stats[predictor_name][pc]
            branch_metrics.total += 1
            if correct:
                branch_metrics.correct += 1
    
    def get_predictor_stats(self, predictor_name: str) -> Dict[str, Any]:
        """Get statistics for a predictor."""
        if predictor_name not in self._predictors:
            return {}
        
        metrics = self._predictors[predictor_name]
        
        stats = {
            'total': metrics.total,
            'correct': metrics.correct,
            'mispredictions': metrics.mispredictions,
            'accuracy': metrics.accuracy,
            'mpki': metrics.mpki,
            'misprediction_rate': metrics.misprediction_rate,
            'taken_actual': metrics.taken_actual,
            'taken_predicted': metrics.taken_predicted,
        }
        
        # Confidence stats
        if metrics.high_confidence_total > 0:
            stats['high_confidence_accuracy'] = (
                metrics.high_confidence_correct / metrics.high_confidence_total
            )
        if metrics.low_confidence_total > 0:
            stats['low_confidence_accuracy'] = (
                metrics.low_confidence_correct / metrics.low_confidence_total
            )
        
        return stats
    
    def get_per_branch_stats(self, predictor_name: str) -> Dict[int, Dict]:
        """Get per-branch statistics."""
        if predictor_name not in self._per_branch_stats:
            return {}
        
        return {
            pc: {
                'total': m.total,
                'correct': m.correct,
                'accuracy': m.correct / m.total if m.total > 0 else 0
            }
            for pc, m in self._per_branch_stats[predictor_name].items()
        }
    
    def get_h2p_branches(self, predictor_name: str, 
                        threshold: float = 0.3) -> List[int]:
        """
        Get hard-to-predict branches.
        
        Args:
            predictor_name: Predictor to analyze
            threshold: Misprediction rate threshold
            
        Returns:
            List of H2P branch PCs
        """
        per_branch = self._per_branch_stats.get(predictor_name, {})
        
        h2p = []
        for pc, metrics in per_branch.items():
            if metrics.total >= 10:  # Minimum samples
                mispred_rate = 1.0 - (metrics.correct / metrics.total)
                if mispred_rate >= threshold:
                    h2p.append(pc)
        
        return h2p
    
    def reset(self) -> None:
        """Reset all metrics."""
        for metrics in self._predictors.values():
            metrics.reset()
        
        for name in self._per_branch_stats:
            self._per_branch_stats[name].clear()
    
    def get_comparison_table(self) -> str:
        """Get comparison table as formatted string."""
        if not self._predictors:
            return "No predictors registered"
        
        lines = [
            "Predictor Comparison:",
            "-" * 60,
            f"{'Predictor':<20} {'Accuracy':>12} {'MPKI':>10} {'Mispred':>12}",
            "-" * 60
        ]
        
        for name, metrics in self._predictors.items():
            lines.append(
                f"{name:<20} {metrics.accuracy*100:>11.4f}% {metrics.mpki:>10.4f} "
                f"{metrics.mispredictions:>12,}"
            )
        
        lines.append("-" * 60)
        return "\n".join(lines)


@dataclass
class PredictorMetrics:
    """Metrics for a single predictor."""
    total: int = 0
    correct: int = 0
    mispredictions: int = 0
    taken_actual: int = 0
    taken_predicted: int = 0
    high_confidence_total: int = 0
    high_confidence_correct: int = 0
    low_confidence_total: int = 0
    low_confidence_correct: int = 0
    
    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    @property
    def mpki(self) -> float:
        """Mispredictions per 1000 instructions."""
        if self.total == 0:
            return 0.0
        return (self.mispredictions / self.total) * 1000
    
    @property
    def misprediction_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.mispredictions / self.total
    
    def reset(self) -> None:
        self.total = 0
        self.correct = 0
        self.mispredictions = 0
        self.taken_actual = 0
        self.taken_predicted = 0
        self.high_confidence_total = 0
        self.high_confidence_correct = 0
        self.low_confidence_total = 0
        self.low_confidence_correct = 0


@dataclass
class BranchMetrics:
    """Metrics for a single branch."""
    total: int = 0
    correct: int = 0


class ResultsExporter:
    """Export simulation results to various formats."""
    
    @staticmethod
    def to_csv(results: SimulationResults, filepath: str) -> None:
        """Export results to CSV."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Trace', results.trace_name])
            writer.writerow(['Branches', results.branches_simulated])
            writer.writerow(['Time (s)', results.elapsed_time])
            writer.writerow([])
            
            # Per-predictor results
            for name, stats in results.predictor_results.items():
                writer.writerow([f'{name} - Accuracy', stats.get('accuracy', 0)])
                writer.writerow([f'{name} - MPKI', stats.get('mpki', 0)])
                writer.writerow([f'{name} - Mispredictions', 
                               stats.get('mispredictions', 0)])
    
    @staticmethod
    def to_json(results: SimulationResults, filepath: str) -> None:
        """Export results to JSON."""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
    
    @staticmethod
    def to_latex_table(results: List[SimulationResults]) -> str:
        """Generate LaTeX table from results."""
        if not results:
            return ""
        
        predictor_names = list(results[0].predictor_results.keys())
        
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Branch Prediction Results}",
            r"\begin{tabular}{l" + "c" * len(predictor_names) + "}",
            r"\hline",
            "Trace & " + " & ".join(predictor_names) + r" \\",
            r"\hline"
        ]
        
        for result in results:
            mpki_values = [
                f"{result.predictor_results[name].get('mpki', 0):.2f}"
                for name in predictor_names
            ]
            lines.append(
                f"{result.trace_name} & " + " & ".join(mpki_values) + r" \\"
            )
        
        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        return "\n".join(lines)
