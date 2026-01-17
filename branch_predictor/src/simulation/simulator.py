"""
Branch Prediction Simulator

Main simulation engine for evaluating branch predictors.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from ..predictors.base import BasePredictor, PredictionResult
from ..predictors.hybrid import HybridPredictor
from ..components.history import GlobalHistoryRegister
from ..trace.parser import TraceParser, BranchTrace
from ..trace.formats import BranchRecord
from .metrics import MetricsCollector, SimulationResults


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    warmup_instructions: int = 1000000
    simulation_instructions: int = 10000000
    history_length: int = 64
    use_path_history: bool = True
    path_history_bits: int = 16
    verbose: bool = True
    log_interval: int = 100000
    collect_per_branch_stats: bool = False


class BranchSimulator:
    """
    Branch Prediction Simulator.
    
    Simulates branch prediction using trace-driven methodology.
    Supports multiple predictors and collects detailed statistics.
    """
    
    def __init__(self, config: Union[SimulationConfig, dict]):
        """
        Initialize simulator.
        
        Args:
            config: Simulation configuration
        """
        if isinstance(config, dict):
            self.config = SimulationConfig(**config)
        else:
            self.config = config
        
        # Initialize history register
        self.history = GlobalHistoryRegister(
            length=self.config.history_length,
            use_path_history=self.config.use_path_history,
            path_bits=self.config.path_history_bits
        )
        
        # Predictors to evaluate
        self.predictors: Dict[str, BasePredictor] = {}
        
        # Metrics collector
        self.metrics = MetricsCollector()
        
        # Trace parser
        self.parser = TraceParser()
        
        # State
        self.branches_processed = 0
        self.warmup_complete = False
        
    def add_predictor(self, name: str, predictor: BasePredictor) -> None:
        """Add a predictor to evaluate."""
        self.predictors[name] = predictor
        self.metrics.register_predictor(name)
    
    def run(self, trace_path: Union[str, Path],
            trace_format: Optional[str] = None) -> SimulationResults:
        """
        Run simulation on a trace file.
        
        Args:
            trace_path: Path to trace file
            trace_format: Optional format hint
            
        Returns:
            SimulationResults with all metrics
        """
        trace_path = Path(trace_path)
        
        if trace_format:
            self.parser = TraceParser(format_name=trace_format)
        
        # Get trace info
        trace_info = self.parser.get_trace_info(trace_path)
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Branch Prediction Simulation")
            print(f"{'='*60}")
            print(f"Trace: {trace_path.name}")
            print(f"Format: {trace_info.format}")
            print(f"Estimated branches: {trace_info.estimated_branches:,}")
            print(f"Predictors: {list(self.predictors.keys())}")
            print(f"{'='*60}\n")
        
        # Reset state
        self._reset()
        
        # Start timing
        start_time = time.time()
        
        # Calculate total branches to process
        total_branches = (self.config.warmup_instructions + 
                         self.config.simulation_instructions)
        
        # Run simulation
        if self.config.verbose:
            progress = tqdm(
                self.parser.parse_file(trace_path, max_branches=total_branches),
                total=total_branches,
                desc="Simulating",
                unit="branches"
            )
        else:
            progress = self.parser.parse_file(trace_path, max_branches=total_branches)
        
        try:
            for branch in progress:
                self._process_branch(branch)
                
                # Log progress
                if (self.config.verbose and 
                    self.branches_processed % self.config.log_interval == 0):
                    self._log_progress(trace_path)
                    
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        
        # Finalize
        elapsed_time = time.time() - start_time
        
        # Compile results
        results = self._compile_results(trace_path, elapsed_time)
        
        if self.config.verbose:
            self._print_results(results)
        
        return results
    
    def run_on_trace(self, trace: BranchTrace) -> SimulationResults:
        """
        Run simulation on pre-loaded trace.
        
        Args:
            trace: BranchTrace object
            
        Returns:
            SimulationResults
        """
        self._reset()
        
        start_time = time.time()
        
        total = min(len(trace), 
                   self.config.warmup_instructions + 
                   self.config.simulation_instructions)
        
        if self.config.verbose:
            progress = tqdm(trace, total=total, desc="Simulating")
        else:
            progress = trace
        
        for i, branch in enumerate(progress):
            if i >= total:
                break
            self._process_branch(branch)
        
        elapsed_time = time.time() - start_time
        
        return self._compile_results("memory", elapsed_time)
    
    def _process_branch(self, branch: BranchRecord) -> None:
        """Process a single branch."""
        self.branches_processed += 1
        
        # Check warmup
        in_warmup = self.branches_processed <= self.config.warmup_instructions
        
        if not in_warmup and not self.warmup_complete:
            self.warmup_complete = True
            # Reset metrics after warmup
            self.metrics.reset()
            if self.config.verbose:
                print("\nWarmup complete. Starting measurement.\n")
        
        # Get current history
        history = self.history.get_history(as_bipolar=True)
        
        # Run each predictor
        for name, predictor in self.predictors.items():
            # Make prediction
            prediction = predictor.predict(branch.pc, history)
            
            # Record outcome
            correct = (prediction.prediction == branch.taken)
            
            if not in_warmup:
                self.metrics.record_prediction(
                    name, branch.pc, prediction, branch.taken,
                    collect_per_branch=self.config.collect_per_branch_stats
                )
            
            # Update predictor
            predictor.update(branch.pc, history, branch.taken, prediction)
            predictor.stats.record_prediction(prediction, branch.taken)
        
        # Update global history
        self.history.update(branch.taken, branch.pc)
    
    def _reset(self) -> None:
        """Reset simulator state."""
        self.history.reset()
        self.metrics.reset()
        self.branches_processed = 0
        self.warmup_complete = False
        
        # Reset predictors
        for predictor in self.predictors.values():
            predictor.reset()
    
    def _compile_results(self, trace_source: Union[str, Path],
                        elapsed_time: float) -> SimulationResults:
        """Compile simulation results."""
        return SimulationResults(
            trace_name=str(trace_source),
            branches_simulated=self.branches_processed - self.config.warmup_instructions,
            warmup_branches=self.config.warmup_instructions,
            elapsed_time=elapsed_time,
            predictor_results={
                name: self.metrics.get_predictor_stats(name)
                for name in self.predictors
            },
            hardware_costs={
                name: pred.get_hardware_cost()
                for name, pred in self.predictors.items()
            },
            config=vars(self.config)
        )
    
    def _log_progress(self, trace_path: Union[str, Path]) -> None:
        """Log progress during simulation."""
        if not self.warmup_complete:
            return
        
        measured = self.branches_processed - self.config.warmup_instructions
        if measured <= 0:
            return
        
        # Get quick stats for first predictor
        first_pred = list(self.predictors.keys())[0]
        stats = self.metrics.get_predictor_stats(first_pred)
        
        tqdm.write(f"Branches: {measured:,} | "
                  f"MPKI: {stats.get('mpki', 0):.2f} | "
                  f"Acc: {stats.get('accuracy', 0)*100:.2f}% | {trace_path}")
    
    def _print_results(self, results: SimulationResults) -> None:
        """Print final results."""
        print(f"\n{'='*60}")
        print("SIMULATION RESULTS")
        print(f"{'='*60}")
        print(f"Branches simulated: {results.branches_simulated:,}")
        print(f"Time elapsed: {results.elapsed_time:.2f}s")
        print(f"Speed: {results.branches_simulated/results.elapsed_time:,.0f} branches/sec")
        print()
        
        # Results per predictor
        for name, stats in results.predictor_results.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {stats.get('accuracy', 0)*100:.4f}%")
            print(f"  MPKI: {stats.get('mpki', 0):.4f}")
            print(f"  Mispredictions: {stats.get('mispredictions', 0):,}")
            
            hw = results.hardware_costs.get(name, {})
            print(f"  Hardware: {hw.get('total_kb', 0):.2f} KB")
        
        print(f"\n{'='*60}")


class ComparativeSimulator:
    """
    Run comparative simulations across multiple traces and predictors.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results: List[SimulationResults] = []
        
    def run_comparison(self, 
                      traces: List[Union[str, Path]],
                      predictors: Dict[str, BasePredictor]) -> Dict:
        """
        Run comparison across traces.
        
        Args:
            traces: List of trace file paths
            predictors: Dictionary of predictors to compare
            
        Returns:
            Aggregated results
        """
        all_results = []
        
        for trace in traces:
            sim = BranchSimulator(self.config)
            
            for name, predictor in predictors.items():
                # Clone predictor for fresh state
                sim.add_predictor(name, predictor)
            
            results = sim.run(trace)
            all_results.append(results)
        
        # Aggregate results
        return self._aggregate_results(all_results)
    
    def _aggregate_results(self, 
                          results: List[SimulationResults]) -> Dict:
        """Aggregate results across traces."""
        if not results:
            return {}
        
        predictor_names = list(results[0].predictor_results.keys())
        
        aggregated = {
            'traces': [r.trace_name for r in results],
            'total_branches': sum(r.branches_simulated for r in results),
            'total_time': sum(r.elapsed_time for r in results),
            'per_predictor': {}
        }
        
        for name in predictor_names:
            mpki_values = [r.predictor_results[name].get('mpki', 0) 
                         for r in results]
            acc_values = [r.predictor_results[name].get('accuracy', 0) 
                        for r in results]
            
            aggregated['per_predictor'][name] = {
                'avg_mpki': np.mean(mpki_values),
                'std_mpki': np.std(mpki_values),
                'avg_accuracy': np.mean(acc_values),
                'per_trace_mpki': dict(zip(aggregated['traces'], mpki_values))
            }
        
        return aggregated
