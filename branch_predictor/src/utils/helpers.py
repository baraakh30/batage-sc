"""
Utility Functions

Helper functions for configuration, logging, and I/O.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_config(config: Dict[str, Any], 
                config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_results(results: Dict[str, Any],
                 output_dir: Union[str, Path],
                 name: str = "results",
                 formats: tuple = ('json', 'csv')) -> Dict[str, Path]:
    """
    Save results to multiple formats.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        name: Base filename
        formats: Output formats ('json', 'csv', 'yaml')
        
    Returns:
        Dictionary of format -> output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{name}_{timestamp}"
    
    output_paths = {}
    
    if 'json' in formats:
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        output_paths['json'] = json_path
    
    if 'yaml' in formats:
        yaml_path = output_dir / f"{base_name}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        output_paths['yaml'] = yaml_path
    
    if 'csv' in formats:
        csv_path = output_dir / f"{base_name}.csv"
        _save_results_csv(results, csv_path)
        output_paths['csv'] = csv_path
    
    return output_paths


def _save_results_csv(results: Dict[str, Any], filepath: Path) -> None:
    """Save results to CSV format."""
    import csv
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Flatten results for CSV
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    writer.writerow([f"{key}.{sub_key}", sub_value])
            else:
                writer.writerow([key, value])


def setup_logging(level: str = "INFO",
                  log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger("branch_predictor")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def format_number(n: Union[int, float], precision: int = 2) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(n) >= 1e9:
        return f"{n/1e9:.{precision}f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.{precision}f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


def format_bytes(n: int) -> str:
    """Format bytes with appropriate unit."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def parse_trace_path(path_str: str) -> Path:
    """
    Parse and validate trace file path.
    
    Handles various path formats and checks existence.
    """
    path = Path(path_str).expanduser().resolve()
    
    if not path.exists():
        # Check common trace directories
        common_dirs = [
            Path("traces"),
            Path("data/branch"),
            Path("../traces"),
        ]
        
        for common_dir in common_dirs:
            candidate = common_dir / path_str
            if candidate.exists():
                return candidate
        
        raise FileNotFoundError(f"Trace file not found: {path}")
    
    return path


def get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    
    # Walk up until we find project markers
    for parent in current.parents:
        if (parent / "config").exists() or (parent / "src").exists():
            return parent
    
    return current.parent.parent.parent


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.elapsed = time.time() - self.start_time
        print(f"{self.name}: {self.elapsed:.2f}s")


def create_predictor_from_config(config: Dict[str, Any], 
                                 predictor_type: str = "hybrid"):
    """
    Create predictor instance from configuration.
    
    Args:
        config: Full configuration dictionary
        predictor_type: Type of predictor to create
        
    Returns:
        Predictor instance
    """
    from ..predictors.hybrid import HybridPredictor, AdaptiveHybridPredictor
    from ..predictors.perceptron import PerceptronPredictor
    from ..predictors.bnn import BinaryNeuralNetwork
    from ..predictors.base import BimodalPredictor, GSharePredictor
    
    predictor_map = {
        'hybrid': HybridPredictor,
        'adaptive_hybrid': AdaptiveHybridPredictor,
        'perceptron': PerceptronPredictor,
        'bnn': BinaryNeuralNetwork,
        'bimodal': BimodalPredictor,
        'gshare': GSharePredictor,
    }
    
    predictor_class = predictor_map.get(predictor_type.lower())
    if not predictor_class:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
    
    # Get type-specific config
    type_config = config.get(predictor_type, config)
    
    return predictor_class(type_config)
