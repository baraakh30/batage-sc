"""
Trace Parser

Unified trace parser supporting multiple formats.
Handles compressed traces and provides streaming interface.
"""

import os
import gzip
import lzma
import bz2
from pathlib import Path
from typing import Iterator, Optional, List, Union
from dataclasses import dataclass

from .formats import (
    TraceFormat, BranchRecord,
    CBPTraceFormat, ChampSimTraceFormat,
    SimpleTextFormat, CBP2025Format
)


@dataclass
class TraceInfo:
    """Information about a trace file."""
    path: str
    format: str
    compression: Optional[str]
    size_bytes: int
    estimated_branches: int


class BranchTrace:
    """
    Container for branch trace data.
    
    Can be used for streaming or caching trace data.
    """
    
    def __init__(self, records: Optional[List[BranchRecord]] = None):
        self._records = records or []
        self._index = 0
        
    def add(self, record: BranchRecord) -> None:
        """Add a branch record."""
        self._records.append(record)
    
    def __iter__(self) -> Iterator[BranchRecord]:
        return iter(self._records)
    
    def __len__(self) -> int:
        return len(self._records)
    
    def __getitem__(self, idx: int) -> BranchRecord:
        return self._records[idx]
    
    def get_statistics(self) -> dict:
        """Get trace statistics."""
        if not self._records:
            return {'count': 0}
        
        taken_count = sum(1 for r in self._records if r.taken)
        unique_pcs = len(set(r.pc for r in self._records))
        
        # Branch type breakdown
        conditional = sum(1 for r in self._records if r.is_conditional)
        calls = sum(1 for r in self._records if r.is_call)
        returns = sum(1 for r in self._records if r.is_return)
        indirect = sum(1 for r in self._records if r.is_indirect)
        
        return {
            'count': len(self._records),
            'taken': taken_count,
            'not_taken': len(self._records) - taken_count,
            'taken_ratio': taken_count / len(self._records),
            'unique_pcs': unique_pcs,
            'conditional': conditional,
            'calls': calls,
            'returns': returns,
            'indirect': indirect
        }


class TraceParser:
    """
    Unified trace parser with format detection and decompression.
    """
    
    # Supported formats
    FORMATS = {
        'cbp': CBPTraceFormat,
        'cbp2025': CBP2025Format,
        'champsim': ChampSimTraceFormat,
        'text': SimpleTextFormat,
    }
    
    # Compression handlers
    COMPRESSION = {
        '.gz': gzip.open,
        '.gzip': gzip.open,
        '.xz': lzma.open,
        '.bz2': bz2.open,
        '.lzma': lzma.open,
    }
    
    def __init__(self, format_name: Optional[str] = None):
        """
        Initialize parser.
        
        Args:
            format_name: Force specific format (auto-detect if None)
        """
        self.format_name = format_name
        self._format: Optional[TraceFormat] = None
        
    def parse_file(self, filepath: Union[str, Path], 
                   max_branches: Optional[int] = None,
                   skip_branches: int = 0) -> Iterator[BranchRecord]:
        """
        Parse a trace file.
        
        Args:
            filepath: Path to trace file
            max_branches: Maximum branches to read (None = all)
            skip_branches: Number of branches to skip (warmup)
            
        Yields:
            BranchRecord for each branch
        """
        filepath = Path(filepath)
        
        # Detect compression
        compression_ext = None
        effective_path = filepath
        
        for ext in self.COMPRESSION:
            if filepath.suffix.lower() == ext:
                compression_ext = ext
                effective_path = filepath.with_suffix('')
                break
        
        # Detect format
        format_obj = self._get_format(effective_path)
        
        # Open file (possibly compressed)
        is_text_format = isinstance(format_obj, SimpleTextFormat)
        
        if compression_ext:
            open_func = self.COMPRESSION[compression_ext]
            mode = 'rt' if is_text_format else 'rb'
            file_handle = open_func(filepath, mode)
        else:
            # Use text mode for text formats, binary for others
            if is_text_format:
                file_handle = open(filepath, 'r')
            else:
                file_handle = open(filepath, 'rb')
        
        try:
            count = 0
            skipped = 0
            
            for record in format_obj.parse(file_handle):
                # Skip warmup branches
                if skipped < skip_branches:
                    skipped += 1
                    continue
                
                yield record
                count += 1
                
                # Check limit
                if max_branches and count >= max_branches:
                    break
                    
        finally:
            file_handle.close()
    
    def load_trace(self, filepath: Union[str, Path],
                   max_branches: Optional[int] = None,
                   skip_branches: int = 0) -> BranchTrace:
        """
        Load entire trace into memory.
        
        Args:
            filepath: Path to trace file
            max_branches: Maximum branches to load
            skip_branches: Branches to skip
            
        Returns:
            BranchTrace with all records
        """
        records = list(self.parse_file(filepath, max_branches, skip_branches))
        return BranchTrace(records)
    
    def get_trace_info(self, filepath: Union[str, Path]) -> TraceInfo:
        """Get information about a trace file."""
        filepath = Path(filepath)
        
        # Check compression
        compression = None
        for ext in self.COMPRESSION:
            if filepath.suffix.lower() == ext:
                compression = ext[1:]  # Remove dot
                break
        
        # Get size
        size = filepath.stat().st_size
        
        # Estimate branches (rough)
        if compression:
            # Compressed files: assume ~10x compression
            estimated = (size * 10) // 20
        else:
            estimated = size // 20  # ~20 bytes per record
        
        # Detect format
        format_name = self.format_name or self._detect_format(filepath)
        
        return TraceInfo(
            path=str(filepath),
            format=format_name,
            compression=compression,
            size_bytes=size,
            estimated_branches=estimated
        )
    
    def _get_format(self, filepath: Path) -> TraceFormat:
        """Get format parser for file."""
        if self.format_name:
            format_class = self.FORMATS.get(self.format_name.lower())
            if format_class:
                return format_class()
        
        # Auto-detect format
        format_name = self._detect_format(filepath)
        format_class = self.FORMATS.get(format_name, SimpleTextFormat)
        return format_class()
    
    def _detect_format(self, filepath: Path) -> str:
        """Detect trace format from filename and content."""
        name = filepath.name.lower()
        
        # Remove compression extension for detection
        check_name = name
        for ext in self.COMPRESSION:
            if check_name.endswith(ext):
                check_name = check_name[:-len(ext)]
                break
        
        # Check for known patterns
        if 'champsim' in name:
            return 'champsim'
        if 'cbp' in name:
            return 'cbp'
        
        # CBP 2025 traces follow pattern: category_N_trace (e.g., int_0_trace.gz)
        # These are binary traces from the CBP 2025 competition
        if check_name.endswith('_trace'):
            return 'cbp2025'
        
        # Text formats
        if filepath.suffix in ('.txt', '.trace'):
            return 'text'
        
        # Default to cbp2025 for unknown binary files
        return 'cbp2025'
    
    @classmethod
    def list_supported_formats(cls) -> List[str]:
        """List supported trace formats."""
        return list(cls.FORMATS.keys())
    
    @classmethod
    def list_supported_compressions(cls) -> List[str]:
        """List supported compression formats."""
        return [ext[1:] for ext in cls.COMPRESSION.keys()]


def create_sample_trace(filepath: Union[str, Path], 
                        num_branches: int = 10000,
                        pattern: str = 'random') -> None:
    """
    Create a sample trace file for testing.
    
    Args:
        filepath: Output path
        num_branches: Number of branches to generate
        pattern: Pattern type ('random', 'loop', 'biased')
    """
    import random
    
    filepath = Path(filepath)
    
    with open(filepath, 'w') as f:
        f.write("# Sample branch trace\n")
        f.write("# Format: PC TAKEN TARGET\n")
        
        pc = 0x400000
        
        for i in range(num_branches):
            # Generate outcome based on pattern
            if pattern == 'random':
                taken = random.random() > 0.5
            elif pattern == 'loop':
                # Loop pattern: mostly taken, occasionally not
                taken = (i % 10) != 9
            elif pattern == 'biased':
                # Biased: 80% taken
                taken = random.random() > 0.2
            else:
                taken = random.random() > 0.5
            
            target = pc + 0x100 if taken else pc + 4
            outcome = 'T' if taken else 'N'
            
            f.write(f"0x{pc:x} {outcome} 0x{target:x}\n")
            
            # Advance PC
            pc += 4
            
            # Occasional jumps
            if i % 100 == 0:
                pc = 0x400000 + random.randint(0, 0x10000)
