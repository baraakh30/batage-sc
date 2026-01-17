"""
Trace Format Definitions

Defines various trace file formats for branch prediction simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, BinaryIO, TextIO
import struct


@dataclass
class BranchRecord:
    """Single branch record from a trace."""
    pc: int              # Program counter
    target: int          # Branch target
    taken: bool          # Branch outcome
    branch_type: int     # Type of branch (conditional, call, return, etc.)
    
    # Optional metadata
    instruction_count: Optional[int] = None
    
    @property
    def is_conditional(self) -> bool:
        return (self.branch_type & 0x1) != 0
    
    @property
    def is_call(self) -> bool:
        return (self.branch_type & 0x2) != 0
    
    @property
    def is_return(self) -> bool:
        return (self.branch_type & 0x4) != 0
    
    @property
    def is_indirect(self) -> bool:
        return (self.branch_type & 0x8) != 0


class TraceFormat(ABC):
    """Abstract base class for trace formats."""
    
    @abstractmethod
    def parse(self, file_handle) -> Iterator[BranchRecord]:
        """
        Parse trace file and yield branch records.
        
        Args:
            file_handle: Open file handle
            
        Yields:
            BranchRecord for each branch in trace
        """
        pass
    
    @abstractmethod
    def get_format_name(self) -> str:
        """Return format name."""
        pass


class CBPTraceFormat(TraceFormat):
    """
    Championship Branch Prediction (CBP) trace format.
    
    CBP 2016/2024 format: Binary file with branch records.
    Each record contains PC, target, and outcome.
    """
    
    def __init__(self, pc_bits: int = 64, target_bits: int = 64):
        self.pc_bits = pc_bits
        self.target_bits = target_bits
        
        # CBP format: typically 8 bytes PC + 8 bytes target + 1 byte flags
        self.record_size = 17
        
    def get_format_name(self) -> str:
        return "CBP"
    
    def parse(self, file_handle: BinaryIO) -> Iterator[BranchRecord]:
        """Parse CBP binary trace format."""
        while True:
            data = file_handle.read(self.record_size)
            if not data or len(data) < self.record_size:
                break
            
            # Unpack: PC (8 bytes), Target (8 bytes), Flags (1 byte)
            pc, target, flags = struct.unpack('<QQB', data)
            
            # Flags: bit 0 = taken, bits 1-4 = branch type
            taken = (flags & 0x1) != 0
            branch_type = (flags >> 1) & 0xF
            
            yield BranchRecord(
                pc=pc,
                target=target,
                taken=taken,
                branch_type=branch_type
            )
    
    def parse_text(self, file_handle: TextIO) -> Iterator[BranchRecord]:
        """Parse text-based CBP format (alternative)."""
        for line in file_handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                pc = int(parts[0], 16) if parts[0].startswith('0x') else int(parts[0])
                taken = parts[1].lower() in ('t', '1', 'taken', 'true')
                target = int(parts[2], 16) if len(parts) > 2 else 0
                
                yield BranchRecord(
                    pc=pc,
                    target=target,
                    taken=taken,
                    branch_type=1  # Assume conditional
                )


class ChampSimTraceFormat(TraceFormat):
    """
    ChampSim trace format.
    
    Used by ChampSim simulator. Binary format with instruction records.
    """
    
    def __init__(self):
        # ChampSim instruction trace format
        self.record_size = 64  # Typical ChampSim record size
        
    def get_format_name(self) -> str:
        return "ChampSim"
    
    def parse(self, file_handle: BinaryIO) -> Iterator[BranchRecord]:
        """
        Parse ChampSim trace format.
        
        Note: ChampSim traces contain full instruction info.
        We extract only branch instructions.
        """
        while True:
            # Read instruction record header
            header = file_handle.read(8)
            if not header or len(header) < 8:
                break
            
            # Parse header to get instruction type and size
            # ChampSim format varies - this is a simplified version
            ip, = struct.unpack('<Q', header)
            
            # Read rest of record
            rest = file_handle.read(self.record_size - 8)
            if not rest:
                break
            
            # Check if this is a branch instruction
            # In real ChampSim format, we'd check instruction type field
            # Simplified: treat every record as potential branch
            
            # Parse branch info from record
            if len(rest) >= 16:
                target, flags = struct.unpack('<QQ', rest[:16])
                
                is_branch = (flags & 0x1) != 0
                if is_branch:
                    taken = (flags & 0x2) != 0
                    branch_type = (flags >> 2) & 0xF
                    
                    yield BranchRecord(
                        pc=ip,
                        target=target,
                        taken=taken,
                        branch_type=branch_type
                    )


class SimpleTextFormat(TraceFormat):
    """
    Simple text trace format for testing.
    
    Format: PC TAKEN [TARGET]
    Example:
        0x400100 T 0x400200
        0x400108 N
    """
    
    def get_format_name(self) -> str:
        return "SimpleText"
    
    def parse(self, file_handle: TextIO) -> Iterator[BranchRecord]:
        """Parse simple text format."""
        for line_num, line in enumerate(file_handle, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            try:
                # Parse PC
                pc_str = parts[0]
                if pc_str.startswith('0x'):
                    pc = int(pc_str, 16)
                else:
                    pc = int(pc_str)
                
                # Parse outcome
                outcome = parts[1].upper()
                taken = outcome in ('T', '1', 'TAKEN', 'TRUE', 'Y')
                
                # Parse target (optional)
                target = 0
                if len(parts) >= 3:
                    target_str = parts[2]
                    if target_str.startswith('0x'):
                        target = int(target_str, 16)
                    else:
                        target = int(target_str)
                
                yield BranchRecord(
                    pc=pc,
                    target=target,
                    taken=taken,
                    branch_type=1  # Assume conditional
                )
                
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                continue


class CBP2025Format(TraceFormat):
    """
    CBP 2025 competition format.
    
    Real trace format from Championship Branch Prediction 2025.
    Based on trace_reader.h from https://github.com/ramisheikh/cbp2025
    
    Trace Format (variable-length records):
    - Inst PC: 8 bytes (uint64)
    - Inst Type: 1 byte (InstClass enum)
    - If load/store (type 1 or 2):
        - Effective Address: 8 bytes
        - Access Size: 1 byte
        - Base Update flag: 1 byte
        - If Store (type 2): Reg Offset flag: 1 byte
    - If branch (type 3,4,5,9,10,11):
        - Taken: 1 byte (bool)
        - If Taken: Target: 8 bytes
    - Num Input Regs: 1 byte
    - Input Reg Names: 1 byte each
    - Num Output Regs: 1 byte
    - Output Reg Names: 1 byte each
    - Output Reg Values: 8 bytes each (16 for SIMD regs 32-63)
    
    InstClass enum values:
        0 = aluInstClass
        1 = loadInstClass
        2 = storeInstClass
        3 = condBranchInstClass
        4 = uncondDirectBranchInstClass
        5 = uncondIndirectBranchInstClass
        6 = fpInstClass
        7 = slowAluInstClass
        8 = undefInstClass
        9 = callDirectInstClass
        10 = callIndirectInstClass
        11 = ReturnInstClass
    """
    
    # InstClass enum values
    INST_ALU = 0
    INST_LOAD = 1
    INST_STORE = 2
    INST_COND_BRANCH = 3
    INST_UNCOND_DIRECT_BRANCH = 4
    INST_UNCOND_INDIRECT_BRANCH = 5
    INST_FP = 6
    INST_SLOW_ALU = 7
    INST_UNDEF = 8
    INST_CALL_DIRECT = 9
    INST_CALL_INDIRECT = 10
    INST_RETURN = 11
    
    # Branch types
    BRANCH_TYPES = {3, 4, 5, 9, 10, 11}
    
    # Memory types
    MEMORY_TYPES = {1, 2}
    
    def get_format_name(self) -> str:
        return "CBP2025"
    
    def _is_simd_reg(self, reg: int) -> bool:
        """Check if register is SIMD (32-63) excluding flag/zero regs."""
        return 32 <= reg <= 63
    
    def parse(self, file_handle: BinaryIO) -> Iterator[BranchRecord]:
        """
        Parse CBP 2025 binary trace format.
        
        Yields only branch instructions (conditional branches primarily).
        """
        instruction_count = 0
        
        while True:
            # Read PC (8 bytes)
            pc_data = file_handle.read(8)
            if not pc_data or len(pc_data) < 8:
                break
            
            pc = struct.unpack('<Q', pc_data)[0]
            
            # Read instruction type (1 byte)
            type_data = file_handle.read(1)
            if not type_data:
                break
            inst_type = type_data[0]
            
            # Initialize branch info
            taken = False
            target = pc + 4  # Default next PC
            
            # Handle memory instructions (load/store)
            if inst_type in self.MEMORY_TYPES:
                # Effective Address (8 bytes)
                file_handle.read(8)
                # Access Size (1 byte)
                file_handle.read(1)
                # Base Update flag (1 byte)
                file_handle.read(1)
                # Store has additional reg offset flag
                if inst_type == self.INST_STORE:
                    file_handle.read(1)
            
            # Handle branch instructions
            if inst_type in self.BRANCH_TYPES:
                # Taken flag (1 byte)
                taken_data = file_handle.read(1)
                if taken_data:
                    taken = bool(taken_data[0])
                
                # If taken, read target (8 bytes)
                if taken:
                    target_data = file_handle.read(8)
                    if target_data and len(target_data) == 8:
                        target = struct.unpack('<Q', target_data)[0]
            
            # Read input registers
            num_in_data = file_handle.read(1)
            if not num_in_data:
                break
            num_in_regs = num_in_data[0]
            
            # Skip input register names
            if num_in_regs > 0:
                file_handle.read(num_in_regs)
            
            # Read output registers
            num_out_data = file_handle.read(1)
            if not num_out_data:
                break
            num_out_regs = num_out_data[0]
            
            # Read output register names and values
            out_regs = []
            for _ in range(num_out_regs):
                reg_data = file_handle.read(1)
                if reg_data:
                    out_regs.append(reg_data[0])
            
            # Skip output register values
            for reg in out_regs:
                if self._is_simd_reg(reg):
                    file_handle.read(16)  # SIMD: 128 bits
                else:
                    file_handle.read(8)   # INT: 64 bits
            
            instruction_count += 1
            
            # Only yield conditional branches for branch prediction
            if inst_type == self.INST_COND_BRANCH:
                # Map InstClass to our branch_type format
                branch_type = 1  # Conditional
                
                yield BranchRecord(
                    pc=pc,
                    target=target,
                    taken=taken,
                    branch_type=branch_type,
                    instruction_count=instruction_count
                )
