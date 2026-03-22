from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Op(Enum):
    # Category 1: Arithmetic & Logic
    ADD  = auto()
    SUB  = auto()
    MUL  = auto()
    DIV  = auto()
    AND  = auto()
    OR   = auto()
    NOT  = auto()

    # Category 2: Data Movement
    LOAD  = auto()    # LOAD #imm -> Rn  (immediate)
    LOADM = auto()    # LOADM MEM[addr] -> Rn  (from memory)
    STORE = auto()    # STORE Rn -> MEM[addr]
    MOV   = auto()    # MOV Rn -> Rm

    # Category 3: Control Flow
    JMP  = auto()    # JMP label
    JEQ  = auto()    # JEQ label  (jump if FLAG == EQ)
    JNE  = auto()    # JNE label
    JGT  = auto()    # JGT label  (jump if FLAG == GT)
    HALT = auto()

    # Category 4: Comparison
    CMP  = auto()    # CMP Rn Rm -> sets FLAG


class Flag(Enum):
    NONE = "N"
    EQ   = "EQ"
    LT   = "LT"
    GT   = "GT"


@dataclass
class Instruction:
    op: Op
    dst: Optional[str] = None     # destination register e.g. "R1"
    src1: Optional[str] = None    # source register 1
    src2: Optional[str] = None    # source register 2 or label
    imm: Optional[int] = None     # immediate value
    addr: Optional[int] = None    # memory address
    label: Optional[str] = None   # jump target label


@dataclass
class MachineState:
    pc: int                        = 0
    registers: dict                = field(default_factory=lambda: {f"R{i}": 0 for i in range(1, 5)})
    memory: list                   = field(default_factory=lambda: [0] * 8)
    flag: Flag                     = Flag.NONE
    halted: bool                   = False


# Human-readable descriptions for the <instruction_set> tag
ISA_DESCRIPTIONS = {
    Op.ADD:   "ADD Rn Rm -> Rd: Adds registers Rn and Rm, stores result in Rd.",
    Op.SUB:   "SUB Rn Rm -> Rd: Subtracts Rm from Rn, stores result in Rd.",
    Op.MUL:   "MUL Rn Rm -> Rd: Multiplies Rn and Rm, stores result in Rd.",
    Op.DIV:   "DIV Rn Rm -> Rd: Integer-divides Rn by Rm, stores result in Rd.",
    Op.AND:   "AND Rn Rm -> Rd: Bitwise AND of Rn and Rm.",
    Op.OR:    "OR Rn Rm -> Rd: Bitwise OR of Rn and Rm.",
    Op.NOT:   "NOT Rn -> Rd: Bitwise NOT of Rn.",
    Op.LOAD:  "LOAD #imm -> Rn: Loads immediate integer value into Rn.",
    Op.LOADM: "LOADM MEM[addr] -> Rn: Loads value at memory address into Rn.",
    Op.STORE: "STORE Rn -> MEM[addr]: Writes Rn to memory address.",
    Op.MOV:   "MOV Rn -> Rm: Copies value of Rn into Rm.",
    Op.JMP:   "JMP label: Unconditional jump to label.",
    Op.JEQ:   "JEQ label: Jump to label if FLAG == EQ.",
    Op.JNE:   "JNE label: Jump to label if FLAG != EQ.",
    Op.JGT:   "JGT label: Jump to label if FLAG == GT.",
    Op.CMP:   "CMP Rn Rm: Compare Rn and Rm. Sets FLAG to EQ, LT, or GT.",
    Op.HALT:  "HALT: Stop execution.",
}
