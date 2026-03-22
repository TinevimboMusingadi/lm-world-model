import copy
from dataclasses import dataclass, field
from .isa import Instruction, MachineState, Op, Flag

MAX_STEPS = 500

@dataclass
class SimulationTrace:
    steps: list[MachineState] = field(default_factory=list)   # snapshot after each instruction
    error: str = ""
    timed_out: bool = False


def step(state: MachineState, instructions: list[Instruction],
         labels: dict[str, int]) -> MachineState:
    """
    Execute a single instruction at state.pc.
    Returns a NEW MachineState (never mutates in place).
    Raises ValueError on illegal instruction.
    """
    s = copy.deepcopy(state)
    if s.pc >= len(instructions):
        s.halted = True
        return s

    instr = instructions[s.pc]
    op = instr.op
    reg = s.registers

    match op:
        case Op.LOAD:
            reg[instr.dst] = instr.imm
        case Op.LOADM:
            reg[instr.dst] = s.memory[instr.addr]
        case Op.STORE:
            s.memory[instr.addr] = reg[instr.src1]
        case Op.MOV:
            reg[instr.dst] = reg[instr.src1]
        case Op.ADD:
            reg[instr.dst] = reg[instr.src1] + reg[instr.src2]
        case Op.SUB:
            reg[instr.dst] = reg[instr.src1] - reg[instr.src2]
        case Op.MUL:
            reg[instr.dst] = reg[instr.src1] * reg[instr.src2]
        case Op.DIV:
            if reg[instr.src2] == 0:
                raise ValueError("DIV by zero")
            reg[instr.dst] = reg[instr.src1] // reg[instr.src2]
        case Op.AND:
            reg[instr.dst] = reg[instr.src1] & reg[instr.src2]
        case Op.OR:
            reg[instr.dst] = reg[instr.src1] | reg[instr.src2]
        case Op.NOT:
            reg[instr.dst] = ~reg[instr.src1]
        case Op.CMP:
            a, b = reg[instr.src1], reg[instr.src2]
            s.flag = Flag.EQ if a == b else (Flag.GT if a > b else Flag.LT)
        case Op.JMP:
            s.pc = labels[instr.label]
            return s    # pc already updated — skip pc += 1 below
        case Op.JEQ:
            if s.flag == Flag.EQ:
                s.pc = labels[instr.label]
                return s
        case Op.JNE:
            if s.flag != Flag.EQ:
                s.pc = labels[instr.label]
                return s
        case Op.JGT:
            if s.flag == Flag.GT:
                s.pc = labels[instr.label]
                return s
        case Op.HALT:
            s.halted = True
            return s
        case _:
            raise ValueError(f"Unknown op: {op}")

    s.pc += 1
    return s


def simulate(instructions: list[Instruction],
             labels: dict[str, int],
             initial_state: MachineState | None = None) -> SimulationTrace:
    """
    Run a full program to completion or MAX_STEPS, recording every state.
    """
    trace = SimulationTrace()
    state = initial_state or MachineState()
    trace.steps.append(copy.deepcopy(state))    # record initial state

    for _ in range(MAX_STEPS):
        if state.halted or state.pc >= len(instructions):
            break
        try:
            state = step(state, instructions, labels)
            trace.steps.append(copy.deepcopy(state))
        except Exception as e:
            trace.error = str(e)
            break
    else:
        trace.timed_out = True

    return trace
