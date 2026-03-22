from .isa import MachineState, SimulationTrace

def serialise_state(state: MachineState) -> str:
    """
    Convert a MachineState to a flat token string.
    Implements FIXED-WIDTH formatting for tokenizer robustness.
    Example:
      [PC=003] [R1=00005] [R2=00003] [R3=00008] [R4=00000] [FLAG=N] [MEM=0503080000000000]
    """
    # 5 digit width with sign for registers
    def fmt_reg(v):
        sign = "-" if v < 0 else "0"
        return f"{sign}{abs(v):04d}"

    reg_tokens = " ".join(f"[{k}={fmt_reg(v)}]" for k, v in sorted(state.registers.items()))
    # format memory as 0-padded hex characters per byte
    mem_token = "[MEM=" + "".join(f"{v & 0xFF:02x}" for v in state.memory) + "]"
    flag_token = f"[FLAG={state.flag.value}]"
    return f"[PC={state.pc:03d}] {reg_tokens} {flag_token} {mem_token}"

def serialise_trace(trace: SimulationTrace) -> str:
    """Serialise all states in a SimulationTrace."""
    return "\n".join(serialise_state(s) for s in trace.steps)


def format_output(final_state: MachineState) -> str:
    """
    Produce the <o> tag content: non-zero memory cells + non-zero registers.
    """
    parts = []
    for addr, val in enumerate(final_state.memory):
        if val != 0:
            parts.append(f"MEM[{addr}]={val}")
    for reg, val in sorted(final_state.registers.items()):
        if val != 0:
            parts.append(f"{reg}={val}")
    return " ".join(parts) if parts else "0"
