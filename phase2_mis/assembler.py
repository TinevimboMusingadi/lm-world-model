import re
from .isa import Instruction, Op

# Regex patterns for each instruction format
_PATTERNS = {
    Op.LOAD:  r"LOAD\s+#(-?\d+)\s+->\s+(R\d)",
    Op.LOADM: r"LOADM\s+MEM\[(\d+)\]\s+->\s+(R\d)",
    Op.STORE: r"STORE\s+(R\d)\s+->\s+MEM\[(\d+)\]",
    Op.MOV:   r"MOV\s+(R\d)\s+->\s+(R\d)",
    Op.ADD:   r"ADD\s+(R\d)\s+(R\d)\s+->\s+(R\d)",
    Op.SUB:   r"SUB\s+(R\d)\s+(R\d)\s+->\s+(R\d)",
    Op.MUL:   r"MUL\s+(R\d)\s+(R\d)\s+->\s+(R\d)",
    Op.DIV:   r"DIV\s+(R\d)\s+(R\d)\s+->\s+(R\d)",
    Op.AND:   r"AND\s+(R\d)\s+(R\d)\s+->\s+(R\d)",
    Op.OR:    r"OR\s+(R\d)\s+(R\d)\s+->\s+(R\d)",
    Op.NOT:   r"NOT\s+(R\d)\s+->\s+(R\d)",
    Op.CMP:   r"CMP\s+(R\d)\s+(R\d)",
    Op.JMP:   r"JMP\s+(\w+)",
    Op.JEQ:   r"JEQ\s+(\w+)",
    Op.JNE:   r"JNE\s+(\w+)",
    Op.JGT:   r"JGT\s+(\w+)",
    Op.HALT:  r"HALT",
}

class AssemblyError(Exception):
    pass

def assemble(source: str) -> tuple[list[Instruction], dict[str, int]]:
    """
    Parse MIS source text into (instructions, labels).
    labels maps label name -> instruction index.
    Raises AssemblyError on any syntax problem.
    """
    instructions = []
    labels = {}
    lines = source.strip().splitlines()

    # First pass: collect labels
    instr_index = 0
    for raw_line in lines:
        line = raw_line.split(";")[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            labels[line[:-1]] = instr_index
        else:
            instr_index += 1

    # Second pass: parse instructions
    for raw_line in lines:
        line = raw_line.split(";")[0].strip()
        if not line or line.endswith(":"):
            continue
        try:
            instr = _parse_line(line)
            instructions.append(instr)
        except Exception as e:
            raise AssemblyError(f"Error parsing line '{line}': {e}")

    return instructions, labels

def _parse_line(line: str) -> Instruction:
    for op, pattern in _PATTERNS.items():
        m = re.fullmatch(pattern, line.strip())
        if m:
            return _build_instruction(op, m)
    raise AssemblyError(f"Unrecognised instruction: {line!r}")

def _build_instruction(op: Op, m: re.Match) -> Instruction:
    if op == Op.LOAD:
        return Instruction(op=op, imm=int(m.group(1)), dst=m.group(2))
    elif op == Op.LOADM:
        return Instruction(op=op, addr=int(m.group(1)), dst=m.group(2))
    elif op == Op.STORE:
        return Instruction(op=op, src1=m.group(1), addr=int(m.group(2)))
    elif op == Op.MOV:
        return Instruction(op=op, src1=m.group(1), dst=m.group(2))
    elif op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.AND, Op.OR):
        return Instruction(op=op, src1=m.group(1), src2=m.group(2), dst=m.group(3))
    elif op == Op.NOT:
        return Instruction(op=op, src1=m.group(1), dst=m.group(2))
    elif op == Op.CMP:
        return Instruction(op=op, src1=m.group(1), src2=m.group(2))
    elif op in (Op.JMP, Op.JEQ, Op.JNE, Op.JGT):
        return Instruction(op=op, label=m.group(1))
    elif op == Op.HALT:
        return Instruction(op=op)
    raise ValueError(f"Unknown op {op}")
