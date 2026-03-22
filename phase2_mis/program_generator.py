import random
from dataclasses import dataclass
from .isa import Op

# Which instruction subsets are available at each level
LEVEL_OPS = {
    1: [Op.LOAD, Op.ADD, Op.SUB, Op.STORE],
    2: [Op.LOAD, Op.ADD, Op.SUB, Op.MUL, Op.MOV, Op.STORE, Op.CMP, Op.JEQ, Op.JNE],
    3: [Op.LOAD, Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOV,
        Op.STORE, Op.LOADM, Op.CMP, Op.JEQ, Op.JNE, Op.JGT, Op.JMP],
}

REGISTERS = ["R1", "R2", "R3", "R4"]

@dataclass
class GeneratedProgram:
    source: str
    complexity: int
    seed: int
    ood_holdout: bool = False    # True if this program uses a held-out instruction combo


def generate_program(complexity: int, seed: int,
                     holdout_combos: set | None = None) -> GeneratedProgram:
    """
    Generate a random MIS program at the given complexity.
    Includes Rigorous Controls: 
      - Data Dependency Chains (new instructions use populated registers)
      - Constant PC Trap (jumps injected)
    """
    rng = random.Random(seed)
    available_ops = LEVEL_OPS[complexity]
    n_instructions = rng.randint(3, 5 + complexity * 4)

    lines = []
    used_ops = set()
    populated_registers = []

    # Always start with LOAD instructions to populate registers (Canonicalizing State)
    n_loads = rng.randint(2, min(4, n_instructions))
    for _ in range(n_loads):
        reg = rng.choice(REGISTERS)
        # Wide initial bounds for robust math checks
        val = rng.randint(-128, 255)
        lines.append(f"LOAD #{val} -> {reg}")
        used_ops.add(Op.LOAD)
        if reg not in populated_registers:
            populated_registers.append(reg)

    label_count = 0

    # Fill remaining instructions from available_ops, enforcing data dependency
    for i in range(n_instructions - n_loads):
        op = rng.choice([o for o in available_ops if o != Op.LOAD])
        
        # Enforce branches to break Constant PC Trap
        if complexity >= 2 and rng.random() > 0.8:
            lines.append(f"CMP {rng.choice(populated_registers)} {rng.choice(populated_registers)}")
            lines.append(f"JGT L{label_count}")
            lines.append(f"ADD {rng.choice(populated_registers)} {rng.choice(populated_registers)} -> {rng.choice(REGISTERS)}")
            lines.append(f"L{label_count}:")
            label_count += 1
            used_ops.update([Op.CMP, Op.JGT, Op.ADD])
            continue

        if op in [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.AND, Op.OR]:
            # Use populated registers as sources
            src1 = rng.choice(populated_registers)
            src2 = rng.choice(populated_registers)
            dst = rng.choice(REGISTERS)
            lines.append(f"{op.name} {src1} {src2} -> {dst}")
            if dst not in populated_registers:
                populated_registers.append(dst)
        elif op == Op.NOT:
            src = rng.choice(populated_registers)
            dst = rng.choice(REGISTERS)
            lines.append(f"{op.name} {src} -> {dst}")
            if dst not in populated_registers:
                populated_registers.append(dst)
        elif op == Op.MOV:
            src = rng.choice(populated_registers)
            dst = rng.choice(REGISTERS)
            lines.append(f"{op.name} {src} -> {dst}")
            if dst not in populated_registers:
                populated_registers.append(dst)
        elif op == Op.STORE:
            src = rng.choice(populated_registers)
            addr = rng.randint(0, 7)
            lines.append(f"{op.name} {src} -> MEM[{addr}]")
        elif op == Op.LOADM:
            addr = rng.randint(0, 7)
            dst = rng.choice(REGISTERS)
            lines.append(f"{op.name} MEM[{addr}] -> {dst}")
            if dst not in populated_registers:
                populated_registers.append(dst)
        else:
            # Skip unhandled complex ops for now
            pass
            
        used_ops.add(op)

    lines.append("HALT")
    source = "\n".join(f"  {l}" if not l.endswith(":") else l for l in lines)

    # Check OOD holdout
    ood = False
    if holdout_combos:
        for combo in holdout_combos:
            if combo.issubset(used_ops):
                ood = True
                break

    return GeneratedProgram(source=source, complexity=complexity,
                            seed=seed, ood_holdout=ood)
