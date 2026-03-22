import argparse
import jsonlines
import os
from pathlib import Path
from tqdm import tqdm
from .program_generator import generate_program
from .assembler import assemble
from .simulator import simulate
from .serialiser import serialise_trace, format_output
from .isa import ISA_DESCRIPTIONS, Op
from data.splitter import assign_split

TOTAL = 82_000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/splits/phase2_raw.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=TOTAL)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    holdouts = {frozenset({Op.ADD, Op.JMP}), frozenset({Op.MUL, Op.MOV})}

    with jsonlines.open(args.output, mode="w") as writer:
        for i in tqdm(range(args.n), desc="Generating Phase II"):
            complexity = [1]*5 + [2]*3 + [3]*2
            c = complexity[i % 10]
            spec = generate_program(c, seed=args.seed + i, holdout_combos=holdouts)
            try:
                prog, labels = assemble(spec.source)
                trace = simulate(prog, labels)
            except Exception as e:
                # skip malformed syntax
                continue

            if trace.error or trace.timed_out:
                continue
                
            isa_desc = "\n".join([f"{k.name}: {v}" for k, v in ISA_DESCRIPTIONS.items()])

            record = {
                "program_id": f"p2_{i:07d}",
                "phase": "phase2",
                "code": spec.source,
                "instruction_set_description": isa_desc,
                "execution_trace": serialise_trace(trace),
                "output": format_output(trace.steps[-1]),
                "error": trace.error,
                "timed_out": trace.timed_out,
                "complexity": c,
                "split": assign_split(i, args.seed)
            }
            writer.write(record)

if __name__ == "__main__":
    main()
