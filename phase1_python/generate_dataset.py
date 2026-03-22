import argparse
import jsonlines
import os
from pathlib import Path
from tqdm import tqdm
from .program_generator import generate_program
from .tracer import trace_program
from .serialiser import build_record
from data.splitter import assign_split

TOTAL = 82_000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/splits/phase1_raw.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=TOTAL)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(args.output, mode="w") as writer:
        for i in tqdm(range(args.n), desc="Generating Phase I"):
            complexity = [1]*5 + [2]*3 + [3]*2
            c = complexity[i % 10]
            spec = generate_program(c, seed=args.seed + i)
            trace = trace_program(spec.source)

            if trace.error or trace.timed_out:
                continue

            record = build_record(spec.source, trace, program_id=f"p1_{i:07d}")
            record["complexity"] = c
            record["split"] = assign_split(i, args.seed)
            writer.write(record)

if __name__ == "__main__":
    main()
