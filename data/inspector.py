import argparse
import jsonlines
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    args = parser.parse_args()

    total = 0
    splits = Counter()
    errors = 0
    timeouts = 0
    complexities = Counter()

    with jsonlines.open(args.file) as reader:
        for r in reader:
            total += 1
            splits[r["split"]] += 1
            if r["error"]: errors += 1
            if r["timed_out"]: timeouts += 1
            complexities[r["complexity"]] += 1
    
    print(f"Total records:   {total}")
    print(f"Split counts:    " + " ".join(f"{k}={v}" for k, v in splits.items()))
    print(f"Error rate:      {errors / max(total, 1) * 100:.1f}%")
    print(f"Timed out:       {timeouts / max(total, 1) * 100:.1f}%")
    print(f"Complexity dist: " + " ".join(f"{k}={v / max(total, 1) * 100:.1f}%" for k, v in complexities.items()))

if __name__ == "__main__":
    main()
