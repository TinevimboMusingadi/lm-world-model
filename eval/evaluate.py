import argparse
import jsonlines
from pathlib import Path
from transformers import AutoTokenizer, pipeline
from .metrics import compute_metrics
from .report import print_report, save_csv
from training.prompt_builder import build_prompt, SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test_indist",
                        choices=["test_indist", "test_ood", "test_long", "val"])
    parser.add_argument("--condition", default="B", choices=["A", "B", "C"])
    parser.add_argument("--output", type=Path, default=Path("results/eval_results.csv"))
    parser.add_argument("--temperature", type=float, default=0.0) # For Temperature tests
    args = parser.parse_args()

    # Load records for the requested split
    records = []
    with jsonlines.open(args.data) as reader:
        for r in reader:
            if r["split"] == args.split:
                records.append(r)

    print(f"Evaluating on {len(records)} programs from split '{args.split}'")

    # Build prompts
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    pipe = pipeline("text-generation", model=str(args.checkpoint),
                    tokenizer=tokenizer, device_map="auto",
                    max_new_tokens=512, 
                    temperature=args.temperature, 
                    do_sample=(args.temperature > 0.0))

    predictions = []
    for record in records:
        prompt = f"<s>[INST] {SYSTEM_PROMPT} [/INST]\n"
        prompt += build_prompt(record, condition="A")   # testing gives only code
        out = pipe(prompt, return_full_text=False)[0]["generated_text"]
        predictions.append({"generated": out})

    result = compute_metrics(predictions, records)
    print_report(result, split=args.split, condition=args.condition, temperature=args.temperature)
    save_csv(result, args.output, split=args.split, condition=args.condition, temperature=args.temperature)

if __name__ == "__main__":
    main()
