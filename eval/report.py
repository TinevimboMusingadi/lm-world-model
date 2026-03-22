from .metrics import EvalResult
import pandas as pd
from pathlib import Path

def print_report(result: EvalResult, split: str, condition: str, temperature: float = 0.0):
    print(f"--- Eval Report ---")
    print(f"Split: {split} | Condition: {condition} | Temperature: {temperature}")
    print(f"Programs Evaluated: {result.n_programs}")
    print(f"Output Accuracy: {result.output_accuracy * 100:.2f}%")
    print(f"Trace Accuracy: {result.trace_accuracy * 100:.2f}%")
    print(f"First Error Step: {result.first_error_step:.2f}")
    print(f"Levenshtein Distance: {result.levenshtein_distance:.2f}")
    print("-------------------")

def save_csv(result: EvalResult, path: Path, split: str, condition: str, temperature: float = 0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    file_exists = os.path.isfile(path)
    
    with open(path, "a") as f:
        if not file_exists:
            f.write("split,condition,temperature,output_accuracy,trace_accuracy,first_error_step,levenshtein_distance,n_programs\n")
        f.write(f"{split},{condition},{temperature},{result.output_accuracy},{result.trace_accuracy},{result.first_error_step},{result.levenshtein_distance},{result.n_programs}\n")
