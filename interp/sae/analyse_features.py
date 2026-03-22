import torch
import numpy as np

def find_top_activating_examples(
    sae,
    activation_cache: dict,
    records: list[dict],
    feature_idx: int,
    layer: int,
    top_k: int = 10
) -> list[dict]:
    feature_dir = sae.W_dec[feature_idx]
    scores = []

    for record in records:
        pid = record["program_id"]
        if pid not in activation_cache:
            continue
        acts = activation_cache[pid][layer]
        feature_activations = acts @ feature_dir
        max_activation = feature_activations.max().item()
        max_pos = feature_activations.argmax().item()
        scores.append({
            "program_id": pid,
            "max_activation": max_activation,
            "max_position": max_pos,
            "code": record["code"],
            "trace_step": _get_trace_step(record, max_pos),
        })

    return sorted(scores, key=lambda x: -x["max_activation"])[:top_k]

def _get_trace_step(record, max_pos):
    return record["execution_trace"].splitlines()[min(max_pos, len(record["execution_trace"].splitlines())-1)]
