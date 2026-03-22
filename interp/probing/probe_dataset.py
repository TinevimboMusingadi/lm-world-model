import torch
import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class ProbeExample:
    activation: np.ndarray
    label: float | int
    program_id: str
    step_idx: int
    token_pos: int


ProbeTarget = Literal["R1", "R2", "R3", "R4", "PC", "FLAG", "MEM0", "opcode"]


def build_probe_dataset(
    activation_cache: dict,
    records: list[dict],
    target: ProbeTarget,
    layer: int,
    token_type: Literal["state_token", "instruction_token", "last_token"] = "state_token"
) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []

    for record in records:
        pid = record["program_id"]
        if pid not in activation_cache:
            continue

        acts = activation_cache[pid]
        layer_acts = acts[layer]

        trace_steps = _parse_trace(record["execution_trace"])

        for step in trace_steps:
            tok_pos = _find_token_position(step, record, token_type)
            if tok_pos is None:
                continue

            activation_vec = layer_acts[tok_pos].cpu().numpy()
            label = _extract_label(step, target)

            if label is not None:
                X_list.append(activation_vec)
                y_list.append(label)

    return np.array(X_list), np.array(y_list)


def _find_token_position(step, record, token_type):
    return 0 # Dummy implementation

def _parse_trace(trace_str: str) -> list[dict]:
    import re
    steps = []
    for line in trace_str.strip().splitlines():
        step = {}
        for m in re.finditer(r'\[(\w+)=([^\]]+)\]', line):
            key, val = m.group(1), m.group(2)
            step[key] = val
        if step:
            steps.append(step)
    return steps


def _extract_label(step: dict, target: ProbeTarget) -> float | None:
    if target in ("R1", "R2", "R3", "R4", "PC"):
        val = step.get(target)
        return float(val) if val is not None else None
    elif target == "FLAG":
        mapping = {"N": 0, "EQ": 1, "LT": 2, "GT": 3}
        return mapping.get(step.get("FLAG"))
    elif target.startswith("MEM"):
        mem_str = step.get("MEM", "")
        idx = int(target[3:])
        if len(mem_str) >= (idx + 1) * 2:
            return float(int(mem_str[idx*2:(idx+1)*2]))
    return None
