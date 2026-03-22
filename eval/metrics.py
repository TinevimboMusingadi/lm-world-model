from dataclasses import dataclass

@dataclass
class EvalResult:
    output_accuracy: float      # OA: exact match on <o>
    trace_accuracy: float       # TA: fraction of trace tokens correct
    first_error_step: float     # FES: mean step index of first divergence
    levenshtein_distance: float # MEAN Levenshtein distance for state fidelity
    n_programs: int
    n_correct_output: int


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_metrics(predictions: list[dict], records: list[dict]) -> EvalResult:
    """
    predictions: list of {"generated": str} dicts
    records: list of TrainingRecord dicts (ground truth)
    """
    from .parser import extract_output, extract_trace

    correct_output = 0
    trace_token_hits = []
    first_error_steps = []
    levenshtein_dists = []

    for pred, record in zip(predictions, records):
        gen_output = extract_output(pred["generated"])
        if gen_output == record["output"].strip():
            correct_output += 1

        gen_trace = extract_trace(pred["generated"])
        gt_trace  = [l.strip() for l in record["execution_trace"].splitlines() if l.strip()]

        # Token-level trace accuracy
        hits = sum(g == p for g, p in zip(gt_trace, gen_trace[:len(gt_trace)]))
        total = max(len(gt_trace), 1)
        trace_token_hits.append(hits / total)

        # First error step
        fes = len(gt_trace)    # default: no error
        for i, (g, p) in enumerate(zip(gt_trace, gen_trace[:len(gt_trace)])):
            if g != p:
                fes = i
                break
        first_error_steps.append(fes)

        # Levenshtein distance for near-miss tracing (State Fidelity)
        gen_trace_str = "\n".join(gen_trace)
        gt_trace_str = "\n".join(gt_trace)
        dist = levenshtein(gen_trace_str, gt_trace_str)
        levenshtein_dists.append(dist)

    n = len(records)
    return EvalResult(
        output_accuracy=correct_output / n if n > 0 else 0,
        trace_accuracy=sum(trace_token_hits) / n if n > 0 else 0,
        first_error_step=sum(first_error_steps) / n if n > 0 else 0,
        levenshtein_distance=sum(levenshtein_dists) / n if n > 0 else 0,
        n_programs=n,
        n_correct_output=correct_output,
    )
