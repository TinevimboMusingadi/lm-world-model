from .tracer import ExecutionTrace

def serialise_trace(trace: ExecutionTrace) -> str:
    """
    Convert an ExecutionTrace into a multi-line XML token string.
    Implements fixed-width formatting for variables to prevent tokenization artifacts.
    [line=02] var_1=00005
    """
    lines = []
    for step in trace.steps:
        if step.event == "line":
            # Fixed width format e.g. padding to 5 digits, preserving sign
            def format_val(v):
                if isinstance(v, int):
                    return f"{v:05d}"
                return str(v)
            
            state_str = " ".join(f"{k}={format_val(v)}" for k, v in sorted(step.locals_snapshot.items()))
            lines.append(f"[line={step.line_no:02d}] {state_str}".strip())
    return "\n".join(lines)


def build_record(code: str, trace: ExecutionTrace, program_id: str) -> dict:
    """
    Build the full training record dict from a code string and its trace.
    Matches data/schemas.py TrainingRecord.
    """
    return {
        "program_id": program_id,
        "phase": "phase1",
        "code": code,
        "instruction_set_description": "",       # not used in phase 1
        "execution_trace": serialise_trace(trace),
        "output": trace.output,
        "error": trace.error,
        "timed_out": trace.timed_out,
        "complexity": -1,                        # set by caller
        "split": "",                             # set by splitter
    }
