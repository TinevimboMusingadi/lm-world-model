import sys
import copy
from dataclasses import dataclass, field
from typing import Any

MAX_STEPS = 200
SAFE_BUILTINS = {"range", "len", "abs", "min", "max", "print", "int", "float", "bool", "str"}

@dataclass
class TraceStep:
    line_no: int
    event: str                    # "line" | "return"
    locals_snapshot: dict         # {var_name: value} - primitives only
    return_value: Any = None      # set on "return" events

@dataclass
class ExecutionTrace:
    steps: list[TraceStep] = field(default_factory=list)
    output: str = ""              # captured stdout
    error: str = ""               # set if execution raised an exception
    timed_out: bool = False

def trace_program(code: str) -> ExecutionTrace:
    """
    Execute `code` string under sys.settrace.
    Returns a full ExecutionTrace, never raises.
    """
    result = ExecutionTrace()
    captured_output = []
    step_count = [0]

    def _capture_print(*args, **kwargs):
        captured_output.append(" ".join(str(a) for a in args))

    safe_globals = {
        "__builtins__": {k: __builtins__[k] for k in SAFE_BUILTINS if k in dir(__builtins__)},
        "print": _capture_print,
    }

    def _tracer(frame, event, arg):
        if step_count[0] >= MAX_STEPS:
            result.timed_out = True
            raise StopIteration("MAX_STEPS exceeded")

        if event in ("line", "return"):
            step_count[0] += 1
            snapshot = {
                k: v for k, v in frame.f_locals.items()
                if isinstance(v, (int, float, bool, str)) and not k.startswith("_")
            }
            result.steps.append(TraceStep(
                line_no=frame.f_lineno,
                event=event,
                locals_snapshot=copy.deepcopy(snapshot),
                return_value=arg if event == "return" else None,
            ))
        return _tracer

    try:
        sys.settrace(_tracer)
        exec(compile(code, "<program>", "exec"), safe_globals)  # noqa: S102
    except StopIteration:
        pass
    except Exception as e:
        result.error = str(e)
    finally:
        sys.settrace(None)

    result.output = "\n".join(captured_output)
    return result
