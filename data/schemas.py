from pydantic import BaseModel, field_validator
from typing import Literal

class TrainingRecord(BaseModel):
    program_id: str
    phase: Literal["phase1", "phase2"]
    code: str
    instruction_set_description: str     # empty string for phase 1
    execution_trace: str
    output: str
    error: str
    timed_out: bool
    complexity: int                      # 1, 2, or 3
    split: str                           # e.g., "train", "val", "test_indist", "test_ood", "test_long"

    @field_validator("execution_trace")
    @classmethod
    def trace_not_empty(cls, v):
        if not v.strip():
            raise ValueError("execution_trace must not be empty")
        return v

    @field_validator("output")
    @classmethod
    def output_not_empty(cls, v):
        if not v.strip():
            raise ValueError("output must not be empty")
        return v
