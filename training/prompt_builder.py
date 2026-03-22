SYSTEM_PROMPT = (
    "You are a program execution engine. "
    "Given a program and its instruction set, you must produce "
    "a complete execution trace showing the machine state after each instruction, "
    "followed by the final output."
)


def build_prompt(record: dict, condition: str) -> str:
    """
    Build the full text prompt for one training record.

    condition:
      "A" - output only (no trace in target)
      "B" - full trace + output
      "C" - masked trace + output (some state tokens replaced with [MASK])
    """
    code_block = f"<code>\n{record['code']}\n</code>"
    isa_desc = record.get("instruction_set_description", "")
    isa_block = (
        f"<instruction_set>\n{isa_desc}\n</instruction_set>"
        if isa_desc.strip()
        else ""
    )

    if condition == "A":
        target = f"<o>\n{record['output']}\n</o>"
    elif condition == "B":
        target = (
            f"<execution_trace>\n{record['execution_trace']}\n</execution_trace>\n"
            f"<o>\n{record['output']}\n</o>"
        )
    elif condition == "C":
        masked = _mask_trace(record['execution_trace'])
        target = (
            f"<execution_trace>\n{masked}\n</execution_trace>\n"
            f"<o>\n{record['output']}\n</o>"
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    parts = [code_block]
    if isa_block:
        parts.append(isa_block)
    parts.append(target)

    return "\n\n".join(parts)


def _mask_trace(trace: str, mask_prob: float = 0.3) -> str:
    import random
    lines = trace.split("\n")
    masked = []
    for line in lines:
        if random.random() < mask_prob:
            masked.append("[MASK]")
        else:
            masked.append(line)
    return "\n".join(masked)
