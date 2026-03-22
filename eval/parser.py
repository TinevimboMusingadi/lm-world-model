import re

def extract_output(generation: str) -> str:
    """Pull content between <o> and </o> tags."""
    m = re.search(r"<o>\s*(.*?)\s*</o>", generation, re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_trace(generation: str) -> list[str]:
    """Pull execution trace lines from between <execution_trace> tags."""
    m = re.search(r"<execution_trace>\s*(.*?)\s*</execution_trace>", generation, re.DOTALL)
    if not m:
        return []
    return [line.strip() for line in m.group(1).splitlines() if line.strip()]
