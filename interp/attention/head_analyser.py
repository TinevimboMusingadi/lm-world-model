import torch
import numpy as np

try:
    from nnsight import LanguageModel
    from nnterp import StandardizedTransformer
except ImportError:
    LanguageModel = None
    StandardizedTransformer = None

def extract_attention_patterns(
    model_path: str,
    prompts: list[str],
    n_layers: int,
    n_heads: int
) -> np.ndarray:
    if StandardizedTransformer is None:
        return np.zeros((n_layers, n_heads, 1, 1))

    st = StandardizedTransformer(model_path, enable_attention_probs=True)

    all_patterns = []
    for prompt in prompts:
        with st.trace(prompt):
            patterns = []
            for layer_idx in range(n_layers):
                attn = st.attention_probabilities[layer_idx].save()
                patterns.append(attn)

        prompt_patterns = torch.stack([p.value[0] for p in patterns])  # (n_layers, n_heads, n_tokens, n_tokens)
        all_patterns.append(prompt_patterns.cpu().numpy())

    return np.mean(all_patterns, axis=0)


def score_operand_retrieval(attn_patterns: np.ndarray,
                            instr_positions: list[int],
                            src_positions: list[tuple[int, int]]) -> np.ndarray:
    n_layers, n_heads, _, _ = attn_patterns.shape
    scores = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        for head in range(n_heads):
            pattern = attn_patterns[layer, head]   # (n_tokens, n_tokens)
            for instr_pos, (src1_pos, src2_pos) in zip(instr_positions, src_positions):
                score = (pattern[instr_pos, src1_pos] + pattern[instr_pos, src2_pos]) / 2
                scores[layer, head] += score

    return scores / max(1, len(instr_positions))
