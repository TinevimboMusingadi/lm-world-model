import torch
import numpy as np
from dataclasses import dataclass

try:
    from nnsight import LanguageModel
except ImportError:
    LanguageModel = None


@dataclass
class PatchResult:
    layer: int
    token_position: int
    logit_diff: float       # change in target token logit vs. clean run
    effect_size: float      # normalised: 1.0 = full recovery, 0.0 = no effect


def run_activation_patch_sweep(
    model_path: str,
    clean_prompt: str,
    corrupted_prompt: str,
    target_token: str,           # the correct output token e.g. "8"
    n_layers: int,
    n_tokens: int
) -> np.ndarray:
    if LanguageModel is None:
        return np.zeros((n_layers, n_tokens))
        
    lm = LanguageModel(model_path, device_map="auto")
    tokenizer = lm.tokenizer
    target_id = tokenizer(target_token, return_tensors="pt").input_ids[0, -1]

    # Step 1: Get clean run logit for target token
    with lm.trace(clean_prompt, scan=False, validate=False):
        clean_logits = lm.lm_head.output.save()
    clean_target_logit = clean_logits.value[0, -1, target_id].item()

    # Step 2: Get corrupted run — cache all layer activations
    corrupted_cache = {}
    with lm.trace(corrupted_prompt, scan=False, validate=False):
        for layer_idx in range(n_layers):
            corrupted_cache[layer_idx] = lm.model.layers[layer_idx].output[0].save()
    corrupted_cache = {k: v.value for k, v in corrupted_cache.items()}

    # Get corrupted logit (baseline for effect normalisation)
    with lm.trace(corrupted_prompt, scan=False, validate=False):
        corrupted_logits = lm.lm_head.output.save()
    corrupted_target_logit = corrupted_logits.value[0, -1, target_id].item()
    baseline_diff = clean_target_logit - corrupted_target_logit

    # Step 3: Sweep — patch one (layer, pos) at a time
    effect_matrix = np.zeros((n_layers, n_tokens))

    for layer_idx in range(n_layers):
        for pos in range(n_tokens):
            with lm.trace(clean_prompt, scan=False, validate=False):
                lm.model.layers[layer_idx].output[0][:, pos, :] = \
                    corrupted_cache[layer_idx][:, pos, :]
                patched_logits = lm.lm_head.output.save()

            patched_logit = patched_logits.value[0, -1, target_id].item()
            logit_diff = clean_target_logit - patched_logit

            effect_matrix[layer_idx, pos] = logit_diff / (baseline_diff + 1e-8)

    return effect_matrix


def plot_patching_heatmap(effect_matrix: np.ndarray, token_labels: list[str],
                          title: str = "Activation Patching Heatmap") -> None:
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Heatmap(
        z=effect_matrix,
        x=token_labels,
        y=[f"L{i}" for i in range(effect_matrix.shape[0])],
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
    ))
    fig.update_layout(title=title, xaxis_title="Token Position", yaxis_title="Layer")
    fig.show()
