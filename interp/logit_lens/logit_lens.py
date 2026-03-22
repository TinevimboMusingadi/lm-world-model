import torch
import numpy as np

try:
    from nnsight import LanguageModel
except ImportError:
    LanguageModel = None

def compute_logit_lens(
    model_path: str,
    prompt: str,
    top_k: int = 5
) -> dict:
    if LanguageModel is None:
        return {"layer_predictions": [], "tokens": [], "n_layers": 0}
        
    lm = LanguageModel(model_path, device_map="auto")
    tokenizer = lm.tokenizer

    tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]
    token_strings = [tokenizer.decode([t]) for t in tokens]

    layer_logits = []

    with lm.trace(prompt):
        for layer_idx in range(lm.config.num_hidden_layers):
            resid = lm.model.layers[layer_idx].output[0]
            normed = lm.model.norm(resid)
            logits = lm.lm_head(normed)
            layer_logits.append(logits.save())

    results = []
    for layer_idx, logit_tensor in enumerate(layer_logits):
        probs = torch.softmax(logit_tensor.value[0], dim=-1)
        top = torch.topk(probs, k=top_k, dim=-1)
        layer_result = []
        for pos in range(len(tokens)):
            top_tokens = [
                (tokenizer.decode([idx.item()]), prob.item())
                for idx, prob in zip(top.indices[pos], top.values[pos])
            ]
            layer_result.append(top_tokens)
        results.append(layer_result)

    return {
        "layer_predictions": results,
        "tokens": token_strings,
        "n_layers": lm.config.num_hidden_layers,
    }
