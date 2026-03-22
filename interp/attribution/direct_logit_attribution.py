import torch
import numpy as np

def compute_dla(
    model,                    # HookedTransformer or nnsight model
    cache: dict,              # activation cache from a forward pass
    target_token_id: int,
    n_layers: int,
    n_heads: int,
    d_model: int
) -> dict:
    W_U = model.lm_head.weight                  # (vocab_size, d_model)
    target_dir = W_U[target_token_id]            # (d_model,)

    attn_contributions = np.zeros((n_layers, n_heads))
    mlp_contributions  = np.zeros(n_layers)

    for layer in range(n_layers):
        z    = cache[f"blocks.{layer}.attn.hook_z"][0, -1]   # (n_heads, d_head)
        W_O  = model.blocks[layer].attn.W_O                  # (n_heads, d_head, d_model)

        for head in range(n_heads):
            head_out = z[head] @ W_O[head]                   # (d_model,)
            contribution = (head_out @ target_dir).item()
            attn_contributions[layer, head] = contribution

        mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, -1]   # (d_model,)
        mlp_contributions[layer] = (mlp_out @ target_dir).item()

    return {
        "attn_heads": attn_contributions,
        "mlp_layers": mlp_contributions,
    }
