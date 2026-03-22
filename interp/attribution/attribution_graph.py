import torch
import numpy as np
import networkx as nx

def build_simplified_attribution_graph(
    model,
    prompt: str,
    target_token_id: int,
    n_layers: int,
    n_heads: int,
    threshold: float = 0.05
) -> nx.DiGraph:
    G = nx.DiGraph()

    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    target_logit = logits[0, -1, target_token_id]
    target_logit.backward()

    G.add_node("output", type="output", token_id=target_token_id)
    for pos, tok in enumerate(tokens[0]):
        G.add_node(f"input_{pos}", type="input", token_id=tok.item())
    for layer in range(n_layers):
        for head in range(n_heads):
            G.add_node(f"attn_{layer}_{head}", type="attention",
                       layer=layer, head=head)
        G.add_node(f"mlp_{layer}", type="mlp", layer=layer)

    W_U = model.W_U
    target_dir = W_U[:, target_token_id]

    for layer in range(n_layers):
        z     = cache[f"blocks.{layer}.attn.hook_z"][0, -1]
        W_O   = model.blocks[layer].attn.W_O
        for head in range(n_heads):
            head_out = z[head] @ W_O[head]
            weight = (head_out @ target_dir).item()
            if abs(weight) > threshold:
                G.add_edge(f"attn_{layer}_{head}", "output", weight=weight)

        mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, -1]
        weight = (mlp_out @ target_dir).item()
        if abs(weight) > threshold:
            G.add_edge(f"mlp_{layer}", "output", weight=weight)

        attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0]
        for head in range(n_heads):
            for pos in range(tokens.shape[1]):
                attn_weight = attn_pattern[head, -1, pos].item()
                if attn_weight > threshold:
                    G.add_edge(f"input_{pos}", f"attn_{layer}_{head}",
                               weight=attn_weight)

    return G

def visualise_attribution_graph(G: nx.DiGraph, output_path: str = None) -> None:
    import plotly.graph_objects as go

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = [n for n in G.nodes()]
    node_colors = [
        "red"    if G.nodes[n]["type"] == "output"    else
        "blue"   if G.nodes[n]["type"] == "input"     else
        "green"  if G.nodes[n]["type"] == "attention" else
        "orange" for n in G.nodes()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.5, color="#888")))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                             text=node_labels, marker=dict(color=node_colors, size=10)))
    if output_path:
        fig.write_html(output_path)
    fig.show()
