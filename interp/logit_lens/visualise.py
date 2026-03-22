import plotly.graph_objects as go
import numpy as np

def plot_logit_lens_heatmap(logit_lens_result: dict, target_position: int,
                             title: str = "") -> None:
    layers = range(logit_lens_result["n_layers"])
    preds = logit_lens_result["layer_predictions"]

    top1_tokens  = [preds[l][target_position][0][0] for l in layers]
    top1_probs   = [preds[l][target_position][0][1] for l in layers]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(layers), y=top1_probs,
        mode="lines+markers",
        text=top1_tokens,
        hovertemplate="Layer %{x}<br>Token: %{text}<br>Prob: %{y:.3f}",
    ))
    fig.update_layout(title=title or f"Logit Lens at Position {target_position}",
                      xaxis_title="Layer", yaxis_title="Top-1 Probability")
    fig.show()
