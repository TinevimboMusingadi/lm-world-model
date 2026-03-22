import torch
import os
from pathlib import Path
import pandas as pd


def track_probes_across_checkpoints(
    checkpoint_dir: str,
    activation_cache_fn,
    records: list[dict],
    targets: list[str] = ["R1", "R2", "R3", "PC", "FLAG"],
    peak_layer: int = 10
) -> pd.DataFrame:
    from interp.probing.probe_dataset import build_probe_dataset
    from interp.probing.linear_probe import train_regression_probe, train_classification_probe

    checkpoints = sorted(Path(checkpoint_dir).glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))

    rows = []
    for ckpt in checkpoints:
        step = int(ckpt.name.split("-")[1])
        print(f"Processing checkpoint {step}...")

        cache = activation_cache_fn(str(ckpt))

        for target in targets:
            X, y = build_probe_dataset(cache, records, target, layer=peak_layer)
            if len(y) < 30:
                continue

            if target in ("FLAG",):
                result = train_classification_probe(X, y, peak_layer, target)
                score = result.accuracy
            else:
                result = train_regression_probe(X, y, peak_layer, target)
                score = result.r2_score

            rows.append({"step": step, "target": target, "score": score})

    df = pd.DataFrame(rows)
    return df


def plot_developmental_curves(df: pd.DataFrame) -> None:
    import plotly.express as px
    fig = px.line(df, x="step", y="score", color="target",
                  title="Probe Accuracy During Training (Developmental Interpretability)",
                  labels={"step": "Training Step", "score": "Probe R² / Accuracy"})
    fig.show()
