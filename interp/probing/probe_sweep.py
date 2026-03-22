from .probe_dataset import build_probe_dataset, ProbeTarget
from .linear_probe import train_regression_probe, train_classification_probe, ProbeResult
import pandas as pd
import plotly.express as px

REGRESSION_TARGETS = ["R1", "R2", "R3", "R4", "PC", "MEM0"]
CLASSIFICATION_TARGETS = ["FLAG", "opcode"]

def sweep_all_layers(
    activation_cache: dict,
    records: list[dict],
    n_layers: int,
    output_csv: str = "interp/results/probe_curves/sweep.csv"
) -> pd.DataFrame:
    results = []

    for layer in range(n_layers):
        print(f"Probing layer {layer}/{n_layers-1}...")
        for target in REGRESSION_TARGETS:
            X, y = build_probe_dataset(activation_cache, records, target, layer)
            if len(y) < 50:
                continue
            r = train_regression_probe(X, y, layer, target)
            results.append(r)

        for target in CLASSIFICATION_TARGETS:
            X, y = build_probe_dataset(activation_cache, records, target, layer)
            if len(y) < 50:
                continue
            r = train_classification_probe(X, y, layer, target)
            results.append(r)

    df = pd.DataFrame([vars(r) for r in results])
    df.to_csv(output_csv, index=False)
    return df

def plot_probe_curves(df: pd.DataFrame, metric: str = "r2_score") -> None:
    fig = px.line(
        df[df["probe_type"] == "regression"],
        x="layer", y=metric, color="target",
        title="Linear Probe Accuracy by Layer (Regression Targets)",
        labels={"r2_score": "R² Score", "layer": "Transformer Layer"},
    )
    fig.show()
