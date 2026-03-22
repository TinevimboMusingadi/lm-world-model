import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

@dataclass
class ProbeResult:
    layer: int
    target: str
    r2_score: float
    accuracy: float
    n_examples: int
    probe_type: str

def train_regression_probe(X: np.ndarray, y: np.ndarray, layer: int,
                           target: str, cv: int = 5) -> ProbeResult:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    probe = Ridge(alpha=1.0)
    scores = cross_val_score(probe, X_scaled, y, cv=cv, scoring="r2")

    return ProbeResult(
        layer=layer,
        target=target,
        r2_score=float(scores.mean()),
        accuracy=float((scores > 0).mean()),
        n_examples=len(y),
        probe_type="regression"
    )

def train_classification_probe(X: np.ndarray, y: np.ndarray, layer: int,
                               target: str, cv: int = 5) -> ProbeResult:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    probe = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(probe, X_scaled, y, cv=cv, scoring="accuracy")

    return ProbeResult(
        layer=layer,
        target=target,
        r2_score=-1.0,
        accuracy=float(scores.mean()),
        n_examples=len(y),
        probe_type="classification"
    )
