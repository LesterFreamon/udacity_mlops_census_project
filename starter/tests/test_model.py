import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score

from src.ml.model import compute_model_metrics


def test_compute_model_metrics():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    preds = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == precision_score(y, preds, zero_division=1)
    assert recall == recall_score(y, preds, zero_division=1)
    assert fbeta == fbeta_score(y, preds, beta=1, zero_division=1)
