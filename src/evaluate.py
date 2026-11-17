"""Métricas binarias correctas sobre validación; nunca consulta el test.

`positive_rate` se preserva como alias heredado de `target_prevalence`.
`predicted_positive_rate` indica la proporción de predicciones >= umbral.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    confusion_matrix, precision_score, recall_score,
)


def evaluate_binary(y_true, probabilities, threshold=0.5):
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError('Las etiquetas y puntuaciones deben ser vectores de igual longitud, no vacíos.')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('El conjunto debe contener ambas clases binarias (0 y 1).')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Las puntuaciones deben ser finitas y estar entre 0 y 1.')
    if not (0 < threshold < 1):
        raise ValueError('El umbral debe estar estrictamente entre 0 y 1.')
    pred = (p >= threshold).astype(int)
    # mean convierte a flotante sin overflow de numpy.int8 con sum de Python.
    prevalence = float(np.mean(y.astype(np.int64)))
    return {
        'roc_auc': float(roc_auc_score(y, p)),
        'pr_auc': float(average_precision_score(y, p)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'threshold': float(threshold),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        'positive_rate': prevalence,  # compatibilidad anterior: etiqueta real
        'target_prevalence': prevalence,
        'predicted_positive_rate': float(pred.mean()),
        'n': int(len(y)),
    }
