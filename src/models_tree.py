"""Árbol y bosque para el contrato v4, con preparación ajustada solo en train."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.features import make_preprocessor


def make_decision_tree(include_g2=False, **params):
    cfg = {'random_state': RANDOM_STATE}
    cfg.update(params)
    return Pipeline([
        ('preprocess', make_preprocessor(include_g2)),
        ('model', DecisionTreeClassifier(**cfg)),
    ])


def make_random_forest(include_g2=False, **params):
    cfg = {'random_state': RANDOM_STATE, 'n_jobs': -1}
    cfg.update(params)
    return Pipeline([
        ('preprocess', make_preprocessor(include_g2)),
        ('model', RandomForestClassifier(**cfg)),
    ])


def top_decile_capture(y_true, scores, ids, fraction=0.1):
    """Cuenta impagos entre los k scores mayores, desempate estable por ID."""
    y = np.asarray(y_true)
    p = np.asarray(scores, dtype=float)
    keys = np.asarray(ids).astype(str)
    if (y.ndim != 1 or p.ndim != 1 or keys.ndim != 1 or
            not len(y) or len(y) != len(p) or len(y) != len(keys) or
            not set(np.unique(y)).issubset({0, 1}) or y.sum() == 0 or
            not np.isfinite(p).all() or not 0 < fraction <= 1):
        raise ValueError('Etiquetas, scores, IDs o fracción inválidos.')
    k = max(1, int(round(len(y) * fraction)))
    order = np.lexsort((keys, -p))[:k]
    hits = int(y[order].sum())
    return {
        'fraction': float(fraction),
        'n_selected': k,
        'positives_captured': hits,
        'capture': float(hits / y.sum()),
    }
