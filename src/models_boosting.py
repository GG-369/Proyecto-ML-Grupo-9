"""Modelos de boosting (Kris). No lee ni conoce el test.

Usa el contrato de variables de `src.features` (G1 / G1+G2) sin modificarlo.
HistGradientBoosting maneja NaN de forma nativa y categorías con codificación
ordinal, por lo que aquí NO se usa el escalado/one-hot de la regresión logística.
Todo el ajuste (encoder) ocurre dentro del Pipeline, solo con datos de train.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.config import RANDOM_STATE
from src.features import select_feature_columns


def _cats_to_object(frame):
    """Unifica dtypes string/category con pd.NA a object con np.nan."""
    out = pd.DataFrame(frame).copy()
    for c in out.columns:
        out[c] = out[c].astype(object).where(out[c].notna(), np.nan)
    return out


def _nums_to_float(frame):
    """Int64/Float64 nullable (pd.NA) -> float64 con np.nan."""
    return pd.DataFrame(frame).astype('float64')


def make_hgb_preprocessor(include_g2=False):
    num, cat = select_feature_columns(include_g2)
    return ColumnTransformer([
        ('num', FunctionTransformer(_nums_to_float), num),
        ('cat', Pipeline([
            ('to_object', FunctionTransformer(_cats_to_object)),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value',
                                       unknown_value=np.nan,
                                       encoded_missing_value=np.nan)),
        ]), cat),
    ])


def make_hgb(include_g2=False, **params):
    """Pipeline = preprocesamiento + HistGradientBoostingClassifier."""
    num, cat = select_feature_columns(include_g2)
    cat_idx = list(range(len(num), len(num) + len(cat)))  # categóricas van al final
    cfg = dict(random_state=RANDOM_STATE, early_stopping=False)
    cfg.update(params)
    return Pipeline([
        ('preprocess', make_hgb_preprocessor(include_g2)),
        ('model', HistGradientBoostingClassifier(categorical_features=cat_idx, **cfg)),
    ])


def make_gb_classic(include_g2=False, **params):
    """GradientBoosting clásico (lento): imputación + one-hot, como referencia."""
    num, cat = select_feature_columns(include_g2)
    pre = ColumnTransformer([
        ('num', Pipeline([
            ('to_float', FunctionTransformer(_nums_to_float)),
            ('imputer', SimpleImputer(strategy='median')),
        ]), num),
        ('cat', Pipeline([
            ('to_object', FunctionTransformer(_cats_to_object)),
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), cat),
    ])
    cfg = dict(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8,
               random_state=RANDOM_STATE)
    cfg.update(params)
    return Pipeline([('preprocess', pre), ('model', GradientBoostingClassifier(**cfg))])


def sample_configs(n, seed=RANDOM_STATE):
    """Búsqueda aleatoria reproducible. `max_iter` NO se busca: se elige el
    mejor número de iteraciones con staged_predict_proba en la subvalidación."""
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        configs.append({
            'learning_rate': float(np.exp(rng.uniform(np.log(0.02), np.log(0.2)))),
            'max_leaf_nodes': int(rng.choice([8, 15, 31, 63])),
            'max_depth': [None, 4, 6, 8][int(rng.integers(4))],
            'min_samples_leaf': int(rng.choice([20, 50, 100, 200, 500])),
            'l2_regularization': float(rng.choice([0.0, 0.1, 1.0, 10.0])),
            'class_weight': [None, 'balanced'][int(rng.integers(2))],
        })
    return configs


def top_decile_capture(y_true, scores, frac=0.10):
    """Proporción de incumplimientos que caen en el `frac` de scores más altos
    y lift sobre la prevalencia."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-s, kind='mergesort')[:k]
    capture = float(y[top].sum() / y.sum())
    lift = float(y[top].mean() / y.mean())
    return {'frac': frac, 'k': int(k), 'capture': capture, 'lift': lift}
