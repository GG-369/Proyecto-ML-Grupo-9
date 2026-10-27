"""Baseline validación temporal. No existe evaluación de test en este módulo."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.config import RANDOM_STATE
from src.features import make_preprocessor, select_feature_columns


def make_baseline(include_g2=False):
    return Pipeline([
        ('preprocess', make_preprocessor(include_g2)),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced',
                                     random_state=RANDOM_STATE)),
    ])


def train_and_predict(train, valid, include_g2=False):
    nums, cats = select_feature_columns(include_g2)
    cols = nums + cats
    if train.empty or valid.empty:
        raise ValueError('Split de train/valid vacío.')
    if set(train['id'].dropna().astype(str)) & set(valid['id'].dropna().astype(str)):
        raise ValueError('Se encontraron IDs comunes entre train y valid.')
    if train['issue_d'].max() >= valid['issue_d'].min():
        raise ValueError('La partición no es estrictamente temporal.')
    pipeline = make_baseline(include_g2)
    pipeline.fit(train[cols], train['target'])
    return pipeline.predict_proba(valid[cols])[:, 1]
