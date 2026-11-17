"""Humo para la integración real del script de Nicolás (datos sintéticos)."""
import numpy as np
import pandas as pd
import pytest

from src.features import G1_CAT, G1_NUM, G2_CAT, G2_NUM
from src.models_tree import make_decision_tree, make_random_forest, top_decile_capture


@pytest.mark.parametrize('factory', [make_decision_tree, make_random_forest])
@pytest.mark.parametrize('with_g2', [False, True])
def test_tree_pipelines_train_and_predict(factory, with_g2):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({col: rng.normal(size=30) for col in G1_NUM + G2_NUM})
    for col in G1_CAT + G2_CAT:
        frame[col] = ['A', 'B'] * 15
    frame.loc[0, 'annual_inc'] = np.nan
    frame.loc[1, 'purpose'] = None
    model = factory(with_g2, min_samples_leaf=2)
    columns = G1_NUM + G1_CAT + ((G2_NUM + G2_CAT) if with_g2 else [])
    model.fit(frame[columns], np.array([0, 1] * 15))
    future = frame.iloc[:2][columns].copy()
    future.loc[0, 'purpose'] = 'nueva_categoria'
    probs = model.predict_proba(future)[:, 1]
    assert probs.shape == (2,)
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()


def test_top_decile_counts_and_ties_by_id():
    y = [0] * 9 + [1]
    scores = [0.8, 0.8] + [0.1] * 8
    ids = ['2', '1'] + [str(i) for i in range(3, 11)]
    result = top_decile_capture(y, scores, ids, fraction=0.2)
    assert result['n_selected'] == 2
    assert result['positives_captured'] == 0
    assert result['capture'] == 0
    with pytest.raises(ValueError):
        top_decile_capture([0, 0], [0.4, 0.2], ['1', '2'])
