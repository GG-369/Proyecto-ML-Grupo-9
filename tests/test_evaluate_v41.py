import numpy as np
import pytest
from src.evaluate import evaluate_binary


def test_no_overflow_with_numpy_int8_labels():
    y = np.array([1]*12_800 + [0]*77_815, dtype=np.int8)
    score = y.astype(float)*0.7 + 0.1
    result = evaluate_binary(y, score)
    assert result['target_prevalence'] == pytest.approx(12_800/90_615)
    assert result['positive_rate'] == pytest.approx(12_800/90_615)
    assert result['predicted_positive_rate'] == pytest.approx(12_800/90_615)


def test_target_prevalence_distinct_from_prediction_rate():
    y = np.array([0,0,0,1,1], dtype=np.int8)
    p = np.array([0.8,0.6,0.2,0.2,0.9])
    r = evaluate_binary(y,p)
    assert r['target_prevalence'] == pytest.approx(0.4)
    assert r['predicted_positive_rate'] == pytest.approx(0.6)
    assert r['precision'] == pytest.approx(1/3)
    assert r['recall'] == pytest.approx(.5)


def test_avoids_invalid_probabilities():
    for p in ([0.5,float('nan')], [-0.1,0.8], [0.4,1.1]):
        with pytest.raises(ValueError,match='puntuaciones'):
            evaluate_binary(np.array([0,1]), np.array(p))
