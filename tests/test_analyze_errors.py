"""La unión por ID protege el análisis de segmentos ante desorden de filas."""
import pandas as pd
import pytest

from scripts.analyze_errors import segment_report


def test_segments_join_by_id_and_check_target(tmp_path):
    predictions = tmp_path / 'pred.csv'
    valid = tmp_path / 'valid.parquet'
    pd.DataFrame({'id': ['1', '2', '3', '4'], 'target': [1, 0, 1, 0],
                  'score_random_forest_g1': [.9, .7, .2, .1]}).to_csv(predictions, index=False)
    frame = pd.DataFrame({'id': ['4', '2', '1', '3'], 'target': [0, 0, 1, 1],
                          'fico_avg': [750, 700, 650, 670],
                          'annual_inc': [110000, 60000, 30000, 50000]})
    frame.to_parquet(valid)
    result = segment_report(predictions, valid)
    assert result['n'] == 4
    assert sum(r['false_negatives'] for r in result['segments']['fico_band']) == 1
    frame.loc[0, 'target'] = 1
    frame.to_parquet(valid)
    with pytest.raises(ValueError, match='no coinciden'):
        segment_report(predictions, valid)
