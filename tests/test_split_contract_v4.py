"""Tests sintéticos de integridad: NO resultados empíricos de Lending Club."""
import pandas as pd
import pytest
from src.splits import label_windows, verify_partition, EXPECTED_COUNTS


def test_temporal_boundaries():
    df=pd.DataFrame({'issue_d':pd.to_datetime([
        '2013-12-01','2014-06-01','2014-07-01','2014-12-01','2015-01-01',
        '2015-06-01','2015-07-01'])})
    assert label_windows(df).tolist()==['train','train','valid','valid','test','test','outside']


def test_invalid_term_rejected():
    from src.features import G1_NUM, G1_CAT
    f=pd.DataFrame({'id':['1'],'issue_d':[pd.Timestamp('2014-10-01')],
                    'target':[1],'term':[60]})
    for c in G1_NUM + G1_CAT:
        if c not in f: f[c]=0
    with pytest.raises(ValueError,match='36 meses'):
        verify_partition(f,'valid')


def test_duplicate_loan_rejected():
    from src.features import G1_NUM, G1_CAT
    f=pd.DataFrame({'id':['1','1'],'issue_d':[pd.Timestamp('2014-10-01')]*2,
                    'target':[1,0],'term':[36]*2})
    for c in G1_NUM + G1_CAT:
        if c not in f: f[c]=0
    with pytest.raises(ValueError,match='duplicados'):
        verify_partition(f,'valid')


def test_expected_counts_sum_to_cohort():
    assert sum(EXPECTED_COUNTS.values())==456348
