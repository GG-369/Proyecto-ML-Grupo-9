import numpy as np
import pandas as pd
from scripts.compare_training_windows_v41 import compare_windows
from src.features import G1_NUM, G1_CAT, G2_NUM, G2_CAT


def make_tiny_frame():
    n=16
    rng=np.random.default_rng(31)
    d={ 'id':[f'loan-{i}' for i in range(n)],
        'issue_d':pd.to_datetime(['2011-01-01']*4+['2012-01-01']*4 +['2013-01-01']*4+['2014-07-01']*4),
        'target':np.array([0,1]*8,dtype=np.int8),
        'term':[36]*n,
    }
    for c in G1_NUM+G2_NUM:
        if c not in d: d[c]=rng.normal(size=n)
    for c in G1_CAT+G2_CAT:
        d[c]=['A','B']*8
    return pd.DataFrame(d)


def test_ablation_uses_only_valid_and_different_training_windows():
    frame=make_tiny_frame()
    result=compare_windows(frame.iloc[:12].copy(),frame.iloc[12:].copy())
    assert result['cohorts']['full_2007_2014h1']['train_n']==12
    assert result['cohorts']['recent_2012_2014h1']['train_n']==8
    assert result['valid_n']==4
    assert 'logreg_g1_g2' in result['comparison']
