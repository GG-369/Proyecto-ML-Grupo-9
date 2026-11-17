"""Prueba end-to-end en minidata sintética; NO resultados reales."""
import json
import numpy as np
import pandas as pd
from src.features import G1_NUM, G1_CAT, G2_NUM, G2_CAT
import src.splits as splits
from scripts.train_baseline_v4 import run


def test_freeze_and_baseline_do_not_export_test(monkeypatch,tmp_path):
    # Se guardan exactamente 4 train, 2 valid, 2 test; test no debe exportarse.
    monkeypatch.setattr(splits,'EXPECTED_TOTAL',8)
    monkeypatch.setattr(splits,'EXPECTED_COUNTS',{'train':4,'valid':2,'test':2})
    # El entorno de pruebas puede no tener pyarrow: emulamos solamente el I/O
    # Parquet con pickle. No altera la lógica de split/modelado.
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', lambda self, path, index=False: self.to_pickle(path))
    monkeypatch.setattr(pd, 'read_parquet', pd.read_pickle)
    processed=tmp_path/'processed'; processed.mkdir()
    rows=8
    raw={'id':[f'id-{i}' for i in range(rows)],
         'issue_d':pd.to_datetime(['2012-01-01','2012-03-01','2013-01-01','2014-06-01',
                                  '2014-07-01','2014-12-01','2015-01-01','2015-06-01']),
         'target':[0,1,0,1,0,1,0,1],
         'term':[36]*rows}
    for c in G1_NUM + G2_NUM:
        if c not in raw: raw[c]=list(np.arange(rows,dtype=float)+1)
    for c in G1_CAT + G2_CAT:
        raw[c]=['A','B']*4
    pd.DataFrame(raw).to_parquet(processed/'part-00000.parquet',index=False)
    (processed/'manifest.json').write_text(json.dumps({'mode':'matured_36','source_file':'synthetic.csv',
                                                      'rows_exported':8}),encoding='utf-8')
    audit=tmp_path/'audit.json'
    audit.write_text(json.dumps({'cohort_totals':{'matured_36_final':{'final':8}}}),encoding='utf-8')
    output=tmp_path/'splits'
    contract=splits.freeze_from_parts(processed,audit,output)
    assert contract['counts']=={'train':4,'valid':2,'test':2}
    assert sorted(p.name for p in output.glob('*.parquet'))==['train.parquet','valid.parquet']
    metrics=run(output,tmp_path/'metrics.json',tmp_path/'preds.csv')
    assert metrics['valid_n']==2
    assert 'logreg_g1' in metrics['results']
    assert not (output/'test.parquet').exists()
