"""Pruebas del contrato: registros artificiales NO representan datos de Lending Club."""
import numpy as np
import pandas as pd
import pytest

from src.features import RAW_NEEDED, G1_NUM, G1_CAT, G2_NUM, G2_CAT
from src.data import prepare_parquet_parts
from src.train import train_and_predict


def test_prepare_rejects_unvalidated_all_final(tmp_path):
    with pytest.raises(ValueError, match='inválido'):
        prepare_parquet_parts(tmp_path/'not-used.csv',tmp_path/'output',mode='all_final')


def test_prepare_only_safe_columns(monkeypatch, tmp_path):
    data = {key: [np.nan] * 4 for key in RAW_NEEDED}
    data.update({
        'id':['1','2','3','4'],
        'issue_d':['Jan-2012','Feb-2012','Apr-2014','Jan-2018'],
        'loan_status':['Fully Paid','Charged Off','Default','Current'],
        'term':['36 months']*4,
        'emp_length':['1 year']*4,
        'earliest_cr_line':['Jan-2000']*4,
        'fico_range_low':[660]*4,
        'fico_range_high':[664]*4,
        'annual_inc':[50000]*4,
    })
    # En el CSV de prueba se agrega intencionalmente una columna con leakage.
    data['recoveries']=[0,300,100,0]
    source=tmp_path/'synthetic.csv.gz'
    pd.DataFrame(data).to_csv(source,index=False,compression='gzip')
    collected=[]
    def check_export(self, dest, index=False):
        collected.append((self.copy(),dest))
    monkeypatch.setattr(pd.DataFrame,'to_parquet',check_export)
    manifest=prepare_parquet_parts(source,tmp_path/'out',mode='legacy',chunksize=2)
    assert manifest['raw_rows_scanned']==4
    assert manifest['rows_exported']==3
    out=pd.concat([frame for frame,_ in collected],ignore_index=True)
    assert out['target'].tolist()==[0,1,1]
    assert 'recoveries' not in out
    assert 'loan_status' not in out
    assert 'id' in out.columns
    assert len(collected)==2


def test_logistic_train_valid_split():
    rng=np.random.default_rng(42)
    n=64
    df=pd.DataFrame(index=range(n))
    df['id']=[str(i) for i in range(n)]
    df['issue_d']=pd.to_datetime(['2012-01-01']*48+['2014-01-01']*16)
    df['target']=np.array([0,1]*32)
    for col in G1_NUM + G2_NUM:
        df[col]=rng.normal(size=n)
    for col in G1_CAT + G2_CAT:
        df[col]=['A','B']*(n//2)
    p=train_and_predict(df.iloc[:48],df.iloc[48:],include_g2=False)
    assert p.shape==(16,)
    assert np.all((p>=0)&(p<=1))
    p2=train_and_predict(df.iloc[:48],df.iloc[48:],include_g2=True)
    assert p2.shape==(16,)
    with pytest.raises(ValueError,match='IDs comunes'):
        train_and_predict(df.iloc[:48],df.iloc[40:])
